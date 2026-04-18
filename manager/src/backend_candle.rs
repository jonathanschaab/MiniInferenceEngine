use async_trait::async_trait;
use candle_core::{Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use candle_transformers::models::quantized_llama::ModelWeights as QuantizedLlamaModel;
use candle_transformers::models::quantized_qwen2::ModelWeights as QuantizedQwen2Model;
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;

use crate::backend::InferenceBackend;
use crate::registry::{ModelArch, ModelConfig, ModelDType, get_model_registry};
use crate::types::EngineStatus;
use crate::types::GenerationParameters;
use crate::types::MemoryStrategy;
use std::sync::{Arc, Mutex};

pub struct ExtractiveCompressor {
    base: BertModel,
    classifier: Linear,
}

impl ExtractiveCompressor {
    pub fn load(vb: VarBuilder, config: &BertConfig) -> candle_core::Result<Self> {
        let base = BertModel::load(vb.pp("roberta"), config)?;
        let classifier = candle_nn::linear(config.hidden_size, 2, vb.pp("classifier"))?;
        Ok(Self { base, classifier })
    }

    pub fn forward(&self, input_ids: &Tensor) -> candle_core::Result<Tensor> {
        let token_type_ids = input_ids.zeros_like()?;
        let hidden_states = self.base.forward(input_ids, &token_type_ids, None)?;
        self.classifier.forward(&hidden_states)
    }
}

pub enum DynamicModel {
    Llama(QuantizedLlamaModel),
    Qwen2(QuantizedQwen2Model),
    XLMRoberta(ExtractiveCompressor),
}

impl DynamicModel {
    pub fn forward(&mut self, x: &Tensor, start_pos: usize) -> candle_core::Result<Tensor> {
        match self {
            Self::Llama(m) => m.forward(x, start_pos),
            Self::Qwen2(m) => m.forward(x, start_pos),
            Self::XLMRoberta(_) => Err(candle_core::Error::Msg(
                "Token Classifiers cannot be used in a generative loop!".to_string(),
            )),
        }
    }
}

pub fn load_engine(
    model_id: &str,
    device: &Device,
) -> Result<(DynamicModel, Tokenizer, Option<std::fs::File>), String> {
    let config = get_model_registry()
        .into_iter()
        .find(|c| c.id == model_id)
        .ok_or_else(|| format!("Model ID {} not found in registry", model_id))?;

    let api = Api::new().map_err(|e| e.to_string())?;

    if config.filename.ends_with(".safetensors") {
        let repo = api.model(config.repo);
        let weights_path = repo
            .get(&config.filename)
            .map_err(|e| format!("Missing weights: {}", e))?;
        let config_path = repo
            .get("config.json")
            .map_err(|e| format!("Missing config.json: {}", e))?;
        let tokenizer_path = api
            .model(config.tokenizer_repo)
            .get("tokenizer.json")
            .map_err(|e| format!("Missing tokenizer: {}", e))?;

        let config_str = std::fs::read_to_string(config_path)
            .map_err(|e| format!("Failed to read config: {}", e))?;
        let conf: BertConfig =
            serde_json::from_str(&config_str).map_err(|e| format!("Bad config JSON: {}", e))?;

        let dtype = match config.compression_dtype {
            Some(ModelDType::F16) => candle_core::DType::F16,
            Some(ModelDType::BF16) => candle_core::DType::BF16,
            _ => candle_core::DType::F32,
        };

        println!("💎 Loading Safetensors instantly via Mmap...");

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, device)
                .map_err(|e| format!("Safetensors Mmap failed: {}", e))?
        };

        let model = ExtractiveCompressor::load(vb, &conf)
            .map_err(|e| format!("Extractive load failed: {}", e))?;
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| format!("Tokenizer load failed: {}", e))?;

        return Ok((DynamicModel::XLMRoberta(model), tokenizer, None));
    }

    let weights_path = api
        .model(config.repo)
        .get(&config.filename)
        .map_err(|e| e.to_string())?;
    let tokenizer_path = api
        .model(config.tokenizer_repo)
        .get("tokenizer.json")
        .map_err(|e| e.to_string())?;

    let mut file = std::fs::File::open(&weights_path).map_err(|e| e.to_string())?;
    let gguf_content =
        candle_core::quantized::gguf_file::Content::read(&mut file).map_err(|e| e.to_string())?;

    let model = match config.arch {
        ModelArch::Llama => DynamicModel::Llama(
            QuantizedLlamaModel::from_gguf(gguf_content, &mut file, device)
                .map_err(|e| e.to_string())?,
        ),
        ModelArch::Qwen2 => DynamicModel::Qwen2(
            QuantizedQwen2Model::from_gguf(gguf_content, &mut file, device)
                .map_err(|e| e.to_string())?,
        ),
        _ => {
            return Err(format!(
                "Unsupported GGUF architecture for model: {}",
                config.id
            ));
        }
    };

    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| e.to_string())?;

    Ok((model, tokenizer, Some(file)))
}

#[allow(clippy::too_many_arguments)]
async fn run_generation_candle<F>(
    prompt: &str,
    model: &mut DynamicModel,
    tokenizer: &Tokenizer,
    device: &Device,
    params: &GenerationParameters,
    max_context_len: usize,
    mut on_tokenization_time: impl FnMut(u128),
    mut on_token: F,
) -> Result<(String, u128), String>
where
    F: FnMut(String) -> Result<(), ()>,
{
    let tok_start = std::time::Instant::now();
    let mut tokens = tokenizer
        .encode(prompt, true)
        .map_err(|e| e.to_string())?
        .get_ids()
        .to_vec();
    let tok_time = tok_start.elapsed().as_millis();
    on_tokenization_time(tok_time);

    let eos_strings = [
        "</s>",
        "<|endoftext|>",
        "<|im_end|>",
        "<|end_of_text|>",
        "<|eot_id|>",
    ];
    let eos_tokens: Vec<u32> = eos_strings
        .iter()
        .filter_map(|s| tokenizer.token_to_id(s))
        .collect();

    let prompt_length = tokens.len();
    let mut logits_processor = LogitsProcessor::new(
        params.seed.unwrap_or_else(rand::random::<u64>),
        params.temperature.map(|t| t as f64),
        params.top_p.map(|p| p as f64),
    );

    let prefill_chunk_size = 256;
    let mut current_pos = 0;

    println!("🔋 Prefilling {} tokens into KV Cache...", tokens.len());

    if tokens.len() > 1 {
        // Generalize using Rust's native slice chunks iterator
        for chunk in tokens[..tokens.len().saturating_sub(1)].chunks(prefill_chunk_size) {
            let input_tensor = Tensor::new(chunk, device)
                .map_err(|e| e.to_string())?
                .unsqueeze(0)
                .map_err(|e| e.to_string())?
                .contiguous()
                .map_err(|e| e.to_string())?;

            let _ = model
                .forward(&input_tensor, current_pos)
                .map_err(|e| e.to_string())?;
            current_pos += chunk.len();
            tokio::task::yield_now().await;
        }
    }

    println!("⚡ Generation started...");

    let mut prev_index = prompt_length;

    for index in 0..params.max_tokens.unwrap_or(500) {
        let context_size = if index == 0 {
            tokens.len() - current_pos
        } else {
            1
        };
        let start_pos = tokens.len().saturating_sub(context_size);

        let input_tensor = Tensor::new(&tokens[start_pos..], device)
            .map_err(|e| e.to_string())?
            .unsqueeze(0)
            .map_err(|e| e.to_string())?
            .contiguous()
            .map_err(|e| e.to_string())?;

        let logits = model
            .forward(&input_tensor, start_pos)
            .map_err(|e| e.to_string())?;
        drop(input_tensor);

        let next_token_logits = logits.squeeze(0).map_err(|e| e.to_string())?;
        let next_token = logits_processor
            .sample(&next_token_logits)
            .map_err(|e| e.to_string())?;

        tokens.push(next_token);
        if eos_tokens.contains(&next_token) {
            break;
        }
        if tokens.len() >= max_context_len {
            println!(
                "⚠️ Reached maximum context length of {} tokens.",
                max_context_len
            );
            break;
        }

        let text = tokenizer
            .decode(&tokens[prev_index..], true)
            .unwrap_or_default();
        // '\u{FFFD}' indicates an incomplete UTF-8 byte sequence across tokens.
        // Wait for the next token to complete the character before flushing.
        if !text.is_empty() && !text.ends_with('\u{FFFD}') {
            if on_token(text).is_err() {
                break;
            }
            prev_index = tokens.len();
        }
        tokio::task::yield_now().await;
    }

    // Flush any remaining partial characters
    if prev_index < tokens.len() {
        let text = tokenizer
            .decode(&tokens[prev_index..], true)
            .unwrap_or_default();
        if !text.is_empty() {
            let _ = on_token(text);
        }
    }

    let final_text = tokenizer
        .decode(&tokens[prompt_length..], true)
        .map_err(|e| e.to_string())?;
    Ok((final_text, tok_time))
}

pub async fn generate_text(
    prompt: &str,
    model: &mut DynamicModel,
    tokenizer: &Tokenizer,
    device: &Device,
    params: &GenerationParameters,
    max_context_len: usize,
) -> Result<(String, u128), String> {
    run_generation_candle(
        prompt,
        model,
        tokenizer,
        device,
        params,
        max_context_len,
        |_| {},
        |_| Ok(()),
    )
    .await
}

pub async fn generate_text_stream(
    prompt: &str,
    model: &mut DynamicModel,
    tokenizer: &Tokenizer,
    device: &Device,
    params: &GenerationParameters,
    max_context_len: usize,
    tx: tokio::sync::mpsc::UnboundedSender<crate::types::StreamEvent>,
) -> Result<(), String> {
    run_generation_candle(
        prompt,
        model,
        tokenizer,
        device,
        params,
        max_context_len,
        |tok_time| {
            let _ = tx.send(crate::types::StreamEvent::TokenizationTime(tok_time));
        },
        |token| {
            tx.send(crate::types::StreamEvent::Token(token))
                .map_err(|_| ())
        },
    )
    .await?;
    let _ = tx.send(crate::types::StreamEvent::Done);
    Ok(())
}

pub async fn compress_text(
    prompt: &str,
    model: &DynamicModel,
    tokenizer: &Tokenizer,
    device: &Device,
    target_len: usize,
    max_chunk_size: usize,
) -> Result<(String, u128), String> {
    if let DynamicModel::XLMRoberta(m) = model {
        let tok_start = std::time::Instant::now();
        let tokens = tokenizer
            .encode(prompt, true)
            .map_err(|e| e.to_string())?
            .get_ids()
            .to_vec();
        let tok_time = tok_start.elapsed().as_millis();
        let mut token_scores: Vec<(usize, u32, f32)> = Vec::new();
        let mut global_idx = 0;

        println!(
            "✂️ Slicing {} tokens into {}-token chunks for RoBERTa...",
            tokens.len(),
            max_chunk_size
        );

        for chunk in tokens.chunks(max_chunk_size) {
            let input_tensor = Tensor::new(chunk, device)
                .map_err(|e| e.to_string())?
                .unsqueeze(0)
                .map_err(|e| e.to_string())?;

            let logits = m.forward(&input_tensor).map_err(|e| e.to_string())?;
            let logits = logits.squeeze(0).map_err(|e| e.to_string())?;
            let probabilities = candle_nn::ops::softmax(&logits, 1).map_err(|e| e.to_string())?;
            let probs_f32 = probabilities
                .to_dtype(candle_core::DType::F32)
                .map_err(|e| e.to_string())?;
            let probs_vec = probs_f32.to_vec2::<f32>().map_err(|e| e.to_string())?;

            for (i, token) in chunk.iter().enumerate() {
                token_scores.push((global_idx, *token, probs_vec[i][1]));
                global_idx += 1;
            }
            tokio::task::yield_now().await;
        }

        token_scores.sort_by(|a, b| b.2.total_cmp(&a.2));
        if token_scores.len() > target_len {
            token_scores.truncate(target_len);
        }
        token_scores.sort_by_key(|a| a.0);

        let kept_tokens: Vec<u32> = token_scores
            .into_iter()
            .map(|(_, token, _)| token)
            .collect();
        let compressed_text = tokenizer
            .decode(&kept_tokens, true)
            .map_err(|e| e.to_string())?;
        println!(
            "✅ Extractive compression complete. Shrunk from {} to {} tokens.",
            tokens.len(),
            kept_tokens.len()
        );

        Ok((compressed_text, tok_time))
    } else {
        Err("compress_text must be used with a Token Classifier!".to_string())
    }
}

pub struct CandleEngine {
    device: Device,
    model: Option<DynamicModel>,
    tokenizer: Option<Tokenizer>,
    _file: Option<std::fs::File>,
    nvml: Option<nvml_wrapper::Nvml>,
    gpu_device_index: u32,
    max_context_len: usize,
}

impl Default for CandleEngine {
    fn default() -> Self {
        Self::new(0)
    }
}

impl CandleEngine {
    pub fn new(gpu_device_index: u32) -> Self {
        Self {
            device: Device::cuda_if_available(gpu_device_index as usize).unwrap_or(Device::Cpu),
            model: None,
            tokenizer: None,
            _file: None,
            nvml: nvml_wrapper::Nvml::init().ok(),
            gpu_device_index,
            max_context_len: 2048,
        }
    }

    pub async fn generate_stream(
        &mut self,
        prompt: &str,
        params: &GenerationParameters,
        tx: tokio::sync::mpsc::UnboundedSender<crate::types::StreamEvent>,
    ) {
        let model = match self.model.as_mut() {
            Some(m) => m,
            None => {
                let _ = tx.send(crate::types::StreamEvent::Error("Model not loaded".into()));
                return;
            }
        };
        let tokenizer = match self.tokenizer.as_ref() {
            Some(t) => t,
            None => {
                let _ = tx.send(crate::types::StreamEvent::Error(
                    "Tokenizer not loaded".into(),
                ));
                return;
            }
        };
        let max_ctx = self.max_context_len;
        if let Err(e) = generate_text_stream(
            prompt,
            model,
            tokenizer,
            &self.device,
            params,
            max_ctx,
            tx.clone(),
        )
        .await
        {
            let _ = tx.send(crate::types::StreamEvent::Error(e));
        }
    }
}

#[async_trait]
impl InferenceBackend for CandleEngine {
    async fn load_model(
        &mut self,
        config: &ModelConfig,
        status: Arc<Mutex<EngineStatus>>,
        _strategy: &MemoryStrategy,
        _required_ctx: usize,
    ) -> Result<usize, String> {
        let (used_before, total, _) =
            crate::get_vram_info(self.nvml.as_ref(), self.gpu_device_index).unwrap_or((0, 0, 0));

        {
            let mut s = status.lock().unwrap();
            s.log_vram(
                "Allocate",
                "Candle",
                &format!("Loading weights for {}", config.id),
                0,
            );
        }

        let (m, t, f) = load_engine(&config.id, &self.device)?;
        self.model = Some(m);
        self.tokenizer = Some(t);
        self._file = f;
        self.max_context_len = config.max_context_len;

        let (used_after, _, free_after) =
            crate::get_vram_info(self.nvml.as_ref(), self.gpu_device_index).unwrap_or((0, 0, 0));
        let diff = used_after.saturating_sub(used_before);
        {
            let mut s = status.lock().unwrap();
            s.log_vram("Allocate", "Candle::Weights", "Model Weights", diff as i64);
            s.set_model_vram(
                config.id.clone(),
                "Candle".to_string(),
                false,
                "Active".to_string(),
                diff,
                0,
                0,
            );
            s.update_nvml(total, used_after, free_after);
        }
        Ok(config.max_context_len)
    }

    async fn generate_text(
        &mut self,
        prompt: &str,
        params: &GenerationParameters,
    ) -> Result<(String, u128), String> {
        let model = self.model.as_mut().ok_or("Model not loaded")?;
        let tokenizer = self.tokenizer.as_ref().ok_or("Tokenizer not loaded")?;
        generate_text(
            prompt,
            model,
            tokenizer,
            &self.device,
            params,
            self.max_context_len,
        )
        .await
    }

    fn supports_extractive_compression(&self) -> bool {
        matches!(self.model, Some(DynamicModel::XLMRoberta(_)))
    }

    async fn compress_text(
        &mut self,
        prompt: &str,
        target_len: usize,
        max_chunk: usize,
    ) -> Result<(String, u128), String> {
        let model = self.model.as_ref().ok_or("Model not loaded")?;
        let tokenizer = self.tokenizer.as_ref().ok_or("Tokenizer not loaded")?;
        compress_text(
            prompt,
            model,
            tokenizer,
            &self.device,
            target_len,
            max_chunk,
        )
        .await
    }

    fn get_vram_usage(&self) -> Option<(u64, u64)> {
        None // Relies on Orchestrator's Nvml Memory Manager
    }

    fn is_statically_allocated(&self) -> bool {
        false
    }

    fn get_offload_pct(&self) -> f32 {
        0.0
    }
}
