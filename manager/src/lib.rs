use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, oneshot};

use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;
use candle_core::{Device, Tensor};
use candle_nn::{Linear, VarBuilder, Module};
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use candle_transformers::models::quantized_llama::ModelWeights as LlamaWeights;
use candle_transformers::models::quantized_qwen2::ModelWeights as Qwen2Weights;
use candle_transformers::generation::LogitsProcessor;
use nvml_wrapper::Nvml;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH, Instant};
use std::fs;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct LoadMetric {
    pub timestamp: u64,
    pub model_id: String,
    pub load_time_ms: u128,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct GenerationMetric {
    pub timestamp: u64,
    pub model_id: String,
    pub prompt_chars: usize,
    pub prompt_tokens: usize,
    pub generation_time_ms: u128,
}

#[derive(Serialize, Deserialize, Clone, Default, Debug)]
pub struct TelemetryStore {
    pub loads: Vec<LoadMetric>,
    pub generations: Vec<GenerationMetric>,
}

impl TelemetryStore {
    pub fn load_from_disk() -> Self {
        if let Ok(data) = fs::read_to_string("stats.json") {
            serde_json::from_str(&data).unwrap_or_default()
        } else {
            Self::default()
        }
    }

    pub fn save_to_disk(&self) {
        // Serialize synchronously (CPU bound, very fast)
        if let Ok(json) = serde_json::to_string_pretty(self) {
            // Spawn a detached background task for the I/O (Disk bound, slow)
            tokio::spawn(async move {
                let _ = tokio::fs::write("stats.json", json).await;
            });
        }
    }

    pub fn record_load(&mut self, model_id: String, load_time_ms: u128) {
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        self.loads.push(LoadMetric { timestamp, model_id, load_time_ms });
        self.save_to_disk();
    }

    pub fn record_generation(&mut self, model_id: String, prompt_chars: usize, prompt_tokens: usize, generation_time_ms: u128) {
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        self.generations.push(GenerationMetric { timestamp, model_id, prompt_chars, prompt_tokens, generation_time_ms });
        self.save_to_disk();
    }
}
// The thread-safe status object we will share between the web server and the engine
#[derive(Serialize, Clone, Default, Debug)]
pub struct EngineStatus {
    pub active_chat_model_id: Option<String>,
    pub last_compressor_model_id: Option<String>,
    pub benchmark_running: bool,
}

pub fn lock_status(status: &Arc<Mutex<EngineStatus>>) -> std::sync::MutexGuard<'_, EngineStatus> {
    status.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

pub fn get_vram_info(device_index: u32) -> Option<(u64, u64, u64)> {
    let nvml = Nvml::init().ok()?;
    let device = nvml.device_by_index(device_index).ok()?;
    let info = device.memory_info().ok()?;
    
    Some((info.used, info.total, info.free))
}

fn estimate_bytes_per_token(arch: &ModelArch, params_billions: f32) -> usize {
    match arch {
        ModelArch::Qwen2 if params_billions > 10.0 => 150_000, // ~150KB for 14B
        ModelArch::Qwen2 => 80_000,                            // Smaller Qwens
        ModelArch::Llama if params_billions < 10.0 => 125_000, // ~125KB for 8B
        _ => 100_000,                                          // Fallback
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub enum CompressionDType {
    F32,
    F16,
}

#[derive(Clone, Serialize, Deserialize)]
pub enum ModelArch {
    Llama,
    Qwen2,
    XLMRoberta,
}

pub struct ExtractiveCompressor {
    base: BertModel,
    classifier: Linear,
}

impl ExtractiveCompressor {
    pub fn load(vb: VarBuilder, config: &BertConfig) -> candle_core::Result<Self> {
        // Step into the "roberta" prefix for the core model
        let base = BertModel::load(vb.pp("roberta"), config)?;
        
        // The classifier head sits at the root
        let classifier = candle_nn::linear(config.hidden_size, 2, vb.pp("classifier"))?;
        
        Ok(Self { base, classifier })
    }

    pub fn forward(&self, input_ids: &Tensor) -> candle_core::Result<Tensor> {
        // RoBERTa doesn't use token_type_ids, so we feed it zeros
        let token_type_ids = input_ids.zeros_like()?; 
        let hidden_states = self.base.forward(input_ids, &token_type_ids, None)?;
        
        // Pass the results through our custom layer
        self.classifier.forward(&hidden_states)
    }
}

pub enum DynamicModel {
    Llama(LlamaWeights),
    Qwen2(Qwen2Weights),
    XLMRoberta(ExtractiveCompressor),
}

impl DynamicModel {
    pub fn forward(&mut self, x: &Tensor, start_pos: usize) -> candle_core::Result<Tensor> {
        match self {
            Self::Llama(m) => m.forward(x, start_pos),
            Self::Qwen2(m) => m.forward(x, start_pos),
            Self::XLMRoberta(_) => {
                // Return a structured error instead of blowing up the application
                Err(candle_core::Error::Msg(
                    "Token Classifiers (like RoBERTa) cannot be used in a generative loop!".to_string()
                ))
            }
        }
    }
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum ModelRole {
    GeneralChat,
    ContextCompressor,
    CodeSpecialist,
    ToolCaller,
    Reasoning,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub id: String,
    pub name: String,
    pub repo: String,
    pub tokenizer_repo: String,
    pub filename: String,
    pub max_context_len: usize,
    pub roles: Vec<ModelRole>,
    pub arch: ModelArch,
    pub compression_dtype: Option<CompressionDType>, // Only for Safetensors models
    pub parameters_billions: f32, 
    pub size_on_disk_gb: f32,
}

// Expose the registry so the web server can send it to the UI
pub fn get_model_registry() -> Vec<ModelConfig> {
    vec![
        ModelConfig { 
            id: "llama-3.1-8b".to_string(), name: "Llama 3.1 (8B)".to_string(),
            repo: "QuantFactory/Meta-Llama-3.1-8B-Instruct-GGUF".to_string(),
            tokenizer_repo: "NousResearch/Meta-Llama-3.1-8B-Instruct".to_string(),
            filename: "Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf".to_string(), 
            max_context_len: 128000, roles: vec![ModelRole::GeneralChat],
            arch: ModelArch::Llama,
            compression_dtype: None,
            parameters_billions: 8.0,
            size_on_disk_gb: 4.58,
        },
        ModelConfig { 
            id: "qwen-2.5-7b".to_string(), 
            name: "Qwen 2.5 (7B)".to_string(),
            // Point to the unified community repo instead of the split official repo
            repo: "bartowski/Qwen2.5-7B-Instruct-GGUF".to_string(), 
            tokenizer_repo: "Qwen/Qwen2.5-7B-Instruct".to_string(),
            // Note the exact capitalization used in the bartowski repo:
            filename: "Qwen2.5-7B-Instruct-Q4_K_M.gguf".to_string(), 
            max_context_len: 128000, 
            roles: vec![ModelRole::GeneralChat, ModelRole::CodeSpecialist],
            arch: ModelArch::Qwen2,
            compression_dtype: None,
            parameters_billions: 7.61,
            size_on_disk_gb: 4.36,
        },
        ModelConfig { 
            id: "qwen-2.5-14b".to_string(), 
            name: "Qwen 2.5 (14B)".to_string(),
            repo: "bartowski/Qwen2.5-14B-Instruct-GGUF".to_string(), 
            tokenizer_repo: "Qwen/Qwen2.5-14B-Instruct".to_string(),
            filename: "Qwen2.5-14B-Instruct-Q4_K_M.gguf".to_string(), 
            max_context_len: 131072, 
            roles: vec![ModelRole::GeneralChat, ModelRole::CodeSpecialist],
            arch: ModelArch::Qwen2,
            compression_dtype: None,
            parameters_billions: 14.0,
            size_on_disk_gb: 8.37,
        },
        ModelConfig { 
            id: "qwen-coder-14b".to_string(), 
            name: "Qwen2.5 Coder (14B)".to_string(),
            repo: "Qwen/Qwen2.5-Coder-14B-Instruct-GGUF".to_string(), 
            tokenizer_repo: "Qwen/Qwen2.5-Coder-14B-Instruct".to_string(),
            filename: "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf".to_string(), 
            max_context_len: 131072,
            roles: vec![ModelRole::CodeSpecialist],
            arch: ModelArch::Qwen2,
            compression_dtype: None,
            parameters_billions: 14.0,
            size_on_disk_gb: 8.37,
        },
        ModelConfig { 
            id: "strand-rust-14b".to_string(), 
            name: "Strand Rust Coder (14B)".to_string(),
            repo: "mradermacher/Strand-Rust-Coder-14B-v1-GGUF".to_string(), 
            tokenizer_repo: "Fortytwo-Network/Strand-Rust-Coder-14B-v1".to_string(),
            filename: "Strand-Rust-Coder-14B-v1.Q4_K_M.gguf".to_string(), 
            max_context_len: 32768,
            roles: vec![ModelRole::CodeSpecialist],
            arch: ModelArch::Qwen2,
            compression_dtype: None,
            parameters_billions: 14.0,
            size_on_disk_gb: 8.37,
        },
        ModelConfig { 
            id: "llmlingua-2-f16".to_string(), 
            name: "LLMLingua-2 (F16 - Lean)".to_string(),
            repo: "microsoft/llmlingua-2-xlm-roberta-large-meetingbank".to_string(), 
            tokenizer_repo: "microsoft/llmlingua-2-xlm-roberta-large-meetingbank".to_string(),
            filename: "model.safetensors".to_string(), 
            max_context_len: 512, 
            roles: vec![ModelRole::ContextCompressor],
            arch: ModelArch::XLMRoberta,
            compression_dtype: Some(CompressionDType::F16),
            parameters_billions: 0.56,
            size_on_disk_gb: 2.08,
        },
        ModelConfig { 
            id: "llmlingua-2-f32".to_string(), 
            name: "LLMLingua-2 (F32 - Precision)".to_string(),
            repo: "microsoft/llmlingua-2-xlm-roberta-large-meetingbank".to_string(), 
            tokenizer_repo: "microsoft/llmlingua-2-xlm-roberta-large-meetingbank".to_string(),
            filename: "model.safetensors".to_string(), 
            max_context_len: 512, 
            roles: vec![ModelRole::ContextCompressor],
            arch: ModelArch::XLMRoberta,
            compression_dtype: Some(CompressionDType::F32),
            parameters_billions: 0.56,
            size_on_disk_gb: 2.08,
        },
        ModelConfig { 
            id: "qwen-compressor".to_string(), 
            name: "Qwen 1.5B (Abstractive)".to_string(),
            repo: "Qwen/Qwen2.5-1.5B-Instruct-GGUF".to_string(), 
            tokenizer_repo: "Qwen/Qwen2.5-1.5B-Instruct".to_string(),
            filename: "qwen2.5-1.5b-instruct-q4_k_m.gguf".to_string(), 
            max_context_len: 32768, 
            roles: vec![ModelRole::ContextCompressor],
            arch: ModelArch::Qwen2,
            compression_dtype: None,
            parameters_billions: 1.54,
            size_on_disk_gb: 1.04,
        },
    ]
}

#[derive(Deserialize, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Deserialize)]
pub struct ApiRequest {
    pub chat_model_id: String,
    pub compressor_model_id: String,
    pub messages: Vec<Message>,
}

#[derive(Serialize)]
pub struct ApiResponse {
    pub answer: String,
}

pub struct UserRequest {
    pub chat_model_id: String,
    pub compressor_model_id: String,
    pub messages: Vec<Message>,
    pub responder: oneshot::Sender<String>,
    pub force_compression: bool,
}

// Helper: Formats the array into a generic string format
fn format_chat(messages: &[Message]) -> String {
    let mut prompt = String::new();
    for msg in messages { prompt.push_str(&format!("<|{}|>\n{}</s>\n", msg.role, msg.content)); }
    prompt.push_str("<|assistant|>\n");
    prompt
}

// THE SNAPSHOT LOADER: This function returns the mapped File so the OS keeps it alive in RAM!
fn load_engine(model_id: &str, device: &Device) -> Result<(DynamicModel, Tokenizer, Option<std::fs::File>), String> {
    let config = get_model_registry().into_iter().find(|c| c.id == model_id)
        .ok_or_else(|| format!("Model ID {} not found in registry", model_id))?; 
    
    let api = Api::new().map_err(|e| e.to_string())?;

    if config.filename.ends_with(".safetensors") {
        let repo = api.model(config.repo);
        
        // --- Safetensors IO wrapped in Results ---
        let weights_path = repo.get(&config.filename).map_err(|e| format!("Missing weights: {}", e))?;
        let config_path = repo.get("config.json").map_err(|e| format!("Missing config.json: {}", e))?;
        let tokenizer_path = api.model(config.tokenizer_repo).get("tokenizer.json").map_err(|e| format!("Missing tokenizer: {}", e))?;

        let config_str = std::fs::read_to_string(config_path).map_err(|e| format!("Failed to read config: {}", e))?;
        let conf: BertConfig = serde_json::from_str(&config_str).map_err(|e| format!("Bad config JSON: {}", e))?;
        
        let dtype = match config.compression_dtype {
            Some(CompressionDType::F16) => candle_core::DType::F16,
            _ => candle_core::DType::F32,
        };

        println!("💎 Loading Safetensors instantly via Mmap...");
        
        let vb = unsafe { 
            VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, device)
                .map_err(|e| format!("Safetensors Mmap failed: {}", e))? 
        };

        let model = ExtractiveCompressor::load(vb, &conf).map_err(|e| format!("Extractive load failed: {}", e))?;
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| format!("Tokenizer load failed: {}", e))?;

        return Ok((DynamicModel::XLMRoberta(model), tokenizer, None));
    }

    // --- PATH B: GENERATIVE MODELS (GGUF) ---
    let weights_path = api.model(config.repo).get(&config.filename).map_err(|e| e.to_string())?;
    let tokenizer_path = api.model(config.tokenizer_repo).get("tokenizer.json").map_err(|e| e.to_string())?;

    let mut file = std::fs::File::open(&weights_path).map_err(|e| e.to_string())?;
    let gguf_content = candle_core::quantized::gguf_file::Content::read(&mut file).map_err(|e| e.to_string())?;
    
    let model = match config.arch {
        ModelArch::Llama => DynamicModel::Llama(LlamaWeights::from_gguf(gguf_content, &mut file, device).map_err(|e| e.to_string())?),
        ModelArch::Qwen2 => DynamicModel::Qwen2(Qwen2Weights::from_gguf(gguf_content, &mut file, device).map_err(|e| e.to_string())?),
        _ => return Err(format!("Unsupported GGUF architecture for model: {}", config.id)),
    };

    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| e.to_string())?;
    
    Ok((model, tokenizer, Some(file)))
}

fn generate_text(prompt: &str, model: &mut DynamicModel, tokenizer: &Tokenizer, device: &Device, max_tokens: usize) -> Result<String, String> {
    let mut tokens = tokenizer.encode(prompt, true).map_err(|e| e.to_string())?.get_ids().to_vec();
    let prompt_length = tokens.len();
    let mut logits_processor = LogitsProcessor::new(299792458, None, None);

    let prefill_chunk_size = 256; 
    let mut current_pos = 0;

    println!("🔋 Prefilling {} tokens into KV Cache...", tokens.len());
    
    if tokens.len() > 1 {
        while current_pos < tokens.len() - 1 {
            let chunk_size = (tokens.len() - 1 - current_pos).min(prefill_chunk_size);
            let chunk = &tokens[current_pos..current_pos + chunk_size];
            
            // Map tensor creation and reshaping errors
            let input_tensor = Tensor::new(chunk, device).map_err(|e| e.to_string())?
                .unsqueeze(0).map_err(|e| e.to_string())?
                .contiguous().map_err(|e| e.to_string())?;

            let _ = model.forward(&input_tensor, current_pos).map_err(|e| e.to_string())?;
            current_pos += chunk_size;
        }
    }

    println!("⚡ Generation started...");

    for index in 0..max_tokens {
        let context_size = if index == 0 { tokens.len() - current_pos } else { 1 };
        let start_pos = tokens.len().saturating_sub(context_size);
        
        let input_tensor = Tensor::new(&tokens[start_pos..], device).map_err(|e| e.to_string())?
            .unsqueeze(0).map_err(|e| e.to_string())?
            .contiguous().map_err(|e| e.to_string())?;

        let logits = model.forward(&input_tensor, start_pos).map_err(|e| e.to_string())?;
        drop(input_tensor);

        let next_token_logits = logits.squeeze(0).map_err(|e| e.to_string())?;
        let next_token = logits_processor.sample(&next_token_logits).map_err(|e| e.to_string())?;
        
        tokens.push(next_token);

        if next_token == 2 || next_token == 151645 || next_token == 151643 || next_token == 128001 || next_token == 128009 { 
            break; 
        }
    }
    
    tokenizer.decode(&tokens[prompt_length..], true).map_err(|e| e.to_string())
}

fn compress_text(prompt: &str, model: &DynamicModel, tokenizer: &Tokenizer, device: &Device, target_len: usize, max_chunk_size: usize) -> Result<String, String> {
    if let DynamicModel::XLMRoberta(m) = model {
        let tokens = tokenizer.encode(prompt, true).map_err(|e| e.to_string())?.get_ids().to_vec();
        
        let mut token_scores: Vec<(usize, u32, f32)> = Vec::new(); 
        let mut global_idx = 0;

        // Make the print statement dynamic
        println!("✂️ Slicing {} tokens into {}-token chunks for RoBERTa...", tokens.len(), max_chunk_size);

        // Replace the hardcoded 500 with the config variable
        for chunk in tokens.chunks(max_chunk_size) {
            let input_tensor = Tensor::new(chunk, device).map_err(|e| e.to_string())?
                .unsqueeze(0).map_err(|e| e.to_string())?;
                
            let logits = m.forward(&input_tensor).map_err(|e| e.to_string())?;
            let logits = logits.squeeze(0).map_err(|e| e.to_string())?; 

            let probabilities = candle_nn::ops::softmax(&logits, 1).map_err(|e| e.to_string())?;
            
            let probs_f32 = probabilities.to_dtype(candle_core::DType::F32).map_err(|e| e.to_string())?;
            let probs_vec = probs_f32.to_vec2::<f32>().map_err(|e| e.to_string())?;

            for (i, token) in chunk.iter().enumerate() {
                let keep_probability = probs_vec[i][1];
                token_scores.push((global_idx, *token, keep_probability));
                global_idx += 1;
            }
        }

        token_scores.sort_by(|a, b| b.2.total_cmp(&a.2));
        
        if token_scores.len() > target_len {
            token_scores.truncate(target_len);
        }

        token_scores.sort_by(|a, b| a.0.cmp(&b.0));

        let kept_tokens: Vec<u32> = token_scores.into_iter().map(|(_, token, _)| token).collect();

        let compressed_text = tokenizer.decode(&kept_tokens, true).map_err(|e| e.to_string())?;
        println!("✅ Extractive compression complete. Shrunk from {} to {} tokens.", tokens.len(), kept_tokens.len());
        
        Ok(compressed_text)
    } else {
        Err("compress_text must be used with a Token Classifier!".to_string())
    }
}

pub async fn run_batcher_loop(
    mut receiver: mpsc::Receiver<UserRequest>, 
    status: Arc<Mutex<EngineStatus>>,
    telemetry: Arc<Mutex<TelemetryStore>>
) {
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    
    let mut active_model_id = String::new();
    let mut active_model: Option<DynamicModel> = None;
    let mut active_tokenizer: Option<Tokenizer> = None;
    let mut _active_file: Option<std::fs::File> = None;
    let mut active_max_context: usize = 2048; 
    let mut active_model_config: Option<ModelConfig> = None;

    println!("⚙️  ORCHESTRATOR ONLINE: Waiting for requests...");

    'main: while let Some(request) = receiver.recv().await {
        println!("📥 Processing new chat request...");

        let last_message = match request.messages.last() {
            Some(msg) => msg.clone(),
            None => {
                println!("⚠️ Rejected request: No messages provided.");
                let _ = request.responder.send("Server Error: Request contained no messages.".to_string());
                continue 'main;
            }
        };

        // 1. Hot-Swap to the requested Chat Model
        if active_model_id != request.chat_model_id {
            println!("🔄 Swapping VRAM to {}...", request.chat_model_id);

            drop(active_model.take()); 
            drop(active_tokenizer.take()); 
            drop(_active_file.take()); 

            let load_start = Instant::now();

            let (m, t, f) = match load_engine(&request.chat_model_id, &device) {
                Ok(engine) => engine,
                Err(e) => {
                    println!("❌ Chat model load failed: {}", e);
                    let _ = request.responder.send(format!("Server Error: Failed to load chat model: {}", e));
                    continue 'main;
                }
            };

            let elapsed = load_start.elapsed().as_millis();
            println!("⏱️ Model loaded in {} ms", elapsed);
            if let Ok(mut t) = telemetry.lock() {
                t.record_load(request.chat_model_id.clone(), elapsed);
            }
            
            active_model = Some(m); active_tokenizer = Some(t); _active_file = f; 
            active_model_id = request.chat_model_id.clone();

            let config = match get_model_registry().into_iter().find(|c| c.id == active_model_id) {
                Some(c) => c,
                None => {
                    let _ = request.responder.send("Server Error: Active model missing from registry.".to_string());
                    continue 'main;
                }
            };
            active_max_context = config.max_context_len;
            active_model_config = Some(config);
            println!("✅ Model limits established. Max context window: {}", active_max_context);

            {
                let mut current_status = lock_status(&status);
                current_status.active_chat_model_id = Some(active_model_id.clone());
            }
        }

        let mut formatted_prompt = format_chat(&request.messages);
        
        let token_count = match active_tokenizer.as_ref().unwrap().encode(formatted_prompt.clone(), true) {
            Ok(enc) => enc.get_ids().len(),
            Err(e) => {
                let _ = request.responder.send(format!("Server Error: Tokenization failed: {}", e));
                continue 'main;
            }
        };

        // --- THE DYNAMIC MEMORY MANAGER ---
        let mut trigger_compression = false;
        let mut dynamic_target_budget = active_max_context;

        // estimate_bytes_per_token logic (roughly 150KB for 14B models)
        let config = active_model_config.as_ref().unwrap();
        let bytes_per_token = estimate_bytes_per_token(&config.arch, config.parameters_billions);

        if let Some((_, _, free_vram)) = get_vram_info(0) {
            let safe_free_vram = free_vram.saturating_sub(500 * 1024 * 1024); 
            let absolute_max_tokens = (safe_free_vram as usize / bytes_per_token).min(active_max_context);
            
            println!("🧮 MEMORY CHECK: Free VRAM can hold ~{} tokens.", absolute_max_tokens);

            if token_count > absolute_max_tokens {
                println!("⚠️ WARNING: Prompt exceeds physical VRAM limits! Triggering dynamic compression.");
                trigger_compression = true;
                // FIX 1: Prevent 0-token budgets. Always leave at least 256 tokens.
                dynamic_target_budget = ((absolute_max_tokens as f32 * 0.80) as usize).max(256); 
            } else if token_count > (active_max_context as f32 * 0.80) as usize {
                trigger_compression = true;
                dynamic_target_budget = ((active_max_context as f32 * 0.50) as usize).max(256);
            } else if request.force_compression {
                println!("⚠️ Benchmarking: Forcing compression execution.");
                trigger_compression = true;
                dynamic_target_budget = ((token_count as f32 * 0.50) as usize).max(256);
            }
        } else {
            // CPU fallback
            if token_count > (active_max_context as f32 * 0.80) as usize {
                trigger_compression = true;
                dynamic_target_budget = ((active_max_context as f32 * 0.50) as usize).max(256);
            } else if request.force_compression {
                println!("⚠️ Benchmarking: Forcing compression execution.");
                trigger_compression = true;
                dynamic_target_budget = ((token_count as f32 * 0.50) as usize).max(256);
            }
        }

        if trigger_compression {
            if let Some((used_start, total, _)) = get_vram_info(0) { 
                println!("📊 VRAM before compressor: {:.2}GB / {:.2}GB", 
                    used_start as f32 / 1024.0_f32.powi(3), 
                    total as f32 / 1024.0_f32.powi(3));
            }

            // --- RECORD COMPRESSOR LOAD TIME ---
            let comp_load_start = Instant::now();
            let (mut comp_m, comp_t, _comp_f) = match load_engine(&request.compressor_model_id, &device) {
                Ok(engine) => engine,
                Err(e) => {
                    let _ = request.responder.send(format!("Server Error: Failed to load compressor: {}", e));
                    continue 'main; 
                }
            };
            
            let comp_load_elapsed = comp_load_start.elapsed().as_millis();
            println!("⏱️ Compressor loaded in {} ms", comp_load_elapsed);
            if let Ok(mut t) = telemetry.lock() {
                t.record_load(request.compressor_model_id.clone(), comp_load_elapsed);
            }
            
            let comp_config = match get_model_registry().into_iter().find(|c| c.id == request.compressor_model_id) {
                Some(c) => c,
                None => {
                    let _ = request.responder.send("Server Error: Compressor missing from registry.".to_string());
                    continue 'main;
                }
            };

            {
                let mut current_status = lock_status(&status);
                current_status.last_compressor_model_id = Some(request.compressor_model_id.clone());
            }

            let target_budget = dynamic_target_budget; 

            // --- RECORD COMPRESSION EXECUTION TIME ---
            let comp_start = Instant::now();
            let summary = match &comp_m {
                DynamicModel::XLMRoberta(_) => {
                    match compress_text(&formatted_prompt, &comp_m, &comp_t, &device, target_budget, comp_config.max_context_len) {
                        Ok(compressed) => compressed,
                        Err(e) => {
                            let _ = request.responder.send(format!("Server Error: Context compression failed: {}", e));
                            continue 'main;
                        }
                    }
                },
                _ => {
                    let mut current_tokens = match comp_t.encode(formatted_prompt.clone(), true) {
                        Ok(enc) => enc.get_ids().to_vec(),
                        Err(e) => {
                            let _ = request.responder.send(format!("Server Error: Compressor encode failed: {}", e));
                            continue 'main;
                        }
                    };
                    
                    let safe_input_limit = comp_config.max_context_len.saturating_sub(600); 

                    while current_tokens.len() > target_budget {
                        let chunk_end = current_tokens.len().min(safe_input_limit);
                        
                        let chunk_text = match comp_t.decode(&current_tokens[0..chunk_end], true) {
                            Ok(text) => text,
                            Err(e) => {
                                let _ = request.responder.send(format!("Server Error: Chunk decode failed: {}", e));
                                continue 'main; 
                            }
                        };

                        let compression_prompt = format!("<|user|>\nSummarize history compactly:\n{}</s>\n<|assistant|>\n", chunk_text);
                        
                        let summary_text = match generate_text(&compression_prompt, &mut comp_m, &comp_t, &device, 400) {
                            Ok(text) => text,
                            Err(e) => {
                                let _ = request.responder.send(format!("Server Error: Generation failed: {}", e));
                                continue 'main;
                            }
                        };

                        let summary_tokens = match comp_t.encode(summary_text, true) {
                            Ok(enc) => enc.get_ids().to_vec(),
                            Err(e) => {
                                let _ = request.responder.send(format!("Server Error: Summary tokenization failed: {}", e));
                                continue 'main;
                            }
                        };

                        let mut next_tokens = summary_tokens;
                        next_tokens.extend_from_slice(&current_tokens[chunk_end..]);

                        if next_tokens.len() >= current_tokens.len() { break; }
                        current_tokens = next_tokens;
                    }
                    
                    match comp_t.decode(&current_tokens, true) {
                        Ok(text) => text,
                        Err(e) => {
                            let _ = request.responder.send(format!("Server Error: Final decode failed: {}", e));
                            continue 'main;
                        }
                    }
                }
            };

            let comp_elapsed = comp_start.elapsed().as_millis();
            println!("⏱️ Compression completed in {} ms", comp_elapsed);
            if let Ok(mut t) = telemetry.lock() {
                t.record_generation(request.compressor_model_id.clone(), formatted_prompt.len(), token_count, comp_elapsed);
            }

            drop(comp_t); drop(_comp_f); drop(comp_m);

            if let Some((used_end, total, _)) = get_vram_info(0) {
                if (used_end as f32 / total as f32) > 0.85 {
                    println!("🧹 Threshold met. Syncing hardware...");
                    if let Err(e) = device.synchronize() {
                        println!("⚠️ Warning: Hardware VRAM sync failed: {}", e);
                    }
                }
            }
            
            println!("🔄 Resuming Chat...");
            let mut new_messages = Vec::new();
            
            if request.messages.len() > 1 {
                // Multi-turn chat: Compress the older history, but keep their newest question intact
                new_messages.push(Message { role: "system".to_string(), content: format!("Compressed Context:\n{}", summary.trim()) });
                new_messages.push(last_message);
            } else {
                // Single massive file drop (or benchmark): The summary IS the new message
                new_messages.push(Message { role: "user".to_string(), content: format!("Review this compressed context:\n{}", summary.trim()) });
            }
            
            formatted_prompt = format_chat(&new_messages);
        }

        println!("📥 Processing prompt...");
        let gen_start = Instant::now();
        
        // Execute the main chat model generation
        match generate_text(&formatted_prompt, active_model.as_mut().unwrap(), active_tokenizer.as_ref().unwrap(), &device, 500) {
            Ok(answer) => {
                let elapsed = gen_start.elapsed().as_millis();
                println!("⏱️ Generation completed in {} ms", elapsed);
                if let Ok(mut t) = telemetry.lock() {
                    t.record_generation(active_model_id.clone(), formatted_prompt.len(), token_count, elapsed);
                }
                
                let _ = request.responder.send(answer.trim().to_string());
            },
            Err(e) => {
                println!("❌ Generation Error: {}", e);
                let _ = request.responder.send(format!("Server Error: Generation failed: {}", e));
            }
        }
    } // Closes the 'main while loop
} // Closes the run_batcher_loop function