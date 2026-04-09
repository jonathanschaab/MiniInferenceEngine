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

pub fn get_vram_info(device_index: u32) -> Option<(u64, u64)> {
    // Fail gracefully at each step if hardware is missing or incompatible
    let nvml = Nvml::init().ok()?;
    let device = nvml.device_by_index(device_index).ok()?;
    let info = device.memory_info().ok()?;
    
    // Returns (Used, Total) in bytes
    Some((info.used, info.total))
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
            compression_dtype: None
        },
        ModelConfig { 
            id: "qwen3-14b".to_string(), 
            name: "Qwen3 (14B Equivalent)".to_string(),
            // Switched to a single-file community repo
            repo: "bartowski/Qwen2.5-14B-Instruct-GGUF".to_string(), 
            tokenizer_repo: "Qwen/Qwen2.5-14B-Instruct".to_string(),
            filename: "Qwen2.5-14B-Instruct-Q4_K_M.gguf".to_string(), 
            max_context_len: 131072, 
            roles: vec![ModelRole::GeneralChat, ModelRole::CodeSpecialist],
            arch: ModelArch::Qwen2,
            compression_dtype: None
        },
        ModelConfig { 
            id: "strand-rust-14b".to_string(), name: "Strand Rust Coder (14B)".to_string(),
            repo: "Qwen/Qwen2.5-Coder-14B-Instruct-GGUF".to_string(), 
            tokenizer_repo: "Qwen/Qwen2.5-Coder-14B-Instruct".to_string(),
            filename: "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf".to_string(), 
            max_context_len: 131072,
            roles: vec![ModelRole::CodeSpecialist],
            arch: ModelArch::Qwen2,
            compression_dtype: None
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
            compression_dtype: Some(CompressionDType::F16)
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
            compression_dtype: Some(CompressionDType::F32)
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
            compression_dtype: None
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

fn generate_text(prompt: &str, model: &mut DynamicModel, tokenizer: &Tokenizer, device: &Device, max_tokens: usize) -> String {
    let mut tokens = tokenizer.encode(prompt, true).unwrap().get_ids().to_vec();
    let prompt_length = tokens.len();
    let mut logits_processor = LogitsProcessor::new(299792458, None, None);

    // --- THE VRAM SAVER: CHUNKED PREFILL ---
    // Instead of computing attention for all tokens at once, we feed them in blocks
    // to build the KV cache progressively without CuBLAS workspace spikes.
    let prefill_chunk_size = 256; 
    let mut current_pos = 0;

    println!("🔋 Prefilling {} tokens into KV Cache...", tokens.len());
    
    // We prefill everything EXCEPT the very last token
    if tokens.len() > 1 {
        while current_pos < tokens.len() - 1 {
            let chunk_size = (tokens.len() - 1 - current_pos).min(prefill_chunk_size);
            let chunk = &tokens[current_pos..current_pos + chunk_size];
            
            let input_tensor = Tensor::new(chunk, device)
                .unwrap()
                .unsqueeze(0)
                .unwrap()
                .contiguous()
                .unwrap();

            // Run the math just to fill the KV Cache (we ignore the output logits here)
            let _ = model.forward(&input_tensor, current_pos).unwrap();
            current_pos += chunk_size;
        }
    }

    println!("⚡ Generation started...");

    for index in 0..max_tokens {
        // Step 1: Send the un-processed tail of the prompt (usually 1 token), 
        // then step by 1 token for all future loops.
        let context_size = if index == 0 { tokens.len() - current_pos } else { 1 };
        let start_pos = tokens.len().saturating_sub(context_size);
        
        // Input MUST be 2D: [batch_size, seq_len]
        let input_tensor = Tensor::new(&tokens[start_pos..], device)
            .unwrap()
            .unsqueeze(0)
            .unwrap()
            .contiguous()
            .unwrap();

        // Run the GPU Math for the final token(s) to get the actual predictions
        let logits = model.forward(&input_tensor, start_pos).unwrap();

        drop(input_tensor);

        // Squeeze the batch dimension to get a pure 1D array of probabilities
        let next_token_logits = logits.squeeze(0).unwrap();

        // Pick the actual next word ID
        let next_token = logits_processor.sample(&next_token_logits).unwrap();
        tokens.push(next_token);

        // Stop if we hit EOS tokens for Llama 2/3, Qwen, or ChatML
        if next_token == 2 || next_token == 151645 || next_token == 151643 || next_token == 128001 || next_token == 128009 { 
            break; 
        }
    }
    
    tokenizer.decode(&tokens[prompt_length..], true).unwrap()
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

        token_scores.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        
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

pub async fn run_batcher_loop(mut receiver: mpsc::Receiver<UserRequest>) {
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    
    let mut active_model_id = String::new();
    let mut active_model: Option<DynamicModel> = None;
    let mut active_tokenizer: Option<Tokenizer> = None;
    let mut _active_file: Option<std::fs::File> = None;
    let mut active_max_context: usize = 2048; 

    println!("⚙️  ORCHESTRATOR ONLINE: Waiting for requests...");

    while let Some(request) = receiver.recv().await {
        println!("📥 Processing new chat request...");

        // --- PREVENT CRASH ON EMPTY MESSAGES ---
        let last_message = match request.messages.last() {
            Some(msg) => msg.clone(),
            None => {
                println!("⚠️ Rejected request: No messages provided.");
                let _ = request.responder.send("Server Error: Request contained no messages.".to_string());
                continue;
            }
        };

        // Hot-Swap to the requested Chat Model
        if active_model_id != request.chat_model_id {
            println!("🔄 Swapping VRAM to {}...", request.chat_model_id);
            drop(active_model.take()); 
            drop(active_tokenizer.take()); 
            drop(_active_file.take()); 
            
            // --- CHAT MODEL LOAD ---
            let (m, t, f) = match load_engine(&request.chat_model_id, &device) {
                Ok(engine) => engine,
                Err(e) => {
                    println!("❌ Chat model load failed: {}", e);
                    let _ = request.responder.send(format!("Server Error: Failed to load chat model: {}", e));
                    continue;
                }
            };
            
            active_model = Some(m); active_tokenizer = Some(t); _active_file = f; 
            active_model_id = request.chat_model_id.clone();

            let config = get_model_registry().into_iter().find(|c| c.id == active_model_id).unwrap();
            active_max_context = config.max_context_len;
            println!("✅ Model limits established. Max context window: {}", active_max_context);
        }

        let mut formatted_prompt = format_chat(&request.messages);
        let token_count = active_tokenizer.as_ref().unwrap().encode(formatted_prompt.clone(), true).unwrap().get_ids().len();

        // The Dynamic Auto-Compressor Intercept
        let compression_threshold = (active_max_context as f32 * 0.80) as usize;

        if token_count > compression_threshold {
            if let Some((used_start, total)) = get_vram_info(0) { 
                println!("📊 VRAM before compressor: {:.2}GB / {:.2}GB", 
                    used_start as f32 / 1024.0 / 1024.0 / 1024.0, 
                    total as f32 / 1024.0 / 1024.0 / 1024.0);
            } else {
                println!("🖥️ Running in CPU fallback mode. VRAM tracking disabled.");
            }

            // --- COMPRESSOR LOAD ---
            let (mut comp_m, comp_t, _comp_f) = match load_engine(&request.compressor_model_id, &device) {
                Ok(engine) => engine,
                Err(e) => {
                    println!("❌ Compressor load failed: {}", e);
                    let _ = request.responder.send(format!("Server Error: Failed to load compressor: {}", e));
                    continue;
                }
            };
            
            let comp_config = get_model_registry().into_iter()
                .find(|c| c.id == request.compressor_model_id)
                .unwrap();
            
            let target_budget = (active_max_context as f32 * 0.2) as usize; 

            let summary = match &comp_m {
                DynamicModel::XLMRoberta(_) => {
                    println!("✂️ Running Extractive Compression (True LLMLingua-2)...");
                    
                    match compress_text(&formatted_prompt, &comp_m, &comp_t, &device, target_budget, comp_config.max_context_len) {
                        Ok(compressed) => compressed,
                        Err(e) => {
                            println!("❌ Compression failed: {}", e);
                            let _ = request.responder.send(format!("Server Error: Context compression failed: {}", e));
                            continue;
                        }
                    }
                },
                _ => {
                    println!("🧠 Running Rolling Abstractive Compression (Qwen Agent)...");
                    let mut current_tokens = comp_t.encode(formatted_prompt.clone(), true).unwrap().get_ids().to_vec();
                    let safe_input_limit = comp_config.max_context_len.saturating_sub(600); 

                    while current_tokens.len() > target_budget {
                        let chunk_end = current_tokens.len().min(safe_input_limit);
                        let chunk_text = comp_t.decode(&current_tokens[0..chunk_end], true).unwrap();
                        let compression_prompt = format!("<|user|>\nSummarize history compactly:\n{}</s>\n<|assistant|>\n", chunk_text);
                        
                        let summary_text = generate_text(&compression_prompt, &mut comp_m, &comp_t, &device, 400);
                        let summary_tokens = comp_t.encode(summary_text, true).unwrap().get_ids().to_vec();

                        let mut next_tokens = summary_tokens;
                        next_tokens.extend_from_slice(&current_tokens[chunk_end..]);

                        if next_tokens.len() >= current_tokens.len() { break; }
                        current_tokens = next_tokens;
                    }
                    comp_t.decode(&current_tokens, true).unwrap()
                }
            };

            drop(comp_t); 
            drop(_comp_f);
            drop(comp_m);

            if let Some((used_end, total)) = get_vram_info(0) {
                if (used_end as f32 / total as f32) > 0.85 {
                    println!("🧹 Threshold met. Syncing hardware...");
                    device.synchronize().unwrap();
                }
            }
            
            println!("🔄 Resuming Chat...");
            let mut new_messages = vec![Message { role: "system".to_string(), content: format!("Compressed Context:\n{}", summary.trim()) }];
            new_messages.push(last_message);
            formatted_prompt = format_chat(&new_messages);
        }

        println!("📥 Processing prompt...");
        let answer = generate_text(&formatted_prompt, active_model.as_mut().unwrap(), active_tokenizer.as_ref().unwrap(), &device, 500);
        let _ = request.responder.send(answer.trim().to_string());
    }
}