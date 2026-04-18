use tokio::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use nvml_wrapper::Nvml;
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;

pub mod telemetry;
pub mod types;
pub mod registry;
pub mod backend;
#[cfg(feature = "backend-candle")]
pub mod backend_candle;
#[cfg(feature = "backend-llamacpp")]
pub mod backend_llamacpp;

pub use telemetry::*;
pub use types::*;
pub use registry::*;
pub use backend::*;
#[cfg(feature = "backend-candle")]
pub use backend_candle::*;
#[cfg(feature = "backend-llamacpp")]
pub use backend_llamacpp::*;

pub fn get_vram_info(nvml: Option<&Nvml>, device_index: u32) -> Option<(u64, u64, u64)> {
    let nvml = nvml?;
    let device = nvml.device_by_index(device_index).ok()?;
    let info = device.memory_info().ok()?;
    
    Some((info.used, info.total, info.free))
}

pub enum ActiveBackend {
    #[cfg(feature = "backend-candle")]
    Candle(Box<CandleEngine>),
    #[cfg(feature = "backend-llamacpp")]
    LlamaCpp(Box<LlamaCppEngine>),
}

impl ActiveBackend {
    pub async fn load_model(&mut self, config: &ModelConfig, status: Arc<Mutex<EngineStatus>>, strategy: &str, required_ctx: usize) -> Result<usize, String> {
        match self {
            #[cfg(feature = "backend-candle")] ActiveBackend::Candle(b) => b.load_model(config, status, strategy, required_ctx).await,
            #[cfg(feature = "backend-llamacpp")] ActiveBackend::LlamaCpp(b) => b.load_model(config, status, strategy, required_ctx).await,
        }
    }
    pub async fn generate_stream(&mut self, prompt: &str, params: &GenerationParameters, tx: tokio::sync::mpsc::UnboundedSender<crate::types::StreamEvent>) {
        match self {
            #[cfg(feature = "backend-candle")] ActiveBackend::Candle(b) => b.generate_stream(prompt, params, tx).await,
            #[cfg(feature = "backend-llamacpp")] ActiveBackend::LlamaCpp(b) => b.generate_stream(prompt, params, tx).await,
        }
    }
    pub async fn generate_text(&mut self, prompt: &str, params: &GenerationParameters) -> Result<(String, u128), String> {
        match self {
            #[cfg(feature = "backend-candle")] ActiveBackend::Candle(b) => b.generate_text(prompt, params).await,
            #[cfg(feature = "backend-llamacpp")] ActiveBackend::LlamaCpp(b) => b.generate_text(prompt, params).await,
        }
    }
    pub async fn compress_text(&mut self, prompt: &str, target_len: usize, max_chunk: usize) -> Result<(String, u128), String> {
        match self {
            #[cfg(feature = "backend-candle")] ActiveBackend::Candle(b) => b.compress_text(prompt, target_len, max_chunk).await,
            #[cfg(feature = "backend-llamacpp")] ActiveBackend::LlamaCpp(b) => b.compress_text(prompt, target_len, max_chunk).await,
        }
    }
    pub fn supports_extractive_compression(&self) -> bool {
        match self {
            #[cfg(feature = "backend-candle")] ActiveBackend::Candle(b) => b.supports_extractive_compression(),
            #[cfg(feature = "backend-llamacpp")] ActiveBackend::LlamaCpp(b) => b.supports_extractive_compression(),
        }
    }
    pub fn get_vram_usage(&self) -> Option<(u64, u64)> {
        match self {
            #[cfg(feature = "backend-candle")] ActiveBackend::Candle(b) => b.get_vram_usage(),
            #[cfg(feature = "backend-llamacpp")] ActiveBackend::LlamaCpp(b) => b.get_vram_usage(),
        }
    }
    pub fn is_statically_allocated(&self) -> bool {
        match self {
            #[cfg(feature = "backend-candle")] ActiveBackend::Candle(b) => b.is_statically_allocated(),
            #[cfg(feature = "backend-llamacpp")] ActiveBackend::LlamaCpp(b) => b.is_statically_allocated(),
        }
    }
    pub fn get_offload_pct(&self) -> f32 {
        match self {
            #[cfg(feature = "backend-candle")] ActiveBackend::Candle(b) => b.get_offload_pct(),
            #[cfg(feature = "backend-llamacpp")] ActiveBackend::LlamaCpp(b) => b.get_offload_pct(),
        }
    }
}

pub fn create_backend(btype: &BackendType) -> Result<ActiveBackend, String> {
    match btype {
        #[cfg(feature = "backend-candle")]
        BackendType::Candle => Ok(ActiveBackend::Candle(Box::default())),
        #[cfg(feature = "backend-llamacpp")]
        BackendType::LlamaCpp => Ok(ActiveBackend::LlamaCpp(Box::new(LlamaCppEngine::new()?))),
        #[allow(unreachable_patterns)]
        _ => Err(format!("Backend {:?} is not enabled in this build.", btype)),
    }
}

// Helper: Formats the array into a generic string format
fn format_chat(messages: &[Message], arch: &ModelArch) -> String {
    let mut prompt = String::new();
    match arch {
        ModelArch::Qwen2 => {
            for msg in messages { prompt.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", msg.role, msg.content)); }
            prompt.push_str("<|im_start|>assistant\n");
        },
        ModelArch::Llama => {
            for msg in messages { prompt.push_str(&format!("<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>", msg.role, msg.content)); }
            prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
        },
        _ => {
            for msg in messages { prompt.push_str(&format!("{}: {}\n", msg.role, msg.content)); }
            prompt.push_str("assistant: ");
        }
    }
    prompt
}

pub async fn run_batcher_loop(
    mut receiver: mpsc::Receiver<UserRequest>, 
    status: Arc<Mutex<EngineStatus>>,
    telemetry: Arc<Mutex<TelemetryStore>>
) {
    let nvml = Nvml::init().ok();

    let mut active_model_id = String::new();
    let mut active_backend: Option<ActiveBackend> = None;
    let mut active_max_context: usize = 2048; 
    let mut active_model_config: Option<ModelConfig> = None;
    let mut active_memory_strategy = String::new();

    let mut tokenizer_cache: std::collections::HashMap<String, Tokenizer> = std::collections::HashMap::new();

    println!("⚙️  ORCHESTRATOR ONLINE: Waiting for requests...");

    'main: while let Some(request) = receiver.recv().await {
        println!("📥 Processing new chat request...");

        let last_message = match request.messages.last() {
            Some(msg) => msg.clone(),
            None => {
                println!("⚠️ Rejected request: No messages provided.");
                let _ = request.responder.send(StreamEvent::Error("Server Error: Request contained no messages.".to_string()));
                continue 'main;
            }
        };

        let requested_max_tokens = request.parameters.max_tokens.unwrap_or(500);
        let ctx_buffer = request.parameters.context_buffer.unwrap_or(0);
        let config_for_prompt = match get_model_registry().into_iter().find(|c| c.id == request.chat_model_id) {
            Some(c) => c,
            None => {
                let _ = request.responder.send(StreamEvent::Error("Server Error: Active model missing from registry.".to_string()));
                continue 'main;
            }
        };

        // Fetch the specific tokenizer to ensure perfect context memory math
        let tokenizer = match tokenizer_cache.get(&config_for_prompt.tokenizer_repo) {
            Some(tok) => tok.clone(),
            None => {
                let repo = config_for_prompt.tokenizer_repo.clone();
                let tok_res = tokio::task::spawn_blocking(move || {
                    let api = Api::new().map_err(|e| e.to_string())?;
                    let path = api.model(repo).get("tokenizer.json").map_err(|e| e.to_string())?;
                    Tokenizer::from_file(path).map_err(|e| e.to_string())
                }).await;

                match tok_res {
                    Ok(Ok(tok)) => {
                        tokenizer_cache.insert(config_for_prompt.tokenizer_repo.clone(), tok.clone());
                        tok.clone()
                    },
                    _ => {
                        let _ = request.responder.send(StreamEvent::Error("Server Error: Failed to load Tokenizer for exact counting.".to_string()));
                        continue 'main;
                    }
                }
            }
        };

        let formatted_prompt_pre = format_chat(&request.messages, &config_for_prompt.arch);
        let token_count_pre = tokenizer.encode(formatted_prompt_pre.clone(), true)
            .map(|enc| enc.get_ids().len())
            .unwrap_or_else(|_| formatted_prompt_pre.len() / 4);
        let actual_required_ctx = (token_count_pre + requested_max_tokens).max(2048);
        let target_allocated_ctx = actual_required_ctx + ctx_buffer;

        let req_strategy = request.parameters.memory_strategy.clone().unwrap_or_else(|| "offload".to_string());

        let mut needs_reload = active_model_id != request.chat_model_id || active_memory_strategy != req_strategy;
        
        if !needs_reload && actual_required_ctx > active_max_context && req_strategy == "offload" {
            println!("🔄 Expanding KV Cache from {} to {} tokens...", active_max_context, target_allocated_ctx);
            needs_reload = true;
        }

        // 1. Hot-Swap to the requested Chat Model
        if needs_reload {
            println!("🔄 Swapping VRAM to {}...", request.chat_model_id);

            if active_backend.is_some() {
                if active_backend.as_ref().unwrap().get_offload_pct() > 0.0 {
                    let mut s = lock_status(&status);
                    s.log_ram("Free", "Orchestrator", &format!("Released offloaded layers for {}", active_model_id), 0);
                }
                drop(active_backend.take());
                let mut s = lock_status(&status);
                s.remove_model_vram(&active_model_id);
                s.log_vram("Free", "Orchestrator", &format!("Released {} from VRAM", active_model_id), 0);
                if let Some((used, total, free)) = get_vram_info(nvml.as_ref(), 0) {
                    s.update_nvml(total, used, free);
                }
            }

            // wipe the ID and config so a failed load doesn't leave a poison state
            active_model_id.clear();
            active_model_config = None;

            // Mark all other potentially loaded models as Idle before loading the new one
            {
                let mut s = lock_status(&status);
                for model in s.models_vram.iter_mut() { model.status = "Idle".to_string(); }
            }

            let config = config_for_prompt.clone();
            let btype = request.target_backend.as_ref()
                .and_then(|req_b| config.supported_backends.iter().find(|sb| format!("{:?}", sb).to_lowercase() == req_b.to_lowercase()))
                .unwrap_or_else(|| config.supported_backends.first().unwrap())
                .clone();

            let mut backend = match create_backend(&btype) {
                Ok(b) => b,
                Err(e) => {
                    let _ = request.responder.send(StreamEvent::Error(format!("Server Error: {}", e)));
                    continue 'main;
                }
            };

            let load_start = Instant::now();
            active_memory_strategy = req_strategy;

            let actual_context = match backend.load_model(&config, status.clone(), &active_memory_strategy, target_allocated_ctx).await {
                Ok(ctx) => ctx,
                Err(e) => {
                println!("❌ Chat model load failed: {}", e);
                {
                    let mut s = lock_status(&status);
                    s.log_vram("Fail", "Orchestrator", &format!("Failed to allocate VRAM for {}", config.id), 0);
                }
                let _ = request.responder.send(StreamEvent::Error(format!("Server Error: Failed to load chat model: {}", e)));
                continue 'main;
            }
            };

            let elapsed = load_start.elapsed().as_millis();
            println!("⏱️ Model loaded in {} ms using {:?}", elapsed, btype);
            if let Ok(mut t) = telemetry.lock() {
                t.record_load(request.chat_model_id.clone(), format!("{:?}", btype), elapsed);
            }
            
            active_backend = Some(backend);
            active_model_id = request.chat_model_id.clone();
            active_max_context = actual_context;
            active_model_config = Some(config);
            println!("✅ Model limits established. Max context window: {}", active_max_context);

            {
                let mut current_status = lock_status(&status);
                current_status.active_chat_model_id = Some(active_model_id.clone());
                current_status.active_backend = Some(format!("{:?}", btype));
            }
        }

        let config = match active_model_config.as_ref() {
            Some(c) => c,
            None => {
                let _ = request.responder.send(StreamEvent::Error("Server Error: No active model configuration found. Please initialize a model first.".to_string()));
                continue 'main;
            }
        };

        let mut formatted_prompt = format_chat(&request.messages, &config.arch);
        
        // Exact token count using the actual Hugging Face tokenizer
        let mut token_count = tokenizer.encode(formatted_prompt.clone(), true)
            .map(|enc| enc.get_ids().len())
            .unwrap_or_else(|_| formatted_prompt.len() / 4);

        // --- THE DYNAMIC MEMORY MANAGER ---
        let mut trigger_compression = false;
        let mut dynamic_target_budget = active_max_context;

        // Use the backend's get_vram_usage if available, otherwise fallback to Orchestrator's NVML
        let vram_info = active_backend.as_ref().unwrap().get_vram_usage()
            .map(|(u, t)| (u, t, t.saturating_sub(u)))
            .or_else(|| get_vram_info(nvml.as_ref(), 0));

        let static_alloc = active_backend.as_ref().unwrap().is_statically_allocated();
        
        if static_alloc {
            println!("🧮 MEMORY CHECK: Statically allocated up to {} tokens.", active_max_context);
            if token_count + requested_max_tokens > active_max_context {
                println!("⚠️ WARNING: Prompt + Max Tokens exceeds KV Cache! Triggering compression.");
                trigger_compression = true;
                dynamic_target_budget = active_max_context.saturating_sub(requested_max_tokens).max(256);
            } else if token_count > (active_max_context as f32 * 0.80) as usize {
                trigger_compression = true;
                dynamic_target_budget = ((active_max_context as f32 * 0.50) as usize).max(256).min(active_max_context);
            } else if request.force_compression {
                println!("⚠️ Benchmarking: Forcing compression execution.");
                trigger_compression = true;
                dynamic_target_budget = ((token_count as f32 * 0.50) as usize).max(256).min(active_max_context);
            }
        } else if let Some((_, _, free_vram)) = vram_info {
                // This rough heuristic is only for the Candle backend's dynamic memory check.
                // Llama.cpp calculates this precisely during its static allocation.
                let bytes_per_token = match &config.arch {
                    ModelArch::Qwen2 if config.parameters_billions > 10.0 => 150_000,
                    ModelArch::Qwen2 => 80_000,
                    ModelArch::Llama if config.parameters_billions < 10.0 => 125_000,
                    _ => 100_000,
                };
                let safe_free_vram = free_vram.saturating_sub(500 * 1024 * 1024); 
                let absolute_max_tokens = (safe_free_vram as usize / bytes_per_token).min(active_max_context);
                
                println!("🧮 MEMORY CHECK: Free VRAM can hold ~{} tokens.", absolute_max_tokens);

                if token_count + requested_max_tokens > absolute_max_tokens {
                    println!("⚠️ WARNING: Prompt exceeds physical VRAM limits! Triggering dynamic compression.");
                    trigger_compression = true;
                    dynamic_target_budget = absolute_max_tokens.saturating_sub(requested_max_tokens).max(256); 
                } else if token_count > (active_max_context as f32 * 0.80) as usize {
                    trigger_compression = true;
                    dynamic_target_budget = ((active_max_context as f32 * 0.50) as usize).max(256).min(absolute_max_tokens);
                } else if request.force_compression {
                    println!("⚠️ Benchmarking: Forcing compression execution.");
                    trigger_compression = true;
                    dynamic_target_budget = ((token_count as f32 * 0.50) as usize).max(256).min(absolute_max_tokens);
                }
        } else {
            // CPU fallback
                if token_count + requested_max_tokens > active_max_context {
                    println!("⚠️ WARNING: Prompt + Max Tokens exceeds KV Cache! Triggering compression.");
                    trigger_compression = true;
                    dynamic_target_budget = active_max_context.saturating_sub(requested_max_tokens).max(256);
                } else if token_count > (active_max_context as f32 * 0.80) as usize {
                    trigger_compression = true;
                    dynamic_target_budget = ((active_max_context as f32 * 0.50) as usize).max(256).min(active_max_context);
                } else if request.force_compression {
                    println!("⚠️ Benchmarking: Forcing compression execution.");
                    trigger_compression = true;
                    dynamic_target_budget = ((token_count as f32 * 0.50) as usize).max(256).min(active_max_context);
                }
        }

        if trigger_compression {
            if let Some((used_start, total, _)) = vram_info { 
                println!("📊 VRAM before compressor: {:.2}GB / {:.2}GB", 
                    used_start as f32 / 1024.0_f32.powi(3), 
                    total as f32 / 1024.0_f32.powi(3));
            }

            // Mark the main chat model as idle while the compressor is active
            {
                let mut s = lock_status(&status);
                s.set_model_status(&active_model_id, "Idle");
            }

            let comp_config = match get_model_registry().into_iter().find(|c| c.id == request.compressor_model_id) {
                Some(c) => c,
                None => {
                    let _ = request.responder.send(StreamEvent::Error("Server Error: Compressor missing from registry.".to_string()));
                    continue 'main;
                }
            };

            let comp_btype = request.target_backend.as_ref()
                .and_then(|req_b| comp_config.supported_backends.iter().find(|sb| format!("{:?}", sb).to_lowercase() == req_b.to_lowercase()))
                .unwrap_or_else(|| comp_config.supported_backends.first().unwrap());

            let mut comp_backend = match create_backend(comp_btype) {
                Ok(b) => b,
                Err(e) => {
                    let _ = request.responder.send(StreamEvent::Error(format!("Server Error: {}", e)));
                    continue 'main;
                }
            };

            let comp_required_ctx = (token_count + requested_max_tokens).max(2048) + ctx_buffer;
            // --- RECORD COMPRESSOR LOAD TIME ---
            let comp_load_start = Instant::now();
            if let Err(e) = comp_backend.load_model(&comp_config, status.clone(), "offload", comp_required_ctx).await {
                {
                    let mut s = lock_status(&status);
                    s.log_vram("Fail", "Orchestrator", &format!("Failed to allocate VRAM for {}", comp_config.id), 0);
                }
                let _ = request.responder.send(StreamEvent::Error(format!("Server Error: Failed to load compressor: {}", e)));
                continue 'main; 
            }
            
            let comp_load_elapsed = comp_load_start.elapsed().as_millis();
            println!("⏱️ Compressor loaded in {} ms using {:?}", comp_load_elapsed, comp_btype);
            if let Ok(mut t) = telemetry.lock() {
                t.record_load(request.compressor_model_id.clone(), format!("{:?}", comp_btype), comp_load_elapsed);
            }

            {
                let mut current_status = lock_status(&status);
                current_status.last_compressor_model_id = Some(request.compressor_model_id.clone());
            }

            let target_budget = dynamic_target_budget; 

            // --- RECORD COMPRESSION EXECUTION TIME ---
            let comp_start = Instant::now();
            
            let (summary, comp_tok_time) = if comp_backend.supports_extractive_compression() {
                match comp_backend.compress_text(&formatted_prompt, target_budget, comp_config.max_context_len).await {
                    Ok(compressed) => compressed,
                    Err(e) => {
                        let _ = request.responder.send(StreamEvent::Error(format!("Server Error: Context compression failed: {}", e)));
                        continue 'main;
                    }
                }
            } else {
                // Abstractive fallback using generative backend
                let compression_prompt = format!("<|user|>\nSummarize this text to be shorter:\n{}</s>\n<|assistant|>\n", formatted_prompt);
                let params = GenerationParameters {
                    max_tokens: Some(target_budget),
                    ..Default::default()
                };
                
                match comp_backend.generate_text(&compression_prompt, &params).await {
                    Ok(text) => text,
                    Err(e) => {
                        let _ = request.responder.send(StreamEvent::Error(format!("Server Error: Generation failed: {}", e)));
                        continue 'main;
                    }
                }
            };

            let comp_elapsed = comp_start.elapsed().as_millis();
            println!("⏱️ Compression completed in {} ms (Tokenization: {} ms)", comp_elapsed, comp_tok_time);
            let comp_btype_str = format!("{:?}", comp_btype);
            let comp_offload_pct = comp_backend.get_offload_pct();
            if let Ok(mut t) = telemetry.lock() {
                t.record_generation(request.compressor_model_id.clone(), comp_btype_str, request.parameters.clone(), comp_offload_pct, formatted_prompt.len(), token_count, comp_tok_time, comp_elapsed);
            }

            if comp_backend.get_offload_pct() > 0.0 {
                let mut s = lock_status(&status);
                s.log_ram("Free", "Orchestrator", &format!("Released offloaded layers for {}", request.compressor_model_id), 0);
            }
            drop(comp_backend);
            
            {
                let mut s = lock_status(&status);
                s.remove_model_vram(&request.compressor_model_id);
                s.log_vram("Free", "Orchestrator", &format!("Released {} from VRAM", request.compressor_model_id), 0);
            }

            // Mark the main chat model as active again now that the compressor is gone
            {
                let mut s = lock_status(&status);
                s.set_model_status(&active_model_id, "Active");
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
            
            formatted_prompt = format_chat(&new_messages, &config.arch);
            token_count = tokenizer.encode(formatted_prompt.clone(), true)
                .map(|enc| enc.get_ids().len())
                .unwrap_or_else(|_| formatted_prompt.len() / 4); // Update exact count for the compressed prompt
        }

        println!("📥 Processing prompt...");
        let gen_start = Instant::now();

        let (inner_tx, mut inner_rx) = tokio::sync::mpsc::unbounded_channel();
        let mut tokenization_time_ms: u128 = 0;
        let responder = request.responder;

        let gen_fut = active_backend.as_mut().unwrap().generate_stream(&formatted_prompt, &request.parameters, inner_tx);
        
        // Concurrently run the backend generator and dynamically intercept internal stream
        // events like 'TokenizationTime' so they don't leak into the web HTTP chunked response!
        let fwd_fut = async {
            while let Some(ev) = inner_rx.recv().await {
                match ev {
                    StreamEvent::TokenizationTime(t) => tokenization_time_ms = t,
                    other => { let _ = responder.send(other); }
                }
            }
        };

        tokio::join!(gen_fut, fwd_fut);
            
        let elapsed = gen_start.elapsed().as_millis();
        println!("⏱️ Generation completed in {} ms (Tokenization: {} ms)", elapsed, tokenization_time_ms);
        let active_backend_name = {
            let s = lock_status(&status);
            s.active_backend.clone().unwrap_or_else(|| "Unknown".to_string())
        };
        let offload_pct = active_backend.as_ref().unwrap().get_offload_pct();
        if let Ok(mut t) = telemetry.lock() {
            t.record_generation(active_model_id.clone(), active_backend_name, request.parameters.clone(), offload_pct, formatted_prompt.len(), token_count, tokenization_time_ms, elapsed);
        }
    } // Closes the 'main while loop
} // Closes the run_batcher_loop function