use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokio::sync::mpsc;

use hf_hub::api::sync::Api;
use nvml_wrapper::Nvml;
use tokenizers::Tokenizer;

const CANDLE_COMPUTE_MARGIN_BYTES: u64 = 500 * 1024 * 1024;

pub mod backend;
#[cfg(feature = "backend-candle")]
pub mod backend_candle;
#[cfg(feature = "backend-llamacpp")]
pub mod backend_llamacpp;
pub mod registry;
pub mod telemetry;
pub mod types;

pub use backend::*;
#[cfg(feature = "backend-candle")]
pub use backend_candle::*;
#[cfg(feature = "backend-llamacpp")]
pub use backend_llamacpp::*;
pub use registry::*;
pub use telemetry::*;
pub use types::*;

pub fn get_vram_info(nvml: Option<&Nvml>, device_index: u32) -> Option<(u64, u64, u64)> {
    let nvml = nvml?;
    let device = nvml.device_by_index(device_index).ok()?;
    let info = device.memory_info().ok()?;

    Some((info.used, info.total, info.free))
}

pub async fn wait_for_vram_release(
    nvml: Option<&Nvml>,
    device_index: u32,
    vram_before: u64,
    expected_release_bytes: u64,
    model_id: &str,
    backend_name: &str,
) {
    if nvml.is_none() || expected_release_bytes == 0 {
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        return;
    }

    let start = Instant::now();
    let timeout = std::time::Duration::from_secs(10); // 10 second max fallback

    // Allow a 64MB variance for OS background processes and driver caching that might fluctuate concurrently
    let target_vram = vram_before
        .saturating_sub(expected_release_bytes)
        .saturating_add(64 * 1024 * 1024);

    while start.elapsed() < timeout {
        if let Some((used, _, _)) = get_vram_info(nvml, device_index)
            && used <= target_vram
        {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }

    println!(
        "⚠️ WARNING: VRAM release timeout exceeded for {} on {}! System might OOM.",
        model_id, backend_name
    );
}

fn get_exact_token_count(prompt: &str, tokenizer: &Tokenizer) -> usize {
    tokenizer
        .encode(prompt, true)
        .map(|enc| enc.get_ids().len())
        .unwrap_or_else(|_| prompt.len() / 4)
}

pub enum ActiveBackend {
    #[cfg(feature = "backend-candle")]
    Candle(Box<CandleEngine>),
    #[cfg(feature = "backend-llamacpp")]
    LlamaCpp(Box<LlamaCppEngine>),
}

impl ActiveBackend {
    pub async fn load_model(
        &mut self,
        config: &ModelConfig,
        status: Arc<Mutex<EngineStatus>>,
        strategy: &MemoryStrategy,
        required_ctx: usize,
    ) -> Result<usize, String> {
        match self {
            #[cfg(feature = "backend-candle")]
            ActiveBackend::Candle(b) => b.load_model(config, status, strategy, required_ctx).await,
            #[cfg(feature = "backend-llamacpp")]
            ActiveBackend::LlamaCpp(b) => {
                b.load_model(config, status, strategy, required_ctx).await
            }
        }
    }
    pub async fn generate_stream(
        &mut self,
        prompt: &str,
        params: &GenerationParameters,
        tx: tokio::sync::mpsc::UnboundedSender<crate::types::StreamEvent>,
    ) {
        match self {
            #[cfg(feature = "backend-candle")]
            ActiveBackend::Candle(b) => b.generate_stream(prompt, params, tx).await,
            #[cfg(feature = "backend-llamacpp")]
            ActiveBackend::LlamaCpp(b) => b.generate_stream(prompt, params, tx).await,
        }
    }
    pub async fn generate_text(
        &mut self,
        prompt: &str,
        params: &GenerationParameters,
    ) -> Result<(String, u128), String> {
        match self {
            #[cfg(feature = "backend-candle")]
            ActiveBackend::Candle(b) => b.generate_text(prompt, params).await,
            #[cfg(feature = "backend-llamacpp")]
            ActiveBackend::LlamaCpp(b) => b.generate_text(prompt, params).await,
        }
    }
    pub async fn compress_text(
        &mut self,
        prompt: &str,
        target_len: usize,
        max_chunk: usize,
    ) -> Result<(String, u128), String> {
        match self {
            #[cfg(feature = "backend-candle")]
            ActiveBackend::Candle(b) => b.compress_text(prompt, target_len, max_chunk).await,
            #[cfg(feature = "backend-llamacpp")]
            ActiveBackend::LlamaCpp(b) => b.compress_text(prompt, target_len, max_chunk).await,
        }
    }
    pub fn supports_extractive_compression(&self) -> bool {
        match self {
            #[cfg(feature = "backend-candle")]
            ActiveBackend::Candle(b) => b.supports_extractive_compression(),
            #[cfg(feature = "backend-llamacpp")]
            ActiveBackend::LlamaCpp(b) => b.supports_extractive_compression(),
        }
    }
    pub fn get_vram_usage(&self) -> Option<(u64, u64)> {
        match self {
            #[cfg(feature = "backend-candle")]
            ActiveBackend::Candle(b) => b.get_vram_usage(),
            #[cfg(feature = "backend-llamacpp")]
            ActiveBackend::LlamaCpp(b) => b.get_vram_usage(),
        }
    }
    pub fn is_statically_allocated(&self) -> bool {
        match self {
            #[cfg(feature = "backend-candle")]
            ActiveBackend::Candle(b) => b.is_statically_allocated(),
            #[cfg(feature = "backend-llamacpp")]
            ActiveBackend::LlamaCpp(b) => b.is_statically_allocated(),
        }
    }
    pub fn get_offload_pct(&self) -> f32 {
        match self {
            #[cfg(feature = "backend-candle")]
            ActiveBackend::Candle(b) => b.get_offload_pct(),
            #[cfg(feature = "backend-llamacpp")]
            ActiveBackend::LlamaCpp(b) => b.get_offload_pct(),
        }
    }
}

pub fn create_backend(btype: &BackendType, gpu_device_index: u32) -> Result<ActiveBackend, String> {
    match btype {
        #[cfg(feature = "backend-candle")]
        BackendType::Candle => Ok(ActiveBackend::Candle(Box::new(CandleEngine::new(
            gpu_device_index,
        )))),
        #[cfg(feature = "backend-llamacpp")]
        BackendType::LlamaCpp => Ok(ActiveBackend::LlamaCpp(Box::new(LlamaCppEngine::new(
            gpu_device_index,
        )?))),
        #[allow(unreachable_patterns)]
        _ => Err(format!("Backend {:?} is not enabled in this build.", btype)),
    }
}

pub async fn run_batcher_loop(
    mut receiver: mpsc::Receiver<UserRequest>,
    status: Arc<Mutex<EngineStatus>>,
    telemetry: Arc<Mutex<TelemetryStore>>,
    gpu_device_index: u32,
) {
    let nvml = Nvml::init().ok();

    let mut active_model_id = String::new();
    let mut active_backend: Option<ActiveBackend> = None;
    let mut active_backend_name = String::new();
    let mut active_backend_type: Option<BackendType> = None;
    let mut active_max_context: usize = 2048;
    let mut active_model_config: Option<ModelConfig> = None;
    let mut active_memory_strategy = MemoryStrategy::Offload;

    let mut tokenizer_cache: std::collections::HashMap<String, Tokenizer> =
        std::collections::HashMap::new();

    println!("⚙️  ORCHESTRATOR ONLINE: Waiting for requests...");

    'main: while let Some(request) = receiver.recv().await {
        println!("📥 Processing new chat request...");

        let last_message = match request.messages.last() {
            Some(msg) => msg.clone(),
            None => {
                println!("⚠️ Rejected request: No messages provided.");
                let _ = request.responder.send(StreamEvent::Error(
                    "Server Error: Request contained no messages.".to_string(),
                ));
                continue 'main;
            }
        };

        let requested_max_tokens = request.parameters.max_tokens.unwrap_or(500);
        let ctx_buffer = request.parameters.context_buffer.unwrap_or(0);
        let mut config_for_prompt = match get_model_registry()
            .await
            .iter()
            .find(|c| c.id == request.chat_model_id)
        {
            Some(c) => c.clone(),
            None => {
                let _ = request.responder.send(StreamEvent::Error(
                    "Server Error: Active model missing from registry.".to_string(),
                ));
                continue 'main;
            }
        };

        let req_yarn = request.parameters.yarn_enabled.unwrap_or(true);
        if req_yarn {
            config_for_prompt.max_context_len = config_for_prompt.max_yarn_context;
        }

        // Fetch the specific tokenizer to ensure perfect context memory math
        let tokenizer = match tokenizer_cache.get(&config_for_prompt.tokenizer_repo) {
            Some(tok) => tok.clone(),
            None => {
                let repo = config_for_prompt.tokenizer_repo.clone();
                let tok_res = tokio::task::spawn_blocking(move || {
                    let api = Api::new().map_err(|e| e.to_string())?;
                    let path = api
                        .model(repo)
                        .get("tokenizer.json")
                        .map_err(|e| e.to_string())?;
                    Tokenizer::from_file(path).map_err(|e| e.to_string())
                })
                .await;

                match tok_res {
                    Ok(Ok(tok)) => {
                        if tokenizer_cache.len() >= 5 {
                            // Simple eviction to prevent memory leak
                            tokenizer_cache.clear();
                        }
                        tokenizer_cache
                            .insert(config_for_prompt.tokenizer_repo.clone(), tok.clone());
                        tok.clone()
                    }
                    _ => {
                        let _ = request.responder.send(StreamEvent::Error(
                            "Server Error: Failed to load Tokenizer for exact counting."
                                .to_string(),
                        ));
                        continue 'main;
                    }
                }
            }
        };

        let req_b = request.target_backend.as_deref().map(|s| s.to_lowercase());
        let target_btype_opt = match req_b.as_deref() {
            Some(b) if b != "auto" => config_for_prompt
                .supported_backends
                .iter()
                .find(|sb| format!("{:?}", sb).to_lowercase() == b)
                .cloned(),
            _ => None,
        };

        let target_btype = match target_btype_opt {
            Some(b) => b,
            None => {
                // AUTO SELECTION LOGIC
                // If the requested model is already loaded, stick with its currently active backend to prevent an unnecessary VRAM reload.
                if active_model_id == request.chat_model_id
                    && let Some(b) = config_for_prompt
                        .supported_backends
                        .iter()
                        .find(|&sb| Some(*sb) == active_backend_type)
                {
                    *b
                } else if config_for_prompt
                    // If we are loading a different model, the VRAM IO cost is being paid anyway. Default to Llama.cpp for best performance.
                    .supported_backends
                    .contains(&BackendType::LlamaCpp)
                {
                    BackendType::LlamaCpp
                } else if let Some(b) = config_for_prompt.supported_backends.first() {
                    // Fallback to whatever the first supported backend is.
                    *b
                } else {
                    let _ = request.responder.send(StreamEvent::Error(
                        "Server Error: No supported backend found for this model.".to_string(),
                    ));
                    continue 'main;
                }
            }
        };
        let target_backend_name = format!("{:?}", target_btype);

        let formatted_prompt_pre = config_for_prompt.arch.format_chat(&request.messages);
        let token_count_pre = get_exact_token_count(&formatted_prompt_pre, &tokenizer);
        let actual_required_ctx = (token_count_pre + requested_max_tokens).max(2048);
        let target_allocated_ctx = actual_required_ctx + ctx_buffer;

        let req_strategy = request
            .parameters
            .memory_strategy
            .clone()
            .unwrap_or(MemoryStrategy::Offload);

        let mut needs_reload = active_model_id != request.chat_model_id
            || active_memory_strategy != req_strategy
            || active_backend_name != target_backend_name;

        if !needs_reload
            && actual_required_ctx > active_max_context
            && req_strategy == MemoryStrategy::Offload
            && active_max_context < config_for_prompt.max_context_len
        {
            println!(
                "🔄 Expanding KV Cache from {} to {} tokens...",
                active_max_context,
                target_allocated_ctx.min(config_for_prompt.max_context_len)
            );
            needs_reload = true;
        }

        // 1. Hot-Swap to the requested Chat Model
        if needs_reload {
            println!("🔄 Swapping VRAM to {}...", request.chat_model_id);

            if let Some(backend) = active_backend.take() {
                let offload_pct = backend.get_offload_pct();

                let expected_release = {
                    let s = lock_status(&status);
                    s.models_vram
                        .iter()
                        .find(|m| m.id == active_model_id)
                        .map(|m| m.weights + m.kv_cache)
                        .unwrap_or(0)
                };
                let vram_before = get_vram_info(nvml.as_ref(), gpu_device_index)
                    .map(|(u, _, _)| u)
                    .unwrap_or(0);

                drop(backend);

                // Dynamically await VRAM cleanup
                wait_for_vram_release(
                    nvml.as_ref(),
                    gpu_device_index,
                    vram_before,
                    expected_release,
                    &active_model_id,
                    &active_backend_name,
                )
                .await;

                let mut s = lock_status(&status);
                if offload_pct > 0.0 {
                    s.log_ram(
                        "Free",
                        "Orchestrator",
                        &format!("Released offloaded layers for {}", active_model_id),
                        0,
                    );
                }
                s.remove_model_vram(&active_model_id);
                s.log_vram(
                    "Free",
                    "Orchestrator",
                    &format!("Released {} from VRAM", active_model_id),
                    0,
                );
                if let Some((used, total, free)) = get_vram_info(nvml.as_ref(), gpu_device_index) {
                    s.update_nvml(total, used, free);
                }
            }

            // wipe the ID and config so a failed load doesn't leave a poison state
            active_model_id.clear();
            active_model_config = None;

            // Mark all other potentially loaded models as Idle before loading the new one
            {
                let mut s = lock_status(&status);
                for model in s.models_vram.iter_mut() {
                    model.status = "Idle".to_string();
                }
            }

            let config = config_for_prompt.clone();
            let mut backend = match create_backend(&target_btype, gpu_device_index) {
                Ok(b) => b,
                Err(e) => {
                    let _ = request
                        .responder
                        .send(StreamEvent::Error(format!("Server Error: {}", e)));
                    continue 'main;
                }
            };

            let load_start = Instant::now();
            active_memory_strategy = req_strategy.clone();

            let actual_context = match backend
                .load_model(
                    &config,
                    status.clone(),
                    &active_memory_strategy,
                    target_allocated_ctx,
                )
                .await
            {
                Ok(ctx) => ctx,
                Err(e) => {
                    println!("❌ Chat model load failed: {}", e);
                    {
                        let mut s = lock_status(&status);
                        s.log_vram(
                            "Fail",
                            "Orchestrator",
                            &format!("Failed to allocate VRAM for {}", config.id),
                            0,
                        );
                    }
                    let _ = request.responder.send(StreamEvent::Error(format!(
                        "Server Error: Failed to load chat model: {}",
                        e
                    )));
                    continue 'main;
                }
            };

            let elapsed = load_start.elapsed().as_millis();
            println!("⏱️ Model loaded in {} ms using {:?}", elapsed, target_btype);
            if let Ok(mut t) = telemetry.lock() {
                t.record_load(
                    request.chat_model_id.clone(),
                    target_backend_name.clone(),
                    elapsed,
                );
            }

            active_backend = Some(backend);
            active_backend_name = target_backend_name.clone();
            active_backend_type = Some(target_btype);
            active_model_id = request.chat_model_id.clone();
            active_max_context = actual_context;
            active_model_config = Some(config);
            println!(
                "✅ Model limits established. Max context window: {}",
                active_max_context
            );

            {
                let mut current_status = lock_status(&status);
                current_status.active_chat_model_id = Some(active_model_id.clone());
                current_status.active_backend = Some(target_backend_name);
            }
        }

        let config = match active_model_config.as_ref() {
            Some(c) => c,
            None => {
                let _ = request.responder.send(StreamEvent::Error("Server Error: No active model configuration found. Please initialize a model first.".to_string()));
                continue 'main;
            }
        };

        // Mark the chat model as active now that we are officially processing the prompt
        {
            let mut current_status = lock_status(&status);
            current_status.set_model_status(&active_model_id, "Active");
        }

        let mut formatted_prompt = config.arch.format_chat(&request.messages);

        // Exact token count using the active backend's tokenizer if available
        let mut token_count = get_exact_token_count(&formatted_prompt, &tokenizer);

        // --- THE DYNAMIC MEMORY MANAGER ---
        let mut trigger_compression = false;
        let mut dynamic_target_budget = active_max_context;

        // Use the backend's get_vram_usage if available, otherwise fallback to Orchestrator's NVML
        let vram_info = active_backend
            .as_ref()
            .unwrap()
            .get_vram_usage()
            .map(|(u, t)| (u, t, t.saturating_sub(u)))
            .or_else(|| get_vram_info(nvml.as_ref(), gpu_device_index));

        let static_alloc = active_backend.as_ref().unwrap().is_statically_allocated();

        if static_alloc {
            println!(
                "🧮 MEMORY CHECK: Statically allocated up to {} tokens.",
                active_max_context
            );
            if token_count + requested_max_tokens > active_max_context {
                println!(
                    "⚠️ WARNING: Prompt + Max Tokens exceeds KV Cache! Triggering compression."
                );
                trigger_compression = true;
                dynamic_target_budget = active_max_context
                    .saturating_sub(requested_max_tokens)
                    .max(256);
            } else if token_count > (active_max_context as f32 * 0.80) as usize {
                trigger_compression = true;
                dynamic_target_budget = ((active_max_context as f32 * 0.50) as usize)
                    .max(256)
                    .min(active_max_context);
            } else if request.force_compression {
                println!("⚠️ Benchmarking: Forcing compression execution.");
                trigger_compression = true;
                dynamic_target_budget = ((token_count as f32 * 0.50) as usize)
                    .max(256)
                    .min(active_max_context);
            }
        } else if let Some((_, _, free_vram)) = vram_info {
            // This rough heuristic is only for the Candle backend's dynamic memory check.
            // Llama.cpp calculates this precisely during its static allocation.
            let bytes_per_token = config.estimate_kv_bytes_per_token();
            let safe_free_vram = free_vram.saturating_sub(CANDLE_COMPUTE_MARGIN_BYTES);
            let absolute_max_tokens =
                (safe_free_vram as usize / bytes_per_token).min(active_max_context);

            println!(
                "🧮 MEMORY CHECK: Free VRAM can hold ~{} tokens.",
                absolute_max_tokens
            );

            if token_count + requested_max_tokens > absolute_max_tokens {
                println!(
                    "⚠️ WARNING: Prompt exceeds physical VRAM limits! Triggering dynamic compression."
                );
                trigger_compression = true;
                dynamic_target_budget = absolute_max_tokens
                    .saturating_sub(requested_max_tokens)
                    .max(256);
            } else if token_count > (active_max_context as f32 * 0.80) as usize {
                trigger_compression = true;
                dynamic_target_budget = ((active_max_context as f32 * 0.50) as usize)
                    .max(256)
                    .min(absolute_max_tokens);
            } else if request.force_compression {
                println!("⚠️ Benchmarking: Forcing compression execution.");
                trigger_compression = true;
                dynamic_target_budget = ((token_count as f32 * 0.50) as usize)
                    .max(256)
                    .min(absolute_max_tokens);
            }
        } else {
            // CPU fallback
            if token_count + requested_max_tokens > active_max_context {
                println!(
                    "⚠️ WARNING: Prompt + Max Tokens exceeds KV Cache! Triggering compression."
                );
                trigger_compression = true;
                dynamic_target_budget = active_max_context
                    .saturating_sub(requested_max_tokens)
                    .max(256);
            } else if token_count > (active_max_context as f32 * 0.80) as usize {
                trigger_compression = true;
                dynamic_target_budget = ((active_max_context as f32 * 0.50) as usize)
                    .max(256)
                    .min(active_max_context);
            } else if request.force_compression {
                println!("⚠️ Benchmarking: Forcing compression execution.");
                trigger_compression = true;
                dynamic_target_budget = ((token_count as f32 * 0.50) as usize)
                    .max(256)
                    .min(active_max_context);
            }
        }

        if trigger_compression {
            if let Some((used_start, total, _)) = vram_info {
                println!(
                    "📊 VRAM before compressor: {:.2}GB / {:.2}GB",
                    used_start as f32 / 1024.0_f32.powi(3),
                    total as f32 / 1024.0_f32.powi(3)
                );
            }

            // Mark the main chat model as idle while the compressor is active
            {
                let mut s = lock_status(&status);
                s.set_model_status(&active_model_id, "Idle");
            }

            let comp_config = match get_model_registry()
                .await
                .iter()
                .find(|c| c.id == request.compressor_model_id)
            {
                Some(c) => c.clone(),
                None => {
                    let _ = request.responder.send(StreamEvent::Error(
                        "Server Error: Compressor missing from registry.".to_string(),
                    ));
                    continue 'main;
                }
            };

            let req_comp_b = request.target_backend.as_deref().map(|s| s.to_lowercase());
            let comp_btype_opt = match req_comp_b.as_deref() {
                Some(b) if b != "auto" => comp_config
                    .supported_backends
                    .iter()
                    .find(|sb| format!("{:?}", sb).to_lowercase() == b),
                _ => None,
            };

            let comp_btype = match comp_btype_opt {
                Some(b) => b,
                None => {
                    // Compressor auto selection logic
                    if comp_config
                        .supported_backends
                        .contains(&BackendType::LlamaCpp)
                    {
                        comp_config
                            .supported_backends
                            .iter()
                            .find(|sb| **sb == BackendType::LlamaCpp)
                            .unwrap()
                    } else if let Some(b) = comp_config.supported_backends.first() {
                        b
                    } else {
                        let _ = request.responder.send(StreamEvent::Error(
                            "Server Error: No supported backend found for compressor model."
                                .to_string(),
                        ));
                        continue 'main;
                    }
                }
            };

            let mut comp_backend = match create_backend(comp_btype, gpu_device_index) {
                Ok(b) => b,
                Err(e) => {
                    let _ = request
                        .responder
                        .send(StreamEvent::Error(format!("Server Error: {}", e)));
                    continue 'main;
                }
            };

            let comp_required_ctx = (token_count + requested_max_tokens).max(2048) + ctx_buffer;
            // --- RECORD COMPRESSOR LOAD TIME ---
            let comp_load_start = Instant::now();
            if let Err(e) = comp_backend
                .load_model(
                    &comp_config,
                    status.clone(),
                    &MemoryStrategy::Offload,
                    comp_required_ctx,
                )
                .await
            {
                {
                    let mut s = lock_status(&status);
                    s.log_vram(
                        "Fail",
                        "Orchestrator",
                        &format!("Failed to allocate VRAM for {}", comp_config.id),
                        0,
                    );
                }
                let _ = request.responder.send(StreamEvent::Error(format!(
                    "Server Error: Failed to load compressor: {}",
                    e
                )));
                continue 'main;
            }

            let comp_load_elapsed = comp_load_start.elapsed().as_millis();
            println!(
                "⏱️ Compressor loaded in {} ms using {:?}",
                comp_load_elapsed, comp_btype
            );
            if let Ok(mut t) = telemetry.lock() {
                t.record_load(
                    request.compressor_model_id.clone(),
                    format!("{:?}", comp_btype),
                    comp_load_elapsed,
                );
            }

            {
                let mut current_status = lock_status(&status);
                current_status.last_compressor_model_id = Some(request.compressor_model_id.clone());
            }

            let target_budget = dynamic_target_budget;

            // --- RECORD COMPRESSION EXECUTION TIME ---
            let comp_start = Instant::now();

            let (summary, comp_tok_time) = if comp_backend.supports_extractive_compression() {
                match comp_backend
                    .compress_text(
                        &formatted_prompt,
                        target_budget,
                        comp_config.max_context_len,
                    )
                    .await
                {
                    Ok(compressed) => compressed,
                    Err(e) => {
                        let _ = request.responder.send(StreamEvent::Error(format!(
                            "Server Error: Context compression failed: {}",
                            e
                        )));
                        continue 'main;
                    }
                }
            } else {
                // Abstractive fallback using generative backend
                let compression_messages = vec![Message {
                    role: "user".to_string(),
                    content: format!("Summarize this text to be shorter:\n{}", formatted_prompt),
                }];
                let compression_prompt = comp_config.arch.format_chat(&compression_messages);
                let params = GenerationParameters {
                    max_tokens: Some(target_budget),
                    ..Default::default()
                };

                match comp_backend
                    .generate_text(&compression_prompt, &params)
                    .await
                {
                    Ok(text) => text,
                    Err(e) => {
                        let _ = request.responder.send(StreamEvent::Error(format!(
                            "Server Error: Generation failed: {}",
                            e
                        )));
                        continue 'main;
                    }
                }
            };

            let comp_elapsed = comp_start.elapsed().as_millis();
            println!(
                "⏱️ Compression completed in {} ms (Tokenization: {} ms)",
                comp_elapsed, comp_tok_time
            );
            let comp_btype_str = format!("{:?}", comp_btype);
            let comp_offload_pct = comp_backend.get_offload_pct();
            if let Ok(mut t) = telemetry.lock() {
                t.record_generation(
                    request.compressor_model_id.clone(),
                    comp_btype_str.clone(),
                    request.parameters.clone(),
                    comp_offload_pct,
                    formatted_prompt.len(),
                    token_count,
                    comp_tok_time,
                    comp_elapsed,
                );
            }

            let expected_release = {
                let s = lock_status(&status);
                s.models_vram
                    .iter()
                    .find(|m| m.id == request.compressor_model_id)
                    .map(|m| m.weights + m.kv_cache)
                    .unwrap_or(0)
            };
            let vram_before = get_vram_info(nvml.as_ref(), gpu_device_index)
                .map(|(u, _, _)| u)
                .unwrap_or(0);

            drop(comp_backend);

            // Dynamically await exact VRAM cleanup
            wait_for_vram_release(
                nvml.as_ref(),
                gpu_device_index,
                vram_before,
                expected_release,
                &request.compressor_model_id,
                &comp_btype_str,
            )
            .await;

            {
                let mut s = lock_status(&status);
                if comp_offload_pct > 0.0 {
                    s.log_ram(
                        "Free",
                        "Orchestrator",
                        &format!(
                            "Released offloaded layers for {}",
                            request.compressor_model_id
                        ),
                        0,
                    );
                }
                s.remove_model_vram(&request.compressor_model_id);
                s.log_vram(
                    "Free",
                    "Orchestrator",
                    &format!("Released {} from VRAM", request.compressor_model_id),
                    0,
                );
                if let Some((used, total, free)) = get_vram_info(nvml.as_ref(), gpu_device_index) {
                    s.update_nvml(total, used, free);
                }
                // Mark the main chat model as active again now that the compressor is gone
                s.set_model_status(&active_model_id, "Active");
            }

            println!("🔄 Resuming Chat...");
            let mut new_messages = Vec::new();

            if request.messages.len() > 1 {
                // Multi-turn chat: Compress the older history, but keep their newest question intact
                new_messages.push(Message {
                    role: "system".to_string(),
                    content: format!("Compressed Context:\n{}", summary.trim()),
                });
                new_messages.push(last_message);
            } else {
                // Single massive file drop (or benchmark): The summary IS the new message
                new_messages.push(Message {
                    role: "user".to_string(),
                    content: format!("Review this compressed context:\n{}", summary.trim()),
                });
            }

            formatted_prompt = config.arch.format_chat(&new_messages);
            token_count = get_exact_token_count(&formatted_prompt, &tokenizer);
        }

        println!("📥 Processing prompt...");
        let gen_start = Instant::now();

        let (inner_tx, mut inner_rx) = tokio::sync::mpsc::unbounded_channel();
        let responder = request.responder;

        let gen_fut = active_backend.as_mut().unwrap().generate_stream(
            &formatted_prompt,
            &request.parameters,
            inner_tx,
        );

        // Concurrently run the backend generator and dynamically intercept internal stream
        // events like 'TokenizationTime' so they don't leak into the web HTTP chunked response!
        let fwd_fut = async {
            let mut terminal_received = false;
            let mut tok_time: u128 = 0;
            while let Some(ev) = inner_rx.recv().await {
                match ev {
                    StreamEvent::TokenizationTime(t) => tok_time = t,
                    StreamEvent::Done => {
                        terminal_received = true;
                        let _ = responder.send(StreamEvent::Done);
                        break;
                    }
                    StreamEvent::Error(e) => {
                        terminal_received = true;
                        let _ = responder.send(StreamEvent::Error(e));
                        break;
                    }
                    other => {
                        if responder.send(other).is_err() {
                            break; // Client disconnected, stop forwarding
                        }
                    }
                }
            }

            // If the backend panicked or dropped the channel without sending a terminal event,
            // synthesize one here so the HTTP stream closes gracefully and the UI doesn't hang.
            if !terminal_received {
                let _ = responder.send(StreamEvent::Done);
            }

            tok_time
        };

        let (_, tokenization_time_ms) = tokio::join!(gen_fut, fwd_fut);

        let elapsed = gen_start.elapsed().as_millis();
        println!(
            "⏱️ Generation completed in {} ms (Tokenization: {} ms)",
            elapsed, tokenization_time_ms
        );
        let offload_pct = active_backend.as_ref().unwrap().get_offload_pct();
        if let Ok(mut t) = telemetry.lock() {
            t.record_generation(
                active_model_id.clone(),
                active_backend_name.clone(),
                request.parameters.clone(),
                offload_pct,
                formatted_prompt.len(),
                token_count,
                tokenization_time_ms,
                elapsed,
            );
        }

        // Mark the model as Idle now that generation is complete
        {
            let mut current_status = lock_status(&status);
            current_status.set_model_status(&active_model_id, "Idle");
        }
    } // Closes the 'main while loop
} // Closes the run_batcher_loop function
