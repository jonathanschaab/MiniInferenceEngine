use async_trait::async_trait;
use hf_hub::api::sync::Api;
use self_cell::self_cell;
use std::num::NonZeroU32;

use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::AddBos;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::sampling::LlamaSampler;

use crate::backend::InferenceBackend;
use crate::registry::ModelConfig;
use crate::types::EngineStatus;
use crate::types::GenerationParameters;
use crate::types::MemoryStrategy;
use std::sync::OnceLock;
use std::sync::{Arc, Mutex};

const LLAMA_CPP_COMPUTE_MARGIN_BYTES: u64 = 1_500 * 1024 * 1024;

static LLAMA_BACKEND: OnceLock<LlamaBackend> = OnceLock::new();

self_cell! {
    struct LlamaInstance {
        owner: LlamaModel,
        #[covariant]
        dependent: LlamaContext,
    }
}

enum EngineCommand {
    LoadModel {
        config: ModelConfig,
        status: Arc<Mutex<EngineStatus>>,
        strategy: MemoryStrategy,
        required_ctx: usize,
        reply: tokio::sync::oneshot::Sender<Result<(usize, f32), String>>,
    },
    GenerateStream {
        prompt: String,
        params: GenerationParameters,
        tx: tokio::sync::mpsc::UnboundedSender<crate::types::StreamEvent>,
    },
    GenerateText {
        prompt: String,
        params: GenerationParameters,
        reply: tokio::sync::oneshot::Sender<Result<(String, u128), String>>,
    },
    Shutdown {
        reply: std::sync::mpsc::Sender<()>,
    },
}

fn process_utf8_buffer(byte_buffer: &mut Vec<u8>) -> String {
    let mut result = String::new();
    match std::str::from_utf8(byte_buffer) {
        Ok(valid_str) => {
            result.push_str(valid_str);
            byte_buffer.clear();
        }
        Err(e) => {
            let valid_len = e.valid_up_to();
            if valid_len > 0 {
                let valid_str = unsafe { std::str::from_utf8_unchecked(&byte_buffer[..valid_len]) };
                result.push_str(valid_str);
                byte_buffer.drain(..valid_len);
            }
            if e.error_len().is_some() {
                result.push_str(&String::from_utf8_lossy(byte_buffer));
                byte_buffer.clear();
            }
        }
    }
    result
}

fn run_generation<F>(
    inst: &mut LlamaInstance,
    prompt: &str,
    params: &GenerationParameters,
    mut on_tokenization_time: impl FnMut(u128),
    mut on_token: F,
) -> Result<(String, u128), String>
where
    F: FnMut(String) -> Result<(), ()>,
{
    let tok_start = std::time::Instant::now();
    let tokens_list = inst
        .with_dependent_mut(|model, _| model.str_to_token(prompt, AddBos::Always))
        .map_err(|e| e.to_string())?;

    let tok_time = tok_start.elapsed().as_millis();
    on_tokenization_time(tok_time);

    inst.with_dependent_mut(|_, ctx| ctx.clear_kv_cache());

    let n_batch = inst.with_dependent_mut(|_, ctx| ctx.n_batch() as usize);
    let mut current_pos = 0;
    let last_index_of_prompt = tokens_list.len().saturating_sub(1);
    let mut logit_index = 0;

    for chunk in tokens_list.chunks(n_batch.max(1)) {
        let mut batch = LlamaBatch::new(chunk.len(), 1);
        for (i, &token) in chunk.iter().enumerate() {
            let pos = current_pos + i;
            let is_last = pos == last_index_of_prompt;
            if is_last {
                logit_index = i as i32;
            }
            batch
                .add(token, pos as i32, &[0], is_last)
                .map_err(|e| e.to_string())?;
        }
        inst.with_dependent_mut(|_, ctx| ctx.decode(&mut batch))
            .map_err(|e| e.to_string())?;
        current_pos += chunk.len();
    }

    let mut output = String::new();
    let mut byte_buffer = Vec::new();
    let mut n_cur = tokens_list.len() as i32;
    let max_new_tokens = params.max_tokens.unwrap_or(500).min(i32::MAX as usize) as i32;
    let mut generated_tokens = 0;

    let mut samplers = Vec::new();
    if let Some(k) = params.top_k {
        samplers.push(LlamaSampler::top_k(k as i32));
    }
    if let Some(p) = params.top_p {
        samplers.push(LlamaSampler::top_p(p, 1));
    }
    if let Some(t) = params.temperature {
        samplers.push(LlamaSampler::temp(t));
    }

    let seed_u64 = params.seed.unwrap_or_else(rand::random::<u64>);
    let seed_u32 = (seed_u64 ^ (seed_u64 >> 32)) as u32;
    samplers.push(LlamaSampler::dist(seed_u32));
    let mut sampler = LlamaSampler::chain_simple(samplers);

    while generated_tokens < max_new_tokens {
        let (new_token_id, is_eog, new_bytes) = inst.with_dependent_mut(|model, ctx| {
            let new_token_id = sampler.sample(ctx, logit_index);
            sampler.accept(new_token_id);

            let is_eog = model.is_eog_token(new_token_id);

            let new_bytes = if !is_eog {
                model
                    .token_to_piece_bytes(new_token_id, 256, false, None)
                    .unwrap_or_default()
            } else {
                Vec::new()
            };

            (new_token_id, is_eog, new_bytes)
        });

        if is_eog {
            break;
        }

        byte_buffer.extend_from_slice(&new_bytes);
        let decoded = process_utf8_buffer(&mut byte_buffer);
        if !decoded.is_empty() {
            output.push_str(&decoded);
            if on_token(decoded).is_err() {
                break;
            }
        }

        let mut batch = LlamaBatch::new(1, 1);
        batch
            .add(new_token_id, n_cur, &[0], true)
            .map_err(|e| e.to_string())?;
        inst.with_dependent_mut(|_, ctx| ctx.decode(&mut batch))
            .map_err(|e| e.to_string())?;

        logit_index = 0;
        n_cur += 1;
        generated_tokens += 1;
    }

    if !byte_buffer.is_empty() {
        let remaining = String::from_utf8_lossy(&byte_buffer).into_owned();
        output.push_str(&remaining);
        let _ = on_token(remaining);
    }

    Ok((output, tok_time))
}

pub struct LlamaCppEngine {
    command_tx: tokio::sync::mpsc::UnboundedSender<EngineCommand>,
    offload_pct: f32,
}

impl LlamaCppEngine {
    pub fn new(gpu_device_index: u32) -> Result<Self, String> {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<EngineCommand>();

        std::thread::spawn(move || {
            let backend = LLAMA_BACKEND.get_or_init(|| {
                LlamaBackend::init().expect("Failed to initialize llama.cpp backend")
            });

            let mut instance: Option<LlamaInstance> = None;
            let nvml = nvml_wrapper::Nvml::init().ok();

            // This dedicated OS thread handles !Send objects safely and sequentially
            while let Some(cmd) = rx.blocking_recv() {
                match cmd {
                    EngineCommand::LoadModel {
                        config,
                        status,
                        strategy,
                        required_ctx,
                        reply,
                    } => {
                        let res = (|| -> Result<(usize, f32), String> {
                            let api = Api::new().map_err(|e| e.to_string())?;
                            let repo = api.model(config.repo.clone());
                            let weights_path = repo
                                .get(&config.filename)
                                .map_err(|e| format!("Missing weights: {}", e))?;

                            let available_vram =
                                crate::get_vram_info(nvml.as_ref(), gpu_device_index)
                                    .map(|(_, _, free)| free)
                                    .unwrap_or(0);

                            let bytes_per_token = config.estimate_kv_bytes_per_token();

                            let compute_margin: u64 = LLAMA_CPP_COMPUTE_MARGIN_BYTES;
                            let mut final_ctx_len =
                                required_ctx.max(2048).min(config.max_context_len);
                            let mut n_gpu_layers = config.num_layers as u32;
                            let weights_vram_cost_est =
                                (config.size_on_disk_gb * 1024.0 * 1024.0 * 1024.0) as u64;

                            // Estimate non-layer weights (embeddings, output head, etc.)
                            // using the exact parameter ratio from the model registry.
                            let non_layer_ratio = config.non_layer_params_billions
                                / config.parameters_billions.max(0.001);
                            let non_layer_weights_est =
                                (weights_vram_cost_est as f32 * non_layer_ratio) as u64;
                            let vram_per_layer = (weights_vram_cost_est
                                .saturating_sub(non_layer_weights_est))
                                / config.num_layers.max(1) as u64;

                            if strategy == MemoryStrategy::Compress {
                                if weights_vram_cost_est + compute_margin > available_vram {
                                    let vram_for_weights = available_vram
                                        .saturating_sub(compute_margin)
                                        .saturating_sub(non_layer_weights_est);
                                    n_gpu_layers =
                                        (vram_for_weights / vram_per_layer.max(1)) as u32;
                                }
                            } else {
                                let kv_vram_cost_per_layer = ((final_ctx_len * bytes_per_token)
                                    as u64)
                                    / config.num_layers.max(1) as u64;
                                let total_cost_per_gpu_layer =
                                    kv_vram_cost_per_layer + vram_per_layer;
                                let vram_for_layers = available_vram
                                    .saturating_sub(compute_margin)
                                    .saturating_sub(non_layer_weights_est);
                                n_gpu_layers =
                                    (vram_for_layers / total_cost_per_gpu_layer.max(1)) as u32;
                                println!(
                                    "🔀 CPU Offloading Active: Fitting {} / {} layers on GPU.",
                                    n_gpu_layers, config.num_layers
                                );
                            }

                            n_gpu_layers = n_gpu_layers.min(config.num_layers as u32);
                            let offload_pct =
                                1.0 - (n_gpu_layers as f32 / config.num_layers.max(1) as f32);
                            if offload_pct > 0.0 {
                                let offloaded_bytes =
                                    (weights_vram_cost_est as f32 * offload_pct) as i64;
                                let mut s = status.lock().unwrap();
                                s.log_ram(
                                    "Allocate",
                                    "LlamaCpp::Offload",
                                    &format!(
                                        "Offloaded {:.1}% of model to CPU RAM",
                                        offload_pct * 100.0
                                    ),
                                    offloaded_bytes,
                                );
                            }

                            let (vram_used_before, vram_total, _) =
                                crate::get_vram_info(nvml.as_ref(), gpu_device_index)
                                    .unwrap_or((0, 0, 0));

                            {
                                let mut s = status.lock().unwrap();
                                s.log_vram(
                                    "Allocate",
                                    "LlamaCpp",
                                    &format!("Loading weights for {}", config.id),
                                    0,
                                );
                            }

                            let model_params = LlamaModelParams::default()
                                .with_n_gpu_layers(n_gpu_layers)
                                .with_main_gpu(gpu_device_index as i32);
                            let model =
                                LlamaModel::load_from_file(backend, weights_path, &model_params)
                                    .map_err(|e| {
                                        format!("Failed to load llama.cpp model: {}", e)
                                    })?;

                            let (vram_used_after, _, vram_free_after) =
                                crate::get_vram_info(nvml.as_ref(), gpu_device_index)
                                    .unwrap_or((0, 0, 0));
                            let weights_vram = vram_used_after.saturating_sub(vram_used_before);

                            {
                                let mut s = status.lock().unwrap();
                                s.log_vram(
                                    "Allocate",
                                    "LlamaCpp::Weights",
                                    "Model Weights",
                                    weights_vram as i64,
                                );
                                s.update_nvml(vram_total, vram_used_after, vram_free_after);
                            }

                            if strategy == MemoryStrategy::Compress {
                                let safe_vram_for_kv =
                                    vram_free_after.saturating_sub(compute_margin);
                                let dynamic_max_ctx = if vram_free_after > compute_margin {
                                    (safe_vram_for_kv as usize / bytes_per_token)
                                        .min(config.max_context_len)
                                } else {
                                    256
                                };
                                final_ctx_len = dynamic_max_ctx.max(256);
                                println!(
                                    "🚀 Max Speed Mode (Compress): Sizing KV Cache for {} tokens based on free VRAM.",
                                    final_ctx_len
                                );
                                {
                                    let mut s = status.lock().unwrap();
                                    s.log_vram("Measure", "LlamaCpp::Plan", &format!("Reserved 1.5GB Compute Margin. Allocating KV Cache for {} tokens", final_ctx_len), 0);
                                }
                            } else {
                                println!(
                                    "💾 Max Context Mode (Offload): Allocating KV Cache for full {} tokens.",
                                    final_ctx_len
                                );
                            }

                            let mut ctx_params = LlamaContextParams::default();
                            if let Some(ctx_len) = NonZeroU32::new(final_ctx_len as u32) {
                                ctx_params = ctx_params.with_n_ctx(Some(ctx_len));
                            }

                            let (vram_used_before_ctx, _, _) =
                                crate::get_vram_info(nvml.as_ref(), gpu_device_index)
                                    .unwrap_or((0, 0, 0));

                            let inst = LlamaInstance::try_new(model, |m| {
                                m.new_context(backend, ctx_params)
                            })
                            .map_err(|e| format!("Failed to create llama.cpp context: {}", e))?;

                            let (vram_used_after_ctx, _, vram_free_final) =
                                crate::get_vram_info(nvml.as_ref(), gpu_device_index)
                                    .unwrap_or((0, 0, 0));
                            let kv_vram = vram_used_after_ctx.saturating_sub(vram_used_before_ctx);

                            {
                                let mut s = status.lock().unwrap();
                                s.log_vram(
                                    "Allocate",
                                    "LlamaCpp::KVCache",
                                    "Context Window Allocated",
                                    kv_vram as i64,
                                );
                                s.set_model_vram(
                                    config.id.clone(),
                                    "LlamaCpp".to_string(),
                                    true,
                                    "Active".to_string(),
                                    weights_vram,
                                    kv_vram,
                                    compute_margin,
                                );
                                s.update_nvml(vram_total, vram_used_after_ctx, vram_free_final);
                            }

                            instance = Some(inst);
                            Ok((final_ctx_len, offload_pct))
                        })();
                        let _ = reply.send(res);
                    }
                    EngineCommand::GenerateStream { prompt, params, tx } => {
                        (|| {
                            let inst = match instance.as_mut() {
                                Some(i) => i,
                                None => {
                                    let _ = tx.send(crate::types::StreamEvent::Error(
                                        "Model not loaded".into(),
                                    ));
                                    return;
                                }
                            };
                            match run_generation(
                                inst,
                                &prompt,
                                &params,
                                |tok_time| {
                                    let _ = tx.send(crate::types::StreamEvent::TokenizationTime(
                                        tok_time,
                                    ));
                                },
                                |token| {
                                    tx.send(crate::types::StreamEvent::Token(token))
                                        .map_err(|_| ())
                                },
                            ) {
                                Ok(_) => {
                                    let _ = tx.send(crate::types::StreamEvent::Done);
                                }
                                Err(e) => {
                                    let _ = tx.send(crate::types::StreamEvent::Error(e));
                                }
                            }
                        })();
                    }
                    EngineCommand::GenerateText {
                        prompt,
                        params,
                        reply,
                    } => {
                        let res = (|| -> Result<(String, u128), String> {
                            let inst = instance.as_mut().ok_or("Model not loaded")?;
                            run_generation(inst, &prompt, &params, |_| {}, |_| Ok(()))
                        })();
                        let _ = reply.send(res);
                    }
                    EngineCommand::Shutdown { reply } => {
                        drop(instance.take()); // Explicitly drop the Llama model and context
                        let _ = reply.send(());
                        break;
                    }
                }
            }
        });

        Ok(Self {
            command_tx: tx,
            offload_pct: 0.0,
        })
    }

    pub async fn generate_stream(
        &mut self,
        prompt: &str,
        params: &GenerationParameters,
        tx: tokio::sync::mpsc::UnboundedSender<crate::types::StreamEvent>,
    ) {
        let _ = self.command_tx.send(EngineCommand::GenerateStream {
            prompt: prompt.to_string(),
            params: params.clone(),
            tx,
        });
    }
}

impl Drop for LlamaCppEngine {
    fn drop(&mut self) {
        let (tx, rx) = std::sync::mpsc::channel();
        if self
            .command_tx
            .send(EngineCommand::Shutdown { reply: tx })
            .is_ok()
        {
            let _ = rx.recv(); // Block until the background thread acknowledges shutdown
        }
    }
}

#[async_trait]
impl InferenceBackend for LlamaCppEngine {
    async fn load_model(
        &mut self,
        config: &ModelConfig,
        status: Arc<Mutex<EngineStatus>>,
        strategy: &MemoryStrategy,
        required_ctx: usize,
    ) -> Result<usize, String> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.command_tx
            .send(EngineCommand::LoadModel {
                config: config.clone(),
                status,
                strategy: strategy.clone(),
                required_ctx,
                reply: tx,
            })
            .map_err(|_| "Engine thread died".to_string())?;

        let (final_ctx_len, offload_pct) = rx
            .await
            .map_err(|_| "Engine thread dropped reply".to_string())??;
        self.offload_pct = offload_pct;
        Ok(final_ctx_len)
    }

    async fn generate_text(
        &mut self,
        prompt: &str,
        params: &GenerationParameters,
    ) -> Result<(String, u128), String> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.command_tx
            .send(EngineCommand::GenerateText {
                prompt: prompt.to_string(),
                params: params.clone(),
                reply: tx,
            })
            .map_err(|_| "Engine thread died".to_string())?;

        rx.await
            .map_err(|_| "Engine thread dropped reply".to_string())?
    }

    fn supports_extractive_compression(&self) -> bool {
        false
    }
    async fn compress_text(
        &mut self,
        _prompt: &str,
        _target_len: usize,
        _max_chunk: usize,
    ) -> Result<(String, u128), String> {
        Err("Not supported".into())
    }
    fn get_vram_usage(&self) -> Option<(u64, u64)> {
        None
    }

    fn is_statically_allocated(&self) -> bool {
        true
    }

    fn get_offload_pct(&self) -> f32 {
        self.offload_pct
    }
}
