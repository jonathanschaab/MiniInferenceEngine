use std::num::NonZeroU32;
use async_trait::async_trait;
use self_cell::self_cell;
use hf_hub::api::sync::Api;

use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::token::data_array::LlamaTokenDataArray;
use llama_cpp_2::model::AddBos;

use std::sync::{Arc, Mutex};
use crate::types::EngineStatus;
use crate::registry::ModelConfig;
use crate::types::GenerationParameters;
use crate::backend::InferenceBackend;

self_cell! {
    struct LlamaInstance {
        owner: LlamaModel,
        #[covariant]
        dependent: LlamaContext,
    }
}

pub struct LlamaCppEngine {
    backend: Arc<LlamaBackend>,
    instance: std::sync::Mutex<Option<LlamaInstance>>,
    offload_pct: f32,
}

// Safety: LlamaContext uses raw C pointers. We safely encapsulate them
// and ensure thread-safe concurrent access by wrapping the instance in a Mutex.
unsafe impl Send for LlamaInstance {}

impl LlamaCppEngine {
    pub fn new() -> Result<Self, String> {
        let backend = LlamaBackend::init().map_err(|e| format!("Failed to init llama backend: {}", e))?;
        Ok(Self {
            backend: Arc::new(backend),
            instance: std::sync::Mutex::new(None),
            offload_pct: 0.0,
        })
    }
    
    pub async fn generate_stream(&mut self, prompt: &str, params: &GenerationParameters, tx: tokio::sync::mpsc::UnboundedSender<crate::types::StreamEvent>) {
        let tok_start = std::time::Instant::now();
        let tokens_list = {
            let mut lock = self.instance.lock().unwrap();
            let instance = match lock.as_mut() {
                Some(i) => i,
                None => { let _ = tx.send(crate::types::StreamEvent::Error("Model not loaded".into())); return; }
            };
            match instance.with_dependent_mut(|model, _| model.str_to_token(prompt, AddBos::Always)) {
                Ok(t) => t,
                Err(e) => { let _ = tx.send(crate::types::StreamEvent::Error(e.to_string())); return; }
            }
        };
        
        let tok_time = tok_start.elapsed().as_millis();
        let _ = tx.send(crate::types::StreamEvent::TokenizationTime(tok_time));
        
        self.instance.lock().unwrap().as_mut().unwrap().with_dependent_mut(|_, ctx| ctx.clear_kv_cache());

        // Chunked Prefill to avoid batch size limits
        let n_batch = self.instance.lock().unwrap().as_mut().unwrap().with_dependent_mut(|_, ctx| ctx.n_batch() as usize);
        let mut current_pos = 0;
        let last_index_of_prompt = tokens_list.len().saturating_sub(1);
        let mut logit_index = 0;
        
        for chunk in tokens_list.chunks(n_batch) {
            {
                let mut lock = self.instance.lock().unwrap();
                let instance = lock.as_mut().unwrap();
                let mut batch = LlamaBatch::new(chunk.len(), 1);
                for (i, &token) in chunk.iter().enumerate() {
                    let pos = current_pos + i;
                    // The last token of the entire prompt has logits enabled.
                    let is_last = pos == last_index_of_prompt;
                    if is_last { logit_index = i as i32; }
                    if let Err(e) = batch.add(token, pos as i32, &[0], is_last) { let _ = tx.send(crate::types::StreamEvent::Error(e.to_string())); return; }
                }
                if let Err(e) = instance.with_dependent_mut(|_, ctx| ctx.decode(&mut batch)) { let _ = tx.send(crate::types::StreamEvent::Error(e.to_string())); return; }
            }
            current_pos += chunk.len();
            tokio::task::yield_now().await;
        }

        let mut n_cur = tokens_list.len() as i32;
        let max_new_tokens = params.max_tokens.unwrap_or(500) as i32;
        let mut generated_tokens = 0;
        let seed = params.seed.unwrap_or(42) as u32;
        
        while generated_tokens < max_new_tokens {
            let (new_token_id, is_eog, text) = {
                let mut lock = self.instance.lock().unwrap();
                let instance = lock.as_mut().unwrap();
                instance.with_dependent_mut(|model, ctx| {
                    let candidates = ctx.candidates_ith(logit_index);
                    let mut candidates_p = LlamaTokenDataArray::from_iter(candidates, false);
                    
                    let new_token_id = candidates_p.sample_token(seed);
                    let is_eog = model.is_eog_token(new_token_id);
                    
                    let text = if !is_eog {
                        model.token_to_piece_bytes(new_token_id, 128, false, None)
                            .ok()
                            .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
                    } else { None };
                    
                    (new_token_id, is_eog, text)
                })
            };

            if is_eog { break; }
            if let Some(t) = text {
                if tx.send(crate::types::StreamEvent::Token(t)).is_err() { break; }
            }
            
            {
                let mut lock = self.instance.lock().unwrap();
                let instance = lock.as_mut().unwrap();
                let mut batch = LlamaBatch::new(1, 1);
                if let Err(e) = batch.add(new_token_id, n_cur, &[0], true) { let _ = tx.send(crate::types::StreamEvent::Error(e.to_string())); break; }
                if let Err(e) = instance.with_dependent_mut(|_, ctx| ctx.decode(&mut batch)) { let _ = tx.send(crate::types::StreamEvent::Error(e.to_string())); break; }
            }
            
            logit_index = 0;
            n_cur += 1;
            generated_tokens += 1;
            
            tokio::task::yield_now().await;
        }
        let _ = tx.send(crate::types::StreamEvent::Done);
    }
}

#[async_trait]
impl InferenceBackend for LlamaCppEngine {
    async fn load_model(&mut self, config: &ModelConfig, status: Arc<Mutex<EngineStatus>>, strategy: &str, required_ctx: usize) -> Result<usize, String> {
        let api = Api::new().map_err(|e| e.to_string())?;
        let repo = api.model(config.repo.clone());
        let weights_path = repo.get(&config.filename).map_err(|e| format!("Missing weights: {}", e))?;
        
        let nvml = nvml_wrapper::Nvml::init().ok();
        let available_vram = crate::get_vram_info(nvml.as_ref(), 0).map(|(_, _, free)| free).unwrap_or(0);
        
        // More precise KV cache calculation based on model architecture
        let bytes_per_token = if config.n_head > 0 && config.n_head_kv > 0 {
            (config.num_layers * config.n_embd * config.n_head_kv / config.n_head) * 4
        } else {
            // Fallback heuristic for models missing detailed params
            100_000
        };

        let compute_margin = 1_500 * 1024 * 1024;

        let mut final_ctx_len = required_ctx.max(2048).min(config.max_context_len);
        let mut n_gpu_layers = config.num_layers as u32;
        let weights_vram_cost_est = (config.size_on_disk_gb * 1024.0 * 1024.0 * 1024.0) as u64;

        if strategy == "compress" {
            // "Compress" strategy: Prioritize fitting model weights into VRAM.
            // Dynamically size KV cache based on remaining VRAM.
            if weights_vram_cost_est + compute_margin > available_vram {
                let vram_for_weights = available_vram.saturating_sub(compute_margin);
                let vram_per_layer = weights_vram_cost_est / config.num_layers.max(1) as u64;
                n_gpu_layers = (vram_for_weights / vram_per_layer.max(1)) as u32;
            }
        } else { // "offload" strategy
            // "Offload" strategy: Prioritize fitting the full KV cache.
            // Offload model layers to CPU if necessary. The KV cache itself is split between VRAM/RAM.
            
            // In llama.cpp, the KV cache is distributed across layers.
            // So, each GPU layer needs VRAM for its weights + its portion of the KV cache.
            let kv_vram_cost_per_layer = ((final_ctx_len * bytes_per_token) as u64) / config.num_layers.max(1) as u64;
            let weights_vram_cost_per_layer = weights_vram_cost_est / config.num_layers.max(1) as u64;
            let total_cost_per_gpu_layer = kv_vram_cost_per_layer + weights_vram_cost_per_layer;

            let vram_for_layers = available_vram.saturating_sub(compute_margin);
            
            n_gpu_layers = (vram_for_layers / total_cost_per_gpu_layer) as u32;

            println!("🔀 CPU Offloading Active: Fitting {} / {} layers on GPU.", n_gpu_layers, config.num_layers);
        }
        
        n_gpu_layers = n_gpu_layers.min(config.num_layers as u32);
        self.offload_pct = 1.0 - (n_gpu_layers as f32 / config.num_layers.max(1) as f32);
        if self.offload_pct > 0.0 {
            let offloaded_bytes = (weights_vram_cost_est as f32 * self.offload_pct) as i64;
            {
                let mut s = status.lock().unwrap();
                s.log_ram("Allocate", "LlamaCpp::Offload", &format!("Offloaded {:.1}% of model to CPU RAM", self.offload_pct * 100.0), offloaded_bytes);
            }
        }

        let (vram_used_before, vram_total, _) = crate::get_vram_info(nvml.as_ref(), 0).unwrap_or((0, 0, 0));

        {
            let mut s = status.lock().unwrap();
            s.log_vram("Allocate", "LlamaCpp", &format!("Loading weights for {}", config.id), 0);
        }

        let backend_clone = self.backend.clone();
        let model = tokio::task::spawn_blocking(move || {
            let model_params = LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers);
            LlamaModel::load_from_file(&*backend_clone, weights_path, &model_params)
        })
        .await
        .map_err(|e| format!("Model loading task panicked: {}", e))?
        .map_err(|e| format!("Failed to load llama.cpp model: {}", e))?;
            
        let (vram_used_after, _, vram_free_after) = crate::get_vram_info(nvml.as_ref(), 0).unwrap_or((0, 0, 0));
        let weights_vram = vram_used_after.saturating_sub(vram_used_before);

        {
            let mut s = status.lock().unwrap();
            s.log_vram("Allocate", "LlamaCpp::Weights", "Model Weights", weights_vram as i64);
            s.update_nvml(vram_total, vram_used_after, vram_free_after);
        }

        if strategy == "compress" {
            let safe_vram_for_kv = vram_free_after.saturating_sub(compute_margin);
            let dynamic_max_ctx = if vram_free_after > compute_margin {
                (safe_vram_for_kv as usize / bytes_per_token).min(config.max_context_len)
            } else {
                256 
            };
            final_ctx_len = dynamic_max_ctx.max(256);
            println!("🚀 Max Speed Mode (Compress): Sizing KV Cache for {} tokens based on free VRAM.", final_ctx_len);
            {
                let mut s = status.lock().unwrap();
                s.log_vram("Measure", "LlamaCpp::Plan", &format!("Reserved 1.5GB Compute Margin. Allocating KV Cache for {} tokens", final_ctx_len), 0);
            }
        } else {
            println!("💾 Max Context Mode (Offload): Allocating KV Cache for full {} tokens.", final_ctx_len);
        }

        let safe_ctx_len = final_ctx_len;
        let mut ctx_params = LlamaContextParams::default();
        if let Some(ctx_len) = NonZeroU32::new(safe_ctx_len as u32) {
            ctx_params = ctx_params.with_n_ctx(Some(ctx_len));
        }
        
        let (vram_used_before_ctx, _, _) = crate::get_vram_info(nvml.as_ref(), 0).unwrap_or((0, 0, 0));

        let instance = LlamaInstance::try_new(
            model,
            |m| m.new_context(&*self.backend, ctx_params)
        ).map_err(|e| format!("Failed to create llama.cpp context: {}", e))?;
            
        let (vram_used_after_ctx, _, vram_free_final) = crate::get_vram_info(nvml.as_ref(), 0).unwrap_or((0, 0, 0));
        let kv_vram = vram_used_after_ctx.saturating_sub(vram_used_before_ctx);

        {
            let mut s = status.lock().unwrap();
            s.log_vram("Allocate", "LlamaCpp::KVCache", "Context Window Allocated", kv_vram as i64);
            s.set_model_vram(config.id.clone(), "LlamaCpp".to_string(), "Active".to_string(), weights_vram, kv_vram, compute_margin as u64);
            s.update_nvml(vram_total, vram_used_after_ctx, vram_free_final);
        }

        *self.instance.lock().unwrap() = Some(instance);
        Ok(final_ctx_len)
    }
    
    async fn generate_text(&mut self, prompt: &str, params: &GenerationParameters) -> Result<(String, u128), String> {
        let tok_start = std::time::Instant::now();
        let tokens_list = {
            let mut lock = self.instance.lock().unwrap();
            let instance = lock.as_mut().ok_or("Model not loaded")?;
            instance.with_dependent_mut(|model, _| {
                model.str_to_token(prompt, AddBos::Always)
            }).map_err(|e| e.to_string())?
        };
        
        let tok_time = tok_start.elapsed().as_millis();
            
        self.instance.lock().unwrap().as_mut().unwrap().with_dependent_mut(|_, ctx| ctx.clear_kv_cache());

        // Chunked Prefill to avoid batch size limits
        let n_batch = self.instance.lock().unwrap().as_mut().unwrap().with_dependent_mut(|_, ctx| ctx.n_batch() as usize);
        let mut current_pos = 0;
        let last_index_of_prompt = tokens_list.len().saturating_sub(1);
        let mut logit_index = 0;
        
        for chunk in tokens_list.chunks(n_batch) {
            {
                let mut lock = self.instance.lock().unwrap();
                let instance = lock.as_mut().unwrap();
                let mut batch = LlamaBatch::new(chunk.len(), 1);
                for (i, &token) in chunk.iter().enumerate() {
                    let pos = current_pos + i;
                    let is_last = pos == last_index_of_prompt;
                    if is_last { logit_index = i as i32; }
                    batch.add(token, pos as i32, &[0], is_last).map_err(|e| e.to_string())?;
                }
                instance.with_dependent_mut(|_, ctx| ctx.decode(&mut batch)).map_err(|e| e.to_string())?;
            }
            current_pos += chunk.len();
            tokio::task::yield_now().await;
        }
        
        let mut output = String::new();
        let mut n_cur = tokens_list.len() as i32;
        let max_new_tokens = params.max_tokens.unwrap_or(500) as i32;
        let mut generated_tokens = 0;
        let seed = params.seed.unwrap_or(42) as u32;
        
        while generated_tokens < max_new_tokens {
            let (new_token_id, is_eog, text) = {
                let mut lock = self.instance.lock().unwrap();
                let instance = lock.as_mut().unwrap();
                instance.with_dependent_mut(|model, ctx| {
                    let candidates = ctx.candidates_ith(logit_index);
                    let mut candidates_p = LlamaTokenDataArray::from_iter(candidates, false);
                    
                    let new_token_id = candidates_p.sample_token(seed);
                    let is_eog = model.is_eog_token(new_token_id);
                    
                    let text = if !is_eog {
                        model.token_to_piece_bytes(new_token_id, 128, false, None)
                            .ok()
                            .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
                    } else { None };
                    
                    (new_token_id, is_eog, text)
                })
            };
            
            if is_eog { break; }
            
            if let Some(t) = text {
                output.push_str(&t);
            }
            
            {
                let mut lock = self.instance.lock().unwrap();
                let instance = lock.as_mut().unwrap();
                let mut batch = LlamaBatch::new(1, 1);
                batch.add(new_token_id, n_cur, &[0], true).map_err(|e| e.to_string())?;
                instance.with_dependent_mut(|_, ctx| ctx.decode(&mut batch)).map_err(|e| e.to_string())?;
            }
            logit_index = 0;
            n_cur += 1;
            generated_tokens += 1;
            
            tokio::task::yield_now().await;
        }
        
        Ok((output, tok_time))
    }
    
    fn supports_extractive_compression(&self) -> bool { false }
    async fn compress_text(&mut self, _prompt: &str, _target_len: usize, _max_chunk: usize) -> Result<(String, u128), String> { Err("Not supported".into()) }
    fn get_vram_usage(&self) -> Option<(u64, u64)> { None }

    fn is_statically_allocated(&self) -> bool { true }

    fn get_offload_pct(&self) -> f32 { self.offload_pct }
}