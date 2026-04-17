use async_trait::async_trait;
use std::sync::{Arc, Mutex};
use crate::types::EngineStatus;
use crate::registry::ModelConfig;
use crate::types::GenerationParameters;

#[async_trait]
pub trait InferenceBackend: Send + Sync {
    /// Load a model into the backend hardware/memory.
    async fn load_model(&mut self, config: &ModelConfig, status: Arc<Mutex<EngineStatus>>, strategy: &str, required_ctx: usize) -> Result<usize, String>;
    
    /// Generate text using the loaded generative model.
    async fn generate_text(&mut self, prompt: &str, params: &GenerationParameters) -> Result<(String, u128), String>;
    
    /// Indicates if this backend supports XLM-RoBERTa style extractive compression (Token Classification).
    /// If false, the orchestrator should fall back to standard text summarization.
    fn supports_extractive_compression(&self) -> bool;
    
    /// Compress context using an Extractive Token Classifier. 
    /// Will only be called if `supports_extractive_compression` returns true.
    async fn compress_text(&mut self, prompt: &str, target_len: usize, max_chunk: usize) -> Result<(String, u128), String>;
    
    /// Optional hardware memory management: returns (used_bytes, total_bytes).
    /// Return `None` if the backend relies on the orchestrator's dynamic VRAM management.
    fn get_vram_usage(&self) -> Option<(u64, u64)>;

    /// Returns true if the backend allocates its full context window upfront (like Llama.cpp).
    fn is_statically_allocated(&self) -> bool;

    /// Returns the percentage of the model/KV cache that was offloaded to the CPU (0.0 to 1.0).
    fn get_offload_pct(&self) -> f32;
}