use crate::registry::ModelConfig;
use crate::types::EngineStatus;
use crate::types::GenerationParameters;
use crate::types::MemoryStrategy;
use async_trait::async_trait;
use std::sync::{Arc, Mutex};

#[async_trait]
pub trait InferenceBackend: Send + Sync {
    /// Load a model into the backend hardware/memory.
    async fn load_model(
        &mut self,
        config: &ModelConfig,
        status: Arc<Mutex<EngineStatus>>,
        strategy: &MemoryStrategy,
        required_ctx: usize,
    ) -> Result<usize, String>;

    /// Generate text using the loaded generative model.
    async fn generate_text(
        &mut self,
        prompt: &str,
        params: &GenerationParameters,
    ) -> Result<(String, u128), String>;

    /// Indicates if this backend supports XLM-RoBERTa style extractive compression (Token Classification).
    /// If false, the orchestrator should fall back to standard text summarization.
    fn supports_extractive_compression(&self) -> bool;

    /// Compress context using an Extractive Token Classifier.
    /// Will only be called if `supports_extractive_compression` returns true.
    async fn compress_text(
        &mut self,
        prompt: &str,
        target_len: usize,
        max_chunk: usize,
    ) -> Result<(String, u128), String>;

    /// Optional hardware memory management: returns (used_bytes, total_bytes).
    /// Return `None` if the backend relies on the orchestrator's dynamic VRAM management.
    fn get_vram_usage(&self) -> Option<(u64, u64)>;

    /// Returns true if the backend allocates its full context window upfront (like Llama.cpp).
    fn is_statically_allocated(&self) -> bool;

    /// Returns the percentage of the model/KV cache that was offloaded to the CPU (0.0 to 1.0).
    fn get_offload_pct(&self) -> f32;
}

pub fn process_utf8_buffer(byte_buffer: &mut Vec<u8>) -> String {
    let mut result = String::new();
    loop {
        if byte_buffer.is_empty() {
            break;
        }
        match std::str::from_utf8(byte_buffer) {
            Ok(valid_str) => {
                result.push_str(valid_str);
                byte_buffer.clear();
                break;
            }
            Err(e) => {
                let valid_len = e.valid_up_to();
                if valid_len > 0 {
                    // SAFETY: e.valid_up_to() guarantees that the slice up to valid_len is valid UTF-8.
                    let valid_str =
                        unsafe { std::str::from_utf8_unchecked(&byte_buffer[..valid_len]) };
                    result.push_str(valid_str);
                    byte_buffer.drain(..valid_len);
                }
                if let Some(error_len) = e.error_len() {
                    result.push('\u{FFFD}'); // Standard replacement character
                    byte_buffer.drain(..error_len);
                } else {
                    break; // Incomplete sequence, wait for more bytes
                }
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_process_utf8_buffer() {
        // 1. Valid complete utf8
        let mut buf = vec![b'h', b'e', b'l', b'l', b'o'];
        assert_eq!(process_utf8_buffer(&mut buf), "hello");
        assert!(buf.is_empty());

        // 2. Incomplete utf8 (emoji split mid-byte)
        let mut buf = vec![0xF0, 0x9F, 0x92]; // Missing final byte for 💫
        assert_eq!(process_utf8_buffer(&mut buf), "");
        assert_eq!(buf, vec![0xF0, 0x9F, 0x92]); // Buffer retains the partial sequence

        buf.push(0xAB);
        assert_eq!(process_utf8_buffer(&mut buf), "💫");
        assert!(buf.is_empty());
    }

    #[test]
    fn test_process_utf8_buffer_invalid_replacement() {
        // Submits 3 completely invalid bytes to the decoder
        let mut buf = vec![0xFF, 0xFE, 0xFD];
        let decoded = process_utf8_buffer(&mut buf);
        assert_eq!(decoded, "\u{FFFD}\u{FFFD}\u{FFFD}"); // Yields 3 standard replacement chars
        assert!(buf.is_empty()); // Buffer gracefully recovered by dropping garbage
    }

    #[test]
    fn test_process_utf8_buffer_mixed_valid_invalid() {
        // Valid ASCII mixed with an invalid byte in the middle
        let mut buf = vec![b'a', b'b', b'c', 0xFF, b'x', b'y', b'z'];
        let decoded = process_utf8_buffer(&mut buf);
        assert_eq!(decoded, "abc\u{FFFD}xyz");
        assert!(buf.is_empty());
    }
}
