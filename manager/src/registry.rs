use crate::types::Message;
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

#[derive(Clone, Copy, Serialize, Deserialize, PartialEq, Debug)]
pub enum ModelDType {
    F32,
    F16,
    BF16,
}

#[derive(Clone, Copy, Serialize, Deserialize, PartialEq, Debug)]
pub enum ModelArch {
    Llama,
    Qwen2,
    XLMRoberta,
}

pub trait PromptFormatter {
    fn format_chat(&self, messages: &[Message]) -> String;
}

impl PromptFormatter for ModelArch {
    fn format_chat(&self, messages: &[Message]) -> String {
        let mut prompt = String::new();
        match self {
            ModelArch::Qwen2 => {
                for msg in messages {
                    prompt.push_str(&format!(
                        "<|im_start|>{}\n{}<|im_end|>\n",
                        msg.role, msg.content
                    ));
                }
                prompt.push_str("<|im_start|>assistant\n");
            }
            ModelArch::Llama => {
                for msg in messages {
                    prompt.push_str(&format!(
                        "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
                        msg.role, msg.content
                    ));
                }
                prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
            }
            _ => {
                for msg in messages {
                    prompt.push_str(&format!("{}: {}\n", msg.role, msg.content));
                }
                prompt.push_str("assistant: ");
            }
        }
        prompt
    }
}

#[derive(Debug, PartialEq, Clone, Copy, Serialize, Deserialize)]
pub enum ModelRole {
    GeneralChat,
    ContextCompressor,
    CodeSpecialist,
    ToolCaller,
    Reasoning,
}

#[derive(Clone, Copy, Serialize, Deserialize, PartialEq, Debug)]
pub enum BackendType {
    Candle,
    LlamaCpp,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub id: String,
    pub name: String,
    pub repo: String,
    pub tokenizer_repo: String,
    pub filename: String,
    pub max_context_len: usize,
    pub num_layers: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub roles: Vec<ModelRole>,
    pub arch: ModelArch,
    pub compression_dtype: Option<ModelDType>,
    pub kv_cache_dtype: ModelDType,
    pub parameters_billions: f32,
    pub non_layer_params_billions: f32,
    pub size_on_disk_gb: f32,
    pub supported_backends: Vec<BackendType>,
    #[serde(default)]
    pub is_default_chat: bool,
    #[serde(default)]
    pub is_default_compressor: bool,
}

impl ModelConfig {
    pub fn estimate_kv_bytes_per_token(&self) -> usize {
        let bytes_per_element = match self.kv_cache_dtype {
            ModelDType::F32 => 4,
            ModelDType::F16 | ModelDType::BF16 => 2,
        };
        if self.n_head > 0 && self.n_head_kv > 0 {
            (2 * self.num_layers * self.n_embd * self.n_head_kv / self.n_head.max(1))
                * bytes_per_element
        } else {
            // Fallback heuristics if precise model architecture details are omitted
            match &self.arch {
                ModelArch::Qwen2 if self.parameters_billions > 10.0 => 150_000,
                ModelArch::Qwen2 => 80_000,
                ModelArch::Llama if self.parameters_billions < 10.0 => 125_000,
                _ => 100_000,
            }
        }
    }
}

// Expose the registry so the web server can send it to the UI
pub fn get_model_registry() -> &'static [ModelConfig] {
    static REGISTRY: OnceLock<Vec<ModelConfig>> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        vec![
            ModelConfig {
                id: "llama-3.1-8b".to_string(),
                name: "Llama 3.1 (8B)".to_string(),
                repo: "QuantFactory/Meta-Llama-3.1-8B-Instruct-GGUF".to_string(),
                tokenizer_repo: "NousResearch/Meta-Llama-3.1-8B-Instruct".to_string(),
                filename: "Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf".to_string(),
                max_context_len: 128000,
                roles: vec![ModelRole::GeneralChat],
                num_layers: 32,
                n_embd: 4096,
                n_head: 32,
                n_head_kv: 8,
                arch: ModelArch::Llama,
                compression_dtype: None,
                kv_cache_dtype: ModelDType::F16,
                parameters_billions: 8.0,
                non_layer_params_billions: 1.05,
                size_on_disk_gb: 4.58,
                supported_backends: vec![BackendType::Candle, BackendType::LlamaCpp],
                is_default_chat: false,
                is_default_compressor: false,
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
                num_layers: 28,
                n_embd: 3584,
                n_head: 28,
                n_head_kv: 4,
                arch: ModelArch::Qwen2,
                compression_dtype: None,
                kv_cache_dtype: ModelDType::F16,
                parameters_billions: 7.61,
                non_layer_params_billions: 0.54,
                size_on_disk_gb: 4.36,
                supported_backends: vec![BackendType::Candle, BackendType::LlamaCpp],
                is_default_chat: true,
                is_default_compressor: false,
            },
            ModelConfig {
                id: "qwen-2.5-14b".to_string(),
                name: "Qwen 2.5 (14B)".to_string(),
                repo: "bartowski/Qwen2.5-14B-Instruct-GGUF".to_string(),
                tokenizer_repo: "Qwen/Qwen2.5-14B-Instruct".to_string(),
                filename: "Qwen2.5-14B-Instruct-Q4_K_M.gguf".to_string(),
                max_context_len: 131072,
                roles: vec![ModelRole::GeneralChat, ModelRole::CodeSpecialist],
                num_layers: 48,
                n_embd: 5120,
                n_head: 40,
                n_head_kv: 8,
                arch: ModelArch::Qwen2,
                compression_dtype: None,
                kv_cache_dtype: ModelDType::F16,
                parameters_billions: 14.0,
                non_layer_params_billions: 0.78,
                size_on_disk_gb: 8.37,
                supported_backends: vec![BackendType::Candle, BackendType::LlamaCpp],
                is_default_chat: false,
                is_default_compressor: false,
            },
            ModelConfig {
                id: "qwen-coder-14b".to_string(),
                name: "Qwen2.5 Coder (14B)".to_string(),
                repo: "Qwen/Qwen2.5-Coder-14B-Instruct-GGUF".to_string(),
                tokenizer_repo: "Qwen/Qwen2.5-Coder-14B-Instruct".to_string(),
                filename: "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf".to_string(),
                max_context_len: 131072,
                num_layers: 48,
                n_embd: 5120,
                n_head: 40,
                n_head_kv: 8,
                roles: vec![ModelRole::CodeSpecialist],
                arch: ModelArch::Qwen2,
                compression_dtype: None,
                kv_cache_dtype: ModelDType::F16,
                parameters_billions: 14.0,
                non_layer_params_billions: 0.78,
                size_on_disk_gb: 8.37,
                supported_backends: vec![BackendType::Candle, BackendType::LlamaCpp],
                is_default_chat: false,
                is_default_compressor: false,
            },
            ModelConfig {
                id: "strand-rust-14b".to_string(),
                name: "Strand Rust Coder (14B)".to_string(),
                repo: "mradermacher/Strand-Rust-Coder-14B-v1-GGUF".to_string(),
                tokenizer_repo: "Fortytwo-Network/Strand-Rust-Coder-14B-v1".to_string(),
                filename: "Strand-Rust-Coder-14B-v1.Q4_K_M.gguf".to_string(),
                max_context_len: 32768,
                num_layers: 40,
                n_embd: 5120,
                n_head: 40,
                n_head_kv: 8,
                roles: vec![ModelRole::CodeSpecialist],
                arch: ModelArch::Qwen2,
                compression_dtype: None,
                kv_cache_dtype: ModelDType::F16,
                parameters_billions: 14.0,
                non_layer_params_billions: 0.78,
                size_on_disk_gb: 8.37,
                supported_backends: vec![BackendType::Candle, BackendType::LlamaCpp],
                is_default_chat: false,
                is_default_compressor: false,
            },
            ModelConfig {
                id: "llmlingua-2-f16".to_string(),
                name: "LLMLingua-2 (F16 - Lean)".to_string(),
                repo: "microsoft/llmlingua-2-xlm-roberta-large-meetingbank".to_string(),
                tokenizer_repo: "microsoft/llmlingua-2-xlm-roberta-large-meetingbank".to_string(),
                filename: "model.safetensors".to_string(),
                max_context_len: 512,
                num_layers: 24,
                n_embd: 0,
                n_head: 0,
                n_head_kv: 0,
                roles: vec![ModelRole::ContextCompressor],
                arch: ModelArch::XLMRoberta,
                compression_dtype: Some(ModelDType::F16),
                kv_cache_dtype: ModelDType::F16,
                parameters_billions: 0.56,
                non_layer_params_billions: 0.0,
                size_on_disk_gb: 2.08,
                supported_backends: vec![BackendType::Candle],
                is_default_chat: false,
                is_default_compressor: false,
            },
            ModelConfig {
                id: "llmlingua-2-f32".to_string(),
                name: "LLMLingua-2 (F32 - Precision)".to_string(),
                repo: "microsoft/llmlingua-2-xlm-roberta-large-meetingbank".to_string(),
                tokenizer_repo: "microsoft/llmlingua-2-xlm-roberta-large-meetingbank".to_string(),
                filename: "model.safetensors".to_string(),
                max_context_len: 512,
                num_layers: 24,
                n_embd: 0,
                n_head: 0,
                n_head_kv: 0,
                roles: vec![ModelRole::ContextCompressor],
                arch: ModelArch::XLMRoberta,
                compression_dtype: Some(ModelDType::F32),
                kv_cache_dtype: ModelDType::F32,
                parameters_billions: 0.56,
                non_layer_params_billions: 0.0,
                size_on_disk_gb: 2.08,
                supported_backends: vec![BackendType::Candle],
                is_default_chat: false,
                is_default_compressor: false,
            },
            ModelConfig {
                id: "qwen-compressor".to_string(),
                name: "Qwen 1.5B (Abstractive)".to_string(),
                repo: "Qwen/Qwen2.5-1.5B-Instruct-GGUF".to_string(),
                tokenizer_repo: "Qwen/Qwen2.5-1.5B-Instruct".to_string(),
                filename: "qwen2.5-1.5b-instruct-q4_k_m.gguf".to_string(),
                max_context_len: 32768,
                num_layers: 28,
                n_embd: 1536,
                n_head: 12,
                n_head_kv: 2,
                roles: vec![ModelRole::ContextCompressor],
                arch: ModelArch::Qwen2,
                compression_dtype: None,
                kv_cache_dtype: ModelDType::F16,
                parameters_billions: 1.54,
                non_layer_params_billions: 0.23,
                size_on_disk_gb: 1.04,
                supported_backends: vec![BackendType::Candle, BackendType::LlamaCpp],
                is_default_chat: false,
                is_default_compressor: true,
            },
        ]
    })
}
