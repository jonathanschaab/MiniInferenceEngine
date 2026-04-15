use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub enum CompressionDType {
    F32,
    F16,
}

#[derive(Clone, Serialize, Deserialize, PartialEq)]
pub enum ModelArch {
    Llama,
    Qwen2,
    XLMRoberta,
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
    pub compression_dtype: Option<CompressionDType>,
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