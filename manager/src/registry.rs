use crate::types::Message;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::OnceCell;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

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
    GptOss,
    Mistral,
    Gemma,
    Deepseek,
    Cohere,
}

pub trait PromptFormatter {
    fn format_chat(&self, messages: &[Message]) -> String;
}

impl PromptFormatter for ModelArch {
    fn format_chat(&self, messages: &[Message]) -> String {
        let mut prompt = String::new();
        match self {
            ModelArch::Qwen2 | ModelArch::GptOss => {
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
            ModelArch::Mistral => {
                prompt.push_str("<s>");
                let mut system_prompt = String::new();
                for msg in messages {
                    if msg.role == "system" {
                        if !system_prompt.is_empty() {
                            system_prompt.push_str("\n\n");
                        }
                        system_prompt.push_str(&msg.content);
                    }
                }

                let mut first_user = true;
                let mut has_user = false;
                for msg in messages {
                    if msg.role == "user" {
                        prompt.push_str("[INST] ");
                        if first_user && !system_prompt.is_empty() {
                            prompt.push_str(&system_prompt);
                            prompt.push_str("\n\n");
                            first_user = false;
                        }
                        prompt.push_str(&msg.content);
                        prompt.push_str(" [/INST]");
                        has_user = true;
                    } else if msg.role != "system" {
                        prompt.push_str(&format!("{}</s>", msg.content));
                    }
                }
                if !has_user && !system_prompt.is_empty() {
                    prompt.push_str(&format!("[INST] {} [/INST]", system_prompt));
                }
            }
            ModelArch::Gemma => {
                for msg in messages {
                    let role = if msg.role == "assistant" {
                        "model"
                    } else {
                        &msg.role
                    };
                    prompt.push_str(&format!(
                        "<start_of_turn>{}\n{}<end_of_turn>\n",
                        role, msg.content
                    ));
                }
                prompt.push_str("<start_of_turn>model\n");
            }
            ModelArch::Deepseek => {
                prompt.push_str("<｜begin of sentence｜>");
                for msg in messages {
                    if msg.role == "system" {
                        prompt.push_str(&format!("{}\n", msg.content));
                    } else if msg.role == "user" {
                        prompt.push_str(&format!("<｜User｜>{}", msg.content));
                    } else if msg.role == "assistant" {
                        prompt.push_str(&format!(
                            "<｜Assistant｜>{}<｜end of sentence｜>",
                            msg.content
                        ));
                    } else {
                        prompt.push_str(&format!("{}: {}", msg.role, msg.content));
                    }
                }
                prompt.push_str("<｜Assistant｜>");
            }
            ModelArch::Cohere => {
                for msg in messages {
                    let role_token = match msg.role.as_str() {
                        "system" => "<|SYSTEM_TOKEN|>",
                        "user" => "<|USER_TOKEN|>",
                        "assistant" => "<|CHATBOT_TOKEN|>",
                        _ => "<|USER_TOKEN|>",
                    };
                    prompt.push_str(&format!(
                        "<|START_OF_TURN_TOKEN|>{}{}<|END_OF_TURN_TOKEN|>",
                        role_token, msg.content
                    ));
                }
                prompt.push_str("<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>");
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
    Vision,
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
    pub max_yarn_context: usize,
    pub sliding_window: Option<usize>,
    pub rope_scaling_factor: Option<f32>,
    pub original_max_position_embeddings: Option<usize>,
    pub num_layers: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub num_local_experts: Option<usize>,
    pub num_experts_per_tok: Option<usize>,
    pub kv_lora_rank: Option<usize>,
    pub qk_rope_head_dim: Option<usize>,
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
    pub provenance: std::collections::HashMap<String, String>,
    #[serde(default)]
    pub is_downloaded: bool,
    #[serde(default)]
    pub is_in_hf_cache: bool,
    #[serde(default)]
    pub is_corrupted: bool,
}

impl ModelConfig {
    pub fn estimate_kv_bytes_per_token(&self) -> usize {
        if self.arch == ModelArch::XLMRoberta {
            return 0; // Context compressors (extractive) don't have a generative KV cache
        }

        let bytes_per_element = match self.kv_cache_dtype {
            ModelDType::F32 => 4,
            ModelDType::F16 | ModelDType::BF16 => 2,
        };

        // Multi-head Latent Attention (MLA) Detection
        if let Some(lora_rank) = self.kv_lora_rank {
            let rope_dim = self.qk_rope_head_dim.unwrap_or(64);
            return self.num_layers * (lora_rank + rope_dim) * bytes_per_element;
        }

        (2 * self.num_layers * self.head_dim * self.n_head_kv) * bytes_per_element
    }

    pub fn estimate_compute_margin_bytes(&self, ubatch_size: usize) -> u64 {
        if self.arch == ModelArch::XLMRoberta {
            return 250 * 1024 * 1024; // Static margin for encoder-only models
        }

        let bytes_per_element = 4; // Activations are typically f32
        let active_experts = self.num_experts_per_tok.unwrap_or(1);
        let ffn_size = self.intermediate_size * active_experts;

        // 1. FFN and Hidden States (scales with physical batch size)
        let activation_memory = (self.n_embd + ffn_size) * ubatch_size * bytes_per_element;
        // 2. Attention Matrix (Flash Attention / Tiled Computation)
        // The full [ubatch_size, n_head, max_context_len] matrix is never materialized.
        // Temporary memory scales closer to the tile size: ubatch_size * n_head * ubatch_size.
        let attention_memory = ubatch_size * self.n_head * ubatch_size * bytes_per_element;
        // 3. Static Overhead (Graph nodes, CUDA context, etc.)
        // Dynamically scale with parameter count (e.g., 150MB base + 15MB per billion params)
        let static_overhead_mb = 150.0 + (self.parameters_billions * 15.0);
        let static_overhead = (static_overhead_mb * 1024.0 * 1024.0) as usize;

        (activation_memory + attention_memory + static_overhead) as u64
    }
}

// Detects if a filename is the first chunk of a split file array and generates the full list of files
pub fn get_split_filenames(first_filename: &str) -> Vec<String> {
    let pattern = "-00001-of-";
    if let Some(pos) = first_filename.find(pattern) {
        let prefix = &first_filename[..pos];
        let suffix_start = pos + pattern.len();
        if let Some(end_pos) = first_filename[suffix_start..].find('.') {
            let total_str = &first_filename[suffix_start..suffix_start + end_pos];
            let ext = &first_filename[suffix_start + end_pos..];
            if let Ok(total) = total_str.parse::<usize>() {
                let mut files = Vec::new();
                let pad = total_str.len();
                for i in 1..=total {
                    files.push(format!(
                        "{}-{:0width$}-of-{}{}",
                        prefix,
                        i,
                        total_str,
                        ext,
                        width = pad
                    ));
                }
                return files;
            }
        }
    }
    vec![first_filename.to_string()]
}

#[derive(Default, Clone)]
struct ModelOverrides {
    pub arch: Option<ModelArch>,
    pub kv_cache_dtype: Option<ModelDType>,
    pub max_context_len: Option<usize>,
    pub sliding_window: Option<usize>,
    pub rope_scaling_factor: Option<f32>,
    pub original_max_position_embeddings: Option<usize>,
    pub num_layers: Option<usize>,
    pub n_embd: Option<usize>,
    pub n_head: Option<usize>,
    pub n_head_kv: Option<usize>,
    pub head_dim: Option<usize>,
    pub intermediate_size: Option<usize>,
    pub num_local_experts: Option<usize>,
    pub num_experts_per_tok: Option<usize>,
    pub kv_lora_rank: Option<usize>,
    pub qk_rope_head_dim: Option<usize>,
    pub size_on_disk_gb: Option<f32>,
}

#[derive(Clone)]
struct ModelRegistration {
    id: &'static str,
    name: &'static str,
    repo: &'static str,
    tokenizer_repo: &'static str,
    filename: &'static str,
    roles: Vec<ModelRole>,
    compression_dtype: Option<ModelDType>,
    supported_backends: Vec<BackendType>,
    is_default_chat: bool,
    is_default_compressor: bool,
    parameters_billions: f32,
    non_layer_params_billions: f32,
    overrides: ModelOverrides,
}

static REGISTRY: OnceCell<Arc<RwLock<Vec<ModelConfig>>>> = OnceCell::const_new();

#[allow(clippy::too_many_arguments)]
async fn resolve_model_size(
    filenames: &[String],
    downloads_dir: &std::path::Path,
    hf_cache: &hf_hub::Cache,
    repo: &str,
    reqwest_client: Option<&reqwest::Client>,
    hf_base_url: &str,
    hf_token: Option<&String>,
    reg_id: &str,
    permanent_error: &mut bool,
) -> Option<(f32, String)> {
    let mut total_bytes = 0;
    let mut all_found = true;

    for fname in filenames {
        let local_path = downloads_dir.join(fname);
        if let Ok(meta) = tokio::fs::metadata(&local_path).await {
            total_bytes += meta.len();
        } else {
            let cache_clone = hf_cache.clone();
            let repo_clone = repo.to_string();
            let fname_clone = fname.clone();
            let gguf_path_opt = tokio::task::spawn_blocking(move || {
                cache_clone
                    .repo(hf_hub::Repo::model(repo_clone))
                    .get(&fname_clone)
            })
            .await
            .unwrap_or(None);

            if let Some(gguf_path) = gguf_path_opt {
                if let Ok(meta) = tokio::fs::metadata(&gguf_path).await {
                    total_bytes += meta.len();
                } else {
                    all_found = false;
                }
            } else {
                all_found = false;
            }
        }
    }

    if all_found && total_bytes > 0 {
        return Some((
            total_bytes as f32 / 1024.0 / 1024.0 / 1024.0,
            "disk".to_string(),
        ));
    }

    if let Some(client) = reqwest_client {
        let mut remote_bytes = 0;
        let mut remote_found = true;

        for fname in filenames {
            let url = format!(
                "{}/{}/resolve/main/{}",
                hf_base_url.trim_end_matches('/'),
                repo,
                fname
            );
            let mut req = client.head(&url);
            if let Some(token) = hf_token {
                req = req.bearer_auth(token);
            }
            if let Ok(res) = req.send().await {
                if !res.status().is_success() {
                    let status = res.status();
                    if status == reqwest::StatusCode::UNAUTHORIZED
                        || status == reqwest::StatusCode::FORBIDDEN
                        || status == reqwest::StatusCode::NOT_FOUND
                    {
                        *permanent_error = true;
                    }
                    remote_found = false;
                    break;
                }
                let size_header = res
                    .headers()
                    .get("X-Linked-Size")
                    .or_else(|| res.headers().get(reqwest::header::CONTENT_LENGTH));
                if let Some(cl) = size_header
                    && let Ok(s) = cl.to_str().unwrap_or("0").parse::<u64>()
                {
                    remote_bytes += s;
                } else {
                    remote_found = false;
                    break;
                }
            } else {
                remote_found = false;
                break;
            }
        }

        if remote_found && remote_bytes > 0 {
            return Some((
                remote_bytes as f32 / 1024.0 / 1024.0 / 1024.0,
                "api".to_string(),
            ));
        }
    } else {
        warn!(
            "Reqwest client unavailable, skipping remote size check for {}",
            reg_id
        );
    }

    None
}

// Expose the registry so the web server can send it to the UI
pub async fn get_model_registry(config: &crate::config::AppConfig) -> Vec<ModelConfig> {
    let registry_lock = REGISTRY.get_or_init(|| async {
        let lock = Arc::new(RwLock::new(Vec::new()));

        let downloads_dir = crate::types::resolve_absolute_path(&config.downloads_directory);
        let hf_base_url = config.hf_base_url.clone();

        let mut builder = hf_hub::api::tokio::ApiBuilder::new().with_endpoint(hf_base_url.clone());
        if let Ok(token) = std::env::var("HF_TOKEN") {
            let masked = if token.len() > 4 {
                format!("{}...", &token[..4])
            } else {
                "***".to_string()
            };
            info!("Found HF_TOKEN in environment (starts with {}). Applying to registry API client.", masked);
            builder = builder.with_token(Some(token));
        } else {
            debug!("No HF_TOKEN found in environment. Using default HF API client.");
        }
        let api_opt = builder.build().ok();
        if api_opt.is_none() {
            warn!("Failed to init HF API. Offline mode fallback active.");
        }

        let shared_reqwest_client = match reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::default())
            .connect_timeout(std::time::Duration::from_secs(config.registry_connect_timeout_seconds))
            .timeout(std::time::Duration::from_secs(config.registry_request_timeout_seconds))
            .tcp_keepalive(std::time::Duration::from_secs(60))
            .pool_idle_timeout(std::time::Duration::from_secs(55))
            .build()
        {
            Ok(c) => Some(c),
            Err(e) => {
                warn!("Failed to build reqwest client for model registry: {}", e);
                None
            }
        };

        let hf_token = std::env::var("HF_TOKEN").ok();

        let registrations = vec![
            ModelRegistration {
                id: "llama-3.1-8b",
                name: "Llama 3.1 (8B)",
                repo: "QuantFactory/Meta-Llama-3.1-8B-Instruct-GGUF",
                tokenizer_repo: "NousResearch/Meta-Llama-3.1-8B-Instruct",
                filename: "Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf",
                roles: vec![ModelRole::GeneralChat],
                compression_dtype: None,
                supported_backends: vec![BackendType::Candle, BackendType::LlamaCpp],
                is_default_chat: false,
                is_default_compressor: false,
                parameters_billions: 8.0,
                non_layer_params_billions: 1.05,
                overrides: ModelOverrides::default(),
            },
            ModelRegistration {
                id: "llama-3.2-3b",
                name: "Llama 3.2 (3B)",
                repo: "bartowski/Llama-3.2-3B-Instruct-GGUF",
                tokenizer_repo: "unsloth/Llama-3.2-3B-Instruct",
                filename: "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
                roles: vec![ModelRole::GeneralChat],
                compression_dtype: None,
                supported_backends: vec![BackendType::Candle, BackendType::LlamaCpp],
                is_default_chat: false,
                is_default_compressor: false,
                parameters_billions: 3.21,
                non_layer_params_billions: 0.40,
                overrides: ModelOverrides::default(),
            },
            ModelRegistration {
                id: "qwen-2.5-1.5b",
                name: "Qwen 2.5 (1.5B)",
                repo: "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
                tokenizer_repo: "Qwen/Qwen2.5-1.5B-Instruct",
                filename: "qwen2.5-1.5b-instruct-q4_k_m.gguf",
                roles: vec![ModelRole::GeneralChat, ModelRole::CodeSpecialist],
                compression_dtype: None,
                supported_backends: vec![BackendType::Candle, BackendType::LlamaCpp],
                is_default_chat: false,
                is_default_compressor: false,
                parameters_billions: 1.54,
                non_layer_params_billions: 0.23,
                overrides: ModelOverrides::default(),
            },
            ModelRegistration {
                id: "qwen-2.5-7b",
                name: "Qwen 2.5 (7B)",
                repo: "bartowski/Qwen2.5-7B-Instruct-GGUF",
                tokenizer_repo: "Qwen/Qwen2.5-7B-Instruct",
                filename: "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
                roles: vec![ModelRole::GeneralChat, ModelRole::CodeSpecialist],
                compression_dtype: None,
                supported_backends: vec![BackendType::Candle, BackendType::LlamaCpp],
                is_default_chat: true,
                is_default_compressor: false,
                parameters_billions: 7.61,
                non_layer_params_billions: 0.54,
                overrides: ModelOverrides::default(),
            },
            ModelRegistration {
                id: "qwen-2.5-14b",
                name: "Qwen 2.5 (14B)",
                repo: "bartowski/Qwen2.5-14B-Instruct-GGUF",
                tokenizer_repo: "Qwen/Qwen2.5-14B-Instruct",
                filename: "Qwen2.5-14B-Instruct-Q4_K_M.gguf",
                roles: vec![ModelRole::GeneralChat, ModelRole::CodeSpecialist],
                compression_dtype: None,
                supported_backends: vec![BackendType::Candle, BackendType::LlamaCpp],
                is_default_chat: false,
                is_default_compressor: false,
                parameters_billions: 14.0,
                non_layer_params_billions: 0.78,
                overrides: ModelOverrides::default(),
            },
            ModelRegistration {
                id: "qwen-coder-14b",
                name: "Qwen2.5 Coder (14B)",
                repo: "Qwen/Qwen2.5-Coder-14B-Instruct-GGUF",
                tokenizer_repo: "Qwen/Qwen2.5-Coder-14B-Instruct",
                filename: "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf",
                roles: vec![ModelRole::CodeSpecialist],
                compression_dtype: None,
                supported_backends: vec![BackendType::Candle, BackendType::LlamaCpp],
                is_default_chat: false,
                is_default_compressor: false,
                parameters_billions: 14.0,
                non_layer_params_billions: 0.78,
                overrides: ModelOverrides::default(),
            },
            ModelRegistration {
                id: "strand-rust-14b",
                name: "Strand Rust Coder (14B)",
                repo: "mradermacher/Strand-Rust-Coder-14B-v1-GGUF",
                tokenizer_repo: "Fortytwo-Network/Strand-Rust-Coder-14B-v1",
                filename: "Strand-Rust-Coder-14B-v1.Q4_K_M.gguf",
                roles: vec![ModelRole::CodeSpecialist],
                compression_dtype: None,
                supported_backends: vec![BackendType::Candle, BackendType::LlamaCpp],
                is_default_chat: false,
                is_default_compressor: false,
                parameters_billions: 14.0,
                non_layer_params_billions: 0.78,
                overrides: ModelOverrides::default(),
            },
            ModelRegistration {
                id: "llmlingua-2-f16",
                name: "LLMLingua-2 (F16 - Lean)",
                repo: "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
                tokenizer_repo: "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
                filename: "model.safetensors",
                roles: vec![ModelRole::ContextCompressor],
                compression_dtype: Some(ModelDType::F16),
                supported_backends: vec![BackendType::Candle],
                is_default_chat: false,
                is_default_compressor: false,
                parameters_billions: 0.56,
                non_layer_params_billions: 0.0,
                overrides: ModelOverrides::default(),
            },
            ModelRegistration {
                id: "llmlingua-2-f32",
                name: "LLMLingua-2 (F32 - Precision)",
                repo: "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
                tokenizer_repo: "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
                filename: "model.safetensors",
                roles: vec![ModelRole::ContextCompressor],
                compression_dtype: Some(ModelDType::F32),
                supported_backends: vec![BackendType::Candle],
                is_default_chat: false,
                is_default_compressor: false,
                parameters_billions: 0.56,
                non_layer_params_billions: 0.0,
                overrides: ModelOverrides::default(),
            },
            ModelRegistration {
                id: "qwen-compressor",
                name: "Qwen 1.5B (Abstractive)",
                repo: "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
                tokenizer_repo: "Qwen/Qwen2.5-1.5B-Instruct",
                filename: "qwen2.5-1.5b-instruct-q4_k_m.gguf",
                roles: vec![ModelRole::ContextCompressor],
                compression_dtype: None,
                supported_backends: vec![BackendType::Candle, BackendType::LlamaCpp],
                is_default_chat: false,
                is_default_compressor: true,
                parameters_billions: 1.54,
                non_layer_params_billions: 0.23,
                overrides: ModelOverrides::default(),
            },
            ModelRegistration {
                id: "gpt-oss-20b",
                name: "GPT-OSS (20B)",
                repo: "unsloth/gpt-oss-20b-GGUF",
                tokenizer_repo: "openai/gpt-oss-20b",
                filename: "gpt-oss-20b-Q4_K_M.gguf",
                roles: vec![ModelRole::GeneralChat],
                compression_dtype: None,
                supported_backends: vec![BackendType::LlamaCpp],
                is_default_chat: false,
                is_default_compressor: false,
                parameters_billions: 20.9,
                non_layer_params_billions: 1.5,
                overrides: ModelOverrides {
                    kv_cache_dtype: Some(ModelDType::BF16),
                    ..Default::default()
                },
            },
            ModelRegistration {
                id: "qwen-3.6-35b-a3b",
                name: "Qwen 3.6 (35B-A3B)",
                repo: "unsloth/Qwen3.6-35B-A3B-GGUF",
                tokenizer_repo: "Qwen/Qwen3.6-35B-A3B",
                filename: "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
                roles: vec![ModelRole::GeneralChat, ModelRole::CodeSpecialist],
                compression_dtype: None,
                supported_backends: vec![BackendType::LlamaCpp],
                is_default_chat: false,
                is_default_compressor: false,
                parameters_billions: 35.0,
                non_layer_params_billions: 1.5,
                overrides: ModelOverrides {
                    ..Default::default()
                },
            },
            ModelRegistration {
                id: "qwen-3.6-27b-gguf",
                name: "Qwen 3.6 (27B)",
                repo: "unsloth/Qwen3.6-27B-GGUF",
                tokenizer_repo: "Qwen/Qwen3.6-27B",
                filename: "Qwen3.6-27B-Q4_K_M.gguf",
                roles: vec![ModelRole::GeneralChat, ModelRole::Vision],
                compression_dtype: None,
                supported_backends: vec![BackendType::LlamaCpp],
                is_default_chat: false,
                is_default_compressor: false,
                parameters_billions: 27.8,
                non_layer_params_billions: 1.3,
                overrides: ModelOverrides {
                    ..Default::default()
                },
            },
            ModelRegistration {
                id: "mixtral-8x7b-instruct-v0.1",
                name: "Mixtral 8x7B Instruct v0.1",
                repo: "mradermacher/Mixtral-8x7B-Instruct-v0.1-GGUF",
                tokenizer_repo: "mistralai/Mixtral-8x7B-Instruct-v0.1",
                filename: "Mixtral-8x7B-Instruct-v0.1.Q4_K_M.gguf",
                roles: vec![ModelRole::GeneralChat, ModelRole::CodeSpecialist],
                compression_dtype: None,
                supported_backends: vec![BackendType::LlamaCpp],
                is_default_chat: false,
                is_default_compressor: false,
                parameters_billions: 46.7,
                non_layer_params_billions: 0.3,
                overrides: ModelOverrides {
                    ..Default::default()
                },
            },
            ModelRegistration {
                id: "gemma-4-e4b",
                name: "Gemma 4 (E4B)",
                repo: "unsloth/gemma-4-E4B-it-GGUF",
                tokenizer_repo: "google/gemma-4-E4B-it",
                filename: "gemma-4-E4B-it-Q4_K_M.gguf",
                roles: vec![ModelRole::GeneralChat],
                compression_dtype: None,
                supported_backends: vec![BackendType::LlamaCpp],
                is_default_chat: false,
                is_default_compressor: false,
                parameters_billions: 4.0,
                non_layer_params_billions: 0.5,
                overrides: ModelOverrides::default(),
            },
            ModelRegistration {
                id: "gemma-4-31b",
                name: "Gemma 4 (31B)",
                repo: "unsloth/gemma-4-31B-it-GGUF",
                tokenizer_repo: "google/gemma-4-31B-it",
                filename: "gemma-4-31B-it-Q4_K_M.gguf",
                roles: vec![ModelRole::GeneralChat],
                compression_dtype: None,
                supported_backends: vec![BackendType::LlamaCpp],
                is_default_chat: false,
                is_default_compressor: false,
                parameters_billions: 31.0,
                non_layer_params_billions: 1.5,
                overrides: ModelOverrides::default(),
            },
            ModelRegistration {
                id: "deepseek-v3-671b",
                name: "DeepSeek V3 (671B)",
                repo: "unsloth/DeepSeek-V3-GGUF",
                tokenizer_repo: "deepseek-ai/DeepSeek-V3",
            filename: "DeepSeek-V3-Q4_K_M/DeepSeek-V3-Q4_K_M-00001-of-00009.gguf",
                roles: vec![ModelRole::GeneralChat, ModelRole::Reasoning, ModelRole::CodeSpecialist],
                compression_dtype: None,
                supported_backends: vec![BackendType::LlamaCpp],
                is_default_chat: false,
                is_default_compressor: false,
                parameters_billions: 671.0,
                non_layer_params_billions: 2.5,
                overrides: ModelOverrides::default(),
            },
            ModelRegistration {
                id: "deepseek-v3-q2-xs",
                name: "DeepSeek V3 (671B - Q2_K_XS)",
                repo: "unsloth/DeepSeek-V3-GGUF",
                tokenizer_repo: "deepseek-ai/DeepSeek-V3",
                filename: "DeepSeek-V3-Q2_K_XS/DeepSeek-V3-Q2_K_XS-00001-of-00005.gguf",
                roles: vec![ModelRole::GeneralChat, ModelRole::Reasoning, ModelRole::CodeSpecialist],
                compression_dtype: None,
                supported_backends: vec![BackendType::LlamaCpp],
                is_default_chat: false,
                is_default_compressor: false,
                parameters_billions: 671.0,
                non_layer_params_billions: 2.5,
                overrides: ModelOverrides::default(),
            },
            ModelRegistration {
                id: "deepseek-coder-v2-lite",
                name: "DeepSeek Coder V2 Lite (16B)",
                repo: "bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF",
                tokenizer_repo: "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
                filename: "DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf",
                roles: vec![ModelRole::GeneralChat, ModelRole::CodeSpecialist],
                compression_dtype: None,
                supported_backends: vec![BackendType::LlamaCpp],
                is_default_chat: false,
                is_default_compressor: false,
                parameters_billions: 16.0,
                non_layer_params_billions: 0.5,
                overrides: ModelOverrides::default(),
            },
            ModelRegistration {
                id: "c4ai-command-r-plus",
                name: "Command R+ (104B)",
                repo: "pmysl/c4ai-command-r-plus-GGUF",
                tokenizer_repo: "CohereLabs/c4ai-command-r-plus",
                filename: "command-r-plus-Q4_K_M-00001-of-00002.gguf",
                roles: vec![ModelRole::GeneralChat, ModelRole::ToolCaller],
                compression_dtype: None,
                supported_backends: vec![BackendType::LlamaCpp],
                is_default_chat: false,
                is_default_compressor: false,
                parameters_billions: 104.0,
                non_layer_params_billions: 1.5,
                overrides: ModelOverrides::default(),
            },
            ModelRegistration {
                id: "deepseek-v4-pro-q2-k-xl",
                name: "DeepSeek V4 Pro (Q2_K-XL)",
                repo: "teamblobfish/DeepSeek-V4-Pro-GGUF",
                tokenizer_repo: "deepseek-ai/DeepSeek-V4-Pro",
                filename: "Q2_K-XL/DeepSeek-V4-Pro-Q2_K-XL-00001-of-00013.gguf",
                roles: vec![ModelRole::GeneralChat, ModelRole::Reasoning, ModelRole::CodeSpecialist],
                compression_dtype: None,
                supported_backends: vec![BackendType::LlamaCpp],
                is_default_chat: false,
                is_default_compressor: false,
                parameters_billions: 1600.0,
                non_layer_params_billions: 2.5,
                overrides: ModelOverrides::default(),
            },
        ];

        let mut initial_configs = Vec::new();
        for reg in &registrations {
            let arch_val = reg.overrides.arch.unwrap_or(ModelArch::Llama);
            let max_context_len_val = reg.overrides.max_context_len.unwrap_or(8192);

            let mut max_yarn_context = max_context_len_val;
            if arch_val == ModelArch::Qwen2 {
                if let Some(sw) = reg.overrides.sliding_window {
                    max_yarn_context = max_yarn_context.max(sw);
                }
            } else if let (Some(factor), Some(orig_ctx)) = (reg.overrides.rope_scaling_factor, reg.overrides.original_max_position_embeddings) {
                let scaled_ctx = (orig_ctx as f32 * factor) as usize;
                max_yarn_context = max_yarn_context.max(scaled_ctx);
            }

            let fallback_size_gb = match reg.compression_dtype {
                Some(ModelDType::F32) => reg.parameters_billions * 4.0,
                Some(ModelDType::F16) | Some(ModelDType::BF16) => reg.parameters_billions * 2.0,
                None => reg.parameters_billions * 0.65, // Assume typical Q4_K_M GGUF
            };

            let mut provenance = std::collections::HashMap::new();
            for name in &[
                "arch",
                "kv_cache_dtype",
                "max_context_len",
                "sliding_window",
                "rope_scaling_factor",
                "original_max_position_embeddings",
                "num_layers",
                "n_embd",
                "n_head",
                "n_head_kv",
                "head_dim",
                "intermediate_size",
                "num_local_experts",
                "num_experts_per_tok",
                "kv_lora_rank",
                "qk_rope_head_dim",
                "size_on_disk_gb",
            ] {
                provenance.insert(name.to_string(), "fallback".to_string());
            }

            let n_head_val = reg.overrides.n_head.unwrap_or(1);
            let n_embd_val = reg.overrides.n_embd.unwrap_or(4096);

            initial_configs.push(ModelConfig {
                id: reg.id.to_string(),
                name: reg.name.to_string(),
                repo: reg.repo.to_string(),
                tokenizer_repo: reg.tokenizer_repo.to_string(),
                filename: reg.filename.to_string(),
                roles: reg.roles.clone(),
                arch: arch_val,
                compression_dtype: reg.compression_dtype,
                kv_cache_dtype: reg.overrides.kv_cache_dtype.unwrap_or(ModelDType::F16),
                supported_backends: reg.supported_backends.clone(),
                is_default_chat: reg.is_default_chat,
                is_default_compressor: reg.is_default_compressor,
                max_context_len: max_context_len_val,
                max_yarn_context,
                sliding_window: reg.overrides.sliding_window,
                rope_scaling_factor: reg.overrides.rope_scaling_factor,
                original_max_position_embeddings: reg.overrides.original_max_position_embeddings,
                num_layers: reg.overrides.num_layers.unwrap_or(32),
                n_embd: n_embd_val,
                n_head: n_head_val,
                n_head_kv: reg.overrides.n_head_kv.unwrap_or(n_head_val),
                head_dim: reg.overrides.head_dim.unwrap_or(n_embd_val / n_head_val.max(1)),
                intermediate_size: reg.overrides.intermediate_size.unwrap_or(n_embd_val * 4),
                num_local_experts: reg.overrides.num_local_experts,
                num_experts_per_tok: reg.overrides.num_experts_per_tok,
                kv_lora_rank: reg.overrides.kv_lora_rank,
                qk_rope_head_dim: reg.overrides.qk_rope_head_dim,
                parameters_billions: reg.parameters_billions,
                non_layer_params_billions: reg.non_layer_params_billions,
                size_on_disk_gb: reg.overrides.size_on_disk_gb.unwrap_or(fallback_size_gb),
                provenance,
                is_downloaded: false,
                is_in_hf_cache: false,
                is_corrupted: false,
            });
        }

        lock.write().await.extend(initial_configs);

        let hf_cache = hf_hub::Cache::default();
        let init_semaphore = Arc::new(tokio::sync::Semaphore::new(4));

        for reg in registrations {
            tokio::spawn(background_resolve_model(
                reg,
                downloads_dir.clone(),
                hf_cache.clone(),
                hf_base_url.clone(),
                hf_token.clone(),
                shared_reqwest_client.clone(),
                api_opt.clone(),
                lock.clone(),
                init_semaphore.clone(),
                5,
            ));
        }

        lock
    }).await;

    registry_lock.read().await.clone()
}

#[allow(clippy::too_many_arguments)]
async fn background_resolve_model(
    reg: ModelRegistration,
    downloads_dir: std::path::PathBuf,
    hf_cache: hf_hub::Cache,
    hf_base_url: String,
    hf_token: Option<String>,
    reqwest_client: Option<reqwest::Client>,
    api_opt: Option<hf_hub::api::tokio::Api>,
    lock_clone: Arc<RwLock<Vec<ModelConfig>>>,
    sem_clone: Arc<tokio::sync::Semaphore>,
    mut backoff: u64,
) {
    loop {
        let permit = match sem_clone.acquire().await {
            Ok(p) => p,
            Err(_) => return, // Semaphore closed, safely aborting background resolution
        };

        let mut provenance = std::collections::HashMap::new();
        let mut permanent_error = false;

        let mut arch = reg.overrides.arch;
        let mut kv_cache_dtype = reg.overrides.kv_cache_dtype;
        let mut max_context_len = reg.overrides.max_context_len;
        let mut sliding_window = reg.overrides.sliding_window;
        let mut rope_scaling_factor = reg.overrides.rope_scaling_factor;
        let mut original_max_position_embeddings = reg.overrides.original_max_position_embeddings;
        let mut num_layers = reg.overrides.num_layers;
        let mut n_embd = reg.overrides.n_embd;
        let mut n_head = reg.overrides.n_head;
        let mut n_head_kv = reg.overrides.n_head_kv;
        let mut head_dim = reg.overrides.head_dim;
        let mut intermediate_size = reg.overrides.intermediate_size;
        let mut num_local_experts = reg.overrides.num_local_experts;
        let mut num_experts_per_tok = reg.overrides.num_experts_per_tok;
        let mut kv_lora_rank = reg.overrides.kv_lora_rank;
        let mut qk_rope_head_dim = reg.overrides.qk_rope_head_dim;
        let mut size_on_disk_gb = reg.overrides.size_on_disk_gb;

        let mut check_override = |opt: bool, name: &str| {
            if opt {
                provenance.insert(name.to_string(), "override".to_string());
            }
        };
        check_override(arch.is_some(), "arch");
        check_override(kv_cache_dtype.is_some(), "kv_cache_dtype");
        check_override(max_context_len.is_some(), "max_context_len");
        check_override(sliding_window.is_some(), "sliding_window");
        check_override(rope_scaling_factor.is_some(), "rope_scaling_factor");
        check_override(
            original_max_position_embeddings.is_some(),
            "original_max_position_embeddings",
        );
        check_override(num_layers.is_some(), "num_layers");
        check_override(n_embd.is_some(), "n_embd");
        check_override(n_head.is_some(), "n_head");
        check_override(n_head_kv.is_some(), "n_head_kv");
        check_override(head_dim.is_some(), "head_dim");
        check_override(intermediate_size.is_some(), "intermediate_size");
        check_override(num_local_experts.is_some(), "num_local_experts");
        check_override(num_experts_per_tok.is_some(), "num_experts_per_tok");
        check_override(kv_lora_rank.is_some(), "kv_lora_rank");
        check_override(qk_rope_head_dim.is_some(), "qk_rope_head_dim");
        check_override(size_on_disk_gb.is_some(), "size_on_disk_gb");

        let repo = reg.repo;
        let filename = reg.filename;

        if size_on_disk_gb.is_none() {
            let filenames = get_split_filenames(filename);
            if let Some((size, source)) = resolve_model_size(
                &filenames,
                &downloads_dir,
                &hf_cache,
                repo,
                reqwest_client.as_ref(),
                &hf_base_url,
                hf_token.as_ref(),
                reg.id,
                &mut permanent_error,
            )
            .await
            {
                size_on_disk_gb = Some(size);
                provenance.insert("size_on_disk_gb".to_string(), source);
            }
        }

        // 2. Fetch config.json from tokenizer repo to dynamically populate architectural details
        let needs_remote_config = arch.is_none()
            || kv_cache_dtype.is_none()
            || max_context_len.is_none()
            || sliding_window.is_none()
            || rope_scaling_factor.is_none()
            || original_max_position_embeddings.is_none()
            || num_layers.is_none()
            || n_embd.is_none()
            || n_head.is_none()
            || n_head_kv.is_none()
            || head_dim.is_none()
            || intermediate_size.is_none()
            || num_local_experts.is_none()
            || num_experts_per_tok.is_none()
            || kv_lora_rank.is_none()
            || qk_rope_head_dim.is_none();

        let mut config_parsed = false;

        if needs_remote_config {
            if let Some(api) = &api_opt {
                match api
                    .model(reg.tokenizer_repo.to_string())
                    .get("config.json")
                    .await
                {
                    Ok(config_path) => {
                        if let Ok(config_str) = tokio::fs::read_to_string(config_path).await
                            && let Ok(json) = serde_json::from_str::<serde_json::Value>(&config_str)
                        {
                            config_parsed = true;

                            let get_val = |key: &str| -> Option<&serde_json::Value> {
                                json.get("text_config")
                                    .and_then(|tc| tc.get(key))
                                    .or_else(|| json.get(key))
                            };

                            let is_optional_key = |key: &str| -> bool {
                                [
                                    "head_dim",
                                    "num_key_value_heads",
                                    "sliding_window",
                                    "num_local_experts",
                                    "intermediate_size",
                                    "num_experts_per_tok",
                                    "kv_lora_rank",
                                    "qk_rope_head_dim",
                                    "dtype",
                                    "torch_dtype",
                                ]
                                .contains(&key)
                            };

                            let get_u64 = |key: &str| -> Option<usize> {
                                if let Some(val) = get_val(key)
                                    && !val.is_null()
                                {
                                    if let Some(v) = val.as_u64() {
                                        return Some(v as usize);
                                    }
                                    warn!(
                                        "Invalid format for '{}' in config.json for {}",
                                        key, reg.id
                                    );
                                } else if !is_optional_key(key) {
                                    warn!("Missing '{}' in config.json for {}", key, reg.id);
                                }
                                None
                            };

                            let get_str = |key: &str| -> Option<String> {
                                if let Some(val) = get_val(key)
                                    && !val.is_null()
                                {
                                    if let Some(v) = val.as_str() {
                                        return Some(v.to_string());
                                    }
                                    warn!(
                                        "Invalid format for '{}' in config.json for {}",
                                        key, reg.id
                                    );
                                } else if !is_optional_key(key) {
                                    warn!("Missing '{}' in config.json for {}", key, reg.id);
                                }
                                None
                            };

                            // 1. Resolve arch first to inform subsequent parsing rules
                            if arch.is_none()
                                && let Some(model_type) = get_str("model_type")
                            {
                                arch = match model_type.as_str() {
                                    "llama" => Some(ModelArch::Llama),
                                    "qwen2" | "qwen3_5" | "qwen3_5_text" | "qwen3_5_moe"
                                    | "qwen3_5_moe_text" => Some(ModelArch::Qwen2),
                                    "xlm-roberta" => Some(ModelArch::XLMRoberta),
                                    "gpt_oss" => Some(ModelArch::GptOss),
                                    "mistral" | "mixtral" => Some(ModelArch::Mistral),
                                    "gemma" | "gemma2" | "gemma4_text" => Some(ModelArch::Gemma),
                                    "deepseek_v2" | "deepseek_v3" | "deepseek_v4" | "deepseek" => {
                                        Some(ModelArch::Deepseek)
                                    }
                                    "cohere" => Some(ModelArch::Cohere),
                                    _ => {
                                        warn!(
                                            "Unrecognized 'model_type' ({}) in config.json for {}",
                                            model_type, reg.id
                                        );
                                        None
                                    }
                                };
                                if arch.is_some() {
                                    provenance
                                        .insert("arch".to_string(), "config.json".to_string());
                                }
                            }

                            if max_context_len.is_none()
                                && let Some(v) = get_u64("max_position_embeddings")
                                    .or_else(|| get_u64("model_max_length"))
                                    .or_else(|| get_u64("max_sequence_length"))
                                    .or_else(|| get_u64("max_seq_len"))
                                    .or_else(|| get_u64("seq_length"))
                            {
                                max_context_len = Some(v);
                                provenance.insert(
                                    "max_context_len".to_string(),
                                    "config.json".to_string(),
                                );
                            }
                            if sliding_window.is_none() && arch == Some(ModelArch::Qwen2) {
                                if let Some(v) = get_u64("sliding_window") {
                                    sliding_window = Some(v);
                                    provenance.insert(
                                        "sliding_window".to_string(),
                                        "config.json".to_string(),
                                    );
                                } else {
                                    debug!(
                                        "Missing 'sliding_window' in config.json for {}",
                                        reg.id
                                    );
                                }
                            }
                            if (rope_scaling_factor.is_none()
                                || original_max_position_embeddings.is_none())
                                && let Some(rope_scaling) = get_val("rope_scaling")
                                && rope_scaling.is_object()
                            {
                                if rope_scaling_factor.is_none()
                                    && let Some(factor) =
                                        rope_scaling.get("factor").and_then(|v| v.as_f64())
                                {
                                    rope_scaling_factor = Some(factor as f32);
                                    provenance.insert(
                                        "rope_scaling_factor".to_string(),
                                        "config.json".to_string(),
                                    );
                                }
                                if original_max_position_embeddings.is_none()
                                    && let Some(orig_ctx) = rope_scaling
                                        .get("original_max_position_embeddings")
                                        .and_then(|v| v.as_u64())
                                {
                                    original_max_position_embeddings = Some(orig_ctx as usize);
                                    provenance.insert(
                                        "original_max_position_embeddings".to_string(),
                                        "config.json".to_string(),
                                    );
                                }
                            }

                            let apply_u64 = |opt: &mut Option<usize>,
                                             json_key: &str,
                                             prov_key: &str,
                                             prov: &mut std::collections::HashMap<
                                String,
                                String,
                            >| {
                                if opt.is_none()
                                    && let Some(v) = get_u64(json_key)
                                {
                                    *opt = Some(v);
                                    prov.insert(prov_key.to_string(), "config.json".to_string());
                                }
                            };

                            apply_u64(
                                &mut num_layers,
                                "num_hidden_layers",
                                "num_layers",
                                &mut provenance,
                            );
                            apply_u64(&mut n_embd, "hidden_size", "n_embd", &mut provenance);
                            apply_u64(
                                &mut n_head,
                                "num_attention_heads",
                                "n_head",
                                &mut provenance,
                            );
                            if n_head_kv.is_none()
                                && let Some(v) = get_u64("num_key_value_heads").or(n_head)
                            {
                                n_head_kv = Some(v);
                                provenance
                                    .insert("n_head_kv".to_string(), "config.json".to_string());
                            }
                            apply_u64(&mut head_dim, "head_dim", "head_dim", &mut provenance);
                            apply_u64(
                                &mut intermediate_size,
                                "intermediate_size",
                                "intermediate_size",
                                &mut provenance,
                            );
                            apply_u64(
                                &mut num_local_experts,
                                "num_local_experts",
                                "num_local_experts",
                                &mut provenance,
                            );
                            apply_u64(
                                &mut num_experts_per_tok,
                                "num_experts_per_tok",
                                "num_experts_per_tok",
                                &mut provenance,
                            );
                            apply_u64(
                                &mut kv_lora_rank,
                                "kv_lora_rank",
                                "kv_lora_rank",
                                &mut provenance,
                            );
                            apply_u64(
                                &mut qk_rope_head_dim,
                                "qk_rope_head_dim",
                                "qk_rope_head_dim",
                                &mut provenance,
                            );
                            if kv_cache_dtype.is_none() {
                                if let Some(dt) =
                                    get_str("dtype").or_else(|| get_str("torch_dtype"))
                                {
                                    kv_cache_dtype = match dt.as_str() {
                                        "float16" => Some(ModelDType::F16),
                                        "bfloat16" => Some(ModelDType::BF16),
                                        "float32" => Some(ModelDType::F32),
                                        _ => {
                                            warn!(
                                                "Unrecognized dtype ({}) in config.json for {}",
                                                dt, reg.id
                                            );
                                            None
                                        }
                                    };
                                    if kv_cache_dtype.is_some() {
                                        provenance.insert(
                                            "kv_cache_dtype".to_string(),
                                            "config.json".to_string(),
                                        );
                                    }
                                } else {
                                    warn!(
                                        "Missing both 'dtype' and 'torch_dtype' in config.json for {}",
                                        reg.id
                                    );
                                }
                            }
                        }
                    }
                    Err(e) => {
                        let msg = e.to_string();
                        if msg.contains("401")
                            || msg.contains("Unauthorized")
                            || msg.contains("403")
                            || msg.contains("Forbidden")
                            || msg.contains("404")
                            || msg.contains("Not Found")
                        {
                            warn!(
                                "Failed to fetch config.json for {}: Permanent HTTP error (401/403/404). Stopping retries. If this is a gated model, make sure you have accepted the license on Hugging Face and set the HF_TOKEN environment variable.",
                                reg.id
                            );
                            permanent_error = true;
                        } else {
                            warn!("Failed to fetch config.json for {}: {}", reg.id, e);
                        }
                    }
                }
            } else {
                warn!(
                    "HF API not initialized, skipping remote config.json fetch for {}",
                    reg.id
                );
            }
        }

        let n_head_val = n_head.unwrap_or(1);
        let n_embd_val = n_embd.unwrap_or(4096);

        for name in &[
            "arch",
            "kv_cache_dtype",
            "max_context_len",
            "sliding_window",
            "rope_scaling_factor",
            "original_max_position_embeddings",
            "num_layers",
            "n_embd",
            "n_head",
            "n_head_kv",
            "head_dim",
            "intermediate_size",
            "num_local_experts",
            "num_experts_per_tok",
            "kv_lora_rank",
            "qk_rope_head_dim",
            "size_on_disk_gb",
        ] {
            if !provenance.contains_key(*name) {
                if config_parsed && *name != "size_on_disk_gb" {
                    provenance.insert(name.to_string(), "default".to_string());
                } else {
                    provenance.insert(name.to_string(), "fallback".to_string());
                }
            }
        }

        let arch_val = arch.unwrap_or(ModelArch::Llama);
        let max_context_len_val = max_context_len.unwrap_or(8192);

        let mut max_yarn_context = max_context_len_val;
        if arch_val == ModelArch::Qwen2 {
            if let Some(sw) = sliding_window {
                max_yarn_context = max_yarn_context.max(sw);
            }
        } else if let (Some(factor), Some(orig_ctx)) =
            (rope_scaling_factor, original_max_position_embeddings)
        {
            let scaled_ctx = (orig_ctx as f32 * factor) as usize;
            max_yarn_context = max_yarn_context.max(scaled_ctx);
        }

        let fallback_size_gb = match reg.compression_dtype {
            Some(ModelDType::F32) => reg.parameters_billions * 4.0,
            Some(ModelDType::F16) | Some(ModelDType::BF16) => reg.parameters_billions * 2.0,
            None => reg.parameters_billions * 0.65, // Assume typical Q4_K_M GGUF
        };

        let config = ModelConfig {
            id: reg.id.to_string(),
            name: reg.name.to_string(),
            repo: reg.repo.to_string(),
            tokenizer_repo: reg.tokenizer_repo.to_string(),
            filename: reg.filename.to_string(),
            roles: reg.roles.clone(),
            arch: arch_val,
            compression_dtype: reg.compression_dtype,
            kv_cache_dtype: kv_cache_dtype.unwrap_or(ModelDType::F16),
            supported_backends: reg.supported_backends.clone(),
            is_default_chat: reg.is_default_chat,
            is_default_compressor: reg.is_default_compressor,

            max_context_len: max_context_len_val,
            max_yarn_context,
            sliding_window,
            rope_scaling_factor,
            original_max_position_embeddings,
            num_layers: num_layers.unwrap_or(32),
            n_embd: n_embd_val,
            n_head: n_head_val,
            n_head_kv: n_head_kv.unwrap_or(n_head_val),
            head_dim: head_dim.unwrap_or(n_embd_val / n_head_val.max(1)),
            intermediate_size: intermediate_size.unwrap_or(n_embd_val * 4),
            num_local_experts,
            num_experts_per_tok,
            kv_lora_rank,
            qk_rope_head_dim,
            parameters_billions: reg.parameters_billions,
            non_layer_params_billions: reg.non_layer_params_billions,
            size_on_disk_gb: size_on_disk_gb.unwrap_or(fallback_size_gb),
            provenance,
            is_downloaded: false, // This will be populated at runtime by the orchestrator
            is_in_hf_cache: false, // This will be populated at runtime by the API
            is_corrupted: false,  // This will be populated at runtime by the API
        };

        let mut success = true;
        for val in config.provenance.values() {
            if val == "fallback" {
                success = false;
                break;
            }
        }

        // Safely update the registry without wiping out runtime state flags
        {
            let mut reg_lock = lock_clone.write().await;
            if let Some(pos) = reg_lock.iter().position(|m| m.id == config.id) {
                let mut updated = config.clone();
                updated.is_downloaded = reg_lock[pos].is_downloaded;
                updated.is_in_hf_cache = reg_lock[pos].is_in_hf_cache;
                updated.is_corrupted = reg_lock[pos].is_corrupted;
                reg_lock[pos] = updated;
            }
        }

        drop(permit); // Release the concurrency semaphore BEFORE sleeping

        if success {
            debug!("Successfully resolved model details for {}", config.id);
            break;
        }

        if permanent_error {
            break;
        }

        if api_opt.is_none() && reqwest_client.is_none() {
            warn!(
                "Offline mode active: cannot fully resolve {}; keeping fallbacks.",
                config.id
            );
            break;
        }

        let missing_keys: Vec<_> = config
            .provenance
            .iter()
            .filter(|(_, v)| *v == "fallback")
            .map(|(k, _)| k.clone())
            .collect();
        warn!(
            "Failed to fully resolve {} from Hugging Face. Retrying in {} seconds...\nMissing properties: {:?}",
            config.id, backoff, missing_keys
        );
        tokio::time::sleep(std::time::Duration::from_secs(backoff)).await;
        backoff = (backoff * 2).min(300);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn mock_config(arch: ModelArch, dtype: ModelDType) -> ModelConfig {
        ModelConfig {
            id: "test".into(),
            name: "test".into(),
            repo: "test".into(),
            tokenizer_repo: "test".into(),
            filename: "test".into(),
            max_context_len: 1024,
            max_yarn_context: 1024,
            sliding_window: None,
            rope_scaling_factor: None,
            original_max_position_embeddings: None,
            num_layers: 32,
            n_embd: 4096,
            n_head: 32,
            n_head_kv: 8,
            head_dim: 128,
            intermediate_size: 14336,
            num_local_experts: None,
            num_experts_per_tok: None,
            kv_lora_rank: None,
            qk_rope_head_dim: None,
            roles: vec![],
            arch,
            compression_dtype: None,
            kv_cache_dtype: dtype,
            parameters_billions: 7.0,
            non_layer_params_billions: 0.5,
            size_on_disk_gb: 4.0,
            supported_backends: vec![],
            is_default_chat: false,
            is_default_compressor: false,
            provenance: HashMap::new(),
            is_downloaded: false,
            is_in_hf_cache: false,
            is_corrupted: false,
        }
    }

    #[test]
    fn test_prompt_formatter_llama() {
        let arch = ModelArch::Llama;
        let msgs = vec![Message {
            role: "user".into(),
            content: "Hello".into(),
        }];
        let prompt = arch.format_chat(&msgs);
        assert_eq!(
            prompt,
            "<|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        );
    }

    #[test]
    fn test_prompt_formatter_qwen() {
        let arch = ModelArch::Qwen2;
        let msgs = vec![Message {
            role: "user".into(),
            content: "Hi".into(),
        }];
        let prompt = arch.format_chat(&msgs);
        assert_eq!(
            prompt,
            "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn test_prompt_formatter_gpt_oss() {
        let arch = ModelArch::GptOss;
        let msgs = vec![Message {
            role: "user".into(),
            content: "Hi".into(),
        }];
        let prompt = arch.format_chat(&msgs);
        assert_eq!(
            prompt,
            "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn test_prompt_formatter_mistral() {
        let arch = ModelArch::Mistral;
        let msgs = vec![Message {
            role: "user".into(),
            content: "Hi".into(),
        }];
        let prompt = arch.format_chat(&msgs);
        assert_eq!(prompt, "<s>[INST] Hi [/INST]");
    }

    #[test]
    fn test_prompt_formatter_deepseek() {
        let arch = ModelArch::Deepseek;
        let msgs = vec![Message {
            role: "user".into(),
            content: "Hi".into(),
        }];
        let prompt = arch.format_chat(&msgs);
        assert_eq!(prompt, "<｜begin of sentence｜><｜User｜>Hi<｜Assistant｜>");
    }

    #[test]
    fn test_prompt_formatter_cohere() {
        let arch = ModelArch::Cohere;
        let msgs = vec![Message {
            role: "user".into(),
            content: "Hi".into(),
        }];
        let prompt = arch.format_chat(&msgs);
        assert_eq!(
            prompt,
            "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>Hi<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
        );
    }

    #[test]
    fn test_prompt_formatter_fallback() {
        let arch = ModelArch::XLMRoberta;
        let msgs = vec![Message {
            role: "system".into(),
            content: "test".into(),
        }];
        let prompt = arch.format_chat(&msgs);
        assert_eq!(prompt, "system: test\nassistant: ");
    }

    #[test]
    fn test_prompt_formatter_empty_messages() {
        let msgs = vec![];
        assert_eq!(
            ModelArch::Llama.format_chat(&msgs),
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        );
        assert_eq!(
            ModelArch::Qwen2.format_chat(&msgs),
            "<|im_start|>assistant\n"
        );
        assert_eq!(ModelArch::XLMRoberta.format_chat(&msgs), "assistant: ");
        assert_eq!(ModelArch::Mistral.format_chat(&msgs), "<s>");
    }

    #[test]
    fn test_estimate_kv_bytes() {
        let config = mock_config(ModelArch::Llama, ModelDType::F16);
        // 2 * 32 * 128 * 8 * 2 = 131,072
        assert_eq!(config.estimate_kv_bytes_per_token(), 131_072);
    }

    #[test]
    fn test_estimate_kv_bytes_f32() {
        let config = mock_config(ModelArch::Llama, ModelDType::F32);
        // 2 * 32 * 128 * 8 * 4 = 262,144
        assert_eq!(config.estimate_kv_bytes_per_token(), 262_144);
    }

    #[test]
    fn test_estimate_kv_bytes_compressor() {
        let config = mock_config(ModelArch::XLMRoberta, ModelDType::F16);
        assert_eq!(config.estimate_kv_bytes_per_token(), 0); // Context compressors have no generative KV cache
    }

    #[test]
    fn test_get_split_filenames() {
        let single = get_split_filenames("model.gguf");
        assert_eq!(single, vec!["model.gguf"]);

        let split = get_split_filenames("model-00001-of-00003.gguf");
        assert_eq!(
            split,
            vec![
                "model-00001-of-00003.gguf",
                "model-00002-of-00003.gguf",
                "model-00003-of-00003.gguf"
            ]
        );
    }

    #[tokio::test]
    #[ignore = "Hits real HF API, prone to 429 Rate Limits in CI"]
    async fn test_registry_fallback_and_background_update() {
        let config = crate::config::AppConfig::default();

        // 1. Eagerly grab the registry.
        let initial_registry = get_model_registry(&config).await;
        assert!(
            !initial_registry.is_empty(),
            "Registry should instantly return the models."
        );

        let target_id = "qwen-2.5-1.5b";
        let initial_model = initial_registry
            .iter()
            .find(|m| m.id == target_id)
            .expect("Model missing");

        // Since tests run in parallel, another test might have already triggered and awaited the background resolution.
        // We only assert the full transition if we caught it in the fallback state.
        if initial_model.provenance.get("arch").map(|s| s.as_str()) == Some("fallback") {
            // Because no overrides were provided for Qwen, it defaults to Llama before network resolution!
            assert_eq!(initial_model.arch, ModelArch::Llama);

            let mut updated = false;

            // Poll for up to 15 seconds for the background task to complete the network requests
            for _ in 0..30 {
                tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                let current_registry = get_model_registry(&config).await;
                let current_model = current_registry.iter().find(|m| m.id == target_id).unwrap();

                if current_model.provenance.get("arch").map(|s| s.as_str()) == Some("config.json") {
                    updated = true;
                    // Validate that the architecture was dynamically corrected by config.json!
                    assert_eq!(current_model.arch, ModelArch::Qwen2);
                    assert_ne!(
                        current_model.provenance.get("size_on_disk_gb").unwrap(),
                        "fallback"
                    );
                    break;
                }
            }

            assert!(
                updated,
                "Registry did not self-update from fallback within 15 seconds. (Network issue?)"
            );
        } else if initial_model.provenance.get("arch").map(|s| s.as_str()) == Some("config.json") {
            // If it was already updated by another test, at least verify it was eventually correct
            assert_eq!(initial_model.arch, ModelArch::Qwen2);
        }
    }

    #[tokio::test]
    async fn test_background_resolve_model_with_retry() {
        let _ = tracing_subscriber::fmt().with_test_writer().try_init();
        let start_time = std::time::Instant::now();

        // Mock server that returns 500s for the first 2 seconds to trigger the backoff loop
        let app = axum::Router::new().route(
            "/{*path}",
            axum::routing::get(move |req: axum::extract::Request| async move {
                info!("[MOCK SERVER] Intercepted GET request to: {}", req.uri());
                if start_time.elapsed().as_secs() < 2 {
                    info!("[MOCK SERVER] Simulating 500 Internal Server Error");
                    return axum::response::Response::builder()
                        .status(axum::http::StatusCode::INTERNAL_SERVER_ERROR)
                        .body(axum::body::Body::empty())
                        .unwrap();
                }

                // If hf-hub is asking for the API metadata (like commit info), mock a valid ModelInfo response
                if req.uri().path().contains("/api/models/") {
                    info!("[MOCK SERVER] Serving mock Hugging Face API ModelInfo");
                    let model_info = serde_json::json!({
                        "id": "test/repo",
                        "sha": "dummy_sha",
                        "siblings": [{"rfilename": "config.json"}]
                    });
                    return axum::response::Response::builder()
                        .header(axum::http::header::CONTENT_TYPE, "application/json")
                        .body(axum::body::Body::from(model_info.to_string()))
                        .unwrap();
                }

                let dummy_config = serde_json::json!({
                    "model_type": "qwen2",
                    "hidden_size": 2048,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 16,
                    "num_hidden_layers": 12,
                    "head_dim": 128,
                    "intermediate_size": 8192,
                    "max_position_embeddings": 32768,
                    "dtype": "bfloat16"
                });
                let body_str = dummy_config.to_string();
                let len = body_str.len();

                let res = axum::response::Response::builder()
                    .header(axum::http::header::CONTENT_TYPE, "application/json")
                    .header("ETag", "\"dummy_etag\"")
                    .header("X-Repo-Commit", "dummy_sha")
                    .header("Accept-Ranges", "bytes");

                if let Some(range) = req.headers().get(axum::http::header::RANGE)
                    && let Ok(range_str) = range.to_str()
                    && let Some(stripped) = range_str.strip_prefix("bytes=")
                {
                    let parts: Vec<&str> = stripped.split('-').collect();
                    let start = parts.first().unwrap_or(&"0").parse::<usize>().unwrap_or(0);

                    if start >= len {
                        return axum::response::Response::builder()
                            .status(axum::http::StatusCode::RANGE_NOT_SATISFIABLE)
                            .header("Content-Range", format!("bytes */{}", len))
                            .body(axum::body::Body::empty())
                            .unwrap();
                    }

                    let chunk_len = len - start;
                    info!(
                        "[MOCK SERVER] Serving dummy config.json (Partial: {} bytes)",
                        chunk_len
                    );
                    return res
                        .status(axum::http::StatusCode::PARTIAL_CONTENT)
                        .header(
                            "Content-Range",
                            format!("bytes {}-{}/{}", start, len - 1, len),
                        )
                        .header(axum::http::header::CONTENT_LENGTH, chunk_len.to_string())
                        .body(axum::body::Body::from(body_str[start..].to_string()))
                        .unwrap();
                }

                info!(
                    "[MOCK SERVER] Serving dummy config.json (Full: {} bytes)",
                    len
                );
                res.header(axum::http::header::CONTENT_LENGTH, len.to_string())
                    .body(axum::body::Body::from(body_str))
                    .unwrap()
            })
            .head(move |req: axum::extract::Request| async move {
                info!("[MOCK SERVER] Intercepted HEAD request to: {}", req.uri());
                if start_time.elapsed().as_secs() < 2 {
                    info!("[MOCK SERVER] Simulating 500 Internal Server Error");
                    return axum::response::Response::builder()
                        .status(axum::http::StatusCode::INTERNAL_SERVER_ERROR)
                        .body(axum::body::Body::empty())
                        .unwrap();
                }
                info!("[MOCK SERVER] Serving X-Linked-Size header");
                axum::response::Response::builder()
                    .header("X-Linked-Size", "8589934592") // Simulate an 8GB response
                    .body(axum::body::Body::empty())
                    .unwrap()
            }),
        );

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let hf_base_url = format!("http://127.0.0.1:{}", port);

        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        // Setup isolated mock registry and lock
        let lock = Arc::new(RwLock::new(vec![mock_config(
            ModelArch::Llama,
            ModelDType::F16,
        )]));
        let sem = Arc::new(tokio::sync::Semaphore::new(1));

        let reg = ModelRegistration {
            id: "test",
            name: "test",
            repo: "test/repo",
            tokenizer_repo: "test/repo",
            filename: "model.gguf",
            roles: vec![],
            compression_dtype: None,
            supported_backends: vec![],
            is_default_chat: false,
            is_default_compressor: false,
            parameters_billions: 7.0,
            non_layer_params_billions: 0.5,
            overrides: ModelOverrides::default(),
        };

        let reqwest_client = reqwest::Client::builder().build().unwrap();

        // Use an isolated cache directory so we don't accidentally read from the global ~/.cache/huggingface
        let temp_cache_dir = std::env::temp_dir().join(format!("test_hf_cache_{}", port));
        let _ = tokio::fs::create_dir_all(&temp_cache_dir).await;
        let hf_cache = hf_hub::Cache::new(temp_cache_dir.clone());

        let api_opt = hf_hub::api::tokio::ApiBuilder::new()
            .with_endpoint(hf_base_url.clone())
            .with_cache_dir(temp_cache_dir.clone())
            .build()
            .unwrap();

        // Run the worker directly with a 1-second initial backoff
        // This allows the 2 seconds of 500s to trigger retries until the mock server eventually passes
        super::background_resolve_model(
            reg,
            std::env::temp_dir(),
            hf_cache,
            hf_base_url,
            None,
            Some(reqwest_client),
            Some(api_opt),
            lock.clone(),
            sem,
            1,
        )
        .await;

        // Verify the worker eventually succeeded and updated the isolated lock
        let updated_configs = lock.read().await;
        let config = &updated_configs[0];

        assert_eq!(config.arch, ModelArch::Qwen2);
        assert_eq!(config.n_embd, 2048);
        assert_eq!(config.size_on_disk_gb, 8.0);
        assert_ne!(config.provenance.get("arch").unwrap(), "fallback");

        // Cleanup
        let _ = tokio::fs::remove_dir_all(temp_cache_dir).await;
    }

    #[tokio::test]
    async fn test_background_resolve_model_semaphore_closed() {
        let lock = Arc::new(RwLock::new(vec![mock_config(
            ModelArch::Llama,
            ModelDType::F16,
        )]));

        // Create a semaphore and immediately close it
        let sem = Arc::new(tokio::sync::Semaphore::new(1));
        sem.close();

        let reg = ModelRegistration {
            id: "test",
            name: "test",
            repo: "test/repo",
            tokenizer_repo: "test/repo",
            filename: "model.gguf",
            roles: vec![],
            compression_dtype: None,
            supported_backends: vec![],
            is_default_chat: false,
            is_default_compressor: false,
            parameters_billions: 7.0,
            non_layer_params_billions: 0.5,
            overrides: ModelOverrides::default(),
        };

        let hf_cache = hf_hub::Cache::new(std::env::temp_dir().join("test_hf_cache_sem_closed"));

        // Execute the background task with a strict 100ms timeout.
        // If the semaphore error is ignored, the function will hang in a retry loop
        // (or attempt network calls) and the timeout will trigger.
        // If handled correctly, it returns instantly.
        let result = tokio::time::timeout(
            std::time::Duration::from_millis(100),
            super::background_resolve_model(
                reg,
                std::env::temp_dir(),
                hf_cache,
                "http://localhost".to_string(),
                None,
                None,
                None,
                lock.clone(),
                sem,
                1,
            ),
        )
        .await;

        assert!(
            result.is_ok(),
            "The background task did not exit immediately when the semaphore was closed!"
        );
    }
}
