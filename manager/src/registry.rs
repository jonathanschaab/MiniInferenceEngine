use crate::types::Message;
use hf_hub::api::sync::Api;
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
    GptOss,
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
    pub max_yarn_context: usize,
    pub sliding_window: Option<usize>,
    pub rope_scaling_factor: Option<f32>,
    pub original_max_position_embeddings: Option<usize>,
    pub num_layers: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
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
        if self.n_head > 0 && self.n_head_kv > 0 {
            (2 * self.num_layers * self.head_dim * self.n_head_kv) * bytes_per_element
        } else {
            // Fallback heuristics if precise model architecture details are omitted
            match &self.arch {
                ModelArch::Qwen2 if self.parameters_billions > 10.0 => 150_000,
                ModelArch::Qwen2 => 80_000,
                ModelArch::Llama if self.parameters_billions < 10.0 => 125_000,
                ModelArch::GptOss => 49_000,
                _ => 100_000,
            }
        }
    }
}

#[derive(Default)]
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
    pub size_on_disk_gb: Option<f32>,
}

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

// Expose the registry so the web server can send it to the UI
pub fn get_model_registry() -> &'static [ModelConfig] {
    static REGISTRY: OnceLock<Vec<ModelConfig>> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        let api_opt = Api::new().ok();
        if api_opt.is_none() {
            println!("⚠️ WARNING: Failed to init HF API. Offline mode fallback active.");
        }
        let cache = hf_hub::Cache::default();

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
        ];

        let configs: Vec<ModelConfig> = std::thread::scope(|s| {
            let mut handles = Vec::new();

            for reg in registrations {
                let api_ref = api_opt.as_ref();
                let cache_ref = &cache;

                handles.push(s.spawn(move || {
                    let mut provenance = std::collections::HashMap::new();

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
                    let mut size_on_disk_gb = reg.overrides.size_on_disk_gb;

                    let mut check_override = |opt: bool, name: &str| {
                        if opt { provenance.insert(name.to_string(), "override".to_string()); }
                    };
                    check_override(arch.is_some(), "arch");
                    check_override(kv_cache_dtype.is_some(), "kv_cache_dtype");
                    check_override(max_context_len.is_some(), "max_context_len");
                    check_override(sliding_window.is_some(), "sliding_window");
                    check_override(rope_scaling_factor.is_some(), "rope_scaling_factor");
                    check_override(original_max_position_embeddings.is_some(), "original_max_position_embeddings");
                    check_override(num_layers.is_some(), "num_layers");
                    check_override(n_embd.is_some(), "n_embd");
                    check_override(n_head.is_some(), "n_head");
                    check_override(n_head_kv.is_some(), "n_head_kv");
                    check_override(head_dim.is_some(), "head_dim");
                    check_override(size_on_disk_gb.is_some(), "size_on_disk_gb");

                    // 1. Check if GGUF is cached locally to get exact size instantly
                    if size_on_disk_gb.is_none()
                        && let Some(gguf_path) = cache_ref.repo(hf_hub::Repo::model(reg.repo.to_string())).get(reg.filename)
                            && let Ok(meta) = std::fs::metadata(&gguf_path) {
                                size_on_disk_gb = Some(meta.len() as f32 / 1024.0 / 1024.0 / 1024.0);
                            provenance.insert("size_on_disk_gb".to_string(), "disk".to_string());
                            }

                    // 2. Fetch config.json from tokenizer repo to dynamically populate architectural details
                    if num_layers.is_none() || n_head.is_none() || arch.is_none() || kv_cache_dtype.is_none() || max_context_len.is_none() || sliding_window.is_none() || rope_scaling_factor.is_none() || original_max_position_embeddings.is_none() {
                        if let Some(api) = api_ref {
                            match api.model(reg.tokenizer_repo.to_string()).get("config.json") {
                                Ok(config_path) => {
                                    if let Ok(config_str) = std::fs::read_to_string(config_path)
                                        && let Ok(json) = serde_json::from_str::<serde_json::Value>(&config_str) {
                                        let get_u64 = |key: &str| -> Option<usize> {
                                            match json.get(key) {
                                                Some(val) if !val.is_null() => {
                                                    if let Some(v) = val.as_u64() {
                                                        Some(v as usize)
                                                    } else {
                                                        println!("⚠️ WARNING: Invalid format for '{}' in config.json for {}", key, reg.id);
                                                        None
                                                    }
                                                }
                                                _ => {
                                                    if key != "head_dim" && key != "num_key_value_heads" && key != "sliding_window" {
                                                        println!("⚠️ WARNING: Missing '{}' in config.json for {}", key, reg.id);
                                                    }
                                                    None
                                                }
                                            }
                                        };

                                        let get_str = |key: &str| -> Option<String> {
                                            match json.get(key) {
                                                Some(val) if !val.is_null() => {
                                                    if let Some(v) = val.as_str() {
                                                        Some(v.to_string())
                                                    } else {
                                                        println!("⚠️ WARNING: Invalid format for '{}' in config.json for {}", key, reg.id);
                                                        None
                                                    }
                                                }
                                                _ => {
                                                    if key != "dtype" && key != "torch_dtype" {
                                                        println!("⚠️ WARNING: Missing '{}' in config.json for {}", key, reg.id);
                                                    }
                                                    None
                                                }
                                            }
                                        };

                                        // 1. Resolve arch first to inform subsequent parsing rules
                                        if arch.is_none()
                                            && let Some(model_type) = get_str("model_type") {
                                                arch = match model_type.as_str() {
                                                    "llama" => Some(ModelArch::Llama),
                                                    "qwen2" => Some(ModelArch::Qwen2),
                                                    "xlm-roberta" => Some(ModelArch::XLMRoberta),
                                                    "gpt_oss" => Some(ModelArch::GptOss),
                                                    _ => {
                                                        println!("⚠️ WARNING: Unrecognized 'model_type' ({}) in config.json for {}", model_type, reg.id);
                                                        None
                                                    }
                                                };
                                                if arch.is_some() {
                                                    provenance.insert("arch".to_string(), "config.json".to_string());
                                                }
                                            }

                                        if max_context_len.is_none()
                                            && let Some(v) = get_u64("max_position_embeddings")
                                                .or_else(|| get_u64("model_max_length"))
                                                .or_else(|| get_u64("max_sequence_length"))
                                                .or_else(|| get_u64("max_seq_len"))
                                                .or_else(|| get_u64("seq_length")) {
                                                max_context_len = Some(v);
                                                provenance.insert("max_context_len".to_string(), "config.json".to_string());
                                            }
                                        if sliding_window.is_none() && arch == Some(ModelArch::Qwen2) {
                                            if let Some(v) = get_u64("sliding_window") {
                                                sliding_window = Some(v);
                                                provenance.insert("sliding_window".to_string(), "config.json".to_string());
                                            } else {
                                                println!("⚠️ WARNING: Missing 'sliding_window' in config.json for {}", reg.id);
                                            }
                                        }
                                        if (rope_scaling_factor.is_none() || original_max_position_embeddings.is_none())
                                            && let Some(rope_scaling) = json.get("rope_scaling")
                                                && rope_scaling.is_object() {
                                                    if rope_scaling_factor.is_none()
                                                        && let Some(factor) = rope_scaling.get("factor").and_then(|v| v.as_f64()) {
                                                            rope_scaling_factor = Some(factor as f32);
                                                            provenance.insert("rope_scaling_factor".to_string(), "config.json".to_string());
                                                        }
                                                    if original_max_position_embeddings.is_none()
                                                        && let Some(orig_ctx) = rope_scaling.get("original_max_position_embeddings").and_then(|v| v.as_u64()) {
                                                            original_max_position_embeddings = Some(orig_ctx as usize);
                                                            provenance.insert("original_max_position_embeddings".to_string(), "config.json".to_string());
                                                        }
                                                }
                                        if num_layers.is_none()
                                            && let Some(v) = get_u64("num_hidden_layers") {
                                                num_layers = Some(v);
                                                provenance.insert("num_layers".to_string(), "config.json".to_string());
                                            }
                                        if n_embd.is_none()
                                            && let Some(v) = get_u64("hidden_size") {
                                                n_embd = Some(v);
                                                provenance.insert("n_embd".to_string(), "config.json".to_string());
                                            }
                                        if n_head.is_none()
                                            && let Some(v) = get_u64("num_attention_heads") {
                                                n_head = Some(v);
                                                provenance.insert("n_head".to_string(), "config.json".to_string());
                                            }
                                        if n_head_kv.is_none()
                                            && let Some(v) = get_u64("num_key_value_heads").or(n_head) {
                                                n_head_kv = Some(v);
                                                provenance.insert("n_head_kv".to_string(), "config.json".to_string());
                                            }
                                        if head_dim.is_none()
                                            && let Some(v) = get_u64("head_dim") {
                                                head_dim = Some(v);
                                                provenance.insert("head_dim".to_string(), "config.json".to_string());
                                            }
                                        if kv_cache_dtype.is_none() {
                                            if let Some(dt) = get_str("dtype").or_else(|| get_str("torch_dtype")) {
                                                kv_cache_dtype = match dt.as_str() {
                                                    "float16" => Some(ModelDType::F16),
                                                    "bfloat16" => Some(ModelDType::BF16),
                                                    "float32" => Some(ModelDType::F32),
                                                    _ => {
                                                        println!("⚠️ WARNING: Unrecognized dtype ({}) in config.json for {}", dt, reg.id);
                                                        None
                                                    }
                                                };
                                                if kv_cache_dtype.is_some() {
                                                    provenance.insert("kv_cache_dtype".to_string(), "config.json".to_string());
                                                }
                                            } else {
                                                println!("⚠️ WARNING: Missing both 'dtype' and 'torch_dtype' in config.json for {}", reg.id);
                                            }
                                        }
                                    }
                                }
                                Err(e) => println!("⚠️ WARNING: Failed to fetch config.json for {}: {}", reg.id, e),
                            }
                        } else {
                            println!("⚠️ WARNING: HF API not initialized, skipping remote config.json fetch for {}", reg.id);
                        }
                    }

                    let n_head_val = n_head.unwrap_or(1);
                    let n_embd_val = n_embd.unwrap_or(4096);

                    for name in &["arch", "kv_cache_dtype", "max_context_len", "sliding_window", "rope_scaling_factor", "original_max_position_embeddings", "num_layers", "n_embd", "n_head", "n_head_kv", "head_dim", "size_on_disk_gb"] {
                        if !provenance.contains_key(*name) {
                            provenance.insert(name.to_string(), "fallback".to_string());
                        }
                    }

                    let arch_val = arch.unwrap_or(ModelArch::Llama);
                    let max_context_len_val = max_context_len.unwrap_or(8192);

                    let mut max_yarn_context = max_context_len_val;
                    if arch_val == ModelArch::Qwen2 {
                        if let Some(sw) = sliding_window {
                            max_yarn_context = max_yarn_context.max(sw);
                        }
                    } else if let (Some(factor), Some(orig_ctx)) = (rope_scaling_factor, original_max_position_embeddings) {
                        let scaled_ctx = (orig_ctx as f32 * factor) as usize;
                        max_yarn_context = max_yarn_context.max(scaled_ctx);
                    }

                    ModelConfig {
                        id: reg.id.to_string(),
                        name: reg.name.to_string(),
                        repo: reg.repo.to_string(),
                        tokenizer_repo: reg.tokenizer_repo.to_string(),
                        filename: reg.filename.to_string(),
                        roles: reg.roles,
                        arch: arch_val,
                        compression_dtype: reg.compression_dtype,
                        kv_cache_dtype: kv_cache_dtype.unwrap_or(ModelDType::F16),
                        supported_backends: reg.supported_backends,
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
                        parameters_billions: reg.parameters_billions,
                        non_layer_params_billions: reg.non_layer_params_billions,
                        size_on_disk_gb: size_on_disk_gb.unwrap_or(5.0),
                        provenance,
                    }
                }));
            }

            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        configs
    })
}
