use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

#[derive(Serialize, Clone, Debug)]
pub struct VramEvent {
    pub timestamp: u64,
    pub action: String,
    pub subsystem: String,
    pub description: String,
    pub bytes: i64,
}

#[derive(Serialize, Clone, Debug)]
pub struct RamEvent {
    pub timestamp: u64,
    pub action: String,
    pub subsystem: String,
    pub description: String,
    pub bytes: i64,
}

#[derive(Serialize, Clone, Default, Debug)]
pub struct ModelMemory {
    pub id: String,
    pub backend: String,
    #[serde(default)]
    pub is_statically_allocated: bool,
    pub status: String,
    pub weights: u64,
    pub kv_cache: u64,
    pub compute: u64,
}

#[derive(Serialize, Clone, Default, Debug)]
pub struct EngineStatus {
    pub active_chat_model_id: Option<String>,
    pub last_compressor_model_id: Option<String>,
    pub active_backend: Option<String>,
    pub benchmark_running: bool,
    pub vram_events: Vec<VramEvent>,
    pub ram_events: Vec<RamEvent>,
    pub vram_total: u64,
    pub vram_used: u64,
    pub vram_free: u64,
    pub vram_engine_claimed: u64,
    pub vram_other_processes: u64,
    pub baseline_other_vram: u64,
    pub ram_total: u64,
    pub ram_used: u64,
    pub ram_free: u64,
    pub ram_process_used: u64,
    pub ram_other_processes: u64,
    pub models_vram: Vec<ModelMemory>,
}

impl EngineStatus {
    pub fn total_engine_vram(&self) -> u64 {
        self.models_vram
            .iter()
            .map(|m| m.weights + m.kv_cache + m.compute)
            .sum()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn set_model_vram(
        &mut self,
        id: String,
        backend: String,
        is_statically_allocated: bool,
        status: String,
        weights: u64,
        kv: u64,
        comp: u64,
    ) {
        if let Some(m) = self.models_vram.iter_mut().find(|m| m.id == id) {
            m.weights = weights;
            m.kv_cache = kv;
            m.compute = comp;
            m.backend = backend;
            m.is_statically_allocated = is_statically_allocated;
            m.status = status;
        } else {
            self.models_vram.push(ModelMemory {
                id,
                backend,
                is_statically_allocated,
                status,
                weights,
                kv_cache: kv,
                compute: comp,
            });
        }
        self.vram_engine_claimed = self.total_engine_vram();
    }

    pub fn remove_model_vram(&mut self, id: &str) {
        self.models_vram.retain(|m| m.id != id);
        self.vram_engine_claimed = self.total_engine_vram();
    }

    pub fn set_model_status(&mut self, id: &str, new_status: &str) {
        if let Some(m) = self.models_vram.iter_mut().find(|m| m.id == id) {
            m.status = new_status.to_string();
        }
    }

    pub fn log_vram(&mut self, action: &str, subsystem: &str, description: &str, bytes: i64) {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        self.vram_events.push(VramEvent {
            timestamp,
            action: action.to_string(),
            subsystem: subsystem.to_string(),
            description: description.to_string(),
            bytes,
        });
        if self.vram_events.len() > 100 {
            self.vram_events.remove(0);
        }
    }

    pub fn log_ram(&mut self, action: &str, subsystem: &str, description: &str, bytes: i64) {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        self.ram_events.push(RamEvent {
            timestamp,
            action: action.to_string(),
            subsystem: subsystem.to_string(),
            description: description.to_string(),
            bytes,
        });
        if self.ram_events.len() > 100 {
            self.ram_events.remove(0);
        }
    }

    pub fn update_nvml(&mut self, total: u64, used: u64, free: u64) {
        self.vram_total = total;
        self.vram_used = used;
        self.vram_free = free;

        if self.baseline_other_vram == 0 {
            self.baseline_other_vram = used;
        }

        if self.models_vram.is_empty() {
            self.baseline_other_vram = used;
            self.vram_other_processes = used;
            self.vram_engine_claimed = 0;
        } else {
            let mut static_claimed = 0;
            for m in &self.models_vram {
                if m.is_statically_allocated {
                    static_claimed += m.weights + m.kv_cache + m.compute;
                } else {
                    static_claimed += m.weights;
                }
            }

            let active_count = self
                .models_vram
                .iter()
                .filter(|m| !m.is_statically_allocated && m.status == "Active")
                .count() as u64;

            if active_count == 0 {
                self.baseline_other_vram = used.saturating_sub(static_claimed);
            }

            let dynamic_usage = used.saturating_sub(self.baseline_other_vram + static_claimed);
            if let Some(usage_per_model) = dynamic_usage.checked_div(active_count) {
                for m in self
                    .models_vram
                    .iter_mut()
                    .filter(|m| !m.is_statically_allocated && m.status == "Active")
                {
                    m.kv_cache = usage_per_model;
                }
            }
            self.vram_engine_claimed = self.total_engine_vram();
            self.vram_other_processes = used.saturating_sub(self.vram_engine_claimed);
            if self.vram_other_processes < self.baseline_other_vram {
                self.baseline_other_vram = self.vram_other_processes;
            }
        }
    }

    pub fn update_sysinfo(&mut self, total: u64, used: u64, free: u64, process_used: u64) {
        self.ram_total = total;
        self.ram_used = used;
        self.ram_free = free;
        self.ram_process_used = process_used;
        self.ram_other_processes = used.saturating_sub(process_used);
    }
}

pub fn lock_status(status: &Arc<Mutex<EngineStatus>>) -> std::sync::MutexGuard<'_, EngineStatus> {
    status
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[derive(Deserialize, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Deserialize, Serialize, Clone, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum MemoryStrategy {
    Offload,
    Compress,
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct GenerationParameters {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub max_tokens: Option<usize>,
    pub seed: Option<u64>,
    pub memory_strategy: Option<MemoryStrategy>,
    pub context_buffer: Option<usize>,
    pub yarn_enabled: Option<bool>,
}

impl Default for GenerationParameters {
    fn default() -> Self {
        Self {
            temperature: Some(0.7),
            top_p: Some(0.95),
            top_k: Some(40),
            max_tokens: Some(500),
            seed: None,
            memory_strategy: Some(MemoryStrategy::Offload),
            context_buffer: Some(1024),
            yarn_enabled: Some(true),
        }
    }
}

#[derive(Deserialize)]
pub struct ApiRequest {
    pub chat_model_id: String,
    pub compressor_model_id: String,
    pub messages: Vec<Message>,
    pub parameters: Option<GenerationParameters>,
    pub target_backend: Option<String>,
}

#[derive(Deserialize)]
pub struct BenchmarkRequest {
    pub models: Vec<String>,
    pub parameters: Option<GenerationParameters>,
    pub target_backends: Vec<String>,
}

#[derive(Debug)]
pub enum StreamEvent {
    Token(String),
    TokenizationTime(u128),
    Done,
    Error(String),
}

pub struct UserRequest {
    pub chat_model_id: String,
    pub compressor_model_id: String,
    pub messages: Vec<Message>,
    pub responder: tokio::sync::mpsc::UnboundedSender<StreamEvent>,
    pub force_compression: bool,
    pub parameters: GenerationParameters,
    pub target_backend: Option<String>,
}
