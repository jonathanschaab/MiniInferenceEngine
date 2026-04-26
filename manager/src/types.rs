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

#[derive(Deserialize, Serialize, Clone, Debug, PartialEq, Eq)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct ChatSession {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub email: String,
    #[serde(default)]
    pub updated_at: u64,
    #[serde(default)]
    pub title: String,
    pub messages: Vec<Message>,
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct ChatSessionSummary {
    pub id: String,
    pub updated_at: u64,
    pub title: String,
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct ChatSessionRecord {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub email: String,
    #[serde(default)]
    pub updated_at: u64,
    pub title: String,
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct ChatMessageRecord {
    pub session_id: String,
    pub message_index: usize,
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
    pub seed: Option<i64>,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_status_vram() {
        let mut status = EngineStatus::default();
        status.set_model_vram(
            "model_a".into(),
            "Candle".into(),
            false,
            "Active".into(),
            100,
            50,
            10,
        );
        status.set_model_vram(
            "model_b".into(),
            "LlamaCpp".into(),
            true,
            "Idle".into(),
            200,
            100,
            20,
        );

        assert_eq!(status.total_engine_vram(), 480); // 160 + 320

        status.remove_model_vram("model_a");
        assert_eq!(status.total_engine_vram(), 320);
    }

    #[test]
    fn test_event_log_limits() {
        let mut status = EngineStatus::default();
        for i in 0..150 {
            status.log_vram("Allocate", "Sys", "Test", i as i64);
            status.log_ram("Allocate", "Sys", "Test", i as i64);
        }
        assert_eq!(status.vram_events.len(), 100);
        assert_eq!(status.ram_events.len(), 100);
    }

    #[test]
    fn test_engine_status_mutators() {
        let mut status = EngineStatus::default();
        status.set_model_vram(
            "model_a".into(),
            "Candle".into(),
            false,
            "Idle".into(),
            100,
            50,
            10,
        );

        // Test set_model_status
        status.set_model_status("model_a", "Active");
        assert_eq!(status.models_vram[0].status, "Active");

        // Test remove missing model (should safely no-op)
        let prev_vram = status.total_engine_vram();
        status.remove_model_vram("nonexistent_model");
        assert_eq!(status.models_vram.len(), 1);
        assert_eq!(status.total_engine_vram(), prev_vram);
    }

    #[test]
    fn test_update_nvml_dynamic_vram() {
        let mut status = EngineStatus::default();

        // 1. Simulate an idle system with 10GB total, 2GB used initially
        status.update_nvml(10000, 2000, 8000);
        assert_eq!(status.baseline_other_vram, 2000);

        // 2. Add two models, one static (LlamaCpp), one dynamic (Candle)
        status.set_model_vram(
            "static_model".into(),
            "LlamaCpp".into(),
            true,
            "Active".into(),
            1000,
            500,
            0,
        );
        status.set_model_vram(
            "dynamic_model".into(),
            "Candle".into(),
            false,
            "Active".into(),
            1000,
            0,
            0,
        );

        // 3. Update NVML again. Assume total VRAM usage jumped to 5500.
        status.update_nvml(10000, 5500, 4500);

        // The dynamic usage pool should be: 5500 (used) - 2000 (baseline) - 1500 (static_model) - 1000 (dynamic_weights) = 1000.
        let dyn_model = status
            .models_vram
            .iter()
            .find(|m| m.id == "dynamic_model")
            .unwrap();
        assert_eq!(dyn_model.kv_cache, 1000);
    }

    #[test]
    fn test_update_sysinfo_math() {
        let mut status = EngineStatus::default();
        status.update_sysinfo(16000, 8000, 8000, 2000);
        assert_eq!(status.ram_other_processes, 6000); // 8000 used - 2000 engine process
    }

    #[test]
    fn test_update_nvml_baseline_ratchet() {
        let mut status = EngineStatus::default();
        // Start at 2000 used, no models
        status.update_nvml(10000, 2000, 8000);
        assert_eq!(status.baseline_other_vram, 2000);

        // Now assume usage drops to 1500. The engine should ratchet the baseline down
        status.update_nvml(10000, 1500, 8500);
        assert_eq!(status.baseline_other_vram, 1500);
    }

    #[test]
    fn test_generation_parameters_default() {
        let params = GenerationParameters::default();
        assert_eq!(params.temperature, Some(0.7));
        assert_eq!(params.max_tokens, Some(500));
        assert_eq!(params.memory_strategy, Some(MemoryStrategy::Offload));
    }

    #[test]
    fn test_memory_strategy_serde() {
        // Tests #[serde(rename_all = "lowercase")] ensuring UI compatibility
        let json = serde_json::to_string(&MemoryStrategy::Offload).unwrap();
        assert_eq!(json, "\"offload\"");
        let strategy: MemoryStrategy = serde_json::from_str("\"compress\"").unwrap();
        assert_eq!(strategy, MemoryStrategy::Compress);
    }

    #[test]
    fn test_lock_status_poison_recovery() {
        let status = Arc::new(Mutex::new(EngineStatus::default()));

        // Poison the mutex on purpose inside a separate thread
        let status_clone = status.clone();
        let _ = std::thread::spawn(move || {
            let mut guard = status_clone.lock().unwrap();
            guard.benchmark_running = true;
            panic!("Intentional panic to poison the mutex");
        })
        .join();

        assert!(status.lock().is_err(), "Mutex should be poisoned");

        // lock_status should safely recover the guard and its state
        let guard = lock_status(&status);
        assert!(guard.benchmark_running);
    }

    #[test]
    fn test_update_nvml_zero_active_dynamic() {
        let mut status = EngineStatus::default();
        status.update_nvml(10000, 2000, 8000);

        // Add a dynamic model but keep it "Idle" (active_count == 0)
        status.set_model_vram(
            "idle_dyn".into(),
            "Candle".into(),
            false,
            "Idle".into(),
            1000,
            0,
            0,
        );

        // With active_count == 0, the baseline recalculates: 3500 used - 1000 static weights = 2500
        status.update_nvml(10000, 3500, 6500);
        assert_eq!(status.baseline_other_vram, 2500);
    }
}
