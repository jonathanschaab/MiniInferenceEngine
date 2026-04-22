use crate::types::GenerationParameters;
use serde::{Deserialize, Serialize};
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc::UnboundedSender;
use tracing::error;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct LoadMetric {
    pub timestamp: u64,
    pub model_id: String,
    #[serde(default = "default_backend")]
    pub backend: String,
    pub load_time_ms: u128,
}

fn default_backend() -> String {
    "Unknown".to_string()
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct GenerationMetric {
    pub timestamp: u64,
    pub model_id: String,
    #[serde(default = "default_backend")]
    pub backend: String,
    #[serde(default)]
    pub parameters: GenerationParameters,
    #[serde(default)]
    pub offload_pct: f32,
    pub prompt_chars: usize,
    pub prompt_tokens: usize,
    #[serde(default)]
    pub tokenization_time_ms: u128,
    pub generation_time_ms: u128,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct TelemetryStore {
    pub loads: Vec<LoadMetric>,
    pub generations: Vec<GenerationMetric>,
    #[serde(skip)]
    pub unsaved_events: usize,
    #[serde(skip)]
    pub writer_tx: Option<UnboundedSender<String>>,
}

impl TelemetryStore {
    pub fn load_from_disk() -> Self {
        if let Ok(data) = fs::read_to_string("stats.json") {
            serde_json::from_str(&data).unwrap_or_default()
        } else {
            Self::default()
        }
    }

    pub fn save_to_disk(&self) {
        if let Ok(json) = serde_json::to_string_pretty(self) {
            if let Some(tx) = &self.writer_tx {
                // Send payload to the dedicated writer task (Thread-safe & Ordered!)
                let _ = tx.send(json);
            } else {
                // FAIL LOUDLY: Include the exact payload that is being dropped!
                error!(
                    "[TELEMETRY FAULT] Writer channel missing! Dropping metric payload to prevent disk I/O race conditions. Check channel initialization in main.rs.\nDropped Payload:\n{}",
                    json
                );
            }
        }
    }

    pub fn record_load(&mut self, model_id: String, backend: String, load_time_ms: u128) {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.loads.push(LoadMetric {
            timestamp,
            model_id,
            backend,
            load_time_ms,
        });

        self.unsaved_events += 1;
        if self.unsaved_events >= 5 {
            self.save_to_disk();
            self.unsaved_events = 0;
        }

        if self.loads.len() > 100 {
            self.loads.drain(0..20);
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record_generation(
        &mut self,
        model_id: String,
        backend: String,
        parameters: GenerationParameters,
        offload_pct: f32,
        prompt_chars: usize,
        prompt_tokens: usize,
        tokenization_time_ms: u128,
        generation_time_ms: u128,
    ) {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.generations.push(GenerationMetric {
            timestamp,
            model_id,
            backend,
            parameters,
            offload_pct,
            prompt_chars,
            prompt_tokens,
            tokenization_time_ms,
            generation_time_ms,
        });

        self.unsaved_events += 1;
        if self.unsaved_events >= 5 {
            self.save_to_disk();
            self.unsaved_events = 0;
        }

        if self.generations.len() > 100 {
            self.generations.drain(0..20); // Remove the oldest 20 records
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::GenerationParameters;

    #[test]
    fn test_telemetry_store_limits() {
        let mut store = TelemetryStore::default();
        for i in 0..150 {
            store.record_load(format!("model_{}", i), "Candle".into(), 100);
            store.record_generation(
                format!("model_{}", i),
                "Candle".into(),
                GenerationParameters::default(),
                0.0,
                10,
                10,
                50,
                150,
            );
        }
        // Enforces the drain behavior triggering on length > 100
        assert!(store.loads.len() <= 100);
        assert!(store.generations.len() <= 100);
    }

    #[test]
    fn test_telemetry_unsaved_events_batching() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let mut store = TelemetryStore {
            writer_tx: Some(tx),
            ..Default::default()
        };

        // Record 4 loads (less than 5)
        for i in 0..4 {
            store.record_load(format!("model_{}", i), "Candle".into(), 100);
        }
        assert_eq!(store.unsaved_events, 4);
        assert!(rx.try_recv().is_err()); // No save triggered yet

        // Record 5th load (triggers save)
        store.record_load("model_4".into(), "Candle".into(), 100);
        assert_eq!(store.unsaved_events, 0);
        assert!(rx.try_recv().is_ok()); // Save was triggered and message sent
    }

    #[test]
    fn test_load_metric_default_backend() {
        let json = r#"{"timestamp": 123, "model_id": "test", "load_time_ms": 456}"#;
        let metric: LoadMetric = serde_json::from_str(json).unwrap();
        assert_eq!(metric.backend, "Unknown");
    }

    #[test]
    fn test_telemetry_save_to_disk_missing_channel() {
        let store = TelemetryStore::default();
        // Verifies the method gracefully drops payloads without panicking if channel gets severed
        store.save_to_disk();
    }
}
