use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};
use std::fs;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct LoadMetric {
    pub timestamp: u64,
    pub model_id: String,
    pub load_time_ms: u128,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct GenerationMetric {
    pub timestamp: u64,
    pub model_id: String,
    pub prompt_chars: usize,
    pub prompt_tokens: usize,
    pub generation_time_ms: u128,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TelemetryStore {
    pub loads: Vec<LoadMetric>,
    pub generations: Vec<GenerationMetric>,
    #[serde(skip)]
    pub unsaved_events: usize,
}

impl Default for TelemetryStore {
    fn default() -> Self {
        Self { loads: Vec::new(), generations: Vec::new(), unsaved_events: 0 }
    }
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
            tokio::spawn(async move {
                let _ = tokio::fs::write("stats.json", json).await;
            });
        }
    }

    pub fn record_load(&mut self, model_id: String, load_time_ms: u128) {
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        self.loads.push(LoadMetric { timestamp, model_id, load_time_ms });
        
        self.unsaved_events += 1;
        if self.unsaved_events >= 5 {
            self.save_to_disk();
            self.unsaved_events = 0;
        }
    }

    pub fn record_generation(&mut self, model_id: String, prompt_chars: usize, prompt_tokens: usize, generation_time_ms: u128) {
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        self.generations.push(GenerationMetric { timestamp, model_id, prompt_chars, prompt_tokens, generation_time_ms });
        
        self.unsaved_events += 1;
        if self.unsaved_events >= 5 {
            self.save_to_disk();
            self.unsaved_events = 0;
        }
    }
}