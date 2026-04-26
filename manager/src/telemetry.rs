use crate::types::GenerationParameters;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::{SystemTime, UNIX_EPOCH};
use surrealdb::{Surreal, engine::any::Any};
use tokio::sync::mpsc::UnboundedSender;
use tracing::error;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct LoadMetric {
    #[serde(default)]
    pub timestamp: u64,
    #[serde(default)]
    pub model_id: String,
    #[serde(default = "default_backend")]
    pub backend: String,
    #[serde(default)]
    pub load_time_ms: u64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum TelemetryEvent {
    Load(LoadMetric),
    Generation(GenerationMetric),
}

fn default_backend() -> String {
    "Unknown".to_string()
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct GenerationMetric {
    #[serde(default)]
    pub timestamp: u64,
    #[serde(default)]
    pub model_id: String,
    #[serde(default = "default_backend")]
    pub backend: String,
    #[serde(default)]
    pub parameters: GenerationParameters,
    #[serde(default)]
    pub offload_pct: f32,
    #[serde(default)]
    pub prompt_chars: usize,
    #[serde(default)]
    pub prompt_tokens: usize,
    #[serde(default)]
    pub tokenization_time_ms: u64,
    #[serde(default)]
    pub generation_time_ms: u64,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct TelemetryStore {
    pub loads: VecDeque<LoadMetric>,
    pub generations: VecDeque<GenerationMetric>,
    #[serde(skip)]
    pub writer_tx: Option<UnboundedSender<TelemetryEvent>>,
}

impl TelemetryStore {
    pub async fn load_from_db(db: &Surreal<Any>) -> Self {
        let loads_future =
            db.query("SELECT * FROM telemetry_loads ORDER BY timestamp DESC LIMIT 100");
        let generations_future =
            db.query("SELECT * FROM telemetry_generations ORDER BY timestamp DESC LIMIT 100");

        let (loads_res, generations_res) = tokio::join!(loads_future, generations_future);

        let mut loads = VecDeque::new();
        match loads_res {
            Ok(mut response) => {
                let mut temp: Vec<LoadMetric> = response.take(0).unwrap_or_else(|e| {
                    error!("Failed to parse telemetry_loads from database: {}", e);
                    Vec::new()
                });
                temp.reverse();
                loads = temp.into();
            }
            Err(e) => error!("Failed to query telemetry_loads from database: {}", e),
        }

        let mut generations = VecDeque::new();
        match generations_res {
            Ok(mut response) => {
                let mut temp: Vec<GenerationMetric> = response.take(0).unwrap_or_else(|e| {
                    error!("Failed to parse telemetry_generations from database: {}", e);
                    Vec::new()
                });
                temp.reverse();
                generations = temp.into();
            }
            Err(e) => error!("Failed to query telemetry_generations from database: {}", e),
        }

        TelemetryStore {
            loads,
            generations,
            writer_tx: None,
        }
    }

    pub fn record_load(&mut self, model_id: String, backend: String, load_time_ms: u64) {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let metric = LoadMetric {
            timestamp,
            model_id,
            backend,
            load_time_ms,
        };
        self.loads.push_back(metric.clone());

        if let Some(tx) = &self.writer_tx {
            let _ = tx.send(TelemetryEvent::Load(metric));
        } else {
            error!("[TELEMETRY FAULT] Writer channel missing! Dropping load metric.");
        }

        if self.loads.len() > 100 {
            self.loads.pop_front();
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
        tokenization_time_ms: u64,
        generation_time_ms: u64,
    ) {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let metric = GenerationMetric {
            timestamp,
            model_id,
            backend,
            parameters,
            offload_pct,
            prompt_chars,
            prompt_tokens,
            tokenization_time_ms,
            generation_time_ms,
        };
        self.generations.push_back(metric.clone());

        if let Some(tx) = &self.writer_tx {
            let _ = tx.send(TelemetryEvent::Generation(metric));
        } else {
            error!("[TELEMETRY FAULT] Writer channel missing! Dropping generation metric.");
        }

        if self.generations.len() > 100 {
            self.generations.pop_front();
        }
    }
}

pub async fn cleanup_telemetry(db: &Surreal<Any>, retention_days: u64) -> Result<(), String> {
    let retention_secs = retention_days * 24 * 3600;
    let tables = ["telemetry_loads", "telemetry_generations"];

    for table in tables {
        let query = format!(
            "SELECT model_id, backend, math::max(timestamp) AS max_ts FROM {} GROUP BY model_id, backend",
            table
        );

        let mut response = db.query(&query).await.map_err(|e| e.to_string())?;
        let groups: Vec<serde_json::Value> = response.take(0).unwrap_or_default();

        for group in groups {
            if let (Some(model_id), Some(backend), Some(max_ts)) = (
                group.get("model_id").and_then(|v| v.as_str()),
                group.get("backend").and_then(|v| v.as_str()),
                group
                    .get("max_ts")
                    .and_then(|v| v.as_u64().or_else(|| v.as_f64().map(|f| f as u64))),
            ) {
                let cutoff = max_ts.saturating_sub(retention_secs);
                let delete_query = format!(
                    "DELETE FROM {} WHERE model_id = $model_id AND backend = $backend AND timestamp < $cutoff",
                    table
                );
                let _ = db
                    .query(&delete_query)
                    .bind(("model_id", model_id.to_string()))
                    .bind(("backend", backend.to_string()))
                    .bind(("cutoff", cutoff))
                    .await;
            }
        }
    }
    Ok(())
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
    fn test_load_metric_default_backend() {
        let json = r#"{"timestamp": 123, "model_id": "test", "load_time_ms": 456}"#;
        let metric: LoadMetric = serde_json::from_str(json).unwrap();
        assert_eq!(metric.backend, "Unknown");
    }

    #[test]
    fn test_telemetry_missing_channel() {
        let mut store = TelemetryStore::default();
        // Should not panic
        store.record_load("test".into(), "Candle".into(), 100);
    }

    #[tokio::test]
    async fn test_telemetry_surrealdb_flow() {
        // Spin up an isolated, in-memory SurrealDB instance for testing
        let db = surrealdb::engine::any::connect("mem://").await.unwrap();
        db.use_ns("test").use_db("test").await.unwrap();

        let mut store = TelemetryStore::default();
        store.record_load("db_model".into(), "Candle".into(), 123);

        // Simulate the background task writing the state to the DB
        let res = db
            .create::<Option<LoadMetric>>("telemetry_loads")
            .content(store.loads[0].clone())
            .await;

        assert!(res.is_ok(), "DB Update Failed: {:?}", res.unwrap_err());

        let loaded_store = TelemetryStore::load_from_db(&db).await;
        assert_eq!(loaded_store.loads.len(), 1);
        assert_eq!(loaded_store.loads[0].model_id, "db_model");
    }
}
