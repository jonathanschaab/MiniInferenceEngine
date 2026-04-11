use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use tokio::sync::oneshot;

#[derive(Serialize, Clone, Default, Debug)]
pub struct EngineStatus {
    pub active_chat_model_id: Option<String>,
    pub last_compressor_model_id: Option<String>,
    pub benchmark_running: bool,
}

pub fn lock_status(status: &Arc<Mutex<EngineStatus>>) -> std::sync::MutexGuard<'_, EngineStatus> {
    status.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[derive(Deserialize, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Deserialize)]
pub struct ApiRequest {
    pub chat_model_id: String,
    pub compressor_model_id: String,
    pub messages: Vec<Message>,
}

#[derive(Deserialize)]
pub struct BenchmarkRequest {
    pub models: Vec<String>,
}

#[derive(Serialize)]
pub struct ApiResponse {
    pub answer: String,
}

pub struct UserRequest {
    pub chat_model_id: String,
    pub compressor_model_id: String,
    pub messages: Vec<Message>,
    pub responder: oneshot::Sender<String>,
    pub force_compression: bool,
}