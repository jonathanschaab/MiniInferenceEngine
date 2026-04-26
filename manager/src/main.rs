use auth::{AuthStore, require_session};
use axum::{
    Json,
    body::Body,
    body::Bytes,
    extract::{Path, State},
    http::StatusCode,
    http::header,
    response::{Html, IntoResponse, Redirect},
};
use hf_hub::api::sync::Api;
use oauth2::basic::BasicClient;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use sysinfo::System;
use tokenizers::Tokenizer;
use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tower_sessions::SessionManagerLayer;
use tower_sessions_surrealdb_store::SurrealSessionStore;
use tracing::{error, info, warn}; // Ensure this is imported for AppState
use tracing_subscriber::EnvFilter;

use manager::{
    ApiRequest, BenchmarkRequest, EngineStatus, Message, ModelArch, ModelConfig, ModelRole,
    StreamEvent, TelemetryStore, UserRequest, get_model_registry, lock_status, run_batcher_loop,
};

// --- CONFIGURATION ---
#[derive(Serialize, Deserialize, Clone)]
#[serde(default)]
pub struct DatabaseConfig {
    pub url: String,
    pub jwt_file_path: String,
    pub namespace: String,
    pub database: String,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            url: "ws://localhost:8001".to_string(),
            jwt_file_path: "database.jwt".to_string(),
            namespace: "mini_inference_engine".to_string(),
            database: "main".to_string(),
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct AppConfig {
    pub bind_address: String,
    pub oauth_redirect_uri: String,
    #[serde(default = "default_oauth_client_secret_path")]
    pub oauth_client_secret_path: String,
    pub admin_emails: Vec<String>,
    pub user_emails: Vec<String>,
    pub secure_cookies: bool,
    #[serde(default)]
    pub gpu_device_index: u32,
    #[serde(default = "default_telemetry_retention_days")]
    pub telemetry_retention_days: u64,
    #[serde(default = "default_log_level_console")]
    pub log_level_console: String,
    #[serde(default = "default_log_level_file")]
    pub log_level_file: String,
    #[serde(default = "default_log_level_memory")]
    pub log_level_memory: String,
    #[serde(default = "default_log_file_name")]
    pub log_file_name: String,
    #[serde(default)]
    pub database: DatabaseConfig,
}

fn default_log_level_console() -> String {
    "info".to_string()
}
fn default_log_level_file() -> String {
    "warn".to_string()
}
fn default_log_level_memory() -> String {
    "debug".to_string()
}
fn default_log_file_name() -> String {
    "server.log".to_string()
}
fn default_oauth_client_secret_path() -> String {
    "client_secret.apps.googleusercontent.com.json".to_string()
}
fn default_telemetry_retention_days() -> u64 {
    30
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            bind_address: "127.0.0.1:3000".to_string(), // Secure local default
            oauth_redirect_uri: "http://localhost:3000/auth/google/callback".to_string(),
            oauth_client_secret_path: default_oauth_client_secret_path(),
            admin_emails: vec![],
            user_emails: vec![],
            secure_cookies: true,
            gpu_device_index: 0,
            telemetry_retention_days: default_telemetry_retention_days(),
            log_level_console: default_log_level_console(),
            log_level_file: default_log_level_file(),
            log_level_memory: default_log_level_memory(),
            log_file_name: default_log_file_name(),
            database: DatabaseConfig::default(),
        }
    }
}

impl AppConfig {
    pub fn load() -> Self {
        if let Ok(data) = std::fs::read_to_string("config.toml") {
            toml::from_str(&data).unwrap_or_default()
        } else if let Ok(data) = std::fs::read_to_string("config.json") {
            // Fallback for backwards compatibility, but save as TOML going forward
            let config: Self = serde_json::from_str(&data).unwrap_or_default();
            let _ = std::fs::write("config.toml", toml::to_string_pretty(&config).unwrap());
            config
        } else {
            let config = Self::default();
            let _ = std::fs::write("config.toml", toml::to_string_pretty(&config).unwrap());
            config
        }
    }
}

pub mod auth;
pub mod setup;

// --- SHARED MEMORY LOG WRITER ---
#[derive(Clone)]
pub struct SharedLogBuffer(pub Arc<Mutex<(usize, std::collections::VecDeque<String>)>>);

pub struct SharedLogWriter {
    buffer: Arc<Mutex<(usize, std::collections::VecDeque<String>)>>,
    local_buf: Vec<u8>,
}

impl std::io::Write for SharedLogWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.local_buf.extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Drop for SharedLogWriter {
    fn drop(&mut self) {
        let s = String::from_utf8_lossy(&self.local_buf);
        let trimmed = s.trim_end();
        if !trimmed.is_empty() {
            let mut guard = self.buffer.lock().unwrap_or_else(|e| e.into_inner());
            guard.1.push_back(trimmed.to_string());
            guard.0 += 1; // Increment the global cursor counter
            if guard.1.len() > 1000 {
                guard.1.pop_front();
            }
        }
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for SharedLogBuffer {
    type Writer = SharedLogWriter;
    fn make_writer(&'a self) -> Self::Writer {
        SharedLogWriter {
            buffer: self.0.clone(),
            local_buf: Vec::new(),
        }
    }
}

pub type LogReloadHandle =
    tracing_subscriber::reload::Handle<tracing_subscriber::EnvFilter, tracing_subscriber::Registry>;

// State to share the transmitter queue across web requests
pub struct AppState {
    pub queue_tx: mpsc::Sender<UserRequest>,
    pub engine_status: Arc<Mutex<EngineStatus>>,
    pub telemetry: Arc<Mutex<TelemetryStore>>,
    pub auth_store: Arc<Mutex<AuthStore>>,
    pub reqwest_client: reqwest::Client,
    pub oauth_client: BasicClient,
    pub config: Arc<AppConfig>,
    pub log_buffer: SharedLogBuffer,
    pub log_reload_handle: LogReloadHandle,
    pub current_log_level: Arc<Mutex<String>>,
    pub db: surrealdb::Surreal<surrealdb::engine::any::Any>,
}

pub(crate) async fn serve_ui(
    session: tower_sessions::Session,
) -> Result<Html<&'static str>, Redirect> {
    if require_session(session).await.is_err() {
        return Err(Redirect::to("/auth/login"));
    }
    Ok(Html(include_str!("../index.html")))
}

// Send the model roster to the Javascript dropdowns
pub(crate) async fn get_models() -> Json<Vec<ModelConfig>> {
    Json(get_model_registry().await)
}

// Handle incoming chat requests
pub(crate) async fn handle_generate(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<ApiRequest>,
) -> impl IntoResponse {
    let (response_tx, response_rx) = mpsc::unbounded_channel();

    // Package the UI's model choices and the chat history
    let request = UserRequest {
        chat_model_id: payload.chat_model_id,
        compressor_model_id: payload.compressor_model_id,
        messages: payload.messages,
        responder: response_tx,
        force_compression: false,
        parameters: payload.parameters.unwrap_or_default(),
        target_backend: payload.target_backend,
    };

    // Send to the GPU thread
    let _ = state.queue_tx.send(request).await;

    // Map the incoming channel into an HTTP streaming body
    let stream = UnboundedReceiverStream::new(response_rx).map(|event| match event {
        StreamEvent::Token(t) => Ok::<_, std::convert::Infallible>(Bytes::from(t)),
        StreamEvent::TokenizationTime(_) => Ok(Bytes::new()),
        StreamEvent::Done => Ok(Bytes::new()),
        StreamEvent::Error(e) => Ok(Bytes::from(format!("Error: {}", e))),
    });

    axum::response::Response::builder()
        .header(header::CONTENT_TYPE, "text/plain; charset=utf-8")
        .header(header::TRANSFER_ENCODING, "chunked")
        .body(Body::from_stream(stream))
        .unwrap()
}

pub(crate) async fn get_status(State(state): State<Arc<AppState>>) -> Json<EngineStatus> {
    let current_status = lock_status(&state.engine_status).clone();
    Json(current_status)
}

// The Automated Benchmark Trigger
pub(crate) async fn trigger_benchmark(
    State(state): State<Arc<AppState>>,
    user: auth::CurrentUser,
    Json(payload): Json<BenchmarkRequest>,
) -> impl IntoResponse {
    if !user.is_admin {
        warn!("Benchmark trigger rejected for non-admin: {}", user.email);
        return (
            StatusCode::FORBIDDEN,
            "Only administrators can run the benchmark suite.",
        )
            .into_response();
    }

    // Check if a benchmark is already running
    {
        let mut status = lock_status(&state.engine_status);
        if status.benchmark_running {
            return (StatusCode::CONFLICT, "Benchmark suite is already running.").into_response();
        }
        // Lock it!
        status.benchmark_running = true;
    }

    let mut params = payload.parameters.unwrap_or_default();
    if params.seed.is_none() {
        params.seed = Some(rand::random::<i64>());
    }
    let target_backends = payload.target_backends;
    let queue_tx = state.queue_tx.clone();
    let engine_status = state.engine_status.clone(); // Clone the Arc so the background thread can reset it
    let selected_models = payload.models;

    tokio::spawn(async move {
        info!("🚀 Starting Automated Benchmark Suite...");
        let full_registry = get_model_registry().await;

        let default_compressor = full_registry
            .iter()
            .find(|m| m.is_default_compressor)
            .or_else(|| {
                full_registry
                    .iter()
                    .find(|m| m.roles.contains(&ModelRole::ContextCompressor))
            })
            .map(|m| m.id.clone())
            .unwrap_or_else(|| "llmlingua-2-f16".to_string());

        let default_chat = full_registry
            .iter()
            .find(|m| m.is_default_chat)
            .or_else(|| {
                full_registry
                    .iter()
                    .find(|m| m.roles.contains(&ModelRole::GeneralChat))
            })
            .map(|m| m.id.clone())
            .unwrap_or_else(|| "qwen-2.5-7b".to_string());

        let prompt_generator = full_registry
            .iter()
            .find(|m| m.is_default_compressor && m.arch != ModelArch::XLMRoberta)
            .or_else(|| {
                full_registry.iter().find(|m| {
                    m.roles.contains(&ModelRole::GeneralChat) && m.parameters_billions < 4.0
                })
            })
            .or_else(|| full_registry.iter().find(|m| m.is_default_chat))
            .or_else(|| {
                full_registry
                    .iter()
                    .find(|m| m.roles.contains(&ModelRole::GeneralChat))
            })
            .map(|m| m.id.clone())
            .unwrap_or_else(|| default_chat.clone());

        let registry: Vec<ModelConfig> = full_registry
            .iter()
            .filter(|m| selected_models.contains(&m.id))
            .cloned()
            .collect();

        if registry.is_empty() {
            warn!("Benchmark aborted: No valid models selected.");
            let mut status = lock_status(&engine_status);
            status.benchmark_running = false;
            return;
        }

        if target_backends.is_empty() {
            warn!("Benchmark aborted: No valid backends selected.");
            let mut status = lock_status(&engine_status);
            status.benchmark_running = false;
            return;
        }

        if let Err(e) = tokio::fs::create_dir_all("benchmark_prompts").await {
            error!(
                "Benchmark aborted: Failed to create prompt directory: {}",
                e
            );
            let mut status = lock_status(&engine_status);
            status.benchmark_running = false;
            return;
        }

        // --- GENERATE REALISTIC PROMPTS ---
        info!("🌱 Verifying benchmark prompt files...");

        let tokenizer_result = tokio::task::spawn_blocking(|| {
            let api = Api::new().map_err(|e| format!("API Init Error: {}", e))?;
            let path = api
                .model("Qwen/Qwen2.5-1.5B-Instruct".to_string())
                .get("tokenizer.json")
                .map_err(|e| format!("Tokenizer Download Error: {}", e))?;

            Tokenizer::from_file(path).map_err(|e| format!("Tokenizer Parse Error: {}", e))
        })
        .await;

        let mut qwen_tokenizer = None;

        match tokenizer_result {
            Ok(Ok(tokenizer)) => {
                qwen_tokenizer = Some(tokenizer);
            }
            Ok(Err(e)) => {
                warn!(
                    "Tokenizer initialization failed: {}. Falling back to padding.",
                    e
                );
            }
            Err(e) => {
                // If the spawn_blocking task actually panics, it is caught here as a JoinError
                error!("Thread execution failed: {}. Falling back to padding.", e);
            }
        }

        let mut all_sizes = std::collections::HashSet::new();
        for model in registry.iter() {
            let mut sizes = vec![1, 10, 100, 1000, 10000];
            let safe_max = (model.max_context_len as f32 * 0.95) as usize;
            sizes.retain(|&s| s < safe_max);
            if safe_max > 0 {
                sizes.push(safe_max);
            }
            for s in sizes {
                all_sizes.insert(s);
            }
        }

        let mut sorted_sizes: Vec<usize> = all_sizes.into_iter().collect();
        sorted_sizes.sort();

        for size in sorted_sizes {
            let filename = format!("benchmark_prompts/prompt_{}.txt", size);

            if !std::path::Path::new(&filename).exists() {
                let mut should_save = false;
                let mut final_content = String::new();

                if size < 1000 {
                    info!(
                        "🧠 Using {} to generate realistic prompt of ~{} tokens...",
                        prompt_generator, size
                    );
                    let prompt_instruction = if size <= 50 {
                        format!(
                            "Write a very short technical sentence about Rust. Limit to {} words.",
                            size
                        )
                    } else {
                        format!(
                            "Write a detailed paragraph about Rust memory management. Target exactly {} words.",
                            size
                        )
                    };

                    let (seed_tx, mut seed_rx) = mpsc::unbounded_channel();
                    let _ = queue_tx
                        .send(UserRequest {
                            chat_model_id: prompt_generator.clone(),
                            compressor_model_id: default_compressor.clone(),
                            messages: vec![Message {
                                role: "user".to_string(),
                                content: prompt_instruction,
                            }],
                            responder: seed_tx,
                            force_compression: false,
                            parameters: params.clone(),
                            target_backend: None,
                        })
                        .await;

                    let mut generated_seed = String::new();
                    while let Some(ev) = seed_rx.recv().await {
                        match ev {
                            StreamEvent::Token(t) => generated_seed.push_str(&t),
                            StreamEvent::Done => break,
                            StreamEvent::Error(e) => {
                                generated_seed.push_str(&e);
                                break;
                            }
                            StreamEvent::TokenizationTime(_) => {} // Ignore for seed generation
                        }
                    }

                    if !generated_seed.is_empty()
                        && !generated_seed.starts_with("Server Error")
                        && let Some(tokenizer) = &qwen_tokenizer
                        && let Ok(encoding) = tokenizer.encode(generated_seed.clone(), true)
                    {
                        let token_len = encoding.get_ids().len();
                        let margin = (size as f32 * 0.35) as usize;

                        if token_len >= size.saturating_sub(margin) && token_len <= size + margin {
                            final_content = generated_seed;
                            should_save = true;
                            info!(
                                "✅ Generated {} tokens (within 35% of {}). Saving.",
                                token_len, size
                            );
                        } else {
                            warn!(
                                "Generation missed margin: {} tokens (target {}).",
                                token_len, size
                            );
                        }
                    }
                } else {
                    // --- ALGORITHMIC SYNTHESIS FOR LARGE CONTEXT ---
                    info!(
                        "🖨️ Synthesizing highly unique {} token data payload...",
                        size
                    );
                    let mut synthetic_data = String::with_capacity(size * 4);
                    synthetic_data.push_str("=== MULTI-USER SYSTEM DIAGNOSTIC LOG EXPORT ===\n");

                    let events = [
                        "PLAYER_CONNECT",
                        "ROOM_TRANSITION",
                        "COMBAT_ACTION",
                        "TELNET_NEGOTIATION",
                        "INVENTORY_SYNC",
                    ];
                    let statuses = ["SUCCESS", "TIMEOUT", "DISCONNECT", "INVALID_CMD", "OK"];

                    let mut current_tokens = 0;
                    let mut counter = 0;
                    let mut chunk_buffer = String::with_capacity(8192);

                    if let Some(tokenizer) = &qwen_tokenizer {
                        if let Ok(enc) = tokenizer.encode(synthetic_data.as_str(), true) {
                            current_tokens = enc.get_ids().len();
                        }

                        while current_tokens < size {
                            let ev = events[counter % events.len()];
                            let st = statuses[(counter / 3) % statuses.len()];
                            let session_id = uuid::Uuid::new_v4();

                            let log_line = format!(
                                "[2026-04-11T11:36:{:02}Z] SESSION: {} | EVENT: {} | STATUS: {} | LATENCY: {}ms | ALLOC: {}KB\n",
                                counter % 60,
                                session_id,
                                ev,
                                st,
                                (counter * 7) % 300,
                                (counter * 13) % 2048
                            );

                            synthetic_data.push_str(&log_line);
                            chunk_buffer.push_str(&log_line);

                            if counter % 50 == 0 {
                                let new_tokens = tokenizer
                                    .encode(chunk_buffer.as_str(), false)
                                    .map(|enc| enc.get_ids().len())
                                    .unwrap_or(0);

                                current_tokens += if new_tokens > 0 {
                                    new_tokens
                                } else {
                                    (chunk_buffer.len() / 4).max(1)
                                };
                                chunk_buffer.clear();
                            }
                            counter += 1;
                        }

                        if let Ok(final_encoding) = tokenizer.encode(synthetic_data, true) {
                            let safe_size = size.min(final_encoding.get_ids().len());
                            let exact_slice = &final_encoding.get_ids()[0..safe_size];

                            if let Ok(decoded) = tokenizer.decode(exact_slice, true) {
                                final_content = decoded;
                                should_save = true;
                                info!(
                                    "✅ Successfully synthesized exactly {} unique tokens.",
                                    safe_size
                                );
                            } else {
                                warn!("Decode failed. Falling back to padding.");
                            }
                        } else {
                            warn!("Encode failed. Falling back to padding.");
                        }
                    }
                }

                if !should_save {
                    warn!("Fallback: Using generic memory padding. (Will NOT save to disk).");
                } else {
                    let _ = tokio::fs::write(&filename, &final_content).await;
                }
            }
        }

        info!("✅ Setup Phase Complete. Beginning standard benchmark sweeps...");

        // --- GENERATIVE MODELS ---
        for model in registry.iter().filter(|m| {
            m.roles.contains(&ModelRole::GeneralChat)
                || m.roles.contains(&ModelRole::CodeSpecialist)
        }) {
            let mut test_sizes = vec![1, 10, 100, 1000, 10000];
            let safe_max = (model.max_context_len as f32 * 0.95) as usize;
            test_sizes.retain(|&s| s < safe_max);
            if safe_max > 0 {
                test_sizes.push(safe_max);
            }

            for size in test_sizes {
                let filename = format!("benchmark_prompts/prompt_{}.txt", size);
                let exact_prompt = tokio::fs::read_to_string(&filename)
                    .await
                    .unwrap_or_else(|_| "system ".repeat(size).trim().to_string());

                for target_b in target_backends.iter() {
                    let supports = model
                        .supported_backends
                        .iter()
                        .any(|b| format!("{:?}", b).to_lowercase() == target_b.to_lowercase());
                    if !supports {
                        continue;
                    }

                    info!(
                        "📊 Benchmarking Generative {} using file {} on Backend {}...",
                        model.name, filename, target_b
                    );

                    let (response_tx, mut response_rx) = mpsc::unbounded_channel();
                    let _ = queue_tx
                        .send(UserRequest {
                            chat_model_id: model.id.clone(),
                            compressor_model_id: default_compressor.clone(),
                            messages: vec![Message {
                                role: "user".to_string(),
                                content: exact_prompt.clone(),
                            }],
                            responder: response_tx,
                            force_compression: false,
                            parameters: params.clone(),
                            target_backend: Some(target_b.clone()),
                        })
                        .await;
                    while let Some(ev) = response_rx.recv().await {
                        if let StreamEvent::Done = ev {
                            break;
                        }
                    }
                }
            }
        }

        // --- COMPRESSOR MODELS ---
        info!("🚀 Transitioning to Compressor Benchmarks...");
        for comp_model in registry
            .iter()
            .filter(|m| m.roles.contains(&ModelRole::ContextCompressor))
        {
            let mut test_sizes = vec![1, 10, 100, 1000, 10000];
            let safe_max = (comp_model.max_context_len as f32 * 0.95) as usize;
            test_sizes.retain(|&s| s < safe_max);
            if safe_max > 0 {
                test_sizes.push(safe_max);
            }

            for size in test_sizes {
                let filename = format!("benchmark_prompts/prompt_{}.txt", size);
                let exact_prompt = tokio::fs::read_to_string(&filename)
                    .await
                    .unwrap_or_else(|_| "system ".repeat(size).trim().to_string());

                for target_b in target_backends.iter() {
                    let supports = comp_model
                        .supported_backends
                        .iter()
                        .any(|b| format!("{:?}", b).to_lowercase() == target_b.to_lowercase());
                    if !supports {
                        continue;
                    }

                    info!(
                        "📊 Benchmarking Compressor {} using file {} on Backend {}...",
                        comp_model.name, filename, target_b
                    );

                    let (response_tx, mut response_rx) = mpsc::unbounded_channel();
                    let _ = queue_tx
                        .send(UserRequest {
                            chat_model_id: default_chat.clone(),
                            compressor_model_id: comp_model.id.clone(),
                            messages: vec![Message {
                                role: "user".to_string(),
                                content: exact_prompt.clone(),
                            }],
                            responder: response_tx,
                            force_compression: true,
                            parameters: params.clone(),
                            target_backend: Some(target_b.clone()),
                        })
                        .await;
                    while let Some(ev) = response_rx.recv().await {
                        if let StreamEvent::Done = ev {
                            break;
                        }
                    }
                }
            }
        }

        info!("✅ Automated Benchmark Suite Complete!");

        // Reset the flag when the thread finishes
        {
            let mut status = lock_status(&engine_status);
            status.benchmark_running = false;
        }
    });

    (
        StatusCode::ACCEPTED,
        "Benchmark sweep started in the background.",
    )
        .into_response()
}

// Route: Serve the Stats UI
pub(crate) async fn serve_stats_ui(
    session: tower_sessions::Session,
) -> Result<Html<&'static str>, Redirect> {
    if require_session(session).await.is_err() {
        return Err(Redirect::to("/auth/login"));
    }
    Ok(Html(include_str!("../stats.html")))
}

pub(crate) async fn serve_settings_ui(
    session: tower_sessions::Session,
) -> Result<Html<&'static str>, Redirect> {
    if require_session(session).await.is_err() {
        return Err(Redirect::to("/auth/login"));
    }
    Ok(Html(include_str!("../settings.html")))
}

pub(crate) async fn serve_models_ui(
    session: tower_sessions::Session,
) -> Result<Html<&'static str>, Redirect> {
    if require_session(session).await.is_err() {
        return Err(Redirect::to("/auth/login"));
    }
    Ok(Html(include_str!("../models.html")))
}

pub(crate) async fn serve_memory_ui(
    session: tower_sessions::Session,
) -> Result<Html<&'static str>, Redirect> {
    if require_session(session).await.is_err() {
        return Err(Redirect::to("/auth/login"));
    }
    Ok(Html(include_str!("../memory.html")))
}

pub(crate) async fn serve_console_ui(
    session: tower_sessions::Session,
    State(state): State<Arc<AppState>>,
) -> Result<Html<&'static str>, Redirect> {
    let email = match require_session(session).await {
        Ok(e) => e,
        Err(_) => return Err(Redirect::to("/auth/login")),
    };
    if !state.config.admin_emails.contains(&email) {
        return Err(Redirect::to("/"));
    }
    Ok(Html(include_str!("../console.html")))
}

pub(crate) async fn serve_console_js() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "application/javascript")],
        include_str!("../console.js"),
    )
}

#[derive(Deserialize)]
pub struct LogQuery {
    pub since: Option<usize>,
}

#[derive(Serialize)]
pub struct LogResponse {
    pub logs: Vec<String>,
    pub next_cursor: usize,
}

pub(crate) async fn get_console_logs(
    user: auth::CurrentUser,
    axum::extract::Query(query): axum::extract::Query<LogQuery>,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    if !user.is_admin {
        return (StatusCode::FORBIDDEN, "Admin access required").into_response();
    }
    let guard = state.log_buffer.0.lock().unwrap_or_else(|e| e.into_inner());

    let total_emitted = guard.0;
    let buffer = &guard.1;
    let oldest_available = total_emitted.saturating_sub(buffer.len());
    let since = query.since.unwrap_or(0);

    let logs = if since < oldest_available {
        buffer.iter().cloned().collect() // Client fell behind; send everything
    } else if since >= total_emitted {
        Vec::new() // Up to date
    } else {
        let start_idx = since - oldest_available;
        buffer.iter().skip(start_idx).cloned().collect() // Send only missing slice
    };

    Json(LogResponse {
        logs,
        next_cursor: total_emitted,
    })
    .into_response()
}

pub(crate) async fn clear_console_logs(
    user: auth::CurrentUser,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    if !user.is_admin {
        return (StatusCode::FORBIDDEN, "Admin access required").into_response();
    }
    let mut guard = state.log_buffer.0.lock().unwrap_or_else(|e| e.into_inner());
    guard.1.clear();
    guard.0 = 0;
    StatusCode::OK.into_response()
}

#[derive(Deserialize, Serialize)]
pub struct LogLevelRequest {
    pub level: String,
}

pub(crate) async fn set_console_loglevel(
    user: auth::CurrentUser,
    State(state): State<Arc<AppState>>,
    Json(payload): Json<LogLevelRequest>,
) -> impl IntoResponse {
    if !user.is_admin {
        return (StatusCode::FORBIDDEN, "Admin access required").into_response();
    }
    let new_filter = match EnvFilter::try_new(&payload.level) {
        Ok(filter) => filter,
        Err(e) => {
            return (StatusCode::BAD_REQUEST, format!("Invalid log level: {}", e)).into_response();
        }
    };
    if let Err(e) = state.log_reload_handle.reload(new_filter) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to reload log level: {}", e),
        )
            .into_response();
    }
    *state
        .current_log_level
        .lock()
        .unwrap_or_else(|e| e.into_inner()) = payload.level.clone();
    (StatusCode::OK, "Log level updated").into_response()
}

pub(crate) async fn list_chat_sessions(
    user: auth::CurrentUser,
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<manager::ChatSessionSummary>>, StatusCode> {
    let mut response = state.db.query("SELECT type::string(meta::id(id)) AS id, updated_at, title FROM chat_sessions WHERE email = $email ORDER BY updated_at DESC")
        .bind(("email", user.email.clone()))
        .await.map_err(|e| { error!("DB List Error: {}", e); StatusCode::INTERNAL_SERVER_ERROR })?;
    let sessions: Vec<manager::ChatSessionSummary> = response.take(0).unwrap_or_default();
    Ok(Json(sessions))
}

pub(crate) async fn get_chat_session(
    user: auth::CurrentUser,
    Path(id): Path<String>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<manager::ChatSession>, StatusCode> {
    let mut response = state
        .db
        .query("SELECT type::string(meta::id(id)) AS id, email, updated_at, title FROM type::thing('chat_sessions', $id)")
        .bind(("id", id.clone()))
        .await
        .map_err(|e| {
            error!("DB Query Error: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    let session_record: Option<manager::ChatSessionRecord> = response.take(0).map_err(|e| {
        error!("DB Take Error: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    match session_record {
        Some(s) if s.email == user.email => {
            let mut response = state
                .db
                .query("SELECT role, content, message_index FROM chat_messages WHERE session_id = $session_id ORDER BY message_index ASC")
                .bind(("session_id", id.clone()))
                .await
                .map_err(|e| {
                    error!("DB Query Error: {}", e);
                    StatusCode::INTERNAL_SERVER_ERROR
                })?;

            let db_messages: Vec<manager::Message> = response.take(0).unwrap_or_default();

            Ok(Json(manager::ChatSession {
                id: s.id,
                email: s.email,
                updated_at: s.updated_at,
                title: s.title,
                messages: db_messages,
            }))
        }
        Some(_) => Err(StatusCode::FORBIDDEN),
        None => Err(StatusCode::NOT_FOUND),
    }
}

pub(crate) async fn save_chat_session(
    user: auth::CurrentUser,
    State(state): State<Arc<AppState>>,
    Json(mut payload): Json<manager::ChatSessionRecord>,
) -> Result<Json<manager::ChatSessionRecord>, StatusCode> {
    payload.email = user.email.clone();
    payload.updated_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // Sanitize the title: trim leading punctuation/whitespace, truncate to 30 chars, and provide a fallback.
    let clean_title: String = payload
        .title
        .trim_start_matches(|c: char| c.is_whitespace() || c.is_ascii_punctuation())
        .chars()
        .take(30)
        .collect::<String>()
        .trim_end()
        .to_string();

    payload.title = if clean_title.is_empty() {
        "New Chat Session".to_string()
    } else {
        clean_title
    };

    if !payload.id.is_empty() {
        if uuid::Uuid::parse_str(&payload.id).is_err() {
            return Err(StatusCode::BAD_REQUEST);
        }

        let mut response = state
            .db
            .query("SELECT type::string(meta::id(id)) AS id, email, updated_at, title FROM type::thing('chat_sessions', $id)")
            .bind(("id", payload.id.clone()))
            .await
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        let existing: Option<manager::ChatSessionRecord> = response.take(0).unwrap_or_default();
        if let Some(s) = existing
            && s.email != user.email
        {
            return Err(StatusCode::FORBIDDEN);
        }
    } else {
        payload.id = uuid::Uuid::new_v4().to_string();
    }

    let mut response = state
        .db
        .query("UPSERT type::thing('chat_sessions', $id) MERGE { email: $email, updated_at: $updated_at, title: $title } RETURN type::string(meta::id(id)) AS id, email, updated_at, title")
        .bind(("id", payload.id.clone()))
        .bind(("email", payload.email.clone()))
        .bind(("updated_at", payload.updated_at))
        .bind(("title", payload.title.clone()))
        .await
        .map_err(|e| {
            error!("DB Save Error: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    let saved: Option<manager::ChatSessionRecord> = response.take(0).map_err(|e| {
        error!("DB Take Error: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    saved.map(Json).ok_or_else(|| {
        error!(
            "DB Save Error: Upsert returned None for session {}",
            payload.id
        );
        StatusCode::INTERNAL_SERVER_ERROR
    })
}

pub(crate) async fn append_chat_message(
    user: auth::CurrentUser,
    Path(id): Path<String>,
    State(state): State<Arc<AppState>>,
    Json(mut payload): Json<manager::ChatMessageRecord>,
) -> Result<StatusCode, StatusCode> {
    let mut response = state
        .db
        .query("SELECT type::string(meta::id(id)) AS id, email, updated_at, title FROM type::thing('chat_sessions', $id)")
        .bind(("id", id.clone()))
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let session: Option<manager::ChatSessionRecord> = response.take(0).unwrap_or_default();

    if let Some(s) = session {
        if s.email != user.email {
            return Err(StatusCode::FORBIDDEN);
        }
    } else {
        return Err(StatusCode::NOT_FOUND);
    }

    payload.session_id = id.clone();
    if let Err(e) = state
        .db
        .create::<Option<manager::ChatMessageRecord>>("chat_messages")
        .content(payload)
        .await
    {
        error!("DB Create Error (chat_messages): {}", e);
        return Err(StatusCode::INTERNAL_SERVER_ERROR);
    }

    let updated_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    if let Err(e) = state
        .db
        .query("UPDATE type::thing('chat_sessions', $id) SET updated_at = $time")
        .bind(("id", id.clone()))
        .bind(("time", updated_at))
        .await
    {
        error!("DB Update Error (chat_sessions timestamp): {}", e);
    }

    Ok(StatusCode::OK)
}

pub(crate) async fn truncate_chat_messages(
    user: auth::CurrentUser,
    Path((id, index)): Path<(String, usize)>,
    State(state): State<Arc<AppState>>,
) -> Result<StatusCode, StatusCode> {
    let mut response = state
        .db
        .query("SELECT type::string(meta::id(id)) AS id, email, updated_at, title FROM type::thing('chat_sessions', $id)")
        .bind(("id", id.clone()))
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let session: Option<manager::ChatSessionRecord> = response.take(0).unwrap_or_default();

    if let Some(s) = session {
        if s.email != user.email {
            return Err(StatusCode::FORBIDDEN);
        }
    } else {
        return Err(StatusCode::NOT_FOUND);
    }

    if let Err(e) = state
        .db
        .query(
            "DELETE FROM chat_messages WHERE session_id = $session_id AND message_index >= $index",
        )
        .bind(("session_id", id.clone()))
        .bind(("index", index))
        .await
    {
        error!("DB Delete Error (truncate chat_messages): {}", e);
        return Err(StatusCode::INTERNAL_SERVER_ERROR);
    }

    let updated_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    if let Err(e) = state
        .db
        .query("UPDATE type::thing('chat_sessions', $id) SET updated_at = $time")
        .bind(("id", id.clone()))
        .bind(("time", updated_at))
        .await
    {
        error!(
            "DB Update Error (chat_sessions timestamp on truncate): {}",
            e
        );
    }

    Ok(StatusCode::OK)
}

pub(crate) async fn delete_chat_session(
    user: auth::CurrentUser,
    Path(id): Path<String>,
    State(state): State<Arc<AppState>>,
) -> Result<StatusCode, StatusCode> {
    let mut response = state
        .db
        .query("SELECT type::string(meta::id(id)) AS id, email, updated_at, title FROM type::thing('chat_sessions', $id)")
        .bind(("id", id.clone()))
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let session: Option<manager::ChatSessionRecord> = response.take(0).unwrap_or_default();
    if let Some(s) = session {
        if s.email != user.email {
            return Err(StatusCode::FORBIDDEN);
        }
        if let Err(e) = state
            .db
            .query("DELETE type::thing('chat_sessions', $id)")
            .bind(("id", id.clone()))
            .await
        {
            error!("DB Delete Error (chat_sessions): {}", e);
            return Err(StatusCode::INTERNAL_SERVER_ERROR);
        }
        if let Err(e) = state
            .db
            .query("DELETE FROM chat_messages WHERE session_id = $session_id")
            .bind(("session_id", id.clone()))
            .await
        {
            error!("DB Delete Error (chat_messages): {}", e);
        }
        Ok(StatusCode::OK)
    } else {
        Err(StatusCode::NOT_FOUND)
    }
}

pub(crate) async fn get_console_loglevel(
    user: auth::CurrentUser,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    if !user.is_admin {
        return (StatusCode::FORBIDDEN, "Admin access required").into_response();
    }
    let level = state
        .current_log_level
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    Json(LogLevelRequest { level }).into_response()
}

pub(crate) async fn serve_chat_js() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "application/javascript")],
        include_str!("../chat.js"),
    )
}

pub(crate) async fn serve_stats_js() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "application/javascript")],
        include_str!("../stats.js"),
    )
}

pub(crate) async fn serve_models_js() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "application/javascript")],
        include_str!("../models.js"),
    )
}

pub(crate) async fn serve_settings_js() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "application/javascript")],
        include_str!("../settings.js"),
    )
}

pub(crate) async fn serve_memory_js() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "application/javascript")],
        include_str!("../memory.js"),
    )
}

pub(crate) async fn serve_common_js() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "application/javascript")],
        include_str!("../common.js"),
    )
}

pub(crate) async fn serve_common_css() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "text/css")],
        include_str!("../common.css"),
    )
}

// Route: Serve the Telemetry JSON
pub(crate) async fn get_stats_data(State(state): State<Arc<AppState>>) -> Json<TelemetryStore> {
    let current_data = state
        .telemetry
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone();
    Json(current_data)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = AppConfig::load();

    let (memory_buffer, log_reload_handle, _file_guard) = setup::init_logging(&config);

    // Create the async channel for the GPU queue
    let (tx, rx) = mpsc::channel(32);

    // Initialize the shared state BEFORE spawning the background thread
    let engine_status = Arc::new(Mutex::new(EngineStatus::default()));

    let db_client = setup::init_db(&config).await?;

    // Background Telemetry Cleanup Task
    let db_for_cleanup = db_client.clone();
    let retention_days = config.telemetry_retention_days;
    tokio::spawn(async move {
        if retention_days == 0 {
            return; // 0 disables retention cleanup
        }
        // Check every 24 hours (The first tick completes immediately on startup)
        let mut cleanup_interval = tokio::time::interval(std::time::Duration::from_secs(24 * 3600));
        loop {
            cleanup_interval.tick().await;
            if let Err(e) = manager::cleanup_telemetry(&db_for_cleanup, retention_days).await {
                error!("Telemetry cleanup task failed: {}", e);
            }
        }
    });

    // Background VRAM Tracker
    let status_for_nvml = engine_status.clone();
    let vram_tracker_gpu_idx = config.gpu_device_index;
    tokio::spawn(async move {
        let mut sys = System::new_all();
        let pid = sysinfo::get_current_pid().expect("Failed to get current PID");
        let nvml = nvml_wrapper::Nvml::init().ok();
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(1));
        loop {
            interval.tick().await;

            sys.refresh_memory();
            sys.refresh_process(pid);

            let mut s = manager::lock_status(&status_for_nvml);

            if let Some(process) = sys.process(pid) {
                s.update_sysinfo(
                    sys.total_memory(),
                    sys.used_memory(),
                    sys.free_memory(),
                    process.memory(),
                );
            }

            if let Some((used, total, free)) =
                manager::get_vram_info(nvml.as_ref(), vram_tracker_gpu_idx)
            {
                s.update_nvml(total, used, free);
            }
        }
    });

    let status_for_batcher = engine_status.clone();

    let (telemetry_tx, mut telemetry_rx) =
        tokio::sync::mpsc::unbounded_channel::<manager::TelemetryEvent>();
    let db_for_telemetry = db_client.clone();
    tokio::spawn(async move {
        // This loop processes writes one-at-a-time, storing individual records instead of a monolithic JSON blob
        while let Some(event) = telemetry_rx.recv().await {
            match event {
                manager::TelemetryEvent::Load(metric) => {
                    if let Err(e) = db_for_telemetry
                        .create::<Option<manager::LoadMetric>>("telemetry_loads")
                        .content(metric)
                        .await
                    {
                        error!("Background DB Write Error (telemetry_loads): {}", e);
                    }
                }
                manager::TelemetryEvent::Generation(metric) => {
                    if let Err(e) = db_for_telemetry
                        .create::<Option<manager::GenerationMetric>>("telemetry_generations")
                        .content(metric)
                        .await
                    {
                        error!("Background DB Write Error (telemetry_generations): {}", e);
                    }
                }
            }
        }
    });

    let mut store = TelemetryStore::load_from_db(&db_client).await;
    store.writer_tx = Some(telemetry_tx); // Wire the channel into the store
    let telemetry = Arc::new(Mutex::new(store));

    let telemetry_for_batcher = telemetry.clone();

    // Boot up the GPU Orchestrator in the background
    let batcher_gpu_idx = config.gpu_device_index;
    tokio::spawn(async move {
        run_batcher_loop(
            rx,
            status_for_batcher,
            telemetry_for_batcher,
            batcher_gpu_idx,
        )
        .await;
    });

    let (auth_tx, mut auth_rx) = tokio::sync::mpsc::unbounded_channel::<auth::UserApiKeys>();
    let db_for_auth = db_client.clone();
    tokio::spawn(async move {
        // This loop processes writes one-at-a-time per user, preventing giant monolithic DB updates
        while let Some(user_keys) = auth_rx.recv().await {
            if let Err(e) = db_for_auth
                .upsert::<Option<auth::UserApiKeys>>(("auth_keys", &user_keys.email))
                .content(user_keys)
                .await
            {
                error!("Background DB Write Error (auth_keys): {}", e);
            }
        }
    });

    let mut store = AuthStore::load(&db_client).await?;
    store.writer_tx = Some(auth_tx);
    let auth_store = Arc::new(Mutex::new(store));

    // Initialize global pooled clients once!
    let reqwest_client = reqwest::Client::new();
    let oauth_client =
        auth::build_oauth_client(&config.oauth_redirect_uri, &config.oauth_client_secret_path)?;

    // Eagerly initialize the model registry in the background
    tokio::spawn(async {
        manager::get_model_registry().await;
    });

    let shared_state = Arc::new(AppState {
        queue_tx: tx,
        engine_status,
        telemetry,
        auth_store,
        reqwest_client,
        oauth_client,
        config: Arc::new(config.clone()),
        log_buffer: memory_buffer,
        log_reload_handle,
        current_log_level: Arc::new(Mutex::new(config.log_level_memory.clone())),
        db: db_client.clone(),
    });

    // Setup Session Layer
    let session_store = SurrealSessionStore::new(db_client.clone(), "sessions".to_string());
    let session_layer = SessionManagerLayer::new(session_store)
        .with_secure(config.secure_cookies)
        .with_same_site(tower_sessions::cookie::SameSite::Lax);

    // WEB & SETTINGS ROUTES
    // These handle their own session logic and redirects, so we don't apply the strict middleware here.
    let web_routes = setup::build_web_routes();

    // ENGINE API ROUTES
    // These get wrapped in our strict Dual-Auth Middleware layer.
    let engine_api_routes = setup::build_engine_api_routes(shared_state.clone());

    // MERGE & MOUNT
    // Combine them, inject the shared state, and apply the session layer globally
    let app = web_routes
        .merge(engine_api_routes)
        .with_state(shared_state)
        .layer(session_layer);

    // Start listening on port 3000
    let listener = tokio::net::TcpListener::bind(&config.bind_address)
        .await
        .unwrap();
    info!("🚀 Server safely listening on {}", config.bind_address);

    let (force_tx, force_rx) = tokio::sync::oneshot::channel();

    let server = axum::serve(listener, app).with_graceful_shutdown(shutdown_signal(force_tx));

    tokio::select! {
        res = server => { res?; }
        _ = force_rx => { error!("Forcing immediate shutdown..."); }
    }

    Ok(())
}

async fn shutdown_signal(force_tx: tokio::sync::oneshot::Sender<()>) {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("Failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("Failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
    info!("Received termination signal, starting graceful shutdown...");

    // Force shutdown if graceful shutdown takes longer than 30 seconds, or if a second signal is received
    tokio::spawn(async move {
        let ctrl_c = async {
            tokio::signal::ctrl_c()
                .await
                .expect("Failed to install Ctrl+C handler");
        };

        #[cfg(unix)]
        let terminate = async {
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                .expect("Failed to install SIGTERM handler")
                .recv()
                .await;
        };

        #[cfg(not(unix))]
        let terminate = std::future::pending::<()>();

        tokio::select! {
            _ = ctrl_c => { error!("Received second termination signal. Forcing immediate exit."); },
            _ = terminate => { error!("Received second termination signal. Forcing immediate exit."); },
            _ = tokio::time::sleep(std::time::Duration::from_secs(30)) => { error!("Graceful shutdown timed out after 30 seconds. Forcing exit."); },
        }
        let _ = force_tx.send(());
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use tracing_subscriber::fmt::MakeWriter;

    #[test]
    fn test_app_config_defaults() {
        let config = AppConfig::default();
        assert_eq!(config.bind_address, "127.0.0.1:3000");
        assert!(config.secure_cookies);
    }

    #[test]
    fn test_app_config_log_defaults() {
        assert_eq!(default_log_level_console(), "info");
        assert_eq!(default_log_level_file(), "warn");
        assert_eq!(default_log_level_memory(), "debug");
        assert_eq!(default_log_file_name(), "server.log");
    }

    #[test]
    fn test_shared_log_writer_circular_buffer() {
        let buffer = SharedLogBuffer(Arc::new(Mutex::new((0, std::collections::VecDeque::new()))));

        for i in 0..1010 {
            let mut writer = buffer.make_writer();
            use std::io::Write;
            let _ = writer.write(format!("Log line {}\n", i).as_bytes());
            // Dropping the writer triggers the flush to the shared buffer
        }

        let guard = buffer.0.lock().unwrap();
        assert_eq!(guard.0, 1010); // Verifies the total emitted cursor is intact
        assert_eq!(guard.1.len(), 1000); // Verifies the circular truncation works
        assert_eq!(guard.1.front().unwrap(), "Log line 10"); // Oldest retained
    }

    #[test]
    fn test_shared_log_writer_empty_drop() {
        let buffer = SharedLogBuffer(Arc::new(Mutex::new((0, std::collections::VecDeque::new()))));
        {
            let mut writer = buffer.make_writer();
            let _ = std::io::Write::write(&mut writer, b"   \n"); // Just whitespace
        }
        let guard = buffer.0.lock().unwrap();
        assert_eq!(guard.0, 0); // Should not increment cursor for empty/whitespace logs
        assert!(guard.1.is_empty());
    }

    #[test]
    fn test_log_query_deserialization() {
        let query: LogQuery = serde_json::from_str(r#"{}"#).unwrap();
        assert_eq!(query.since, None);

        let query: LogQuery = serde_json::from_str(r#"{"since": 42}"#).unwrap();
        assert_eq!(query.since, Some(42));
    }

    use axum::Json;
    use axum::extract::{Path, State};
    use std::collections::HashMap;

    async fn create_test_app_state() -> Arc<AppState> {
        let (queue_tx, _) = mpsc::channel(1);
        let (telemetry_tx, _) = mpsc::unbounded_channel();
        let (auth_tx, _) = tokio::sync::mpsc::unbounded_channel();

        let db = surrealdb::engine::any::connect("mem://").await.unwrap();
        db.use_ns("test").use_db("test").await.unwrap();

        // Ensure tables exist for testing purposes
        db.query("DEFINE TABLE chat_sessions SCHEMALESS;")
            .await
            .unwrap();
        db.query("DEFINE TABLE chat_messages SCHEMALESS;")
            .await
            .unwrap();
        db.query("DEFINE TABLE auth_keys SCHEMALESS;")
            .await
            .unwrap();
        db.query("DEFINE INDEX chat_sessions_email_idx ON TABLE chat_sessions COLUMNS email;")
            .await
            .unwrap();
        db.query("DEFINE INDEX chat_messages_session_idx ON TABLE chat_messages COLUMNS session_id, message_index;")
            .await
            .unwrap();
        db.query("DEFINE INDEX telemetry_loads_timestamp_idx ON TABLE telemetry_loads COLUMNS timestamp;")
            .await
            .unwrap();
        db.query("DEFINE INDEX telemetry_generations_timestamp_idx ON TABLE telemetry_generations COLUMNS timestamp;")
            .await
            .unwrap();

        let (_, log_reload_handle) =
            tracing_subscriber::reload::Layer::new(tracing_subscriber::EnvFilter::new("info"));

        Arc::new(AppState {
            queue_tx,
            engine_status: Arc::new(Mutex::new(EngineStatus::default())),
            telemetry: Arc::new(Mutex::new(TelemetryStore {
                loads: std::collections::VecDeque::new(),
                generations: std::collections::VecDeque::new(),
                writer_tx: Some(telemetry_tx),
            })),
            auth_store: Arc::new(Mutex::new(AuthStore {
                api_keys: HashMap::new(),
                key_index: HashMap::new(),
                writer_tx: Some(auth_tx),
            })),
            reqwest_client: reqwest::Client::new(),
            oauth_client: auth::build_oauth_client(
                "http://localhost:3000/auth/google/callback",
                "client_secret.apps.googleusercontent.com.json",
            )
            .unwrap(), // Dummy client
            config: Arc::new(AppConfig::default()),
            log_buffer: SharedLogBuffer(Arc::new(Mutex::new((
                0,
                std::collections::VecDeque::new(),
            )))),
            log_reload_handle,
            current_log_level: Arc::new(Mutex::new("info".to_string())),
            db,
        })
    }

    fn mock_user(email: &str, is_admin: bool) -> auth::CurrentUser {
        auth::CurrentUser {
            email: email.to_string(),
            is_admin,
        }
    }

    #[tokio::test]
    #[ignore = "Requires Oauth Token (Suite 2)"]
    async fn test_chat_session_lifecycle() {
        let state = create_test_app_state().await;
        let user_email = "test@example.com";
        let user = mock_user(user_email, false);

        // 1. Create a new session
        let new_session_payload = manager::ChatSessionRecord {
            id: "".to_string(),
            email: user_email.to_string(),
            updated_at: 0,
            title: "First Session".to_string(),
        };
        let response = save_chat_session(
            user.clone(),
            State(state.clone()),
            Json(new_session_payload),
        )
        .await;
        assert!(
            response.is_ok(),
            "Failed to save new session: {:?}",
            response.err()
        ); // Enhance assertion message
        let new_session = response.unwrap().0;
        assert!(!new_session.id.is_empty());
        assert_eq!(new_session.title, "First Session");
        let session_id = new_session.id.clone();

        // 2. List sessions for the user
        let sessions = list_chat_sessions(user.clone(), State(state.clone()))
            .await
            .unwrap()
            .0;
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].id, session_id);
        assert_eq!(sessions[0].title, "First Session");

        // 3. Get the session details (including messages, initially empty)
        let session_detail_res =
            get_chat_session(user.clone(), Path(session_id.clone()), State(state.clone())).await;
        assert!(
            session_detail_res.is_ok(),
            "Failed to get session details: {:?}",
            session_detail_res.err()
        );
        let session_detail = session_detail_res.unwrap().0;
        assert_eq!(session_detail.id, session_id);
        assert!(session_detail.messages.is_empty());

        // 4. Append a message
        let message_payload = manager::ChatMessageRecord {
            session_id: session_id.clone(),
            message_index: 0,
            role: "user".to_string(),
            content: "Hello AI".to_string(),
        };
        let append_res = append_chat_message(
            user.clone(),
            Path(session_id.clone()),
            State(state.clone()),
            Json(message_payload),
        )
        .await;
        assert_eq!(append_res, Ok(StatusCode::OK));

        // 5. Append another message
        let message_payload_2 = manager::ChatMessageRecord {
            session_id: session_id.clone(),
            message_index: 1,
            role: "assistant".to_string(),
            content: "Hello human".to_string(),
        };
        let append_res_2 = append_chat_message(
            user.clone(),
            Path(session_id.clone()),
            State(state.clone()),
            Json(message_payload_2),
        )
        .await;
        assert_eq!(append_res_2, Ok(StatusCode::OK));

        // 6. Get session again and verify messages
        let session_detail_with_msgs =
            get_chat_session(user.clone(), Path(session_id.clone()), State(state.clone()))
                .await
                .unwrap()
                .0;
        assert_eq!(session_detail_with_msgs.messages.len(), 2);
        assert_eq!(session_detail_with_msgs.messages[0].content, "Hello AI");
        assert_eq!(session_detail_with_msgs.messages[1].content, "Hello human");

        // 7. Truncate messages (e.g., regenerate from a point)
        let truncate_res = truncate_chat_messages(
            user.clone(),
            Path((session_id.clone(), 1)),
            State(state.clone()),
        )
        .await;
        assert_eq!(truncate_res, Ok(StatusCode::OK));

        // 8. Verify truncation
        let session_after_truncate =
            get_chat_session(user.clone(), Path(session_id.clone()), State(state.clone()))
                .await
                .unwrap()
                .0;
        assert_eq!(session_after_truncate.messages.len(), 1);
        assert_eq!(session_after_truncate.messages[0].content, "Hello AI");

        // 9. Attempt to access/modify another user's session (FORBIDDEN)
        let other_user = mock_user("other@example.com", false);
        let other_user_session_get_res = get_chat_session(
            other_user.clone(),
            Path(session_id.clone()),
            State(state.clone()),
        )
        .await;
        assert_eq!(
            other_user_session_get_res.unwrap_err(),
            StatusCode::FORBIDDEN
        );

        let other_user_session_update_payload = manager::ChatSessionRecord {
            id: session_id.clone(),
            email: other_user.email.clone(), // This should be ignored by the backend logic due to ID check
            updated_at: 0,
            title: "Malicious Update".to_string(),
        };
        let other_user_session_update_res = save_chat_session(
            other_user.clone(),   // This should be `other_user` not `user.clone()`
            State(state.clone()), // Use the cloned state
            Json(other_user_session_update_payload), // Use the payload for the other user
        )
        .await;
        assert_eq!(
            other_user_session_update_res.unwrap_err(),
            StatusCode::FORBIDDEN
        ); // Expect a FORBIDDEN status
    }

    #[tokio::test]
    #[ignore = "Requires Oauth Token (Suite 2)"]
    async fn test_chat_session_renaming() {
        let state = create_test_app_state().await;
        let user_email = "test@example.com";
        let user = mock_user(user_email, false);

        // 1. Create a new session
        let new_session_payload = manager::ChatSessionRecord {
            id: "".to_string(),
            email: user_email.to_string(),
            updated_at: 0,
            title: "Original Title".to_string(),
        };
        let response = save_chat_session(
            user.clone(),
            State(state.clone()),
            Json(new_session_payload),
        )
        .await;

        let new_session = response.unwrap().0;
        let session_id = new_session.id.clone();
        assert_eq!(new_session.title, "Original Title");

        // 2. Rename the session
        let rename_payload = manager::ChatSessionRecord {
            id: session_id.clone(),
            email: user_email.to_string(), // Keep same email
            updated_at: 0,
            title: "Renamed Title".to_string(),
        };
        let rename_response =
            save_chat_session(user.clone(), State(state.clone()), Json(rename_payload)).await;
        let renamed_session = rename_response.unwrap().0;

        assert_eq!(
            renamed_session.id, session_id,
            "The ID should remain the same after an update."
        );
        assert_eq!(
            renamed_session.title, "Renamed Title",
            "The title should be updated."
        );

        // 3. Verify the listing reflects the new title and didn't duplicate the database record
        let sessions = list_chat_sessions(user.clone(), State(state.clone()))
            .await
            .unwrap()
            .0;
        assert_eq!(sessions.len(), 1, "There should still only be one session.");
        assert_eq!(sessions[0].id, session_id);
        assert_eq!(sessions[0].title, "Renamed Title");
    }
}
