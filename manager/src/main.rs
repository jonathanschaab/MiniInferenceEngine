use auth::{AuthStore, require_session};
use axum::{
    Json, Router,
    body::Body,
    body::Bytes,
    extract::State,
    http::StatusCode,
    http::header,
    response::{Html, IntoResponse, Redirect},
    routing::{delete, get, post},
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
use tower_sessions::{MemoryStore, SessionManagerLayer};
use tracing::{error, info, warn}; // Ensure this is imported for AppState
use tracing_subscriber::{EnvFilter, Layer, layer::SubscriberExt, util::SubscriberInitExt};

use manager::{
    ApiRequest, BenchmarkRequest, EngineStatus, Message, ModelArch, ModelConfig, ModelRole,
    StreamEvent, TelemetryStore, UserRequest, get_model_registry, lock_status, run_batcher_loop,
};

// --- CONFIGURATION ---
#[derive(Serialize, Deserialize, Clone)]
pub struct AppConfig {
    pub bind_address: String,
    pub oauth_redirect_uri: String,
    pub admin_emails: Vec<String>,
    pub user_emails: Vec<String>,
    pub secure_cookies: bool,
    #[serde(default)]
    pub gpu_device_index: u32,
    #[serde(default = "default_log_level_console")]
    pub log_level_console: String,
    #[serde(default = "default_log_level_file")]
    pub log_level_file: String,
    #[serde(default = "default_log_level_memory")]
    pub log_level_memory: String,
    #[serde(default = "default_log_file_name")]
    pub log_file_name: String,
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

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            bind_address: "127.0.0.1:3000".to_string(), // Secure local default
            oauth_redirect_uri: "http://localhost:3000/auth/google/callback".to_string(),
            admin_emails: vec![],
            user_emails: vec![],
            secure_cookies: true,
            gpu_device_index: 0,
            log_level_console: default_log_level_console(),
            log_level_file: default_log_level_file(),
            log_level_memory: default_log_level_memory(),
            log_file_name: default_log_file_name(),
        }
    }
}

impl AppConfig {
    pub fn load() -> Self {
        if let Ok(data) = std::fs::read_to_string("config.json") {
            serde_json::from_str(&data).unwrap_or_default()
        } else {
            let config = Self::default();
            let _ = std::fs::write(
                "config.json",
                serde_json::to_string_pretty(&config).unwrap(),
            );
            config
        }
    }
}

pub mod auth;

// --- SHARED MEMORY LOG WRITER ---
#[derive(Clone)]
pub struct SharedLogBuffer(pub Arc<Mutex<std::collections::VecDeque<String>>>);

pub struct SharedLogWriter {
    buffer: Arc<Mutex<std::collections::VecDeque<String>>>,
}

impl std::io::Write for SharedLogWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let s = String::from_utf8_lossy(buf).into_owned();
        let mut logs = self.buffer.lock().unwrap();
        // Exclude the trailing newline added by the fmt layer since UI handles lines
        logs.push_back(s.trim_end().to_string());
        if logs.len() > 1000 {
            logs.pop_front();
        }
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for SharedLogBuffer {
    type Writer = SharedLogWriter;
    fn make_writer(&'a self) -> Self::Writer {
        SharedLogWriter {
            buffer: self.0.clone(),
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
}

async fn serve_ui(session: tower_sessions::Session) -> Result<Html<&'static str>, Redirect> {
    if require_session(session).await.is_err() {
        return Err(Redirect::to("/auth/login"));
    }
    Ok(Html(include_str!("../index.html")))
}

// Send the model roster to the Javascript dropdowns
async fn get_models() -> Json<Vec<ModelConfig>> {
    Json(get_model_registry().await)
}

// Handle incoming chat requests
async fn handle_generate(
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

async fn get_status(State(state): State<Arc<AppState>>) -> Json<EngineStatus> {
    let current_status = lock_status(&state.engine_status).clone();
    Json(current_status)
}

// The Automated Benchmark Trigger
async fn trigger_benchmark(
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
        params.seed = Some(rand::random::<u64>());
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
async fn serve_stats_ui(session: tower_sessions::Session) -> Result<Html<&'static str>, Redirect> {
    if require_session(session).await.is_err() {
        return Err(Redirect::to("/auth/login"));
    }
    Ok(Html(include_str!("../stats.html")))
}

async fn serve_settings_ui(
    session: tower_sessions::Session,
) -> Result<Html<&'static str>, Redirect> {
    if require_session(session).await.is_err() {
        return Err(Redirect::to("/auth/login"));
    }
    Ok(Html(include_str!("../settings.html")))
}

async fn serve_models_ui(session: tower_sessions::Session) -> Result<Html<&'static str>, Redirect> {
    if require_session(session).await.is_err() {
        return Err(Redirect::to("/auth/login"));
    }
    Ok(Html(include_str!("../models.html")))
}

async fn serve_memory_ui(session: tower_sessions::Session) -> Result<Html<&'static str>, Redirect> {
    if require_session(session).await.is_err() {
        return Err(Redirect::to("/auth/login"));
    }
    Ok(Html(include_str!("../memory.html")))
}

async fn serve_console_ui(
    session: tower_sessions::Session,
) -> Result<Html<&'static str>, Redirect> {
    if require_session(session).await.is_err() {
        return Err(Redirect::to("/auth/login"));
    }
    Ok(Html(include_str!("../console.html")))
}

async fn serve_console_js() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "application/javascript")],
        include_str!("../console.js"),
    )
}

async fn get_console_logs(
    State(state): State<Arc<AppState>>,
) -> Json<std::collections::VecDeque<String>> {
    let logs = state.log_buffer.0.lock().unwrap().clone();
    Json(logs)
}

async fn clear_console_logs(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    state.log_buffer.0.lock().unwrap().clear();
    StatusCode::OK
}

#[derive(Deserialize, Serialize)]
pub struct LogLevelRequest {
    pub level: String,
}

async fn set_console_loglevel(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<LogLevelRequest>,
) -> impl IntoResponse {
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
    *state.current_log_level.lock().unwrap() = payload.level.clone();
    (StatusCode::OK, "Log level updated").into_response()
}

async fn get_console_loglevel(State(state): State<Arc<AppState>>) -> Json<LogLevelRequest> {
    let level = state.current_log_level.lock().unwrap().clone();
    Json(LogLevelRequest { level })
}

async fn serve_chat_js() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "application/javascript")],
        include_str!("../chat.js"),
    )
}

async fn serve_stats_js() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "application/javascript")],
        include_str!("../stats.js"),
    )
}

async fn serve_models_js() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "application/javascript")],
        include_str!("../models.js"),
    )
}

async fn serve_settings_js() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "application/javascript")],
        include_str!("../settings.js"),
    )
}

async fn serve_memory_js() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "application/javascript")],
        include_str!("../memory.js"),
    )
}

async fn serve_common_js() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "application/javascript")],
        include_str!("../common.js"),
    )
}

async fn serve_common_css() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "text/css")],
        include_str!("../common.css"),
    )
}

// Route: Serve the Telemetry JSON
async fn get_stats_data(State(state): State<Arc<AppState>>) -> Json<TelemetryStore> {
    let current_data = state
        .telemetry
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone();
    Json(current_data)
}

#[tokio::main]
async fn main() {
    let config = AppConfig::load();

    // --- 1. CONSOLE LAYER ---
    let console_layer = tracing_subscriber::fmt::layer()
        .with_writer(std::io::stdout)
        .with_filter(EnvFilter::new(&config.log_level_console));

    // --- 2. FILE LAYER ---
    let file_appender = tracing_appender::rolling::never(".", &config.log_file_name);
    // Bind the _file_guard to keep the background writer active for the life of main()
    let (file_writer, _file_guard) = tracing_appender::non_blocking(file_appender);
    let file_layer = tracing_subscriber::fmt::layer()
        .with_writer(file_writer)
        .with_ansi(false) // Do not write ANSI color codes to the log file!
        .with_filter(EnvFilter::new(&config.log_level_file));

    // --- 3. IN-MEMORY BUFFER LAYER (RELOADABLE) ---
    let memory_buffer = SharedLogBuffer(Arc::new(Mutex::new(std::collections::VecDeque::new())));
    let memory_filter = EnvFilter::new(&config.log_level_memory);
    let (reloadable_memory_filter, log_reload_handle) =
        tracing_subscriber::reload::Layer::new(memory_filter);
    let memory_layer = tracing_subscriber::fmt::layer()
        .with_writer(memory_buffer.clone())
        .with_ansi(false) // Send clean strings to the web UI
        .with_filter(reloadable_memory_filter);

    // Apply all registered layers
    tracing_subscriber::registry()
        .with(memory_layer)
        .with(file_layer)
        .with(console_layer)
        .init();

    // Create the async channel for the GPU queue
    let (tx, rx) = mpsc::channel(32);

    // Initialize the shared state BEFORE spawning the background thread
    let engine_status = Arc::new(Mutex::new(EngineStatus::default()));

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

    let (telemetry_tx, mut telemetry_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    tokio::spawn(async move {
        // This loop processes writes one-at-a-time, perfectly preventing race conditions
        while let Some(json) = telemetry_rx.recv().await {
            let temp_file = "stats.json.tmp";
            if tokio::fs::write(temp_file, &json).await.is_ok() {
                let _ = tokio::fs::rename(temp_file, "stats.json").await;
            }
        }
    });

    let mut store = TelemetryStore::load_from_disk();
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

    let (auth_tx, mut auth_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    tokio::spawn(async move {
        // This loop processes writes one-at-a-time, enforcing strict order and atomic replacement
        while let Some(json) = auth_rx.recv().await {
            let temp_file = "api_keys.json.tmp";
            if tokio::fs::write(temp_file, &json).await.is_ok() {
                let _ = tokio::fs::rename(temp_file, "api_keys.json").await;
            }
        }
    });

    let mut store = AuthStore::load();
    store.writer_tx = Some(auth_tx);
    let auth_store = Arc::new(Mutex::new(store));

    // Initialize global pooled clients once!
    let reqwest_client = reqwest::Client::new();
    let oauth_client = auth::build_oauth_client(&config.oauth_redirect_uri);

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
    });

    // Setup Session Layer
    let session_store = MemoryStore::default();
    let session_layer = SessionManagerLayer::new(session_store)
        .with_secure(config.secure_cookies)
        .with_same_site(tower_sessions::cookie::SameSite::Lax);

    // WEB & SETTINGS ROUTES
    // These handle their own session logic and redirects, so we don't apply the strict middleware here.
    let web_routes = Router::new()
        // Public OAuth
        .route("/auth/login", get(auth::login_handler))
        .route("/auth/google/callback", get(auth::callback_handler))
        .route("/auth/logout", get(auth::logout_handler))
        // Protected UIs (They redirect if session is missing)
        .route("/", get(serve_ui))
        .route("/settings", get(serve_settings_ui))
        .route("/models", get(serve_models_ui))
        .route("/stats", get(serve_stats_ui))
        .route("/memory", get(serve_memory_ui))
        .route("/console", get(serve_console_ui))
        .route("/js/chat.js", get(serve_chat_js))
        .route("/js/models.js", get(serve_models_js))
        .route("/js/stats.js", get(serve_stats_js))
        .route("/js/settings.js", get(serve_settings_js))
        .route("/js/memory.js", get(serve_memory_js))
        .route("/js/console.js", get(serve_console_js))
        .route("/js/common.js", get(serve_common_js))
        .route("/css/common.css", get(serve_common_css))
        // Settings APIs (They check session manually)
        .route(
            "/api/settings/keys",
            get(auth::list_keys_handler).post(auth::create_key_handler),
        )
        .route("/api/settings/keys/:hash", delete(auth::delete_key_handler));

    // ENGINE API ROUTES
    // These get wrapped in our strict Dual-Auth Middleware layer.
    let engine_api_routes = Router::new()
        .route("/api/generate", post(handle_generate))
        .route("/api/stats/collect", post(trigger_benchmark))
        .route("/api/models", get(get_models))
        .route("/api/status", get(get_status))
        .route("/api/stats/data", get(get_stats_data))
        .route(
            "/api/console/logs",
            get(get_console_logs).delete(clear_console_logs),
        )
        .route(
            "/api/console/loglevel",
            get(get_console_loglevel).post(set_console_loglevel),
        )
        .route_layer(axum::middleware::from_fn_with_state(
            shared_state.clone(),
            auth::dual_auth_middleware,
        ));

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
    axum::serve(listener, app).await.unwrap();
}
