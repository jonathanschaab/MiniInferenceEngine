use axum::{
    extract::State,
    response::{Html, IntoResponse, Redirect},
    http::StatusCode,
    routing::{get, post, delete},
    Json, Router,
};
use std::sync::{Arc, Mutex};
use tokio::sync::{mpsc, oneshot};
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;
use tower_sessions::{MemoryStore, SessionManagerLayer};
use auth::{AuthStore, require_session};
use serde::{Deserialize, Serialize};
use oauth2::basic::BasicClient; // Ensure this is imported for AppState

use manager::{
    get_model_registry, run_batcher_loop, ApiRequest, ApiResponse, ModelConfig, UserRequest, EngineStatus, lock_status, TelemetryStore, Message, ModelRole, BenchmarkRequest
};

// --- CONFIGURATION ---
#[derive(Serialize, Deserialize, Clone)]
pub struct AppConfig {
    pub bind_address: String,
    pub oauth_redirect_uri: String,
    pub admin_emails: Vec<String>,
    pub user_emails: Vec<String>,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            bind_address: "127.0.0.1:3000".to_string(), // Secure local default
            oauth_redirect_uri: "http://localhost:3000/auth/google/callback".to_string(),
            admin_emails: vec![],
            user_emails: vec![],
        }
    }
}

impl AppConfig {
    pub fn load() -> Self {
        if let Ok(data) = std::fs::read_to_string("config.json") {
            serde_json::from_str(&data).unwrap_or_default()
        } else {
            let config = Self::default();
            let _ = std::fs::write("config.json", serde_json::to_string_pretty(&config).unwrap());
            config
        }
    }
}

pub mod auth;

// State to share the transmitter queue across web requests
pub struct AppState {
    pub queue_tx: mpsc::Sender<UserRequest>,
    pub engine_status: Arc<Mutex<EngineStatus>>,
    pub telemetry: Arc<Mutex<TelemetryStore>>,
    pub auth_store: Arc<Mutex<AuthStore>>,
    pub reqwest_client: reqwest::Client,
    pub oauth_client: BasicClient,
    pub config: Arc<AppConfig>,
}

async fn serve_ui(session: tower_sessions::Session) -> Result<Html<&'static str>, Redirect> {
    if require_session(session).await.is_err() {
        return Err(Redirect::to("/auth/login"));
    }
    Ok(Html(include_str!("../index.html")))
}

// Send the model roster to the Javascript dropdowns
async fn get_models() -> Json<Vec<ModelConfig>> {
    Json(get_model_registry())
}

// Handle incoming chat requests
async fn handle_generate(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<ApiRequest>,
) -> Json<ApiResponse> {
    
    let (response_tx, response_rx) = oneshot::channel();

    // Package the UI's model choices and the chat history
    let request = UserRequest {
        chat_model_id: payload.chat_model_id,
        compressor_model_id: payload.compressor_model_id,
        messages: payload.messages,
        responder: response_tx,
        force_compression: false,
    };

    // Send to the GPU thread
    let _ = state.queue_tx.send(request).await;

    // Wait for the LLM to finish generating
    let generated_text = response_rx.await.unwrap_or_else(|_| "Error: GPU disconnected".to_string());
    
    Json(ApiResponse { answer: generated_text })
}

async fn get_status(State(state): State<Arc<AppState>>) -> Json<EngineStatus> {
    let current_status = lock_status(&state.engine_status).clone();
    Json(current_status)
}

// The Automated Benchmark Trigger
async fn trigger_benchmark(
    State(state): State<Arc<AppState>>,
    user: auth::CurrentUser,
    Json(payload): Json<BenchmarkRequest>
) -> impl IntoResponse {

    if !user.is_admin {
        println!("⚠️ Benchmark trigger rejected for non-admin: {}", user.email);
        return (StatusCode::FORBIDDEN, "Only administrators can run the benchmark suite.").into_response();
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

    let queue_tx = state.queue_tx.clone();
    let engine_status = state.engine_status.clone(); // Clone the Arc so the background thread can reset it
    let selected_models = payload.models;
    
    tokio::spawn(async move {
        println!("🚀 Starting Automated Benchmark Suite...");
        let full_registry = get_model_registry();
        
        let default_compressor = full_registry.iter()
            .find(|m| m.roles.contains(&ModelRole::ContextCompressor))
            .map(|m| m.id.clone())
            .unwrap_or_else(|| "llmlingua-2-f16".to_string());

        let default_chat = full_registry.iter()
            .find(|m| m.roles.contains(&ModelRole::GeneralChat))
            .map(|m| m.id.clone())
            .unwrap_or_else(|| "qwen-2.5-7b".to_string());

        let registry: Vec<ModelConfig> = full_registry.into_iter()
            .filter(|m| selected_models.contains(&m.id))
            .collect();

        if registry.is_empty() {
            println!("⚠️ Benchmark aborted: No valid models selected.");
            let mut status = lock_status(&engine_status);
            status.benchmark_running = false;
            return;
        }

        let _ = tokio::fs::create_dir_all("benchmark_prompts").await;

        // --- GENERATE REALISTIC PROMPTS ---
        println!("🌱 Verifying benchmark prompt files...");

        let tokenizer_result = tokio::task::spawn_blocking(|| {
            let api = Api::new().map_err(|e| format!("API Init Error: {}", e))?;
            let path = api.model("Qwen/Qwen2.5-1.5B-Instruct".to_string())
                .get("tokenizer.json")
                .map_err(|e| format!("Tokenizer Download Error: {}", e))?;
            
            Tokenizer::from_file(path).map_err(|e| format!("Tokenizer Parse Error: {}", e))
        }).await;
        
        let mut qwen_tokenizer = None;
        
        match tokenizer_result {
            Ok(Ok(tokenizer)) => {
                qwen_tokenizer = Some(tokenizer);
            }
            Ok(Err(e)) => {
                println!("⚠️ Tokenizer initialization failed: {}. Falling back to padding.", e);
            }
            Err(e) => {
                // If the spawn_blocking task actually panics, it is caught here as a JoinError
                println!("⚠️ Thread execution failed: {}. Falling back to padding.", e);
            }
        }
        
        let mut all_sizes = std::collections::HashSet::new();
        for model in registry.iter() {
            let mut sizes = vec![1, 10, 100, 1000, 10000];
            let safe_max = (model.max_context_len as f32 * 0.95) as usize;
            sizes.retain(|&s| s < safe_max);
            if safe_max > 0 { sizes.push(safe_max); }
            for s in sizes { all_sizes.insert(s); }
        }

        let mut sorted_sizes: Vec<usize> = all_sizes.into_iter().collect();
        sorted_sizes.sort();

        for size in sorted_sizes {
            let filename = format!("benchmark_prompts/prompt_{}.txt", size);
            
            if !std::path::Path::new(&filename).exists() {
                let mut should_save = false;
                let mut final_content = String::new();

                if size < 1000 {
                    println!("🧠 Using Qwen 1.5B to generate realistic prompt of ~{} tokens...", size);
                    let prompt_instruction = if size <= 50 {
                        format!("Write a very short technical sentence about Rust. Limit to {} words.", size)
                    } else {
                        format!("Write a detailed paragraph about Rust memory management. Target exactly {} words.", size)
                    };

                    let (seed_tx, seed_rx) = oneshot::channel();
                    let _ = queue_tx.send(UserRequest {
                        chat_model_id: "qwen-compressor".to_string(), 
                        compressor_model_id: default_compressor.clone(),
                        messages: vec![Message { role: "user".to_string(), content: prompt_instruction }],
                        responder: seed_tx,
                        force_compression: false,
                    }).await;

                    if let Ok(generated_seed) = seed_rx.await {
                        if !generated_seed.starts_with("Server Error") {
                            if let Some(tokenizer) = &qwen_tokenizer {
                                if let Ok(encoding) = tokenizer.encode(generated_seed.clone(), true) {
                                    let token_len = encoding.get_ids().len();
                                    let margin = (size as f32 * 0.35) as usize;
                                    
                                    if token_len >= size.saturating_sub(margin) && token_len <= size + margin {
                                        final_content = generated_seed;
                                        should_save = true;
                                        println!("✅ Generated {} tokens (within 35% of {}). Saving.", token_len, size);
                                    } else {
                                        println!("⚠️ Generation missed margin: {} tokens (target {}).", token_len, size);
                                    }
                                }
                            }
                        }
                    }
                } else {
                    // --- ALGORITHMIC SYNTHESIS FOR LARGE CONTEXT ---
                    println!("🖨️ Synthesizing highly unique {} token data payload...", size);
                    let mut synthetic_data = String::with_capacity(size * 4);
                    synthetic_data.push_str("=== MULTI-USER SYSTEM DIAGNOSTIC LOG EXPORT ===\n");
                    
                    let events = ["PLAYER_CONNECT", "ROOM_TRANSITION", "COMBAT_ACTION", "TELNET_NEGOTIATION", "INVENTORY_SYNC"];
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
                                counter % 60, session_id, ev, st, (counter * 7) % 300, (counter * 13) % 2048
                            );
                            
                            synthetic_data.push_str(&log_line);
                            chunk_buffer.push_str(&log_line);
                            
                            if counter % 50 == 0 {
                                if let Ok(enc) = tokenizer.encode(chunk_buffer.as_str(), false) {
                                    current_tokens += enc.get_ids().len();
                                }
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
                                println!("✅ Successfully synthesized exactly {} unique tokens.", safe_size);
                            } else {
                                println!("⚠️ Decode failed. Falling back to padding.");
                            }
                        } else {
                            println!("⚠️ Encode failed. Falling back to padding.");
                        }
                    }
                }

                if !should_save {
                    println!("⚠️ Fallback: Using generic memory padding. (Will NOT save to disk).");
                } else {
                    let _ = tokio::fs::write(&filename, &final_content).await;
                }
            }
        }
        
        println!("✅ Setup Phase Complete. Beginning standard benchmark sweeps...");

        // --- GENERATIVE MODELS ---
        for model in registry.iter().filter(|m| m.roles.contains(&ModelRole::GeneralChat) || m.roles.contains(&ModelRole::CodeSpecialist)) {
            let mut test_sizes = vec![1, 10, 100, 1000, 10000];
            let safe_max = (model.max_context_len as f32 * 0.95) as usize;
            test_sizes.retain(|&s| s < safe_max);
            if safe_max > 0 { test_sizes.push(safe_max); }

            for size in test_sizes {
                let filename = format!("benchmark_prompts/prompt_{}.txt", size);
                let exact_prompt = tokio::fs::read_to_string(&filename).await.unwrap_or_else(|_| "system ".repeat(size).trim().to_string());
                
                println!("📊 Benchmarking Generative {} using file {}...", model.name, filename);
                
                let (response_tx, response_rx) = oneshot::channel();
                let _ = queue_tx.send(UserRequest {
                    chat_model_id: model.id.clone(),
                    compressor_model_id: default_compressor.clone(),
                    messages: vec![Message { role: "user".to_string(), content: exact_prompt }],
                    responder: response_tx,
                    force_compression: false,
                }).await;
                let _ = response_rx.await;
            }
        }

        // --- COMPRESSOR MODELS ---
        println!("🚀 Transitioning to Compressor Benchmarks...");
        for comp_model in registry.iter().filter(|m| m.roles.contains(&ModelRole::ContextCompressor)) {
            let mut test_sizes = vec![1, 10, 100, 1000, 10000];
            let safe_max = (comp_model.max_context_len as f32 * 0.95) as usize;
            test_sizes.retain(|&s| s < safe_max);
            if safe_max > 0 { test_sizes.push(safe_max); }

            for size in test_sizes {
                let filename = format!("benchmark_prompts/prompt_{}.txt", size);
                let exact_prompt = tokio::fs::read_to_string(&filename).await.unwrap_or_else(|_| "system ".repeat(size).trim().to_string());
                
                println!("📊 Benchmarking Compressor {} using file {}...", comp_model.name, filename);
                
                let (response_tx, response_rx) = oneshot::channel();
                let _ = queue_tx.send(UserRequest {
                    chat_model_id: default_chat.clone(), 
                    compressor_model_id: comp_model.id.clone(),
                    messages: vec![Message { role: "user".to_string(), content: exact_prompt }],
                    responder: response_tx,
                    force_compression: true, 
                }).await;
                let _ = response_rx.await;
            }
        }

        println!("✅ Automated Benchmark Suite Complete!");

        // Reset the flag when the thread finishes
        {
            let mut status = lock_status(&engine_status);
            status.benchmark_running = false;
        }
    });

    (StatusCode::ACCEPTED, "Benchmark sweep started in the background.").into_response()
}

// Route: Serve the Stats UI
async fn serve_stats_ui() -> Html<&'static str> {
    Html(include_str!("../stats.html"))
}

async fn serve_settings_ui(session: tower_sessions::Session) -> Result<Html<&'static str>, Redirect> {
    if require_session(session).await.is_err() {
        return Err(Redirect::to("/auth/login"));
    }
    Ok(Html(include_str!("../settings.html")))
}

// Route: Serve the Telemetry JSON
async fn get_stats_data(State(state): State<Arc<AppState>>) -> Json<TelemetryStore> {
    let current_data = state.telemetry.lock().unwrap().clone();
    Json(current_data)
}

#[tokio::main]
async fn main() {
    // Create the async channel for the GPU queue
    let (tx, rx) = mpsc::channel(32);

    // Initialize the shared state BEFORE spawning the background thread
    let engine_status = Arc::new(Mutex::new(EngineStatus::default()));
    let status_for_batcher = engine_status.clone();

    let (telemetry_tx, mut telemetry_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    tokio::spawn(async move {
        // This loop processes writes one-at-a-time, perfectly preventing race conditions
        while let Some(json) = telemetry_rx.recv().await {
            let _ = tokio::fs::write("stats.json", json).await;
        }
    });

    let mut store = TelemetryStore::load_from_disk();
    store.writer_tx = Some(telemetry_tx); // Wire the channel into the store
    let telemetry = Arc::new(Mutex::new(store));

    let telemetry_for_batcher = telemetry.clone();

    // Boot up the GPU Orchestrator in the background
    tokio::spawn(async move {
        run_batcher_loop(rx, status_for_batcher, telemetry_for_batcher).await;
    });

    let config = AppConfig::load();
    let auth_store = Arc::new(Mutex::new(AuthStore::load()));

    // Initialize global pooled clients once!
    let reqwest_client = reqwest::Client::new();
    let oauth_client = auth::build_oauth_client(&config.oauth_redirect_uri);

    let shared_state = Arc::new(AppState { 
        queue_tx: tx,
        engine_status,
        telemetry: telemetry,
        auth_store,
        reqwest_client,
        oauth_client,
        config: Arc::new(config.clone()),
    });

    // Setup Session Layer
    let session_store = MemoryStore::default();
    let session_layer = SessionManagerLayer::new(session_store)
        .with_secure(false); // Set to true if using HTTPS in prod

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
        .route("/stats", get(serve_stats_ui))
        // Settings APIs (They check session manually)
        .route("/api/settings/keys", get(auth::list_keys_handler).post(auth::create_key_handler))
        .route("/api/settings/keys/:hash", delete(auth::delete_key_handler));

    // ENGINE API ROUTES
    // These get wrapped in our strict Dual-Auth Middleware layer.
    let engine_api_routes = Router::new()
        .route("/api/generate", post(handle_generate))
        .route("/api/stats/collect", post(trigger_benchmark))
        .route("/api/models", get(get_models))
        .route("/api/status", get(get_status))
        .route("/api/stats/data", get(get_stats_data))
        .route_layer(axum::middleware::from_fn_with_state(shared_state.clone(), auth::dual_auth_middleware));

    // MERGE & MOUNT
    // Combine them, inject the shared state, and apply the session layer globally
    let app = web_routes
        .merge(engine_api_routes)
        .with_state(shared_state)
        .layer(session_layer);

    // Start listening on port 3000
    let listener = tokio::net::TcpListener::bind(&config.bind_address).await.unwrap();
    println!("🚀 Server safely listening on {}", config.bind_address);
    axum::serve(listener, app).await.unwrap();
}
