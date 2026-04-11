use axum::{
    extract::State,
    response::{Html, IntoResponse},
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use std::sync::{Arc, Mutex};
use tokio::sync::{mpsc, oneshot};

use manager::{
    get_model_registry, run_batcher_loop, ApiRequest, ApiResponse, ModelConfig, UserRequest, EngineStatus, lock_status, TelemetryStore, Message, ModelRole
};

// State to share the transmitter queue across web requests
pub struct AppState {
    pub queue_tx: mpsc::Sender<UserRequest>,
    pub engine_status: Arc<Mutex<EngineStatus>>,
    pub telemetry: Arc<Mutex<TelemetryStore>>,
}

// Route 1: Serve the HTML UI
async fn serve_ui() -> Html<&'static str> {
    Html(include_str!("../index.html"))
}

// Route 2: Send the model roster to the Javascript dropdowns
async fn get_models() -> Json<Vec<ModelConfig>> {
    Json(get_model_registry())
}

// Route 3: Handle incoming chat requests
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

// Route 5: The Automated Benchmark Trigger
async fn trigger_benchmark(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let queue_tx = state.queue_tx.clone();
    
    tokio::spawn(async move {
        println!("🚀 Starting Automated Benchmark Suite...");
        let registry = get_model_registry();
        
        let default_compressor = registry.iter()
            .find(|m| m.roles.contains(&ModelRole::ContextCompressor))
            .map(|m| m.id.clone())
            .unwrap_or_else(|| "llmlingua-2-f16".to_string());

        let default_chat = registry.iter()
            .find(|m| m.roles.contains(&ModelRole::GeneralChat))
            .map(|m| m.id.clone())
            .unwrap_or_else(|| "qwen-2.5-7b".to_string());

        let _ = std::fs::create_dir_all("benchmark_prompts");
        let base_word = "system "; 

        // --- GENERATIVE MODELS ---
        for model in registry.iter().filter(|m| m.roles.contains(&ModelRole::GeneralChat) || m.roles.contains(&ModelRole::CodeSpecialist)) {
            
            let mut test_sizes = vec![1, 10, 100, 1000, 10000];
            let safe_max = (model.max_context_len as f32 * 0.95) as usize;
            
            test_sizes.retain(|&s| s < safe_max);
            if safe_max > 0 { test_sizes.push(safe_max); }

            for size in test_sizes {
                let filename = format!("benchmark_prompts/prompt_{}.txt", size);

                if !std::path::Path::new(&filename).exists() {
                    std::fs::write(&filename, base_word.repeat(size).trim()).unwrap();
                }

                let exact_prompt = std::fs::read_to_string(&filename).unwrap_or_else(|_| "Error".to_string());
                println!("📊 Benchmarking Generative {} using file {}...", model.name, filename);
                
                let (response_tx, response_rx) = oneshot::channel();
                let request = UserRequest {
                    chat_model_id: model.id.clone(),
                    compressor_model_id: default_compressor.clone(),
                    messages: vec![Message { role: "user".to_string(), content: exact_prompt }],
                    responder: response_tx,
                    force_compression: false,
                };

                let _ = queue_tx.send(request).await;
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

                if !std::path::Path::new(&filename).exists() {
                    std::fs::write(&filename, base_word.repeat(size).trim()).unwrap();
                }

                let exact_prompt = std::fs::read_to_string(&filename).unwrap_or_else(|_| "Error".to_string());
                println!("📊 Benchmarking Compressor {} using file {}...", comp_model.name, filename);
                
                let (response_tx, response_rx) = oneshot::channel();
                let request = UserRequest {
                    chat_model_id: default_chat.clone(), // Use a dummy chat model to satisfy the struct
                    compressor_model_id: comp_model.id.clone(),
                    messages: vec![Message { role: "user".to_string(), content: exact_prompt }],
                    responder: response_tx,
                    force_compression: true, // Bypass memory thresholds and force the compressor to run!
                };

                let _ = queue_tx.send(request).await;
                let _ = response_rx.await;
            }
        }

        println!("✅ Automated Benchmark Suite Complete!");
    });

    (StatusCode::ACCEPTED, "Benchmark sweep started in the background.")
}

// Route: Serve the Stats UI
async fn serve_stats_ui() -> Html<&'static str> {
    Html(include_str!("../stats.html"))
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

    // 2. Initialize the shared state BEFORE spawning the background thread
    let engine_status = Arc::new(Mutex::new(EngineStatus::default()));
    let status_for_batcher = engine_status.clone();
    let telemetry = Arc::new(Mutex::new(TelemetryStore::load_from_disk()));
    let telemetry_for_batcher = telemetry.clone();

    // Boot up the GPU Orchestrator in the background
    tokio::spawn(async move {
        run_batcher_loop(rx, status_for_batcher, telemetry_for_batcher).await;
    });

    let shared_state = Arc::new(AppState { 
        queue_tx: tx,
        engine_status,
        telemetry: telemetry.clone(),
    });

    // Build the Axum web server routes
    let app = Router::new()
        .route("/", get(serve_ui))
        .route("/api/models", get(get_models))
        .route("/api/generate", post(handle_generate))
        .route("/api/status", get(get_status))
        .route("/api/stats/collect", post(trigger_benchmark))
        .route("/api/stats/data", get(get_stats_data))
        .route("/stats", get(serve_stats_ui))
        .with_state(shared_state);

    // Start listening on port 3000
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    println!("🚀 Server running on http://0.0.0.0:3000");
    axum::serve(listener, app).await.unwrap();
}
