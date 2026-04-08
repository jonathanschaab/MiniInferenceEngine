use axum::{
    extract::State,
    response::Html,
    routing::{get, post},
    Json, Router,
};
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};

// Import exactly what we need from our library (lib.rs)
use manager::{
    get_model_registry, run_batcher_loop, ApiRequest, ApiResponse, ModelConfig, UserRequest
};

// State to share the transmitter queue across web requests
pub struct AppState {
    pub queue_tx: mpsc::Sender<UserRequest>,
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
    };

    // Send to the GPU thread
    let _ = state.queue_tx.send(request).await;

    // Wait for the LLM to finish generating
    let generated_text = response_rx.await.unwrap_or_else(|_| "Error: GPU disconnected".to_string());
    
    Json(ApiResponse { answer: generated_text })
}

#[tokio::main]
async fn main() {
    // Create the async channel for the GPU queue
    let (tx, rx) = mpsc::channel(32);

    // Boot up the GPU Orchestrator in the background
    tokio::spawn(async move {
        run_batcher_loop(rx).await;
    });

    let shared_state = Arc::new(AppState { queue_tx: tx });

    // Build the Axum web server routes
    let app = Router::new()
        .route("/", get(serve_ui))
        .route("/models", get(get_models))
        .route("/generate", post(handle_generate))
        .with_state(shared_state);

    // Start listening on port 3000
    let listener = tokio::net::TcpListener::bind("127.0.0.1:3000").await.unwrap();
    println!("🚀 Server running on http://127.0.0.1:3000");
    axum::serve(listener, app).await.unwrap();
}
