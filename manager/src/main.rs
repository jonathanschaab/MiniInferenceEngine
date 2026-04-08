use axum::{
    extract::State,
    routing::post,
    Json, Router,
};
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::sync::{mpsc, oneshot};

// Import our core engine logic from lib.rs
use manager::{run_batcher_loop, ApiRequest, ApiResponse, UserRequest};

// 1. The Global State
// We wrap our MPSC sender in an Arc (Atomic Reference Counted pointer) 
// so multiple web threads can safely share the same queue access.
struct AppState {
    queue_tx: mpsc::Sender<UserRequest>,
}

#[tokio::main]
async fn main() {
    // 2. Initialize our Async Queue
    let (queue_tx, queue_rx) = mpsc::channel::<UserRequest>(100);

    // 3. Boot up the GPU Batcher in a background thread
    tokio::spawn(async move {
        run_batcher_loop(queue_rx).await;
    });

    // 4. Create the shared state for the web framework
    let shared_state = Arc::new(AppState { queue_tx });

    // 5. Build the Axum API Router
    let app = Router::new()
        .route("/generate", post(handle_generate))
        .with_state(shared_state);

    // 6. Start listening for traffic on port 3000!
    let listener = TcpListener::bind("127.0.0.1:3000").await.unwrap();
    println!("🚀 Server running on http://127.0.0.1:3000");
    axum::serve(listener, app).await.unwrap();
}

// --- THE WEB HANDLER (The Producer) ---

async fn handle_generate(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<ApiRequest>,
) -> Json<ApiResponse> {
    
    let (response_tx, response_rx) = oneshot::channel();

    let request = UserRequest {
        prompt: payload.prompt,
        responder: response_tx,
    };

    let _ = state.queue_tx.send(request).await;

    // If the GPU thread crashes or drops the pipe, we return a fallback String.
    let generated_text = response_rx.await.unwrap_or_else(|_| "Error: GPU pipeline disconnected".to_string());

    Json(ApiResponse { answer: generated_text })
}
