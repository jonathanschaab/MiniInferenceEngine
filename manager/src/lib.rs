use tokio::sync::{mpsc, oneshot};
use serde::{Deserialize, Serialize};

// What the user sends us via HTTP POST: {"prompt": "Hello GPU"}
#[derive(Deserialize)]
pub struct ApiRequest {
    pub prompt: String,
}

// What we send back: {"answer": [4.0, 4.0, 10.0, 8.0]}
#[derive(Serialize)]
pub struct ApiResponse {
    pub answer: Vec<f32>,
}

// This is what sits inside our MPSC queue
pub struct UserRequest {
    // 1. The payload
    pub prompt: String, 
    
    // 2. The return pipe 
    // It specifically expects to send a Vec<f32> (our math answer) back
    pub responder: oneshot::Sender<Vec<f32>>, 
}

// The C++ function signature we want to call.
// The "C" tells Rust to use the standard C-ABI (Application Binary Interface).
unsafe extern "C" {
    fn run_forward_pass(
        host_X: *const f32, 
        host_W: *const f32, 
        host_Y: *mut f32, 
        N: i32
    );
}

// A Rust wrapper around the unsafe C++ call
pub fn execute_gpu_math(batch_matrix: Vec<f32>, weight_matrix: Vec<f32>, matrix_size: i32) -> Vec<f32> {
    // A. Create an empty Vector of zeroes to hold the output answers from the GPU
    let total_elements = (matrix_size * matrix_size) as usize;
    let mut output_matrix = vec![0.0; total_elements];

    // B. The Unsafe Boundary
    // We are trusting the C++ DLL not to crash or access bad memory.
    unsafe {
        run_forward_pass(
            batch_matrix.as_ptr(),       // *const f32 (Read-only pointer)
            weight_matrix.as_ptr(),      // *const f32 (Read-only pointer)
            output_matrix.as_mut_ptr(),  // *mut f32   (Mutable pointer for the answer)
            matrix_size
        );
    }

    // C. Return the populated answers back to Rust
    output_matrix
}

// The Infinite Background Loop (The Consumer)
pub async fn run_batcher_loop(mut receiver: mpsc::Receiver<UserRequest>) {
    // A. "Load the Model Weights" into RAM. 
    // In a real LLM, this is 80GB of weights. For now, it's our 2x2 test matrix
    let weight_matrix = vec![2.0, 0.0, 1.0, 2.0];
    let matrix_size = 2; // 2x2

    println!("⚙️  GPU Batcher Thread started. Waiting for requests...");

    // B. The Infinite Loop
    // .recv().await will peacefully put this thread to sleep until a request arrives
    while let Some(request) = receiver.recv().await {
        println!("📥 Batcher grabbed request from queue: '{}'", request.prompt);

        // C. Simulate Tokenization / Padding
        // We will fake tokenizing the prompt and just use our 2x2 test input
        let batch_matrix = vec![1.0, 2.0, 3.0, 4.0];

        // D. Cross the FFI Bridge. Run the math on the GPU.
        // Note: In production, we'd use a thread-pool here so we don't block the async runtime, 
        // but for our mini-engine, this is perfect.
        let answer = execute_gpu_math(batch_matrix, weight_matrix.clone(), matrix_size);

        // E. Send the answer back to the specific user.
        // We use if/err because the user might have closed their web browser 
        // while the GPU was thinking!
        if request.responder.send(answer).is_err() {
            eprintln!("⚠️  Warning: User dropped connection before receiving answer.");
        }
    }
    
    println!("🛑 Queue closed. Batcher shutting down.");
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::{mpsc, oneshot};

    #[test]
    fn test_gpu_math_bridge() {
        // 1. Define our 2x2 test matrices
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let w = vec![2.0, 0.0, 1.0, 2.0];
        let matrix_size = 2;

        // 2. Call our Rust wrapper (which triggers the C++ DLL over FFI)
        let y = execute_gpu_math(x, w, matrix_size);

        // 3. Assert the results match our hand-calculated math exactly
        assert_eq!(y, vec![4.0, 4.0, 10.0, 8.0]);
    }

    #[tokio::test]
    async fn test_async_pipeline() {
        // 1. Create the giant MPSC queue (let's say it can hold 100 pending requests)
        let (queue_tx, queue_rx) = mpsc::channel::<UserRequest>(100);

        // 2. Spawn our Batcher loop in the background
        tokio::spawn(async move {
            run_batcher_loop(queue_rx).await;
        });

        // 3. SIMULATE USER 1
        let (user1_tx, user1_rx) = oneshot::channel();
        let req1 = UserRequest {
            prompt: String::from("What is the capital of France?"),
            responder: user1_tx,
        };
        queue_tx.send(req1).await.unwrap();

        // 4. SIMULATE USER 2
        let (user2_tx, user2_rx) = oneshot::channel();
        let req2 = UserRequest {
            prompt: String::from("Write a poem about rust."),
            responder: user2_tx,
        };
        queue_tx.send(req2).await.unwrap();

        // 5. Wait for the answers to come back down the individual pipes
        let answer1 = user1_rx.await.unwrap();
        let answer2 = user2_rx.await.unwrap();

        // 6. Verify the GPU did the math correctly for both users
        assert_eq!(answer1, vec![4.0, 4.0, 10.0, 8.0]);
        assert_eq!(answer2, vec![4.0, 4.0, 10.0, 8.0]);
        
        println!("✅ Both concurrent users received correct GPU answers safely!");
    }
}