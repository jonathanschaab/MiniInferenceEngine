use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, oneshot};

use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_llama::ModelWeights;
use candle_transformers::generation::LogitsProcessor;

#[derive(Deserialize)]
pub struct ApiRequest {
    pub prompt: String,
}

#[derive(Serialize)]
pub struct ApiResponse {
    pub answer: String,
}

pub struct UserRequest {
    pub prompt: String,
    pub responder: oneshot::Sender<String>,
}

pub async fn run_batcher_loop(mut receiver: mpsc::Receiver<UserRequest>) {
    println!("🔥 COLD START: Downloading and loading model into VRAM...");

    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);

    let api = Api::new().unwrap();

    let tokenizer_repo = api.model("TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string());
    let tokenizer_filename = tokenizer_repo.get("tokenizer.json").unwrap();
    let tokenizer = Tokenizer::from_file(tokenizer_filename).unwrap();

    let model_repo = api.model("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF".to_string());
    let weights_filename = model_repo.get("tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf").unwrap();

    let mut file = std::fs::File::open(&weights_filename).unwrap();
    let gguf_content = candle_core::quantized::gguf_file::Content::read(&mut file).unwrap();
    
    // Notice model is now `mut` because the KV Cache changes during generation!
    let mut model = ModelWeights::from_gguf(gguf_content, &mut file, &device).unwrap();

    println!("✅ Model successfully loaded into VRAM!");
    println!("⚙️  HOT LOOP: Engine ready. Waiting for web requests...");

    while let Some(request) = receiver.recv().await {
        println!("📥 Processing prompt: '{}'", request.prompt);
        
        // 1. Format the prompt for TinyLlama's specific chat template
        let formatted_prompt = format!("<|system|>\nYou are a helpful AI assistant.</s>\n<|user|>\n{}</s>\n<|assistant|>\n", request.prompt);

        // 2. Tokenize the text into integer IDs
        let mut tokens = tokenizer.encode(formatted_prompt, true).unwrap().get_ids().to_vec();
        let prompt_length = tokens.len(); // Save this so we don't output the prompt back to the user
        
        // 3. Initialize the sampler (Seed: 299792458, Temp: None means Greedy/Most-Likely search)
        let mut logits_processor = LogitsProcessor::new(299792458, None, None);

        // 4. The Autoregressive Loop (Generate up to 200 words)
        for index in 0..200 {
            
            // A. KV CACHE LOGIC: 
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            
            // Input MUST be 2D: [batch_size, seq_len] -> [1, context_size]
            let input_tensor = Tensor::new(&tokens[start_pos..], &device)
                .unwrap()
                .unsqueeze(0)
                .unwrap();

            // B. Run the GPU Math
            // Candle automatically extracts the last token's probabilities to save memory! 
            // Output shape is ALWAYS [batch_size, vocab_size] -> [1, 32000]
            let logits = model.forward(&input_tensor, start_pos).unwrap();

            // C. Squeeze the batch dimension to get a pure 1D array of probabilities
            // [1, 32000] -> [32000]
            let logits = logits.squeeze(0).unwrap();

            // D. Pick the actual next word ID
            let next_token = logits_processor.sample(&logits).unwrap();
            tokens.push(next_token);

            // E. Stop if the model outputs the EOS (End of Sequence) token. 
            if next_token == 2 {
                break;
            }
        }

        // 5. Decode only the newly generated IDs back into English text
        let generated_ids = &tokens[prompt_length..];
        let final_text = tokenizer.decode(generated_ids, true).unwrap();

        println!("📤 Generation complete!");
        
        // 6. Send the text back to the Axum web thread
        let _ = request.responder.send(final_text);
    }
}