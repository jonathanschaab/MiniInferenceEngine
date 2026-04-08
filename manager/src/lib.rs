use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, oneshot};

use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_llama::ModelWeights;
use candle_transformers::generation::LogitsProcessor;

// Message struct to handle Chat History
#[derive(Deserialize, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Deserialize)]
pub struct ApiRequest {
    pub messages: Vec<Message>,
}

#[derive(Serialize)]
pub struct ApiResponse {
    pub answer: String,
}

pub struct UserRequest {
    pub messages: Vec<Message>, // takes the full array
    pub responder: oneshot::Sender<String>,
}

// Helper: Formats the array into TinyLlama's specific string format
fn format_chat(messages: &[Message]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        prompt.push_str(&format!("<|{}|>\n{}</s>\n", msg.role, msg.content));
    }
    prompt.push_str("<|assistant|>\n");
    prompt
}

// Helper: The core GPU execution loop, abstracted so we can call it multiple times
fn generate_text(
    prompt: &str, 
    model: &mut ModelWeights, 
    tokenizer: &Tokenizer, 
    device: &Device, 
    max_tokens: usize
) -> String {

    let mut tokens = tokenizer.encode(prompt, true).unwrap().get_ids().to_vec();
    let prompt_length = tokens.len();
    let mut logits_processor = LogitsProcessor::new(299792458, None, None);

    for index in 0..max_tokens {
        let context_size = if index > 0 { 1 } else { tokens.len() };
        let start_pos = tokens.len().saturating_sub(context_size);
        
        let input_tensor = Tensor::new(&tokens[start_pos..], device).unwrap().unsqueeze(0).unwrap();
        let logits = model.forward(&input_tensor, start_pos).unwrap();
        let logits = logits.squeeze(0).unwrap();

        let next_token = logits_processor.sample(&logits).unwrap();
        tokens.push(next_token);

        if next_token == 2 { break; } // Stop on EOS
    }

    tokenizer.decode(&tokens[prompt_length..], true).unwrap()
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
    let mut model = ModelWeights::from_gguf(gguf_content, &mut file, &device).unwrap();

    println!("✅ Model successfully loaded into VRAM!");
    println!("⚙️  HOT LOOP: Engine ready. Waiting for web requests...");

    while let Some(request) = receiver.recv().await {
        
        let mut formatted_prompt = format_chat(&request.messages);
        let token_count = tokenizer.encode(formatted_prompt.clone(), true).unwrap().get_ids().len();

        // THE COMPRESSION TRIGGER
        // If the context is getting dangerously close to 2048, compress it!
        if token_count > 1500 {
            println!("⚠️ Context limit approaching ({} tokens). Triggering Auto-Compression...", token_count);
            
            // Build a prompt asking the AI to summarize its own memory
            let compression_prompt = format!(
                "<|system|>\nYou are a helpful AI.</s>\n<|user|>\nSummarize the core facts and context of this conversation compactly:\n\n{}</s>\n<|assistant|>\n", 
                formatted_prompt
            );

            // Run the AI internally!
            let summary = generate_text(&compression_prompt, &mut model, &tokenizer, &device, 300);

            // Rebuild the prompt using the Summary as the new System memory + the latest user message
            let mut new_messages = vec![
                Message { 
                    role: "system".to_string(), 
                    content: format!("Previous conversation summary: {}", summary.trim()) 
                }
            ];
            new_messages.push(request.messages.last().unwrap().clone()); // Keep the user's actual question
            
            formatted_prompt = format_chat(&new_messages);
            println!("✅ Compression complete. Context shrank to {} tokens.", tokenizer.encode(formatted_prompt.clone(), true).unwrap().get_ids().len());
        } else {
            println!("📥 Processing prompt ({} tokens)...", token_count);
        }

        // Generate the final answer and send it back to the web UI
        let final_answer = generate_text(&formatted_prompt, &mut model, &tokenizer, &device, 500);
        let _ = request.responder.send(final_answer.trim().to_string());
    }
}