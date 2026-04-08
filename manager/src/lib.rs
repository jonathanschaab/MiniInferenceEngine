use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, oneshot};

use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_llama::ModelWeights as LlamaWeights;
use candle_transformers::models::quantized_qwen2::ModelWeights as Qwen2Weights;
use candle_transformers::generation::LogitsProcessor;

#[derive(Clone, Serialize, Deserialize)]
pub enum ModelArch {
    Llama,
    Qwen2,
}

pub enum DynamicModel {
    Llama(LlamaWeights),
    Qwen2(Qwen2Weights),
}

impl DynamicModel {
    // A universal forward pass that routes the math to the correct architecture!
    pub fn forward(&mut self, x: &Tensor, start_pos: usize) -> candle_core::Result<Tensor> {
        match self {
            Self::Llama(m) => m.forward(x, start_pos),
            Self::Qwen2(m) => m.forward(x, start_pos),
        }
    }
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum ModelRole {
    GeneralChat,
    ContextCompressor,
    CodeSpecialist,
    ToolCaller,
    Reasoning,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub id: String,
    pub name: String,
    pub repo: String,
    pub tokenizer_repo: String,
    pub filename: String,
    pub max_context_len: usize,
    pub roles: Vec<ModelRole>,
    pub arch: ModelArch,
}

// Expose the registry so the web server can send it to the UI
pub fn get_model_registry() -> Vec<ModelConfig> {
    vec![
        ModelConfig { 
            id: "llama-3.1-8b".to_string(), name: "Llama 3.1 (8B)".to_string(),
            repo: "QuantFactory/Meta-Llama-3.1-8B-Instruct-GGUF".to_string(),
            tokenizer_repo: "NousResearch/Meta-Llama-3.1-8B-Instruct".to_string(),
            filename: "Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf".to_string(), 
            max_context_len: 8192, roles: vec![ModelRole::GeneralChat],
            arch: ModelArch::Llama // <-- Tell the engine to use Llama math
        },
        ModelConfig { 
            id: "qwen3-14b".to_string(), 
            name: "Qwen3 (14B Equivalent)".to_string(),
            // Switched to a single-file community repo
            repo: "bartowski/Qwen2.5-14B-Instruct-GGUF".to_string(), 
            tokenizer_repo: "Qwen/Qwen2.5-14B-Instruct".to_string(),
            // Note the updated capitalization for this specific repo!
            filename: "Qwen2.5-14B-Instruct-Q4_K_M.gguf".to_string(), 
            max_context_len: 4096, 
            roles: vec![ModelRole::GeneralChat, ModelRole::CodeSpecialist],
            arch: ModelArch::Qwen2
        },
        ModelConfig { 
            id: "strand-rust-14b".to_string(), name: "Strand Rust Coder (14B)".to_string(),
            repo: "Qwen/Qwen2.5-Coder-14B-Instruct-GGUF".to_string(), 
            tokenizer_repo: "Qwen/Qwen2.5-Coder-14B-Instruct".to_string(),
            filename: "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf".to_string(), 
            max_context_len: 4096, roles: vec![ModelRole::CodeSpecialist],
            arch: ModelArch::Qwen2 // <-- Tell the engine to use Qwen math
        },
        ModelConfig { 
            id: "llmlingua-2".to_string(), name: "LLMLingua-2 (Compressor)".to_string(),
            repo: "Qwen/Qwen2.5-1.5B-Instruct-GGUF".to_string(), 
            tokenizer_repo: "Qwen/Qwen2.5-1.5B-Instruct".to_string(),
            filename: "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf".to_string(), 
            max_context_len: 16384, roles: vec![ModelRole::ContextCompressor],
            arch: ModelArch::Qwen2 // <-- Tell the engine to use Qwen math
        },
    ]
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

#[derive(Serialize)]
pub struct ApiResponse {
    pub answer: String,
}

pub struct UserRequest {
    pub chat_model_id: String,
    pub compressor_model_id: String,
    pub messages: Vec<Message>,
    pub responder: oneshot::Sender<String>,
}

// Helper: Formats the array into a generic string format
fn format_chat(messages: &[Message]) -> String {
    let mut prompt = String::new();
    for msg in messages { prompt.push_str(&format!("<|{}|>\n{}</s>\n", msg.role, msg.content)); }
    prompt.push_str("<|assistant|>\n");
    prompt
}

// THE SNAPSHOT LOADER: This function returns the mapped File so the OS keeps it alive in RAM!
fn load_engine(model_id: &str, device: &Device) -> (DynamicModel, Tokenizer, std::fs::File) {
    let config = get_model_registry().into_iter().find(|c| c.id == model_id).unwrap();
    let api = Api::new().unwrap();
    
    let weights_path = api.model(config.repo).get(&config.filename).unwrap();
    let tokenizer_path = api.model(config.tokenizer_repo).get("tokenizer.json").unwrap();

    let mut file = std::fs::File::open(&weights_path).unwrap();
    let gguf_content = candle_core::quantized::gguf_file::Content::read(&mut file).unwrap();
    
    // Switch the brain dynamically!
    let model = match config.arch {
        ModelArch::Llama => DynamicModel::Llama(LlamaWeights::from_gguf(gguf_content, &mut file, device).unwrap()),
        ModelArch::Qwen2 => DynamicModel::Qwen2(Qwen2Weights::from_gguf(gguf_content, &mut file, device).unwrap()),
    };

    let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();
    (model, tokenizer, file)
}

fn generate_text(prompt: &str, model: &mut DynamicModel, tokenizer: &Tokenizer, device: &Device, max_tokens: usize) -> String {
    let mut tokens = tokenizer.encode(prompt, true).unwrap().get_ids().to_vec();
    let prompt_length = tokens.len();
    let mut logits_processor = LogitsProcessor::new(299792458, None, None);

    for index in 0..max_tokens {
        // KV CACHE LOGIC: First pass sends the whole prompt to fill the cache.
        // Future passes send ONLY the 1 newest token to save massive compute!
        let context_size = if index > 0 { 1 } else { tokens.len() };
        let start_pos = tokens.len().saturating_sub(context_size);
        
        // Input MUST be 2D: [batch_size, seq_len]
        let input_tensor = Tensor::new(&tokens[start_pos..], device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();

        // Run the GPU Math
        // Candle's quantized models ALWAYS return [batch_size, vocab_size] to save VRAM!
        let logits = model.forward(&input_tensor, start_pos).unwrap();

        // Squeeze the batch dimension to get a pure 1D array of probabilities: [vocab_size]
        let next_token_logits = logits.squeeze(0).unwrap();

        // Pick the actual next word ID
        let next_token = logits_processor.sample(&next_token_logits).unwrap();
        tokens.push(next_token);

        // Stop if we hit EOS tokens for Llama 2 (2), Qwen (151645), ChatML (151643), or Llama 3 (128001/128009)
        if next_token == 2 || next_token == 151645 || next_token == 151643 || next_token == 128001 || next_token == 128009 { 
            break; 
        }
    }
    
    tokenizer.decode(&tokens[prompt_length..], true).unwrap()
}

pub async fn run_batcher_loop(mut receiver: mpsc::Receiver<UserRequest>) {
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    
    let mut active_model_id = String::new();
    let mut active_model: Option<DynamicModel> = None;
    let mut active_tokenizer: Option<Tokenizer> = None;
    let mut _active_file: Option<std::fs::File> = None;

    println!("⚙️  ORCHESTRATOR ONLINE: Waiting for requests...");

    while let Some(request) = receiver.recv().await {
        
        // 1. Hot-Swap to the requested Chat Model
        if active_model_id != request.chat_model_id {
            println!("🔄 Swapping VRAM to {}...", request.chat_model_id);
            drop(active_model.take()); 
            drop(active_tokenizer.take()); 
            drop(_active_file.take()); 
            
            let (m, t, f) = load_engine(&request.chat_model_id, &device);
            active_model = Some(m); active_tokenizer = Some(t); _active_file = Some(f);
            active_model_id = request.chat_model_id.clone();
        }

        let mut formatted_prompt = format_chat(&request.messages);
        let token_count = active_tokenizer.as_ref().unwrap().encode(formatted_prompt.clone(), true).unwrap().get_ids().len();

        // 2. The Auto-Compressor Intercept
        if token_count > 1500 {
            println!("⚠️ Context limit reached. Loading Compressor Model ALONGSIDE Chat Model...");
            
            // DO NOT DROP THE CHAT MODEL! VRAM handles 8.4GB + 1.5GB easily!
            let (mut comp_m, comp_t, _comp_f) = load_engine(&request.compressor_model_id, &device);
            
            let compression_prompt = format!("<|user|>\nSummarize the core facts of this conversation compactly:\n{}</s>\n<|assistant|>\n", formatted_prompt);
            println!("🧠 Running Compression Agent...");
            let summary = generate_text(&compression_prompt, &mut comp_m, &comp_t, &device, 400);

            // CLEAR COMPRESSOR MODEL VRAM ONLY
            drop(comp_m); 
            drop(comp_t); 
            drop(_comp_f);
            
            println!("🔄 Resuming Chat...");
            let mut new_messages = vec![Message { role: "system".to_string(), content: format!("Summary: {}", summary.trim()) }];
            new_messages.push(request.messages.last().unwrap().clone());
            formatted_prompt = format_chat(&new_messages);
        }

        println!("📥 Processing prompt ({} tokens)...", token_count);
        let answer = generate_text(&formatted_prompt, active_model.as_mut().unwrap(), active_tokenizer.as_ref().unwrap(), &device, 500);
        let _ = request.responder.send(answer.trim().to_string());
    }
}
