# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MiniInferenceEngine is a Rust-based local LLM inference server with a web UI. It downloads, caches, and runs quantized models (GGUF/safetensors) on GPU/CPU, serving a streaming chat API with context compression support.

## Build & Run

```bash
# Build (Candle backend is default; Llama.cpp requires system libs)
cd manager
cargo build                            # default: candle-core backend
cargo build --no-default-features --features backend-llamacpp   # Llama.cpp only
cargo build --all-features              # both backends

# Run the server
cargo run

# Clippy + tests (CPU suite, no GPU needed)
cargo test --all-features
cargo clippy --all-targets --all-features
```

### Prerequisites
- Rust (stable)
- CMake, build-essential, clang, libclang-dev (for `llama-cpp-sys-2` FFI)
- NVIDIA CUDA toolkit (for GPU inference; `CUDA_COMPUTE_CAP` env var overrides auto-detection)
- Node.js 20+ (for web UI linting only)

### Configuration
`manager/config.toml` controls all runtime settings:
- `bind_address`, `oauth_*`, `admin_emails`, `user_emails` — server and auth
- `downloads_directory`, `hf_base_url` — model storage and HF API endpoint
- `gpu_device_index`, `max_concurrent_downloads` — GPU and download tuning
- `log_level_console/file/memory` — logging verbosity
- `database.*` — SurrealDB connection (optional, for persistent sessions/chat)

`config.json` is auto-upgraded to TOML format on load.

## Architecture

```
manager/
├── src/
│   ├── main.rs          — Server entrypoint: AppState, route setup, background tasks, graceful shutdown
│   ├── lib.rs            — Re-exports + run_batcher_loop (GPU orchestrator), ActiveBackend enum
│   ├── types.rs          — Core types: EngineStatus, ModelConfig, Message, StreamEvent, etc.
│   ├── registry.rs       — Model registry: 16+ model registrations, config.json auto-discovery, GGUF split detection
│   ├── backend.rs        — InferenceBackend trait + process_utf8_buffer
│   ├── backend_candle.rs — Candle (PyTorch-like) backend: safetensors mmap, generative + extractive compression
│   ├── backend_llamacpp.rs — Llama.cpp backend: dedicated OS thread, static context allocation
│   ├── downloader.rs     — Resumable chunked downloads, SHA-256 checkpointing, ETag/hash verification
│   ├── auth.rs           — Google OAuth2 flow, session management, dual-auth (session OR API key)
│   ├── telemetry.rs      — Load/generation metrics, SurrealDB persistence, retention cleanup
│   └── setup.rs          — DB init, logging (3-layer: console/file/memory), route builders
│
└── web/                  — Vanilla JS + HTML frontend
    ├── index.html, chat.js    — Main chat UI
    ├── stats.html, stats.js   — Performance/telemetry dashboard
    ├── models.html, models.js — Model management UI
    ├── settings.html, settings.js — Settings + API key management
    ├── queue.html, queue.js   — Download queue monitoring
    ├── console.html, console.js — Admin log console
    ├── memory.html, memory.js — VRAM/RAM tracking
    └── common.js, common.css  — Shared utilities and styles
```

### Key Components

**GPU Orchestrator** (`run_batcher_loop` in `lib.rs`):
- Single async loop consuming `mpsc::Receiver<UserRequest>`
- Handles model hot-swapping, VRAM management (NVML-based), context compression (LLMLingua-2 extractive or abstractive fallback)
- Dynamic memory budgeting: estimates KV cache from free VRAM, triggers compression when prompt exceeds budget

**Backend Abstraction** (`InferenceBackend` trait in `backend.rs`):
- `CandleEngine` — safetensors mmap loading, generative (Llama/Qwen) + extractive (XLM-RoBERTa) compression
- `LlamaCppEngine` — dedicated OS thread (!Send objects), statically allocated KV cache, CPU offload percentage tracking
- `ActiveBackend` enum dispatches to the correct variant

**Model Registry** (`registry.rs`):
- `OnceCell`-backed singleton, lazily initialized on first `get_model_registry()` call
- Each model registration spawns a task that fetches config.json from the tokenizer repo to populate architecture params (arch, layers, attention heads, sliding window, rope scaling, KV cache dtype)
- Detects GGUF split files (`-00001-of-00003.gguf` pattern), resolves size from disk or HF API
- `ModelArch` supports prompt formatting for: Llama, Qwen2, Mistral, Gemma, DeepSeek, Cohere, GptOss

**Downloader** (`downloader.rs`):
- Concurrent chunked downloads with per-file SHA-256 validation
- Checkpoint/resume: serializes `Sha256` internal state to hex, flushes to `.meta` files every 5s
- Handles upstream file changes (hash mismatch → restart), 429 rate limiting with exponential backoff
- Corrupted files renamed to `.corrupted`; completed downloads atomically rename `.tmp` → target

**Auth** (`auth.rs`):
- Google OAuth2 flow (login → callback → session store)
- Dual-auth middleware: accepts either valid session cookie OR Bearer API key
- API key storage: per-user in SurrealDB, indexed by SHA-256 hash for O(1) lookup

**Telemetry** (`telemetry.rs`):
- Load metrics (model, backend, time) and generation metrics (model, params, timing) stored in SurrealDB
- 24h retention cleanup task; in-memory deque limited to 100 entries

**File Watcher** (in `main.rs`):
- Uses `notify` to watch `downloads/` and HuggingFace cache directories
- Debounced incremental updates to `EngineStatus.downloaded_models` / `corrupted_models`
- Falls back to polling every 30s if file watcher unavailable

## Code Style

This project's code style is defined in [`.gemini/styleguide.md`](.gemini/styleguide.md). For reference, the critical rules are:

- **No `unsafe`** — all code must be safe Rust
- **No panics** — never use `unwrap()`, `expect()`, `unreachable!()` in production code; handle errors gracefully
- **`rustfmt`** with default settings, 4-space indentation
- **Modern Rust** (edition 2024) — use let chains, `if let` guards, `is_none_or()` etc.
- **Config drift** — new `AppConfig` fields must be added as commented-out defaults in `config.toml`

See the styleguide for naming conventions, testing requirements, and dependency management policies.

## Running Tests

```bash
# CPU tests (all tests except GPU-required ones marked #[ignore])
cd manager
cargo test --all-features

# UI tests (requires running server first)
npx playwright test
```

GPU-marked tests: `test_vram_info_with_gpu`, `test_candle_gpu_init`, `test_llamacpp_gpu_init`.

## Important Details

- **No tests file needed** — this codebase uses inline `#[cfg(test)]` modules throughout (see each source file)
- **MMAP safety** — Candle uses `VarBuilder::from_mmaped_safetensors`; downloader prevents overwriting active mmap files via VRAM-locked model checks
- **Llama.cpp thread** — runs in a dedicated `std::thread` (not async task) because `llama-cpp-2` types are `!Send`; communicates via `tokio::sync::mpsc`
- **Config migration** — `AppConfig::load()` auto-converts legacy `config.json` to `config.toml` on first run
- **Static CRT linking** — MSVC target uses `/MD` via `rustflags` in `.cargo/config.toml`
