//! # 04 — Simple Inference
//!
//! Run a local LLM (Qwen3.5-0.8B) and generate text with token streaming.
//!
//! This is the simplest usage of Module 2 (Inference): load a GGUF model,
//! create the engine, and generate a streamed response.
//!
//! ## Usage
//! ```bash
//! cargo run -p simple-inference -- --model-path path/to/Qwen3.5-0.8B-Q4_K_M.gguf
//! ```

use std::io::Write;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use imece_core::inference::backend::{AsyncLlamaBackend, LlamaCppBackend, LlamaCppKvCache};
use imece_core::inference::engine::InferenceEngine;
use imece_core::inference::executor::ProcessExecutor;
use imece_core::inference::kv_cache::KvCacheController;
use imece_core::inference::types::{InferenceConfig, StopSequence};

// ── CLI ──────────────────────────────────────────────────────────────────

struct Config {
    model_path: String,
    prompt: String,
    max_tokens: usize,
    n_ctx: u32,
    n_threads: u32,
}

fn parse_args() -> Config {
    let args: Vec<String> = std::env::args().collect();

    let mut config = Config {
        model_path: String::from("models/Qwen3.5-0.8B-Q4_K_M.gguf"),
        prompt: String::from("Explain how Rust's ownership model prevents data races in three sentences."),
        max_tokens: 512,
        n_ctx: 2048,
        n_threads: 4,
    };

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model-path" if i + 1 < args.len() => {
                config.model_path = args[i + 1].clone();
                i += 2;
            }
            "--prompt" if i + 1 < args.len() => {
                config.prompt = args[i + 1].clone();
                i += 2;
            }
            "--max-tokens" if i + 1 < args.len() => {
                config.max_tokens = args[i + 1].parse().unwrap_or(512);
                i += 2;
            }
            "--n-ctx" if i + 1 < args.len() => {
                config.n_ctx = args[i + 1].parse().unwrap_or(2048);
                i += 2;
            }
            "--n-threads" if i + 1 < args.len() => {
                config.n_threads = args[i + 1].parse().unwrap_or(4);
                i += 2;
            }
            "--help" | "-h" => {
                println!("Usage: simple-inference [OPTIONS]");
                println!();
                println!("Options:");
                println!("  --model-path <PATH>    Path to GGUF model file [default: models/Qwen3.5-0.8B-Q4_K_M.gguf]");
                println!("  --prompt <TEXT>         Prompt to generate from [default: Rust ownership explanation]");
                println!("  --max-tokens <N>        Maximum tokens to generate [default: 512]");
                println!("  --n-ctx <N>             Context window size [default: 2048]");
                println!("  --n-threads <N>         CPU threads for inference [default: 4]");
                println!("  --help, -h             Show this help message");
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                std::process::exit(1);
            }
        }
    }

    config
}

// ── Main ─────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    // Initialize tracing (shows engine-level logs on stderr).
    tracing_subscriber::fmt::init();

    let config = parse_args();

    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║          IMECE — Simple Inference Example                   ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // ── Step 1: Load the llama.cpp backend ────────────────────────────
    //
    // This is a blocking FFI call that loads the GGUF model into memory.
    // GPU layers are resolved automatically via IMECE_GPU_LAYERS env var
    // or build-time detection (--features cuda → all layers offloaded).

    println!("Loading model: {}", config.model_path);
    println!("  Context window : {} tokens", config.n_ctx);
    println!("  CPU threads    : {}", config.n_threads);

    let backend = Arc::new(
        LlamaCppBackend::load(&config.model_path, config.n_ctx, config.n_threads)
            .unwrap_or_else(|e| {
                eprintln!("Error: Failed to load model: {}", e);
                eprintln!();
                eprintln!("Make sure you have a GGUF model file. Recommended:");
                eprintln!("  Qwen3.5-0.8B-Q4_K_M.gguf (~500 MB)");
                eprintln!();
                eprintln!("Usage: cargo run -p simple-inference -- --model-path <PATH>");
                std::process::exit(1);
            }),
    );

    println!("✓ Model loaded (context_size={})", backend.context_size());
    println!();

    // ── Step 2: Create the async-safe wrapper ────────────────────────
    //
    // AsyncLlamaBackend offloads blocking llama_decode calls (~50–500ms)
    // to tokio::task::spawn_blocking, preventing Tokio runtime stalls.

    let async_backend = AsyncLlamaBackend::new(Arc::clone(&backend));

    // ── Step 3: Create the KV-Cache controller ───────────────────────
    //
    // KvCacheController wraps the raw KV-Cache manager and provides
    // rollback logic + telemetry. In this simple example, no rollbacks
    // will occur — but the engine requires it.

    let kv_cache = LlamaCppKvCache::new(Arc::clone(&backend), 0, 0);
    let kv_controller = KvCacheController::new(kv_cache);

    // ── Step 4: Create the executor ──────────────────────────────────
    //
    // ProcessExecutor handles sandboxed code execution. In this example,
    // we don't expect the model to generate action blocks, but the engine
    // requires an executor for completeness.

    let executor = ProcessExecutor::new();

    // ── Step 5: Configure inference ──────────────────────────────────

    let inference_config = InferenceConfig {
        max_tokens: config.max_tokens,
        temperature: 0.7,
        top_k: 40,
        top_p: 0.95,
        max_rollback_retries: 3,
        stop_sequences: vec![
            StopSequence::terminal("<|im_end|>"),
            StopSequence::terminal("<|endoftext|>"),
        ],
        execution_timeout_ms: 30_000,
    };

    // ── Step 6: Assemble the engine ──────────────────────────────────

    let mut engine = InferenceEngine::new(
        async_backend,
        kv_controller,
        executor,
        inference_config,
    );

    // ── Step 7: Format the prompt ────────────────────────────────────
    //
    // Qwen3.5 uses ChatML-style template:
    //   <|im_start|>system\n...<|im_end|>\n
    //   <|im_start|>user\n...<|im_end|>\n
    //   <|im_start|>assistant\n

    let prompt = format!(
        "<|im_start|>system\nYou are a helpful assistant. Answer concisely.<|im_end|>\n\
         <|im_start|>user\n{}<|im_end|>\n\
         <|im_start|>assistant\n",
        config.prompt
    );

    // ── Step 8: Run inference with streaming ─────────────────────────

    println!("── Prompt ──────────────────────────────────────────────────");
    println!("  {}", config.prompt);
    println!();
    println!("── Response (streaming) ────────────────────────────────────");
    print!("  ");

    let start = std::time::Instant::now();
    let cancel = AtomicBool::new(false);

    let session = engine
        .run_streaming(&prompt, &cancel, |token_text| {
            print!("{}", token_text);
            std::io::stdout().flush().ok();
        })
        .await
        .unwrap_or_else(|e| {
            eprintln!("\nError during inference: {}", e);
            std::process::exit(1);
        });

    let elapsed = start.elapsed();

    println!();
    println!();

    // ── Step 9: Print session statistics ─────────────────────────────

    let tokens_per_sec = if elapsed.as_secs_f64() > 0.0 {
        session.total_tokens_generated as f64 / elapsed.as_secs_f64()
    } else {
        0.0
    };

    println!("── Session Stats ───────────────────────────────────────────");
    println!("  Tokens generated : {}", session.total_tokens_generated);
    println!("  Time elapsed     : {:.2}s", elapsed.as_secs_f64());
    println!("  Speed            : {:.1} tokens/sec", tokens_per_sec);
    println!("  Rollbacks        : {}", session.total_rollbacks);
    println!("  Tokens erased    : {}", session.total_tokens_erased);
    println!();
    println!("✓ Completed — zero API calls, fully local");
    println!();
}
