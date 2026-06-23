//! # 06 — KV-Cache Rollback Agent
//!
//! **The flagship IMECE demo.** Demonstrates the KV-Cache "Time Travel"
//! Rollback protocol — IMECE's core differentiator.
//!
//! ## What Happens
//!
//! 1. The LLM is prompted to write Python code to fetch JSON from a URL.
//! 2. The LLM generates code using `requests` (which isn't installed).
//! 3. `ProcessExecutor` runs the code in a sandbox → fails with
//!    `ModuleNotFoundError: No module named 'requests'`.
//! 4. **KV-Cache Rollback** triggers: the erroneous tokens are erased from
//!    the llama.cpp KV-Cache, the error observation is injected, and
//!    generation resumes from the corrected position.
//! 5. The LLM "corrects itself mid-thought" and generates working code
//!    using `urllib` instead.
//!
//! ## The "Time Travel" Protocol
//!
//! ```text
//! ┌──────────────┐     ┌───────────┐     ┌──────────┐     ┌──────────────┐
//! │   GENERATE   │────▶│ INTERCEPT │────▶│ EXECUTE  │────▶│  EVALUATE    │
//! │ (tokens)     │     │ (stop seq)│     │ (sandbox)│     │ (success/err)│
//! └──────────────┘     └───────────┘     └──────────┘     └──┬───────────┘
//!       ▲                                                     │
//!       │                  ┌───────────────────┐              │
//!       └──────────────────│  KV-CACHE ROLLBACK │◀────────────┘
//!                          │  ("Time Travel")   │   (on error)
//!                          └───────────────────┘
//! ```
//!
//! ## Usage
//! ```bash
//! cargo run -p rollback-agent -- --model-path models/Qwen3.5-0.8B-Q4_K_M.gguf
//! ```

use std::io::Write;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use imece_core::inference::backend::{AsyncLlamaBackend, LlamaCppBackend, LlamaCppKvCache};
use imece_core::inference::engine::{InferenceEngine, SessionEvent};
use imece_core::inference::executor::ProcessExecutor;
use imece_core::inference::kv_cache::KvCacheController;
use imece_core::inference::types::{ExecutionOutcome, InferenceConfig, StopSequence};

// ── CLI ──────────────────────────────────────────────────────────────────

struct Config {
    model_path: String,
    n_ctx: u32,
    n_threads: u32,
    max_tokens: usize,
    max_retries: usize,
}

fn parse_args() -> Config {
    let args: Vec<String> = std::env::args().collect();

    let mut config = Config {
        model_path: String::from("models/Qwen3.5-0.8B-Q4_K_M.gguf"),
        n_ctx: 4096,
        n_threads: 4,
        max_tokens: 1024,
        max_retries: 3,
    };

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model-path" if i + 1 < args.len() => {
                config.model_path = args[i + 1].clone();
                i += 2;
            }
            "--n-ctx" if i + 1 < args.len() => {
                config.n_ctx = args[i + 1].parse().unwrap_or(4096);
                i += 2;
            }
            "--n-threads" if i + 1 < args.len() => {
                config.n_threads = args[i + 1].parse().unwrap_or(4);
                i += 2;
            }
            "--max-tokens" if i + 1 < args.len() => {
                config.max_tokens = args[i + 1].parse().unwrap_or(1024);
                i += 2;
            }
            "--max-retries" if i + 1 < args.len() => {
                config.max_retries = args[i + 1].parse().unwrap_or(3);
                i += 2;
            }
            "--help" | "-h" => {
                println!("Usage: rollback-agent [OPTIONS]");
                println!();
                println!("Demonstrates KV-Cache \"Time Travel\" Rollback — IMECE's core differentiator.");
                println!();
                println!("Options:");
                println!("  --model-path <PATH>   GGUF model file [default: models/Qwen3.5-0.8B-Q4_K_M.gguf]");
                println!("  --n-ctx <N>            Context window size [default: 4096]");
                println!("  --n-threads <N>        CPU threads [default: 4]");
                println!("  --max-tokens <N>       Max tokens to generate [default: 1024]");
                println!("  --max-retries <N>      Max rollback retries [default: 3]");
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
    tracing_subscriber::fmt::init();

    let config = parse_args();

    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     IMECE — KV-Cache \"Time Travel\" Rollback Agent           ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // ── Load Model ───────────────────────────────────────────────────

    println!("Loading model: {}", config.model_path);

    let backend = Arc::new(
        LlamaCppBackend::load(&config.model_path, config.n_ctx, config.n_threads)
            .unwrap_or_else(|e| {
                eprintln!("Error: Failed to load model: {}", e);
                eprintln!();
                eprintln!("Download a GGUF model (e.g., Qwen3.5-0.8B-Q4_K_M.gguf)");
                eprintln!("and pass it via: --model-path <PATH>");
                std::process::exit(1);
            }),
    );

    println!("✓ Model loaded");
    println!("  Context  : {} tokens", backend.context_size());
    println!("  Rollback : {} max retries", config.max_retries);
    println!();

    // ── Build Engine ─────────────────────────────────────────────────

    let async_backend = AsyncLlamaBackend::new(Arc::clone(&backend));

    let kv_cache = LlamaCppKvCache::new(Arc::clone(&backend), 0, 0);
    let kv_controller = KvCacheController::new(kv_cache);

    // ProcessExecutor maps "python" → python3 -c <code>, "bash" → sh -c <code>.
    let executor = ProcessExecutor::new();

    // Configure stop sequences:
    //   - </action>  → action boundary (triggers sandbox execution)
    //   - <|im_end|> → terminal (end of assistant response)
    let inference_config = InferenceConfig {
        max_tokens: config.max_tokens,
        temperature: 0.7,
        top_k: 40,
        top_p: 0.95,
        max_rollback_retries: config.max_retries,
        stop_sequences: vec![
            StopSequence::action("</action>"),
            StopSequence::terminal("<|im_end|>"),
            StopSequence::terminal("<|endoftext|>"),
        ],
        execution_timeout_ms: 15_000,
    };

    let mut engine = InferenceEngine::new(
        async_backend,
        kv_controller,
        executor,
        inference_config,
    );

    // ── Craft the Prompt ─────────────────────────────────────────────
    //
    // System prompt teaches the model to wrap executable code in <action> tags.
    // The user prompt is designed so the model will likely use `requests`
    // (not installed in the sandbox), triggering the rollback protocol.

    let system_prompt = "\
You are an autonomous coding agent. You can execute code by wrapping it in action tags.

To execute Python code, write:
<action type=\"python\">
your python code here
</action>

After execution, you will receive an Observation with the result or error.
If there is an error, fix your code and try again.

IMPORTANT: Only use Python standard library modules. Do NOT use pip packages like 'requests'.";

    let user_prompt = "Write a Python script that fetches JSON data from \
https://httpbin.org/json and prints the slideshow title.";

    let prompt = format!(
        "<|im_start|>system\n{}<|im_end|>\n\
         <|im_start|>user\n{}<|im_end|>\n\
         <|im_start|>assistant\n",
        system_prompt, user_prompt
    );

    // ── Run Inference ────────────────────────────────────────────────

    println!("── Prompt ──────────────────────────────────────────────────");
    println!("  {}", user_prompt);
    println!();
    println!("── Agent Response (streaming) ────────────────────────────────");
    println!();

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

    // ── Display Session Events ───────────────────────────────────────
    //
    // This is where the KV-Cache rollback story becomes visible.
    // Each event shows what happened during the inference session.

    if !session.events.is_empty() {
        println!("── Session Events ──────────────────────────────────────────");
        println!();

        for (i, event) in session.events.iter().enumerate() {
            match event {
                SessionEvent::ActionExecuted { action_type, outcome } => {
                    match outcome {
                        ExecutionOutcome::Success { stdout } => {
                            println!("  [{}] ✅ Action '{}' → Success", i + 1, action_type);
                            if !stdout.is_empty() {
                                let display: String = stdout.chars().take(200).collect();
                                println!("      stdout: {}", display);
                            }
                        }
                        ExecutionOutcome::Failure { exit_code, stderr, .. } => {
                            println!("  [{}] ❌ Action '{}' → Failed (exit_code={})", i + 1, action_type, exit_code);
                            let display: String = stderr.chars().take(200).collect();
                            println!("      stderr: {}", display);
                        }
                        ExecutionOutcome::Timeout { timeout_ms } => {
                            println!("  [{}] ⏱  Action '{}' → Timeout ({}ms)", i + 1, action_type, timeout_ms);
                        }
                    }
                }
                SessionEvent::Rollback { position, tokens_erased, retry_number } => {
                    println!(
                        "  [{}] ⏪ KV-Cache Rollback! pos={}, erased={} tokens, retry #{}",
                        i + 1, position, tokens_erased, retry_number
                    );
                    println!("      → Erased erroneous tokens from KV-Cache");
                    println!("      → Injected error observation at position {}", position);
                    println!("      → Resumed generation (zero prompt recalculation)");
                }
                SessionEvent::Warning(msg) => {
                    println!("  [{}] ⚠  Warning: {}", i + 1, msg);
                }
            }
            println!();
        }
    }

    // ── Session Statistics ────────────────────────────────────────────

    let tokens_per_sec = if elapsed.as_secs_f64() > 0.0 {
        session.total_tokens_generated as f64 / elapsed.as_secs_f64()
    } else {
        0.0
    };

    println!("── Session Stats ───────────────────────────────────────────");
    println!("  Tokens generated   : {}", session.total_tokens_generated);
    println!("  Time elapsed       : {:.2}s ({:.1} tok/s)", elapsed.as_secs_f64(), tokens_per_sec);
    println!("  Total rollbacks    : {}", session.total_rollbacks);
    println!("  Total tokens erased: {}", session.total_tokens_erased);
    println!("  Session events     : {}", session.events.len());
    println!();

    if session.total_rollbacks > 0 {
        println!("  🧠 The LLM corrected itself via KV-Cache \"Time Travel\":");
        println!("     - {} rollback(s) saved {} tokens of re-computation",
            session.total_rollbacks, session.total_tokens_erased);
        println!("     - Zero prompt recalculation — instant recovery");
        println!("     - No other framework can do this.");
    } else {
        println!("  ℹ  No rollbacks occurred — the model got it right first try!");
        println!("     Try a harder prompt or a smaller model to trigger rollback.");
    }

    println!();
}
