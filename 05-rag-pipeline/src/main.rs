//! # 05 — RAG Pipeline
//!
//! End-to-end Retrieval-Augmented Generation combining all "read" modules:
//! Embedding (Module 4) + Memory/DMCE (Module 1) + Inference (Module 2).
//!
//! ## Pipeline
//! 1. Embed documents into a `MemoryStore` using Voyage-4 Nano (Module 4).
//! 2. Embed the user query and build a DMCE memory chain (Module 1).
//! 3. Format the chain as LLM context and generate an answer (Module 2).
//!
//! ## Usage
//! ```bash
//! cargo run -p rag-pipeline -- \
//!   --model-path models/Qwen3.5-0.8B-Q4_K_M.gguf \
//!   --embedding-dir models/voyage-4-nano-onnx
//! ```

use std::io::Write;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use imece_core::embedding::backend::EmbeddingBackend;
use imece_core::embedding::config::{
    EmbeddingServiceConfig, MrlDimension, OutputPrecision, VoyageNanoConfig,
};
use imece_core::inference::backend::{AsyncLlamaBackend, LlamaCppBackend, LlamaCppKvCache};
use imece_core::inference::engine::InferenceEngine;
use imece_core::inference::executor::ProcessExecutor;
use imece_core::inference::kv_cache::KvCacheController;
use imece_core::inference::types::{InferenceConfig, StopSequence};
use imece_core::memory::chain::DmceEngine;
use imece_core::memory::node::{MemoryNode, Role};
use imece_core::memory::store::MemoryStore;

// ── CLI ──────────────────────────────────────────────────────────────────

struct Config {
    model_path: String,
    embedding_dir: String,
    query: String,
    max_tokens: usize,
}

fn parse_args() -> Config {
    let args: Vec<String> = std::env::args().collect();

    let mut config = Config {
        model_path: String::from("models/Qwen3.5-0.8B-Q4_K_M.gguf"),
        embedding_dir: String::from("models/voyage-4-nano-onnx"),
        query: String::from("How does Rust prevent data races?"),
        max_tokens: 512,
    };

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model-path" if i + 1 < args.len() => {
                config.model_path = args[i + 1].clone();
                i += 2;
            }
            "--embedding-dir" if i + 1 < args.len() => {
                config.embedding_dir = args[i + 1].clone();
                i += 2;
            }
            "--query" if i + 1 < args.len() => {
                config.query = args[i + 1].clone();
                i += 2;
            }
            "--max-tokens" if i + 1 < args.len() => {
                config.max_tokens = args[i + 1].parse().unwrap_or(512);
                i += 2;
            }
            "--help" | "-h" => {
                println!("Usage: rag-pipeline [OPTIONS]");
                println!();
                println!("Options:");
                println!("  --model-path <PATH>      GGUF model file [default: models/Qwen3.5-0.8B-Q4_K_M.gguf]");
                println!("  --embedding-dir <PATH>   ONNX embedding model dir [default: models/voyage-4-nano-onnx]");
                println!("  --query <TEXT>            Query to answer [default: \"How does Rust prevent data races?\"]");
                println!("  --max-tokens <N>          Max tokens to generate [default: 512]");
                println!("  --help, -h               Show this help message");
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

// ── Knowledge Base ───────────────────────────────────────────────────────

/// A curated knowledge base about systems programming and Rust.
/// In production, these would be loaded from files or a database.
fn knowledge_base() -> Vec<(&'static str, Role)> {
    vec![
        // Rust memory safety
        ("Rust's ownership model ensures that every value has a single owner. When the owner goes out of scope, the value is dropped automatically, preventing memory leaks and dangling pointers.", Role::Agent),
        ("The borrow checker enforces at compile time that you can have either one mutable reference OR any number of immutable references to a value, but never both simultaneously. This prevents data races.", Role::Agent),
        ("Lifetimes in Rust are compile-time annotations that ensure references never outlive the data they point to. The compiler infers most lifetimes automatically.", Role::Agent),
        ("Rust's Send and Sync traits mark types that can be safely transferred between threads (Send) or shared across threads (Sync). Types containing raw pointers are neither Send nor Sync by default.", Role::Agent),
        ("The Drop trait provides deterministic resource cleanup. When a value goes out of scope, its Drop implementation runs immediately — no garbage collector delay.", Role::Agent),
        // Concurrency
        ("Rust's type system prevents data races at compile time. A data race occurs when two threads access the same memory location concurrently, at least one is a write, and no synchronization is used.", Role::Agent),
        ("Arc<Mutex<T>> is the standard pattern for sharing mutable state across threads in Rust. Arc provides thread-safe reference counting, while Mutex provides interior mutability with locking.", Role::Agent),
        ("Channels (std::sync::mpsc) enable message-passing concurrency. The sender can be cloned for multi-producer scenarios while the receiver remains single-consumer.", Role::Agent),
        // Systems programming
        ("Virtual memory maps each process's address space to physical memory pages managed by the OS. Page faults trigger the OS to load pages from disk into RAM.", Role::Agent),
        ("CPU cache lines are typically 64 bytes. Accessing data sequentially (cache-friendly) can be 100x faster than random access due to spatial locality.", Role::Agent),
        // Machine learning
        ("Transformer models use multi-head self-attention to capture relationships between all positions in a sequence, enabling parallel processing unlike RNNs.", Role::Agent),
        ("Retrieval-Augmented Generation (RAG) enhances LLM responses by retrieving relevant documents and injecting them as context into the prompt.", Role::Agent),
    ]
}

// ── Main ─────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let config = parse_args();

    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║          IMECE — RAG Pipeline Example                       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // ── Phase 1: Initialize Embedding Backend ────────────────────────

    println!("── Phase 1: Embedding ──────────────────────────────────────");
    println!();

    let emb_config = EmbeddingServiceConfig::VoyageNano(VoyageNanoConfig {
        model_dir: config.embedding_dir.clone(),
        mrl_dimension: MrlDimension::D256,
        output_precision: OutputPrecision::Float32,
        num_threads: 4,
        max_length: 512,
    });

    let emb_backend = emb_config.create_backend().unwrap_or_else(|e| {
        eprintln!("Error: Failed to initialize embedding backend: {}", e);
        eprintln!("Usage: cargo run -p rag-pipeline -- --embedding-dir <PATH>");
        std::process::exit(1);
    });

    let dim = emb_backend.dimension();
    println!("  Embedding backend : {}", emb_backend.name());
    println!("  Dimension         : {}", dim);
    println!();

    // ── Phase 2: Populate Memory Store ───────────────────────────────

    println!("── Phase 2: Memory Store ────────────────────────────────────");
    println!();

    let mut store = MemoryStore::new_in_memory(dim).expect("Failed to create memory store");
    let docs = knowledge_base();

    for (text, role) in &docs {
        let embedding = emb_backend
            .embed_document(text)
            .unwrap_or_else(|e| {
                eprintln!("Error embedding document: {}", e);
                std::process::exit(1);
            })
            .to_f32();

        let node = MemoryNode::new(text.to_string(), *role, embedding);
        store.insert(&node).expect("Failed to insert node");
    }

    println!("  Indexed {} documents (dim={})", store.len(), dim);
    println!();

    // ── Phase 3: Build DMCE Memory Chain ─────────────────────────────

    println!("── Phase 3: DMCE Chain Evolution ────────────────────────────");
    println!();
    println!("  Query: \"{}\"", config.query);
    println!();

    let query_emb = emb_backend
        .embed_query(&config.query)
        .unwrap_or_else(|e| {
            eprintln!("Error embedding query: {}", e);
            std::process::exit(1);
        })
        .to_f32();

    let dmce = DmceEngine::new(
        0.6, // β — APT truncation threshold
        10,  // Top-K candidate pool size
        5,   // Maximum chain length (keep context compact for small LLM)
    );

    let chain_result = dmce.evolve_with_diagnostics(&store, &query_emb);

    println!("  Chain length : {} node(s)", chain_result.chain.len());
    println!("  Steps        : {}", chain_result.steps_executed);
    println!("  APT triggered: {}", if chain_result.apt_triggered { "yes" } else { "no" });
    println!();

    // Format the chain as context for the LLM.
    let mut context = String::new();
    for (i, node) in chain_result.chain.iter().enumerate() {
        context.push_str(&format!("[{}] {}\n", i + 1, node.text));
    }

    println!("  Retrieved context:");
    for (i, node) in chain_result.chain.iter().enumerate() {
        let truncated: String = node.text.chars().take(80).collect();
        println!("    [{}] \"{}...\"", i + 1, truncated);
    }
    println!();

    // ── Phase 4: Load LLM & Generate Answer ──────────────────────────

    println!("── Phase 4: LLM Inference ───────────────────────────────────");
    println!();
    println!("  Loading model: {}", config.model_path);

    let backend = Arc::new(
        LlamaCppBackend::load(&config.model_path, 2048, 4).unwrap_or_else(|e| {
            eprintln!("Error: Failed to load model: {}", e);
            eprintln!("Usage: cargo run -p rag-pipeline -- --model-path <PATH>");
            std::process::exit(1);
        }),
    );

    println!("  ✓ Model loaded (context_size={})", backend.context_size());
    println!();

    let async_backend = AsyncLlamaBackend::new(Arc::clone(&backend));
    let kv_cache = LlamaCppKvCache::new(Arc::clone(&backend), 0, 0);
    let kv_controller = KvCacheController::new(kv_cache);
    let executor = ProcessExecutor::new();

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

    let mut engine = InferenceEngine::new(
        async_backend,
        kv_controller,
        executor,
        inference_config,
    );

    // Build the RAG prompt: system instruction + retrieved context + user query.
    let prompt = format!(
        "<|im_start|>system\n\
         You are a helpful assistant. Answer the user's question using ONLY the \
         provided context below. Be concise and accurate.\n\n\
         Context:\n{}<|im_end|>\n\
         <|im_start|>user\n{}<|im_end|>\n\
         <|im_start|>assistant\n",
        context, config.query
    );

    println!("── RAG Answer (streaming) ───────────────────────────────────");
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

    // ── Summary ──────────────────────────────────────────────────────

    let tokens_per_sec = if elapsed.as_secs_f64() > 0.0 {
        session.total_tokens_generated as f64 / elapsed.as_secs_f64()
    } else {
        0.0
    };

    println!("── Pipeline Summary ────────────────────────────────────────");
    println!("  Documents indexed  : {}", docs.len());
    println!("  DMCE chain length  : {}", chain_result.chain.len());
    println!("  Tokens generated   : {}", session.total_tokens_generated);
    println!("  Inference time     : {:.2}s ({:.1} tok/s)", elapsed.as_secs_f64(), tokens_per_sec);
    println!("  Rollbacks          : {}", session.total_rollbacks);
    println!();
    println!("  Pipeline: Embed → DMCE → LLM — zero API calls, fully local");
    println!();
}
