//! # 03 — Semantic Search
//!
//! A persistent semantic search engine using LanceDB for vector storage
//! with an interactive query loop.
//!
//! This example demonstrates Module 1 (Memory/LanceDB) + Module 4 (Embedding)
//! working together for a practical search use case:
//!
//! 1. **Index:** Embed documents and store them in a persistent LanceDB table.
//! 2. **Search:** Embed a query, retrieve Top-K most similar documents.
//!
//! The LanceDB store persists to disk, so indexed data survives restarts.
//!
//! ## Usage
//! ```bash
//! # Index + Search (default)
//! cargo run -p semantic-search -- --model-dir path/to/voyage-4-nano-onnx
//!
//! # Index only
//! cargo run -p semantic-search -- --model-dir path/to/voyage-4-nano-onnx --index
//!
//! # Search only (requires prior indexing)
//! cargo run -p semantic-search -- --model-dir path/to/voyage-4-nano-onnx --search
//! ```

use std::io::{self, BufRead, Write};

use imece_core::embedding::backend::EmbeddingBackend;
use imece_core::embedding::config::{
    EmbeddingServiceConfig, MrlDimension, OutputPrecision, VoyageNanoConfig,
};
use imece_core::memory::lance_store::LanceMemoryStore;
use imece_core::memory::node::{MemoryNode, Role};

// ── CLI ──────────────────────────────────────────────────────────────────

struct Config {
    model_dir: String,
    data_dir: String,
    do_index: bool,
    do_search: bool,
    top_k: usize,
}

fn parse_args() -> Config {
    let args: Vec<String> = std::env::args().collect();

    let mut config = Config {
        model_dir: String::from("models/voyage-4-nano-onnx"),
        data_dir: String::from("./lance_data"),
        do_index: false,
        do_search: false,
        top_k: 5,
    };

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model-dir" if i + 1 < args.len() => {
                config.model_dir = args[i + 1].clone();
                i += 2;
            }
            "--data-dir" if i + 1 < args.len() => {
                config.data_dir = args[i + 1].clone();
                i += 2;
            }
            "--top-k" if i + 1 < args.len() => {
                config.top_k = args[i + 1].parse().unwrap_or(5);
                i += 2;
            }
            "--index" => {
                config.do_index = true;
                i += 1;
            }
            "--search" => {
                config.do_search = true;
                i += 1;
            }
            "--help" | "-h" => {
                println!("Usage: semantic-search [OPTIONS]");
                println!();
                println!("Options:");
                println!("  --model-dir <PATH>   ONNX model directory [default: models/voyage-4-nano-onnx]");
                println!("  --data-dir <PATH>    LanceDB storage directory [default: ./lance_data]");
                println!("  --top-k <N>          Number of results to return [default: 5]");
                println!("  --index              Index documents only");
                println!("  --search             Search only (requires prior indexing)");
                println!("  --help, -h           Show this help message");
                println!();
                println!("If neither --index nor --search is specified, both are performed.");
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                std::process::exit(1);
            }
        }
    }

    // Default: do both
    if !config.do_index && !config.do_search {
        config.do_index = true;
        config.do_search = true;
    }

    config
}

// ── Knowledge Base ───────────────────────────────────────────────────────

/// Built-in knowledge base for demonstration purposes.
/// In a real application, these would be loaded from files.
fn knowledge_base() -> Vec<(&'static str, Role)> {
    vec![
        // Rust
        ("Rust's ownership model prevents data races at compile time by enforcing a single owner for each value.", Role::Agent),
        ("The borrow checker in Rust enforces that you can have either one mutable reference or any number of immutable references to data.", Role::Agent),
        ("Lifetimes in Rust are annotations that tell the compiler how long references should remain valid.", Role::Agent),
        ("Rust's trait system enables polymorphism through static dispatch with monomorphization.", Role::Agent),
        ("Cargo is Rust's build system and package manager, handling dependency resolution and compilation.", Role::Agent),
        // Systems Programming
        ("Virtual memory provides each process with its own isolated address space mapped to physical memory by the OS.", Role::Agent),
        ("A mutex (mutual exclusion) prevents concurrent access to shared resources by allowing only one thread to hold the lock.", Role::Agent),
        ("CPU cache lines are typically 64 bytes, and cache-friendly data access patterns can significantly improve performance.", Role::Agent),
        // Machine Learning
        ("Neural networks learn by adjusting weights through backpropagation, minimizing a loss function via gradient descent.", Role::Agent),
        ("The transformer architecture uses multi-head self-attention to capture relationships between all positions in a sequence.", Role::Agent),
        ("Embeddings are dense vector representations that capture semantic meaning of words, sentences, or documents.", Role::Agent),
        ("Retrieval-Augmented Generation (RAG) enhances LLM responses by injecting relevant retrieved context into the prompt.", Role::Agent),
        // General CS
        ("A hash table provides O(1) average-case lookup by mapping keys to array indices through a hash function.", Role::Agent),
        ("Recursion solves problems by breaking them into smaller subproblems, with a base case to terminate the recursion.", Role::Agent),
        ("Big-O notation describes the upper bound of an algorithm's time or space complexity as input size grows.", Role::Agent),
    ]
}

// ── Display Helpers ──────────────────────────────────────────────────────

fn role_label(role: &Role) -> &'static str {
    match role {
        Role::User => "👤 User",
        Role::Agent => "🤖 Agent",
        Role::System => "⚙️  System",
    }
}

// ── Index Phase ──────────────────────────────────────────────────────────

async fn index_documents(
    store: &mut LanceMemoryStore,
    backend: &dyn EmbeddingBackend,
) {
    let docs = knowledge_base();

    println!("📚 Indexing {} documents into LanceDB...", docs.len());
    println!();

    let mut nodes = Vec::with_capacity(docs.len());

    for (i, (text, role)) in docs.iter().enumerate() {
        let embedding = backend
            .embed_document(text)
            .unwrap_or_else(|e| {
                eprintln!("Error embedding document {}: {}", i + 1, e);
                std::process::exit(1);
            })
            .to_f32();

        let node = MemoryNode::new(text.to_string(), *role, embedding);
        nodes.push(node);
    }

    store
        .insert_batch(&nodes)
        .await
        .unwrap_or_else(|e| {
            eprintln!("Error inserting into LanceDB: {}", e);
            std::process::exit(1);
        });

    let count = store.len().await.unwrap_or(0);
    println!(
        "   ✓ Indexed {} documents (dim={}, stored at LanceDB)",
        count,
        backend.dimension()
    );
    println!();
}

// ── Search Phase ─────────────────────────────────────────────────────────

async fn search_loop(
    store: &LanceMemoryStore,
    backend: &dyn EmbeddingBackend,
    top_k: usize,
) {
    println!("🔍 Entering interactive search mode. Type a query and press Enter.");
    println!("   Type /quit to exit.");
    println!();

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("> ");
        stdout.flush().unwrap();

        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap() == 0 {
            // EOF
            break;
        }

        let query = line.trim();

        if query.is_empty() {
            continue;
        }

        if query == "/quit" || query == "/exit" || query == "/q" {
            println!();
            println!("Goodbye!");
            break;
        }

        // Embed the query
        let query_emb = match backend.embed_query(query) {
            Ok(emb) => emb.to_f32(),
            Err(e) => {
                eprintln!("   Error embedding query: {}", e);
                continue;
            }
        };

        // Search
        let results = match store.top_k(&query_emb, top_k, &[]).await {
            Ok(r) => r,
            Err(e) => {
                eprintln!("   Error searching: {}", e);
                continue;
            }
        };

        if results.is_empty() {
            println!("   No results found.");
            println!();
            continue;
        }

        println!();
        println!("   Top-{} results:", results.len());
        println!();

        for (i, scored) in results.iter().enumerate() {
            let bar_len = ((scored.score.max(0.0)) * 25.0) as usize;
            let bar: String = "█".repeat(bar_len);

            println!(
                "    {}. [{:.4}] {} {} \"{}\"",
                i + 1,
                scored.score,
                bar,
                role_label(&scored.node.role),
                scored.node.text
            );
        }

        println!();
    }
}

// ── Main ─────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    let config = parse_args();

    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║           IMECE — Semantic Search Example                   ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // ── Initialize embedding backend ─────────────────────────────────

    let emb_config = EmbeddingServiceConfig::VoyageNano(VoyageNanoConfig {
        model_dir: config.model_dir,
        mrl_dimension: MrlDimension::D256,
        output_precision: OutputPrecision::Float32,
        num_threads: 4,
        max_length: 512,
    });

    let backend = emb_config.create_backend().unwrap_or_else(|e| {
        eprintln!("Error: Failed to initialize embedding backend: {}", e);
        eprintln!("Usage: cargo run -p semantic-search -- --model-dir <PATH>");
        std::process::exit(1);
    });

    let dim = backend.dimension();
    println!("Backend  : {}", backend.name());
    println!("Dimension: {}", dim);
    println!("Storage  : {}", config.data_dir);
    println!();

    // ── Open or create the LanceDB store ─────────────────────────────

    let mut store = LanceMemoryStore::new(&config.data_dir, dim)
        .await
        .unwrap_or_else(|e| {
            eprintln!("Error: Failed to open LanceDB at '{}': {}", config.data_dir, e);
            std::process::exit(1);
        });

    // ── Index phase ──────────────────────────────────────────────────

    if config.do_index {
        index_documents(&mut store, backend.as_ref()).await;
    }

    // ── Search phase ─────────────────────────────────────────────────

    if config.do_search {
        let count = store.len().await.unwrap_or(0);
        if count == 0 {
            eprintln!("Error: LanceDB store is empty. Run with --index first.");
            std::process::exit(1);
        }
        println!("   Store contains {} documents.", count);
        println!();

        search_loop(&store, backend.as_ref(), config.top_k).await;
    }
}
