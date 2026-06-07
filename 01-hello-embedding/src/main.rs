//! # 01 — Hello Embedding
//!
//! Generate local text embeddings using imece_core's embedding subsystem
//! and compute pairwise cosine similarities.
//!
//! This example demonstrates the most basic usage of Module 4 (Embedding):
//! loading the Voyage-4 Nano ONNX model, embedding texts, and comparing
//! them via cosine similarity.
//!
//! ## Usage
//! ```bash
//! cargo run -p hello-embedding -- --model-dir path/to/voyage-4-nano-onnx
//! ```

use imece_core::embedding::backend::EmbeddingBackend;
use imece_core::embedding::config::{
    EmbeddingServiceConfig, MrlDimension, OutputPrecision, VoyageNanoConfig,
};
use ndarray::Array1;

// ── CLI ──────────────────────────────────────────────────────────────────

fn parse_model_dir() -> String {
    let args: Vec<String> = std::env::args().collect();

    let mut model_dir = String::from("models/voyage-4-nano-onnx");

    let mut i = 1;
    while i < args.len() {
        if args[i] == "--model-dir" && i + 1 < args.len() {
            model_dir = args[i + 1].clone();
            i += 2;
        } else if args[i] == "--help" || args[i] == "-h" {
            println!("Usage: hello-embedding [OPTIONS]");
            println!();
            println!("Options:");
            println!("  --model-dir <PATH>  Path to ONNX model directory [default: models/voyage-4-nano-onnx]");
            println!("  --help, -h          Show this help message");
            std::process::exit(0);
        } else {
            eprintln!("Unknown argument: {}", args[i]);
            std::process::exit(1);
        }
    }

    model_dir
}

// ── Cosine Similarity ────────────────────────────────────────────────────

fn cosine_similarity(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    let dot = a.dot(b);
    let norm_a = a.dot(a).sqrt();
    let norm_b = b.dot(b).sqrt();
    let denom = norm_a * norm_b;

    if denom < f32::EPSILON {
        0.0
    } else {
        dot / denom
    }
}

// ── Main ─────────────────────────────────────────────────────────────────

fn main() {
    let model_dir = parse_model_dir();

    // ── Header ───────────────────────────────────────────────────────

    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║             IMECE — Hello Embedding Example                 ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // ── Configure the embedding backend ──────────────────────────────

    let config = EmbeddingServiceConfig::VoyageNano(VoyageNanoConfig {
        model_dir,
        mrl_dimension: MrlDimension::D256,
        output_precision: OutputPrecision::Int8,
        num_threads: 4,
        max_length: 512,
    });

    let backend = config.create_backend().unwrap_or_else(|e| {
        eprintln!("Error: Failed to initialize embedding backend: {}", e);
        eprintln!();
        eprintln!("Make sure you have the Voyage-4 Nano ONNX model files:");
        eprintln!("  - model.onnx");
        eprintln!("  - tokenizer.json");
        eprintln!();
        eprintln!("Usage: cargo run -p hello-embedding -- --model-dir <PATH>");
        std::process::exit(1);
    });

    println!("Backend  : {}", backend.name());
    println!("Dimension: {}", backend.dimension());
    println!("Precision: Int8 (QAT-optimized)");
    println!();

    // ── Define texts to embed ────────────────────────────────────────

    let texts = [
        "Rust's ownership model prevents data races at compile time.",
        "Python uses a Global Interpreter Lock (GIL) for thread safety.",
        "The borrow checker ensures memory safety without garbage collection.",
        "Machine learning models require large amounts of training data.",
    ];

    println!("── Embedding Texts ──────────────────────────────────────────");
    println!();
    for (i, text) in texts.iter().enumerate() {
        println!("  [{}] \"{}\"", i + 1, text);
    }
    println!();

    // ── Generate embeddings ──────────────────────────────────────────
    // Using embed_document() since these are document-like texts.
    // For search queries, you would use embed_query() instead.

    let embeddings: Vec<Array1<f32>> = texts
        .iter()
        .map(|text| {
            backend
                .embed_document(text)
                .unwrap_or_else(|e| {
                    eprintln!("Error embedding text: {}", e);
                    std::process::exit(1);
                })
                .to_f32()
        })
        .collect();

    // ── Compute pairwise cosine similarity ───────────────────────────

    let n = embeddings.len();

    println!("── Cosine Similarity Matrix ─────────────────────────────────");
    println!();

    // Header row
    print!("          ");
    for j in 0..n {
        print!("  [{:>1}]   ", j + 1);
    }
    println!();

    // Matrix rows
    for i in 0..n {
        print!("  [{:>1}]   ", i + 1);
        for j in 0..n {
            let sim = cosine_similarity(&embeddings[i], &embeddings[j]);
            print!(" {:.4}  ", sim);
        }
        println!();
    }
    println!();

    // ── Find most and least similar pairs ────────────────────────────

    let mut best = (0, 1, f32::NEG_INFINITY);
    let mut worst = (0, 1, f32::INFINITY);

    for i in 0..n {
        for j in (i + 1)..n {
            let sim = cosine_similarity(&embeddings[i], &embeddings[j]);
            if sim > best.2 {
                best = (i, j, sim);
            }
            if sim < worst.2 {
                worst = (i, j, sim);
            }
        }
    }

    println!("── Analysis ─────────────────────────────────────────────────");
    println!();
    println!(
        "  Most similar pair : [{}] ↔ [{}] (score: {:.4})",
        best.0 + 1,
        best.1 + 1,
        best.2
    );
    println!(
        "  Least similar pair: [{}] ↔ [{}] (score: {:.4})",
        worst.0 + 1,
        worst.1 + 1,
        worst.2
    );
    println!();

    // ── Demonstrate query vs document embedding ──────────────────────

    println!("── Query vs Document Embedding ──────────────────────────────");
    println!();

    let query_text = "How does Rust prevent data races?";
    println!("  Query: \"{}\"", query_text);
    println!();

    let query_emb = backend
        .embed_query(query_text)
        .unwrap_or_else(|e| {
            eprintln!("Error embedding query: {}", e);
            std::process::exit(1);
        })
        .to_f32();

    println!("  Similarity to each document:");
    for (i, (text, doc_emb)) in texts.iter().zip(embeddings.iter()).enumerate() {
        let sim = cosine_similarity(&query_emb, doc_emb);
        let bar_len = ((sim.max(0.0)) * 30.0) as usize;
        let bar: String = "█".repeat(bar_len);
        println!("    [{}] {:.4} {} \"{}\"", i + 1, sim, bar, text);
    }
    println!();
}
