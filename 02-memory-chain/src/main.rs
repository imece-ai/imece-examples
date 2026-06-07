//! # 02 — Memory Chain (DMCE)
//!
//! Build contextual memory chains using the Dynamic Memory Chain Evolution
//! (DMCE) algorithm from the Chain-of-Memory paper (arXiv:2601.14287v1).
//!
//! This example demonstrates:
//! - Populating a `MemoryStore` with `MemoryNode`s
//! - Running DMCE with `evolve_with_diagnostics()` for detailed telemetry
//! - Comparing how different `β` values affect chain length via APT
//!
//! ## Usage
//! ```bash
//! cargo run -p memory-chain -- --model-dir path/to/voyage-4-nano-onnx
//! ```

use imece_core::embedding::backend::EmbeddingBackend;
use imece_core::embedding::config::{
    EmbeddingServiceConfig, MrlDimension, OutputPrecision, VoyageNanoConfig,
};
use imece_core::memory::chain::DmceEngine;
use imece_core::memory::node::{MemoryNode, Role};
use imece_core::memory::store::MemoryStore;

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
            println!("Usage: memory-chain [OPTIONS]");
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

// ── Memory Data ──────────────────────────────────────────────────────────

/// Sample conversation fragments to populate the memory store.
/// Organized into thematic clusters to demonstrate DMCE's ability
/// to select coherent chains.
fn sample_memories() -> Vec<(&'static str, Role)> {
    vec![
        // Cluster: Rust memory safety
        ("Rust's ownership model prevents data races at compile time.", Role::Agent),
        ("The borrow checker enforces exclusive or shared access to data.", Role::Agent),
        ("Lifetimes in Rust ensure references never outlive the data they point to.", Role::Agent),
        ("Drop trait in Rust provides deterministic resource cleanup.", Role::Agent),
        // Cluster: Python
        ("Python uses a GIL that prevents true multi-threaded CPU parallelism.", Role::Agent),
        ("Python's garbage collector uses reference counting with cycle detection.", Role::Agent),
        ("List comprehensions are a Pythonic way to create transformed lists.", Role::Agent),
        // Cluster: Machine learning
        ("Neural networks learn hierarchical feature representations from data.", Role::Agent),
        ("Gradient descent minimizes the loss function by updating model weights.", Role::Agent),
        ("Transformers use self-attention to capture long-range dependencies.", Role::Agent),
        ("Overfitting occurs when a model memorizes training data instead of generalizing.", Role::Agent),
        // Cluster: Systems programming
        ("Virtual memory maps process address spaces to physical memory pages.", Role::Agent),
        ("Context switches save and restore CPU register state between processes.", Role::Agent),
        ("Cache locality optimization is critical for high-performance code.", Role::Agent),
        // Cluster: User questions
        ("How does Rust handle memory without a garbage collector?", Role::User),
        ("Can you explain how neural networks learn?", Role::User),
        ("What is the difference between threads and processes?", Role::User),
        // Cluster: System observations
        ("User session started at 14:32 UTC.", Role::System),
        ("Model inference completed in 245ms (32 tokens).", Role::System),
        ("Memory usage: 2.1 GB / 8.0 GB available.", Role::System),
    ]
}

// ── Display Helpers ──────────────────────────────────────────────────────

fn role_icon(role: &Role) -> &'static str {
    match role {
        Role::User => "👤",
        Role::Agent => "🤖",
        Role::System => "⚙️ ",
    }
}

fn print_chain_result(
    query_text: &str,
    result: &imece_core::memory::chain::ChainResult,
    beta: f32,
) {
    println!("  Query : \"{}\"", query_text);
    println!("  β     : {:.2}", beta);
    println!("  Steps : {}", result.steps_executed);
    println!("  APT   : {}", if result.apt_triggered { "✂️  triggered (chain truncated)" } else { "— not triggered" });
    println!("  Chain : {} node(s)", result.chain.len());
    println!();

    // Display step scores
    if !result.step_scores.is_empty() {
        println!("  Step Scores (S_gate):");
        for (i, score) in result.step_scores.iter().enumerate() {
            let bar_len = ((*score).max(0.0) * 40.0) as usize;
            let bar: String = "█".repeat(bar_len);

            let apt_marker = if i > 0 && i == result.step_scores.len() - 1 && result.apt_triggered {
                " ← APT truncation"
            } else {
                ""
            };

            println!("    Step {}: {:.6} {}{}", i + 1, score, bar, apt_marker);
        }
        println!();
    }

    // Display chain nodes
    println!("  Selected Chain:");
    for (i, node) in result.chain.iter().enumerate() {
        println!(
            "    {}. {} {} \"{}\"",
            i + 1,
            role_icon(&node.role),
            match node.role {
                Role::User => "[User]  ",
                Role::Agent => "[Agent] ",
                Role::System => "[System]",
            },
            node.text
        );
    }
    println!();
}

// ── Main ─────────────────────────────────────────────────────────────────

fn main() {
    let model_dir = parse_model_dir();

    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║            IMECE — Memory Chain (DMCE) Example              ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // ── Initialize embedding backend ─────────────────────────────────

    let config = EmbeddingServiceConfig::VoyageNano(VoyageNanoConfig {
        model_dir,
        mrl_dimension: MrlDimension::D256,
        output_precision: OutputPrecision::Float32, // Float32 for clearer cosine scores
        num_threads: 4,
        max_length: 512,
    });

    let backend = config.create_backend().unwrap_or_else(|e| {
        eprintln!("Error: Failed to initialize embedding backend: {}", e);
        eprintln!("Usage: cargo run -p memory-chain -- --model-dir <PATH>");
        std::process::exit(1);
    });

    let dim = backend.dimension();
    println!("Backend  : {}", backend.name());
    println!("Dimension: {}", dim);
    println!();

    // ── Populate the memory store ────────────────────────────────────

    let mut store = MemoryStore::new_in_memory(dim).expect("Failed to create memory store");

    let memories = sample_memories();
    println!("── Populating Memory Store ──────────────────────────────────");
    println!();

    for (text, role) in &memories {
        let embedding = backend
            .embed_document(text)
            .unwrap_or_else(|e| {
                eprintln!("Error embedding '{}': {}", text, e);
                std::process::exit(1);
            })
            .to_f32();

        let node = MemoryNode::new(text.to_string(), *role, embedding);
        store.insert(&node).expect("Failed to insert memory node");
    }

    println!("  Stored {} memory nodes (dim={})", store.len(), dim);
    println!();

    // ── Run DMCE with different β values ─────────────────────────────

    let queries = [
        "How does Rust ensure memory safety?",
        "Explain how deep learning works",
        "Tell me about operating system internals",
    ];

    let beta_values = [0.3_f32, 0.6, 0.9];

    for query_text in &queries {
        println!("══════════════════════════════════════════════════════════════");
        println!();

        let query_emb = backend
            .embed_query(query_text)
            .unwrap_or_else(|e| {
                eprintln!("Error embedding query '{}': {}", query_text, e);
                std::process::exit(1);
            })
            .to_f32();

        for &beta in &beta_values {
            let engine = DmceEngine::new(
                beta,  // β — APT truncation threshold
                10,    // Top-K candidate pool size
                8,     // Maximum chain length
            );

            let result = engine.evolve_with_diagnostics(&store, &query_emb);
            print_chain_result(query_text, &result, beta);
        }
    }

    // ── Summary ──────────────────────────────────────────────────────

    println!("══════════════════════════════════════════════════════════════");
    println!();
    println!("── Summary ─────────────────────────────────────────────────");
    println!();
    println!("  β = 0.3 (permissive)  → Longer chains, richer context");
    println!("  β = 0.6 (balanced)    → Good trade-off for ≤8GB VRAM");
    println!("  β = 0.9 (aggressive)  → Shorter chains, less VRAM usage");
    println!();
    println!("  The DMCE gating score S_gate(m) = cos(m.e, q) × cos(m.e, C_z)");
    println!("  enforces both global relevance AND contextual consistency.");
    println!();
}
