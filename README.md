# IMECE Examples

Practical examples for the [imece_core](https://crates.io/crates/imece_core) framework — a Rust-native, local-first autonomous agent framework for edge devices.

These examples demonstrate how to use each module of `imece_core` independently and in combination. They are designed to be self-contained, progressively complex, and easy to follow.

> **Note:** `imece_core` is licensed under AGPL v3. These examples are MIT-licensed so you can freely adapt them as starting points for your own projects.

---

## Available Examples

| # | Example | Modules | LLM Required | Description |
|---|---------|---------|:---:|-------------|
| 01 | [hello-embedding](./01-hello-embedding/) | Embedding | ❌ | Local embedding generation with Voyage-4 Nano, cosine similarity |
| 02 | [memory-chain](./02-memory-chain/) | Memory + Embedding | ❌ | DMCE chain-of-memory evolution with diagnostic visualization |
| 03 | [semantic-search](./03-semantic-search/) | Memory (LanceDB) + Embedding | ❌ | Persistent semantic search engine with interactive query loop |

> **🚧 Coming Soon:** Additional examples covering the Inference (KV-Cache Rollback) and Actor (Multi-Agent Swarm) modules are planned. These will demonstrate LLM-powered use cases including single-agent chat, coder-reviewer swarms, RAG pipelines, and the full autonomous agent pipeline.

---

## Prerequisites

- **Rust** 1.70+ (2021 edition)
- **Embedding model** (for all examples): Voyage-4 Nano ONNX model files
  - `model.onnx` — the ONNX-exported model
  - `tokenizer.json` — HuggingFace tokenizer

Place these files in a `models/voyage-4-nano-onnx/` directory, or pass a custom path via the `--model-dir` flag.

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/imece-ai/imece-examples.git
cd imece-examples

# Run the first example (requires embedding model)
cargo run -p hello-embedding -- --model-dir models/voyage-4-nano-onnx

# Run the memory chain example
cargo run -p memory-chain -- --model-dir models/voyage-4-nano-onnx

# Run the interactive semantic search
cargo run -p semantic-search -- --model-dir models/voyage-4-nano-onnx
```

---

## Project Structure

```
imece-examples/
├── Cargo.toml                  # Workspace root
├── LICENSE                     # MIT
├── README.md
├── 01-hello-embedding/         # ⭐ Beginner — local embedding generation
│   ├── Cargo.toml
│   ├── README.md
│   └── src/main.rs
├── 02-memory-chain/            # ⭐ Beginner — DMCE chain evolution
│   ├── Cargo.toml
│   ├── README.md
│   └── src/main.rs
└── 03-semantic-search/         # ⭐⭐ Intermediate — persistent search
    ├── Cargo.toml
    ├── README.md
    └── src/main.rs
```

---

## License

These examples are licensed under the [MIT License](./LICENSE).

`imece_core` itself is licensed under AGPL v3 — see the [imece-core repository](https://github.com/imece-ai/imece-core) for details.
