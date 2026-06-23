# IMECE Examples

Practical examples for the [imece_core](https://crates.io/crates/imece_core) framework — a Rust-native, local-first autonomous agent framework for edge devices.

These examples demonstrate how to use each module of `imece_core` independently and in combination. They are designed to be self-contained, progressively complex, and easy to follow.

> **Note:** `imece_core` is licensed under AGPL v3. These examples are MIT-licensed so you can freely adapt them as starting points for your own projects.

---

## Available Examples

| # | Example | Modules | LLM Required | Level | Description |
|---|---------|---------|:---:|:---:|-------------|
| 01 | [hello-embedding](./01-hello-embedding/) | Embedding | ❌ | ⭐ | Local embedding generation with Voyage-4 Nano, cosine similarity |
| 02 | [memory-chain](./02-memory-chain/) | Memory + Embedding | ❌ | ⭐ | DMCE chain-of-memory evolution with diagnostic visualization |
| 03 | [semantic-search](./03-semantic-search/) | Memory (LanceDB) + Embedding | ❌ | ⭐⭐ | Persistent semantic search engine with interactive query loop |
| 04 | [simple-inference](./04-simple-inference/) | Inference | ✅ | ⭐⭐ | Load a GGUF model and generate text with token streaming |
| 05 | [rag-pipeline](./05-rag-pipeline/) | Memory + Embedding + Inference | ✅ | ⭐⭐⭐ | End-to-end RAG: embed → DMCE chain → LLM answer |
| 06 | [rollback-agent](./06-rollback-agent/) | Inference (KV-Cache Rollback) | ✅ | ⭐⭐⭐ | **Flagship demo** — KV-Cache "Time Travel" self-correction |

---

## Prerequisites

### All Examples
- **Rust** 1.91+ (2021 edition)

### Examples 01–05 (Embedding)
- **Embedding model**: Voyage-4 Nano ONNX model files
  - `model.onnx` — the ONNX-exported model
  - `tokenizer.json` — HuggingFace tokenizer

Place these in a `models/voyage-4-nano-onnx/` directory, or pass a custom path via `--model-dir` / `--embedding-dir`.

### Examples 04–06 (LLM Inference)
- **CMake** 3.14+ (for llama.cpp compilation)
- **C++ compiler** (GCC / Clang)
- **GGUF model file** — recommended: [Qwen3.5-0.8B-Q4_K_M](https://huggingface.co/Qwen/Qwen3.5-0.8B-GGUF) (~500 MB)

Place the `.gguf` file in a `models/` directory, or pass a custom path via `--model-path`.

### Example 06 (Rollback Agent)
- **Python 3** installed (for sandbox code execution)

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/imece-ai/imece-examples.git
cd imece-examples

# ── Embedding-only examples (no LLM needed) ──

# Run the first example (requires embedding model)
cargo run -p hello-embedding -- --model-dir models/voyage-4-nano-onnx

# Run the memory chain example
cargo run -p memory-chain -- --model-dir models/voyage-4-nano-onnx

# Run the interactive semantic search
cargo run -p semantic-search -- --model-dir models/voyage-4-nano-onnx

# ── LLM-powered examples (requires GGUF model) ──

# Simple text generation with token streaming
cargo run -p simple-inference -- --model-path models/Qwen3.5-0.8B-Q4_K_M.gguf

# End-to-end RAG pipeline (embedding + DMCE + LLM)
cargo run -p rag-pipeline -- \
  --model-path models/Qwen3.5-0.8B-Q4_K_M.gguf \
  --embedding-dir models/voyage-4-nano-onnx

# 🧠 KV-Cache "Time Travel" Rollback (the flagship demo)
cargo run -p rollback-agent -- --model-path models/Qwen3.5-0.8B-Q4_K_M.gguf
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
├── 03-semantic-search/         # ⭐⭐ Intermediate — persistent search
│   ├── Cargo.toml
│   ├── README.md
│   └── src/main.rs
├── 04-simple-inference/        # ⭐⭐ Intermediate — LLM text generation
│   ├── Cargo.toml
│   ├── README.md
│   └── src/main.rs
├── 05-rag-pipeline/            # ⭐⭐⭐ Advanced — end-to-end RAG
│   ├── Cargo.toml
│   ├── README.md
│   └── src/main.rs
└── 06-rollback-agent/          # ⭐⭐⭐ Advanced — KV-Cache Rollback
    ├── Cargo.toml
    ├── README.md
    └── src/main.rs
```

---

## License

These examples are licensed under the [MIT License](./LICENSE).

`imece_core` itself is licensed under AGPL v3 — see the [imece-core repository](https://github.com/imece-ai/imece-core) for details.
