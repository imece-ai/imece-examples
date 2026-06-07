# 01 — Hello Embedding

The simplest possible example: generate local text embeddings using `imece_core`'s embedding subsystem and compute pairwise cosine similarities.

## What This Demonstrates

- Configuring `VoyageNanoConfig` for the ONNX embedding backend
- Using `embed_query()` vs `embed_document()` (asymmetric retrieval prompts)
- MRL truncation (2048-d → 256-d) and int8 quantization
- Computing cosine similarity between embedding vectors

## Modules Used

| Module | Purpose |
|--------|---------|
| **Embedding** (Module 4) | Local vector generation via Voyage-4 Nano ONNX |

## Prerequisites

Place the Voyage-4 Nano ONNX model files in a directory:
- `model.onnx`
- `tokenizer.json`

## Usage

```bash
cargo run -p hello-embedding -- --model-dir path/to/voyage-4-nano-onnx
```

## Expected Output

```
╔══════════════════════════════════════════════════════════════╗
║               IMECE — Hello Embedding Example               ║
╚══════════════════════════════════════════════════════════════╝

Backend : voyage-4-nano
Dimension: 256
Precision: Int8

── Embedding Texts ──────────────────────────────────────────

  [1] "Rust's ownership model prevents data races at compile time."
  [2] "Python uses a Global Interpreter Lock (GIL) for thread safety."
  [3] "The borrow checker ensures memory safety without garbage collection."
  [4] "Machine learning models require large amounts of training data."

── Cosine Similarity Matrix ─────────────────────────────────

              [1]      [2]      [3]      [4]
  [1]       1.0000   0.xxxx   0.xxxx   0.xxxx
  [2]       0.xxxx   1.0000   0.xxxx   0.xxxx
  [3]       0.xxxx   0.xxxx   1.0000   0.xxxx
  [4]       0.xxxx   0.xxxx   0.xxxx   1.0000

── Analysis ─────────────────────────────────────────────────

  Most similar pair : [1] ↔ [3] (score: 0.xxxx)
  Least similar pair: [1] ↔ [4] (score: 0.xxxx)
```
