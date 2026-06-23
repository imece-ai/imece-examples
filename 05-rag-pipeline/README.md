# 05 — RAG Pipeline

End-to-end **Retrieval-Augmented Generation** combining three IMECE modules:

1. **Embedding** (Module 4) — embed documents and queries locally
2. **Memory / DMCE** (Module 1) — build semantically coherent memory chains
3. **Inference** (Module 2) — generate grounded answers via local LLM

## What This Example Demonstrates

- Initializing the Voyage-4 Nano embedding backend
- Populating a `MemoryStore` with document embeddings
- Building a DMCE memory chain with `evolve_with_diagnostics()`
- Formatting the chain as RAG context for the LLM prompt
- Running inference with the augmented prompt
- End-to-end pipeline statistics

## Prerequisites

- **Rust** 1.91+, **CMake** 3.14+
- **GGUF model** — [Qwen3.5-0.8B-Q4_K_M](https://huggingface.co/Qwen/Qwen3.5-0.8B-GGUF)
- **Embedding model** — Voyage-4 Nano ONNX files (`model.onnx` + `tokenizer.json`)

## Usage

```bash
cargo run -p rag-pipeline -- \
  --model-path models/Qwen3.5-0.8B-Q4_K_M.gguf \
  --embedding-dir models/voyage-4-nano-onnx

# Custom query
cargo run -p rag-pipeline -- \
  --model-path models/Qwen3.5-0.8B-Q4_K_M.gguf \
  --embedding-dir models/voyage-4-nano-onnx \
  --query "What is the difference between Arc and Rc in Rust?"
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path` | `models/Qwen3.5-0.8B-Q4_K_M.gguf` | Path to GGUF model |
| `--embedding-dir` | `models/voyage-4-nano-onnx` | Path to ONNX embedding model |
| `--query` | "How does Rust prevent data races?" | Query to answer |
| `--max-tokens` | `512` | Maximum tokens to generate |

## Pipeline Architecture

```
Query → [Module 4: Embed] → [Module 1: DMCE Chain] → Context
                                                        ↓
                                           [Module 2: LLM Inference]
                                                        ↓
                                                  Grounded Answer
```

## Expected Output

```
╔══════════════════════════════════════════════════════════════╗
║          IMECE — RAG Pipeline Example                       ║
╚══════════════════════════════════════════════════════════════╝

── Phase 1: Embedding ──────────────────────────────────────
  Embedding backend : voyage-4-nano
  Dimension         : 256

── Phase 2: Memory Store ────────────────────────────────────
  Indexed 12 documents (dim=256)

── Phase 3: DMCE Chain Evolution ────────────────────────────
  Query: "How does Rust prevent data races?"
  Chain length : 4 node(s)
  APT triggered: yes

── Phase 4: LLM Inference ───────────────────────────────────
  ✓ Model loaded (context_size=2048)

── RAG Answer (streaming) ───────────────────────────────────
  Rust prevents data races through its ownership model and borrow checker...

── Pipeline Summary ────────────────────────────────────────
  Documents indexed  : 12
  DMCE chain length  : 4
  Tokens generated   : 93
  Inference time     : 4.51s (20.6 tok/s)

  Pipeline: Embed → DMCE → LLM — zero API calls, fully local
```
