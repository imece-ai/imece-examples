# 04 — Simple Inference

Run a local LLM and generate text with token-by-token streaming — the simplest usage of IMECE's Inference module (Module 2).

## What This Example Demonstrates

- Loading a GGUF model via `LlamaCppBackend::load()`
- Wrapping the backend with `AsyncLlamaBackend` for Tokio compatibility
- Creating a `KvCacheController` for rollback-ready inference
- Running `InferenceEngine::run_streaming()` with a token callback
- Session statistics (tokens generated, speed)

## Prerequisites

- **Rust** 1.91+ (2021 edition)
- **CMake** 3.14+ (for llama.cpp compilation)
- **GGUF model file** — recommended: [Qwen3.5-0.8B-Q4_K_M](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF) (~500 MB)

## Usage

```bash
# Default prompt
cargo run -p simple-inference -- --model-path models/Qwen3.5-0.8B-Q4_K_M.gguf

# Custom prompt
cargo run -p simple-inference -- \
  --model-path models/Qwen3.5-0.8B-Q4_K_M.gguf \
  --prompt "What is the difference between a mutex and a semaphore?"

# With CUDA GPU offloading
cargo run -p simple-inference --features cuda -- \
  --model-path models/Qwen3.5-0.8B-Q4_K_M.gguf
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path` | `models/Qwen3.5-0.8B-Q4_K_M.gguf` | Path to GGUF model file |
| `--prompt` | Rust ownership explanation | Prompt text |
| `--max-tokens` | `512` | Maximum tokens to generate |
| `--n-ctx` | `2048` | Context window size |
| `--n-threads` | `4` | CPU threads for inference |

## Expected Output

```
╔══════════════════════════════════════════════════════════════╗
║          IMECE — Simple Inference Example                   ║
╚══════════════════════════════════════════════════════════════╝

Loading model: models/Qwen3.5-0.8B-Q4_K_M.gguf
✓ Model loaded (context_size=2048)

── Prompt ──────────────────────────────────────────────────
  Explain how Rust's ownership model prevents data races in three sentences.

── Response (streaming) ────────────────────────────────────
  Rust's ownership model ensures that each value has exactly one owner...

── Session Stats ───────────────────────────────────────────
  Tokens generated : 87
  Time elapsed     : 4.23s
  Speed            : 20.6 tokens/sec
  Rollbacks        : 0
  Tokens erased    : 0

✓ Completed — zero API calls, fully local
```

## Next Steps

- **05-rag-pipeline**: Combine embeddings + memory + inference for RAG
- **06-rollback-agent**: See KV-Cache "Time Travel" rollback in action
