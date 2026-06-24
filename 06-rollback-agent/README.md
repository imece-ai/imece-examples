# 06 — KV-Cache Rollback Agent

**The flagship IMECE demo.** This example showcases the framework's core differentiator: **KV-Cache "Time Travel" Rollback** — hardware-level self-correction that no other agent framework offers.

## What Happens

1. The LLM receives a coding task and generates a Python script.
2. The `ProcessExecutor` runs the script in a sandbox → **it fails** (e.g., `ModuleNotFoundError`).
3. The `InferenceEngine` triggers **KV-Cache Rollback**: erroneous tokens are erased directly from llama.cpp's KV-Cache memory.
4. An error observation is injected at the rollback point.
5. The LLM **resumes generation from the corrected position** — zero prompt recalculation, zero context bloat.
6. The corrected script executes successfully.

## The "Time Travel" Protocol

```
┌──────────────┐     ┌───────────┐     ┌──────────┐     ┌──────────────┐
│   GENERATE   │────▶│ INTERCEPT │────▶│ EXECUTE  │────▶│  EVALUATE    │
│ (tokens)     │     │ (stop seq)│     │ (sandbox)│     │ (success/err)│
└──────────────┘     └───────────┘     └──────────┘     └──┬───────────┘
      ▲                                                     │
      │                  ┌───────────────────┐              │
      └──────────────────│  KV-CACHE ROLLBACK │◀────────────┘
                         │  ("Time Travel")   │   (on error)
                         └───────────────────┘
```

**Why this matters:**
- Traditional frameworks: discard context → re-prompt → recalculate everything → expensive
- IMECE: erase bad tokens → inject error → resume → **zero cost recovery**

## Prerequisites

- **Rust** 1.91+, **CMake** 3.14+
- **GGUF model** — [Qwen3.5-0.8B-Q4_K_M](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF)
- **Python 3** installed (for sandbox execution)

## Usage

```bash
cargo run -p rollback-agent -- --model-path models/Qwen3.5-0.8B-Q4_K_M.gguf

# With verbose tracing (see engine-level rollback logs)
RUST_LOG=debug cargo run -p rollback-agent -- --model-path models/Qwen3.5-0.8B-Q4_K_M.gguf
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path` | `models/Qwen3.5-0.8B-Q4_K_M.gguf` | Path to GGUF model |
| `--n-ctx` | `4096` | Context window size |
| `--n-threads` | `4` | CPU threads |
| `--max-tokens` | `1024` | Max tokens to generate |
| `--max-retries` | `3` | Max rollback retries |

## Expected Output

```
╔══════════════════════════════════════════════════════════════╗
║     IMECE — KV-Cache "Time Travel" Rollback Agent           ║
╚══════════════════════════════════════════════════════════════╝

Loading model: models/Qwen3.5-0.8B-Q4_K_M.gguf
✓ Model loaded
  Context  : 4096 tokens
  Rollback : 3 max retries

── Prompt ──────────────────────────────────────────────────
  Write a Python script that fetches JSON data from
  https://httpbin.org/json and prints the slideshow title.

── Agent Response (streaming) ────────────────────────────────

I'll write a Python script to fetch the JSON data.

<action type="python">
import requests
response = requests.get("https://httpbin.org/json")
data = response.json()
print(data["slideshow"]["title"])
</action>
Let me fix that — I'll use the standard library instead.

<action type="python">
import urllib.request
import json
with urllib.request.urlopen("https://httpbin.org/json") as response:
    data = json.loads(response.read().decode())
    print(data["slideshow"]["title"])
</action>

── Session Events ──────────────────────────────────────────

  [1] ❌ Action 'python' → Failed (exit_code=1)
      stderr: ModuleNotFoundError: No module named 'requests'

  [2] ⏪ KV-Cache Rollback! pos=87, erased=55 tokens, retry #1
      → Erased erroneous tokens from KV-Cache
      → Injected error observation at position 87
      → Resumed generation (zero prompt recalculation)

  [3] ✅ Action 'python' → Success
      stdout: Sample Slide Show

── Session Stats ───────────────────────────────────────────
  Tokens generated   : 178
  Time elapsed       : 8.72s (20.4 tok/s)
  Total rollbacks    : 1
  Total tokens erased: 55
  Session events     : 3

  🧠 The LLM corrected itself via KV-Cache "Time Travel":
     - 1 rollback(s) saved 55 tokens of re-computation
     - Zero prompt recalculation — instant recovery
     - No other framework can do this.
```

## How It Works (Technical Detail)

1. **`InferenceConfig.stop_sequences`** includes `</action>` as an *action boundary*. When the model generates this tag, token generation pauses.

2. **`ActionParser::extract_action()`** parses the `<action type="python">...</action>` block from the generated text.

3. **`ProcessExecutor::execute()`** runs `python3 -c <code>` with timeout enforcement and stdout/stderr capture.

4. **On failure**, `KvCacheController::rollback()` calls `llama_memory_seq_rm()` to erase the KV-Cache range `[t_k, t_n)`, truncates the token buffer, and injects the observation tokens.

5. **The generation loop continues** — the model "wakes up" after the observation and generates corrected code, never seeing its previous mistake in the KV-Cache.

## Key Insight

> The LLM experiences rollback as "thinking mid-sentence, realizing a mistake, and correcting it instantly." It never re-reads the prompt. The KV-Cache entries for the prompt are **preserved** — only the erroneous generation is erased. This is what makes it zero-cost.
