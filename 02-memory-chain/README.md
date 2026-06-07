# 02 — Memory Chain (DMCE)

Build a contextual memory chain using the Dynamic Memory Chain Evolution (DMCE) algorithm with Adaptive Path Truncation (APT).

## What This Demonstrates

- Creating `MemoryNode`s with role-tagged embeddings
- Populating an in-memory `MemoryStore`
- Running the DMCE chain evolution algorithm with `evolve_with_diagnostics()`
- Visualizing gating scores and APT truncation behavior
- Comparing chain outputs across different `β` (APT threshold) values

## Modules Used

| Module | Purpose |
|--------|---------|
| **Memory** (Module 1) | MemoryStore, MemoryNode, DmceEngine |
| **Embedding** (Module 4) | Local vector generation via Voyage-4 Nano ONNX |

## Algorithm Summary

```
Flat-Index Store M → Top-K Candidates P → DMCE Iteration
  ┌─── For each step t:
  │    Compute chain centroid C_z^(t)
  │    Score: S_gate(m) = cos(m.e, q) × cos(m.e, C_z^(t))
  │    Select: m* = argmax S_gate
  │    APT check: s*_t < β × s_{t-1} → TRUNCATE
  └─── Output: ordered chain C_z
```

## Usage

```bash
cargo run -p memory-chain -- --model-dir path/to/voyage-4-nano-onnx
```

## Expected Output

The example runs three queries with different `β` values and displays the chain evolution step-by-step, showing how APT controls chain length.
