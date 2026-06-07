# 03 — Semantic Search

A persistent semantic search engine using LanceDB for vector storage and interactive query capabilities.

## What This Demonstrates

- `LanceMemoryStore` for persistent vector storage (disk-backed LanceDB)
- Batch document embedding and indexing
- Interactive semantic search loop via stdin
- Top-K retrieval with cosine similarity scoring
- Persistent data that survives process restarts

## Modules Used

| Module | Purpose |
|--------|---------|
| **Memory** (Module 1) | LanceMemoryStore for persistent vector storage |
| **Embedding** (Module 4) | Local vector generation via Voyage-4 Nano ONNX |

## Usage

### Index mode — embed and store documents

```bash
cargo run -p semantic-search -- --model-dir path/to/voyage-4-nano-onnx --index
```

This indexes a built-in knowledge base into a LanceDB store at `./lance_data/`.

### Search mode — interactive queries

```bash
cargo run -p semantic-search -- --model-dir path/to/voyage-4-nano-onnx --search
```

Enter queries interactively and get top-K results with similarity scores.

### Both (default)

```bash
cargo run -p semantic-search -- --model-dir path/to/voyage-4-nano-onnx
```

Indexes the knowledge base, then enters search mode.

## Expected Output

```
📚 Indexing 15 documents into LanceDB...
   ✓ Indexed 15 documents (256-d, Float32)

🔍 Entering interactive search mode. Type a query and press Enter.
   Type /quit to exit.

> How does Rust handle memory?

   Top-5 results:

    1. [0.8934] 🤖 "Rust's ownership model prevents data races at compile time."
    2. [0.8721] 🤖 "The borrow checker enforces exclusive or shared access to data."
    3. [0.8503] 🤖 "Lifetimes in Rust ensure references never outlive their data."
    ...
```
