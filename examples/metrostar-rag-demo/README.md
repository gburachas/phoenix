# MetroStar RAG Demo

This example demonstrates how to build a **Retrieval-Augmented Generation (RAG)**
pipeline with observability and evaluation powered by **MetroStar Phoenix**.

## Overview

These scripts show end-to-end RAG workflows including:

- **Document ingestion** via PyMuPDF (`pymupdf_reader.py`)
- **RAG evaluation** with RAGAS metrics (`run_ragas_poc.py`, `run_ragas_test.py`)
- **Model evaluation** with Gemma 3 (`evaluate_gemma3.py`, `phoenix_rag_eval_gemma3.py`)
- **LLM-as-Judge** evaluation using GPT-5 (`evaluate_with_gpt5_judge.py`)
- **Phoenix observability** — tracing, upload, and analysis
  (`launch_phoenix.py`, `upload_to_phoenix.py`, `analyze_phoenix_traces.py`)
- **Multi-model comparison** (`run_all_models.sh`)
- **Interactive RAG chat** (`rag_chat.sh`)

## Prerequisites

```bash
# Install MetroStar Phoenix (from repo root)
pip install -e ".[dev]"

# Additional dependencies for RAG examples
pip install ragas llama-index pymupdf qdrant-client
```

## Quick Start

```bash
# 1. Start Phoenix server
python launch_phoenix.py

# 2. Run the RAGAS evaluation proof-of-concept
python run_ragas_poc.py

# 3. Upload results to Phoenix for visualization
python upload_to_phoenix.py

# 4. Analyze traces in Phoenix
python analyze_phoenix_traces.py
```

## File Descriptions

| File | Purpose |
|------|---------|
| `launch_phoenix.py` | Start a local Phoenix server for tracing |
| `pymupdf_reader.py` | PDF document reader for corpus ingestion |
| `run_ragas_poc.py` | RAGAS evaluation proof-of-concept |
| `run_ragas_test.py` | Extended RAGAS test pipeline |
| `evaluate_gemma3.py` | Evaluate Gemma 3 model responses |
| `phoenix_rag_eval_gemma3.py` | RAG evaluation with Phoenix observability |
| `evaluate_with_gpt5_judge.py` | LLM-as-Judge evaluation using GPT-5 |
| `upload_to_phoenix.py` | Upload evaluation results to Phoenix |
| `analyze_phoenix_traces.py` | Analyze collected traces in Phoenix |
| `run_all_models.sh` | Run evaluation across multiple models |
| `rag_chat.sh` | Interactive RAG chat session |

## Architecture

```
Corpus (PDFs/docs) → PyMuPDF Reader → Vector Store (Qdrant)
                                        ↓
User Query → LLM (Gemma3/GPT) → RAG Response
                                        ↓
                          Phoenix Tracing → Evaluation (RAGAS / LLM Judge)
                                        ↓
                          MetroStar Phoenix UI → Analysis Dashboard
```

## Related

- [MetroStar Phoenix Dev Plan](../../docs/) — Sprint roadmap
- [RAGAS Documentation](https://docs.ragas.io/) — Evaluation framework
- [Phoenix Documentation](https://docs.arize.com/phoenix) — Observability platform
