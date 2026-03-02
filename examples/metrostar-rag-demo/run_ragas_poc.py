#!/usr/bin/env python3
"""
RAG + Ragas Evaluation POC — fully local with Ollama + Phoenix
==============================================================

This script:
  1. Launches an Arize Phoenix server (traces + UI on port 6006)
  2. Downloads prompt-engineering papers from Hugging Face (cached)
  3. Builds a LlamaIndex VectorStoreIndex using Ollama embeddings
  4. Generates a Ragas synthetic testset (with PyMuPDF heading-aware extraction)
  5. Queries the RAG for every test question
  6. Runs Ragas evaluation (faithfulness, answer_correctness, context_recall, context_precision)
  7. Pushes evaluation annotations to Phoenix for visualization

Prerequisites
-------------
- Docker containers running:
    ollembed  →  0.0.0.0:8089 → 11434  (embedding model, e.g. mxbai-embed-large:335m)
    ollama    →  0.0.0.0:8090 → 11434  (LLM model, e.g. gemma3:4b)
- Conda env "arize" (or any env with the deps below installed):
    pip install ragas==0.4.3 rapidfuzz pypdf "arize-phoenix[llama-index,embeddings]" \\
        "openai>=1.0.0" pandas "httpx<0.28" "openinference-instrumentation>=0.1.38" \\
        langchain-openai langchain-community llama-index-llms-ollama \\
        llama-index-embeddings-ollama

Usage
-----
    python run_ragas_poc.py                      # defaults
    python run_ragas_poc.py --test-size 10        # more test questions
    python run_ragas_poc.py --clean               # wipe caches and regenerate everything

After the script finishes, Phoenix stays running at http://127.0.0.1:6006
Press Ctrl+C to shut it down.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OLLAMA_LLM_URL = "http://127.0.0.1:8090"
OLLAMA_EMBED_URL = "http://127.0.0.1:8089"
PHOENIX_PORT = 6007
DATA_DIR = Path("./prompt-engineering-papers")
CACHE_DIR = Path("./cache")
REPO_URL = "https://huggingface.co/datasets/explodinggradients/prompt-engineering-papers"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RAG + Ragas POC (Ollama + Phoenix)")
    p.add_argument("--test-size", type=int, default=5, help="Number of synthetic test questions")
    p.add_argument("--num-files", type=int, default=2, help="Max PDFs to ingest")
    p.add_argument("--clean", action="store_true", help="Wipe all caches before running")
    p.add_argument("--ollama-llm-url", default=OLLAMA_LLM_URL)
    p.add_argument("--ollama-embed-url", default=OLLAMA_EMBED_URL)
    p.add_argument("--phoenix-port", type=int, default=PHOENIX_PORT)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def pick_ollama_model(base_url: str, kind: str = "llm") -> str:
    """Auto-pick the smallest available model from an Ollama instance."""
    resp = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=10.0)
    resp.raise_for_status()
    models = resp.json().get("models", []) or []
    if kind == "embed":
        filtered = [m for m in models if "embed" in (m.get("name") or "").lower()]
    else:
        filtered = [m for m in models if "embed" not in (m.get("name") or "").lower()]
    if not filtered:
        avail = ", ".join(sorted({m.get("name", "?") for m in models}))
        raise RuntimeError(f"No {kind} models on {base_url}. Available: {avail}")
    filtered.sort(key=lambda m: m.get("size") or 0)
    return filtered[0]["name"]


def check_ollama(llm_url: str, embed_url: str) -> tuple[str, str]:
    """Verify both Ollama instances are reachable and pick models."""
    print("🔍 Checking Ollama containers …")
    llm_name = pick_ollama_model(llm_url, "llm")
    embed_name = pick_ollama_model(embed_url, "embed")
    print(f"   LLM   : {llm_name}  @ {llm_url}")
    print(f"   Embed : {embed_name} @ {embed_url}")
    return llm_name, embed_name


def download_dataset(data_dir: Path) -> None:
    """Clone the HuggingFace dataset (skip if already present)."""
    print("📥 Downloading dataset …")
    if data_dir.exists() and any(data_dir.iterdir()):
        print(f"   Already present at {data_dir.resolve()} — skipped")
        return
    subprocess.run(["git", "clone", REPO_URL, str(data_dir)], check=True)
    try:
        subprocess.run(["git", "-C", str(data_dir), "lfs", "pull"], check=True)
    except Exception as exc:
        print(f"   ⚠️  git lfs pull failed (may be OK): {exc}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    CACHE_DIR.mkdir(exist_ok=True)

    if args.clean:
        print("🧹 Wiping caches …")
        for p in CACHE_DIR.iterdir():
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()

    # 0 — Verify Ollama
    llm_model, embed_model = check_ollama(args.ollama_llm_url, args.ollama_embed_url)

    # 1 — Connect to Phoenix Docker (or launch embedded as fallback)
    import os
    import phoenix as px

    phoenix_url = f"http://127.0.0.1:{args.phoenix_port}"
    os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = phoenix_url

    # Check if Docker Phoenix is already running
    try:
        r = httpx.get(phoenix_url, timeout=5.0, follow_redirects=True)
        r.raise_for_status()
        print(f"\n✅ Phoenix Docker already running at {phoenix_url}")
    except Exception:
        print(f"\n⚠️  No Phoenix at {phoenix_url} — launching embedded Phoenix")
        os.environ.pop("PHOENIX_COLLECTOR_ENDPOINT", None)
        os.environ["PHOENIX_GRPC_PORT"] = "4318"
        session = px.launch_app(port=args.phoenix_port)
        phoenix_url = session.url
        print(f"   Phoenix UI: {phoenix_url}")

    # Instrument LlamaIndex + LangChain
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
    from phoenix.otel import register

    tp = register(endpoint=f"{phoenix_url}/v1/traces", set_global_tracer_provider=False)
    LlamaIndexInstrumentor().instrument(skip_dep_check=True, tracer_provider=tp)
    LangChainInstrumentor().instrument(skip_dep_check=True, tracer_provider=tp)

    # 2 — Download + load documents
    download_dataset(DATA_DIR)

    # Load PDFs once with PyMuPDF — replaces SimpleDirectoryReader (pypdf).
    # Same heading-rich text feeds both VectorStoreIndex and Ragas KG.
    from pymupdf_reader import load_pdfs_with_headings, load_pdfs_as_llamaindex_docs
    print(f"\n📄 Loading PDFs with heading extraction (max {args.num_files}) …")
    lc_docs = load_pdfs_with_headings(str(DATA_DIR), max_files=args.num_files)
    documents = load_pdfs_as_llamaindex_docs(str(DATA_DIR), max_files=args.num_files)
    for doc in lc_docs:
        nh = len(doc.metadata.get("headings", []))
        print(f"   📄 {doc.metadata['file_name'][:60]} — {len(doc.page_content)} chars, {nh} headings")
    print(f"   Loaded {len(documents)} LlamaIndex docs (PyMuPDF, heading-aware)")

    # 3 — Build / load RAG index
    from llama_index.core import Settings, StorageContext, VectorStoreIndex, load_index_from_storage
    from llama_index.embeddings.ollama import OllamaEmbedding
    from llama_index.llms.ollama import Ollama
    from openinference.instrumentation import dangerously_using_project

    embed = OllamaEmbedding(model_name=embed_model, base_url=args.ollama_embed_url)
    llm = Ollama(model=llm_model, base_url=args.ollama_llm_url, request_timeout=600.0)
    Settings.embed_model = embed
    Settings.llm = llm

    # Sanity-check embedding connectivity
    _ = embed.get_text_embedding("healthcheck")

    index_dir = CACHE_DIR / "vector_index_ollama"
    if index_dir.exists() and any(index_dir.iterdir()):
        print(f"\n♻️  Loading persisted index from {index_dir}")
        sc = StorageContext.from_defaults(persist_dir=str(index_dir))
        vector_index = load_index_from_storage(sc)
    else:
        print(f"\n🧱 Building vector index (persist → {index_dir}) …")
        with dangerously_using_project("indexing"):
            vector_index = VectorStoreIndex.from_documents(documents, embed_model=embed)
        index_dir.mkdir(parents=True, exist_ok=True)
        vector_index.storage_context.persist(persist_dir=str(index_dir))

    query_engine = vector_index.as_query_engine(similarity_top_k=2, llm=llm)

    # Quick smoke test
    print("\n🔎 Smoke test …")
    resp = query_engine.query("What is prompt engineering?")
    print(f"   A: {(resp.response or '')[:200]}…")

    # 4 — Testset generation (Ragas)
    testset_file = CACHE_DIR / "ragas_testset.parquet"
    if testset_file.exists():
        print(f"\n📂 Loading cached testset from {testset_file}")
        test_df = pd.read_parquet(testset_file)
    else:
        print(f"\n🧪 Generating synthetic testset ({args.test_size} questions) …")
        from ragas.testset import TestsetGenerator
        from ragas.testset.transforms.default import default_transforms

        generator = TestsetGenerator.from_llama_index(llm=llm, embedding_model=embed)

        # lc_docs from pymupdf_reader have Markdown heading markers in page_content,
        # so HeadlineSplitter can find headings via text.find() — no workaround needed.
        transforms = default_transforms(
            documents=lc_docs,
            llm=generator.llm,
            embedding_model=generator.embedding_model,
        )

        with dangerously_using_project("ragas-testset"):
            # Use generate_with_langchain_docs so KG nodes get the same
            # heading-rich PyMuPDF text that default_transforms was built from.
            testset = generator.generate_with_langchain_docs(
                lc_docs,
                testset_size=args.test_size,
                transforms=transforms,
                with_debugging_logs=True,
            )

        test_df = testset.to_pandas()
        test_df = (
            test_df.sort_values("user_input")
            .drop_duplicates(subset=["user_input"], keep="first")
            .reset_index(drop=True)
        )
        test_df.to_parquet(testset_file)
        print(f"   💾 Saved {len(test_df)} questions → {testset_file}")

    # Normalise column names
    if "user_input" in test_df.columns and "question" not in test_df.columns:
        test_df = test_df.rename(columns={"user_input": "question"})
    if "reference" in test_df.columns and "ground_truth" not in test_df.columns:
        test_df = test_df.rename(columns={"reference": "ground_truth"})
    print(f"   📊 Testset: {len(test_df)} rows, columns: {list(test_df.columns)}")

    # 5 — Generate RAG answers for each test question
    eval_cache = CACHE_DIR / "ragas_eval_dataset.parquet"
    if eval_cache.exists():
        print(f"\n📂 Loading cached eval dataset from {eval_cache}")
        ragas_evals_df = pd.read_parquet(eval_cache)
    else:
        from tqdm.auto import tqdm

        print(f"\n💬 Querying RAG for {len(test_df)} questions …")
        responses = []
        with dangerously_using_project("llama-index"):
            for q in tqdm(test_df["question"].values, desc="RAG queries"):
                r = query_engine.query(q)
                responses.append(
                    {
                        "answer": r.response,
                        "contexts": [c.node.get_content() for c in r.source_nodes],
                    }
                )

        ragas_evals_df = pd.DataFrame(
            {
                "question": test_df["question"].values,
                "answer": [r["answer"] for r in responses],
                "contexts": [r["contexts"] for r in responses],
                "ground_truth": test_df["ground_truth"].values.tolist(),
            }
        )
        ragas_evals_df.to_parquet(eval_cache)
        print(f"   💾 Saved eval dataset → {eval_cache}")

    from datasets import Dataset

    ragas_eval_dataset = Dataset.from_pandas(ragas_evals_df)

    # 6 — Ragas evaluation
    scores_cache = CACHE_DIR / "ragas_eval_scores.parquet"
    if scores_cache.exists():
        print(f"\n📂 Loading cached eval scores from {scores_cache}")
        eval_scores_df = pd.read_parquet(scores_cache)
    else:
        from langchain_community.chat_models import ChatOllama
        from langchain_community.embeddings import OllamaEmbeddings
        from ragas import evaluate
        from ragas.metrics import (
            answer_correctness,
            context_precision,
            context_recall,
            faithfulness,
        )

        ragas_llm = ChatOllama(base_url=args.ollama_llm_url, model=llm_model)
        ragas_embeddings = OllamaEmbeddings(base_url=args.ollama_embed_url, model=embed_model)

        print(f"\n📏 Running Ragas evaluation (judge: {llm_model}) …")
        with dangerously_using_project("ragas-evals"):
            result = evaluate(
                dataset=ragas_eval_dataset,
                metrics=[faithfulness, answer_correctness, context_recall, context_precision],
                llm=ragas_llm,
                embeddings=ragas_embeddings,
            )

        eval_scores_df = pd.DataFrame(result.scores)
        eval_scores_df.to_parquet(scores_cache)
        print(f"   💾 Saved scores → {scores_cache}")

    print("\n📊 Evaluation scores:")
    print(eval_scores_df.to_string(index=False))
    print(f"\n   Mean scores:")
    for col in eval_scores_df.columns:
        print(f"     {col}: {eval_scores_df[col].mean():.3f}")

    # 7 — Push annotations to Phoenix
    print(f"\n📤 Pushing annotations to Phoenix ({session.url}) …")
    try:
        from phoenix.trace.dsl.helpers import SpanQuery

        client = px.Client()
        time.sleep(2)  # wait for spans to become available

        spans_df = None
        try:
            from phoenix.session.evaluation import get_qa_with_reference
            spans_df = get_qa_with_reference(client, project_name="llama-index")
        except Exception as exc:
            print(f"   ⚠️  Could not fetch spans for annotation: {exc}")

        if spans_df is not None and not spans_df.empty:
            span_questions = (
                spans_df[["input"]]
                .sort_values("input")
                .drop_duplicates(subset=["input"], keep="first")
                .reset_index()
                .rename({"input": "question"}, axis=1)
            )
            eval_data_df = ragas_evals_df.merge(span_questions, on="question").set_index(
                "context.span_id"
            )
            eval_scores_df.index = eval_data_df.index

            import asyncio
            from phoenix.client import AsyncClient

            async def _push():
                px_client = AsyncClient()
                for col in eval_scores_df.columns:
                    evals = eval_scores_df[[col]].rename(columns={col: "score"})
                    await px_client.spans.log_span_annotations_dataframe(
                        dataframe=evals,
                        annotation_name=col,
                        annotator_kind="LLM",
                    )

            asyncio.run(_push())
            print("   ✅ Annotations pushed to Phoenix")
        else:
            print("   ⚠️  No spans found — skipping annotation push")
    except Exception as exc:
        print(f"   ⚠️  Annotation push failed (non-fatal): {exc}")

    # Done!
    print(f"\n{'='*60}")
    print(f"✅ POC complete!  Phoenix UI → {session.url}")
    print(f"{'='*60}")
    print("\nPress Ctrl+C to stop Phoenix and exit.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Shutting down Phoenix …")
        px.close_app()


if __name__ == "__main__":
    main()
