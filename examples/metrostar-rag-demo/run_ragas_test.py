#!/usr/bin/env python3
"""
Run RAGAS evaluation on the LLM prompt-engineering papers dataset.

Produces an HTML report with per-question scores and aggregate metrics.
Supports multiple LLM runs for side-by-side comparison.

Every stage can independently use Ollama or Azure OpenAI:

  Stage              Flag               Default
  ─────              ────               ───────
  RAG answering      --rag-llm          ollama   (--llm-model picks which)
  Testset generation --testset-llm      ollama
  Evaluation judge   --eval-llm         ollama
  Embeddings         --embed-provider   ollama   (--embed-model picks which)

Usage:
    python run_ragas_test.py                                          # defaults
    python run_ragas_test.py --test-size 10                            # sanity run
    python run_ragas_test.py --llm-model gemma3:12b                    # different RAG LLM
    python run_ragas_test.py --testset-llm azure                       # GPT-5 generates testset
    python run_ragas_test.py --eval-llm azure                          # GPT-5 as judge
    python run_ragas_test.py --rag-llm azure                           # Azure RAG (reference)
    python run_ragas_test.py --embed-provider azure                    # Azure embeddings
    python run_ragas_test.py --testset-llm azure --eval-llm azure \
                             --rag-llm ollama --llm-model gemma3:4b    # recommended combo
    python run_ragas_test.py --eval-llm ollama --eval-model gpt-oss:20b  # different judge model
    python run_ragas_test.py --testset-llm ollama --testset-model gpt-oss:20b  # different testset model
    python run_ragas_test.py --compare                                 # compare cached results
"""
from __future__ import annotations

import argparse
import datetime
import html as html_lib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from dotenv import load_dotenv

# Load .env for Azure OpenAI credentials (GPT5_MINI_*, GPT5_CHAT_*, etc.)
_env_file = Path(__file__).resolve().parent / ".env"
if _env_file.exists():
    load_dotenv(_env_file)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OLLAMA_LLM_URL = "http://127.0.0.1:8090"
OLLAMA_EMBED_URL = "http://127.0.0.1:8089"
PHOENIX_PORT = 6007
PHOENIX_GRPC_PORT = 4320          # Docker maps 4320 → 4317 inside container
PHOENIX_PROJECT = "Ragas test on LLM papers"

REPO_URL = "https://huggingface.co/datasets/explodinggradients/prompt-engineering-papers"
BASE_DIR = Path(__file__).resolve().parent
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
DATA_DIR = NOTEBOOKS_DIR / "prompt-engineering-papers"
CACHE_DIR = NOTEBOOKS_DIR / "cache"
RESULTS_DIR = BASE_DIR / "results"

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RAGAS evaluation on LLM papers dataset")
    p.add_argument("--test-size", type=int, default=20,
                   help="Number of synthetic test questions to generate (default: 20)")
    p.add_argument("--num-files", type=int, default=10,
                   help="Max PDF files to ingest (default: 10)")
    p.add_argument("--llm-model", type=str, default=None,
                   help="Ollama LLM model name (auto-detect if omitted)")
    p.add_argument("--embed-model", type=str, default=None,
                   help="Ollama embedding model name (auto-detect if omitted)")
    p.add_argument("--ollama-llm-url", default=OLLAMA_LLM_URL)
    p.add_argument("--ollama-embed-url", default=OLLAMA_EMBED_URL)
    p.add_argument("--phoenix-port", type=int, default=PHOENIX_PORT)
    p.add_argument("--output-html", nargs="?", const="auto", default=None,
                   help="Generate HTML report (optionally specify path; default auto-names it)")
    p.add_argument("--clean", action="store_true",
                   help="Wipe all caches before running")
    p.add_argument("--compare", action="store_true",
                   help="Generate comparison HTML from all cached results and exit")
    p.add_argument("--results-dir", type=str, default=None,
                   help="Override results directory (for --compare with subfolders)")
    p.add_argument("--compare-output", type=str, default=None,
                   help="Output path for comparison HTML (used with --compare)")
    p.add_argument("--skip-testset-cache", action="store_true",
                   help="Force regeneration of the testset (ignore cache)")

    # ── LLM / embedding backend selection ─────────────────────────────
    p.add_argument("--rag-llm", choices=["ollama", "azure"], default="ollama",
                   help="LLM backend for RAG question answering (default: ollama)")
    p.add_argument("--testset-llm", choices=["ollama", "azure"], default="ollama",
                   help="LLM backend for RAGAS testset generation (default: ollama)")
    p.add_argument("--eval-llm", choices=["ollama", "azure"], default="ollama",
                   help="LLM backend for RAGAS evaluation / judge (default: ollama)")
    p.add_argument("--embed-provider", choices=["ollama", "azure"], default="ollama",
                   help="Embedding backend for vector index & RAGAS (default: ollama)")
    p.add_argument("--testset-model", type=str, default=None,
                   help="Ollama model for testset generation when --testset-llm=ollama "
                        "(defaults to --llm-model; useful when testset LLM ≠ RAG LLM)")
    p.add_argument("--eval-model", type=str, default=None,
                   help="Ollama model for evaluation/judge when --eval-llm=ollama "
                        "(defaults to --llm-model; useful when judge LLM ≠ RAG LLM)")
    p.add_argument("--azure-model-prefix", default="GPT5_MINI",
                   help="Env-var prefix for Azure LLM (GPT5_MINI, GPT5_CHAT, …)")
    p.add_argument("--azure-embed-model", default="text-embedding-3-small",
                   help="Azure OpenAI embedding deployment name")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pick_ollama_model(base_url: str, kind: str = "llm") -> str:
    """Auto-pick an Ollama model from /api/tags (smallest by size)."""
    r = httpx.get(f"{base_url}/api/tags", timeout=10.0)
    r.raise_for_status()
    models = r.json().get("models", [])
    if not models:
        raise RuntimeError(f"No models found at {base_url}")
    if kind == "embed":
        embed_models = [m for m in models if "embed" in m["name"].lower()]
        if embed_models:
            models = embed_models
    else:
        models = [m for m in models if "embed" not in m["name"].lower()]
    models.sort(key=lambda m: m.get("size", float("inf")))
    chosen = models[0]["name"]
    print(f"   Auto-picked {kind} model: {chosen}")
    return chosen


def check_ollama(llm_url: str, embed_url: str,
                 llm_model: str | None = None,
                 embed_model: str | None = None) -> tuple[str, str]:
    """Verify both Ollama instances are reachable and pick models."""
    print("🔍 Checking Ollama instances …")
    for label, url in [("LLM", llm_url), ("Embed", embed_url)]:
        r = httpx.get(url, timeout=5.0)
        r.raise_for_status()
        print(f"   ✅ {label} @ {url}")
    if not llm_model:
        llm_model = pick_ollama_model(llm_url, "llm")
    else:
        print(f"   Using LLM model: {llm_model}")
    if not embed_model:
        embed_model = pick_ollama_model(embed_url, "embed")
    else:
        print(f"   Using embed model: {embed_model}")
    return llm_model, embed_model


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


def connect_phoenix(port: int, project_name: str = PHOENIX_PROJECT) -> str:
    """Connect to Phoenix Docker or launch embedded as fallback. Returns URL."""
    import phoenix as px

    phoenix_url = f"http://127.0.0.1:{port}"
    os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = phoenix_url
    os.environ["PHOENIX_PROJECT_NAME"] = project_name

    try:
        r = httpx.get(phoenix_url, timeout=5.0, follow_redirects=True)
        r.raise_for_status()
        print(f"\n✅ Phoenix Docker running at {phoenix_url}")
    except Exception:
        print(f"\n⚠️  No Phoenix at {phoenix_url} — launching embedded Phoenix")
        os.environ.pop("PHOENIX_COLLECTOR_ENDPOINT", None)
        os.environ["PHOENIX_GRPC_PORT"] = "4318"
        session = px.launch_app(port=port)
        phoenix_url = session.url
        print(f"   Phoenix UI: {phoenix_url}")

    return phoenix_url


def instrument(phoenix_url: str, project_name: str = PHOENIX_PROJECT) -> None:
    """Set up OpenTelemetry tracing to Phoenix.

    Key fixes vs. earlier versions:
    * Uses ``set_global_tracer_provider=True`` so that *all* LlamaIndex /
      LangChain spans are captured (including RAGAS internal LLM calls).
    * Sends traces to the HTTP protobuf endpoint (``/v1/traces``) which
      the Phoenix Docker container exposes on the *UI port* (6007→6006).
    * Each experiment combo gets its own ``project_name`` so traces are
      grouped separately in the Phoenix UI.
    """
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
    from phoenix.otel import register

    # Uninstrument first in case this is called more than once in the same
    # process (e.g. when run_all_models.sh sources the script repeatedly).
    try:
        LlamaIndexInstrumentor().uninstrument()
    except Exception:
        pass
    try:
        LangChainInstrumentor().uninstrument()
    except Exception:
        pass

    tp = register(
        project_name=project_name,
        endpoint=f"{phoenix_url}/v1/traces",
        protocol="http/protobuf",
        set_global_tracer_provider=True,   # ← critical: makes spans visible
    )
    LlamaIndexInstrumentor().instrument(skip_dep_check=True, tracer_provider=tp)
    LangChainInstrumentor().instrument(skip_dep_check=True, tracer_provider=tp)
    print(f"   🔭 Traces → {phoenix_url}/v1/traces  (project: {project_name})")
    print(f"   🔭 Phoenix UI → {phoenix_url}")


# ---------------------------------------------------------------------------
# HTML report generation
# ---------------------------------------------------------------------------

METRICS = ["faithfulness", "answer_correctness", "context_recall", "context_precision"]


def _score_color(val: float) -> str:
    """Return a CSS color for a metric value in [0,1]."""
    if val >= 0.8:
        return "#2d8a4e"
    if val >= 0.5:
        return "#c9a227"
    return "#c0392b"


def _truncate(text: str, max_len: int = 300) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "…"


def generate_html_report(
    eval_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    llm_model: str,
    embed_model: str,
    test_size: int,
    duration_sec: float,
    output_path: str,
    testset_llm: str = "",
    judge_llm: str = "",
    phoenix_project: str = PHOENIX_PROJECT,
) -> str:
    """Generate a self-contained HTML report for the RAGAS evaluation run."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n_questions = len(scores_df)

    # Aggregate stats
    agg: dict[str, float] = {}
    for m in METRICS:
        if m in scores_df.columns:
            agg[m] = scores_df[m].mean()

    # Build per-question rows
    rows_html = []
    for i in range(n_questions):
        q_row = eval_df.iloc[i] if i < len(eval_df) else {}
        s_row = scores_df.iloc[i]

        question = html_lib.escape(str(q_row.get("question", "")))
        answer = html_lib.escape(_truncate(str(q_row.get("answer", ""))))
        ground_truth = html_lib.escape(_truncate(str(q_row.get("ground_truth", ""))))

        contexts_raw = q_row.get("contexts", [])
        if isinstance(contexts_raw, str):
            try:
                contexts_raw = json.loads(contexts_raw)
            except Exception:
                contexts_raw = [contexts_raw]
        if contexts_raw is None:
            contexts_raw = []
        contexts_html = "<br>".join(
            f"<em>{html_lib.escape(_truncate(str(c), 200))}</em>"
            for c in contexts_raw
        )

        score_cells = []
        for m in METRICS:
            v = s_row.get(m, float("nan"))
            if pd.isna(v):
                score_cells.append('<td style="text-align:center;">—</td>')
            else:
                color = _score_color(v)
                score_cells.append(
                    f'<td style="text-align:center; color:{color}; font-weight:bold;">'
                    f"{v:.3f}</td>"
                )

        rows_html.append(f"""
        <tr>
            <td>{i+1}</td>
            <td style="max-width:300px;">{question}</td>
            <td style="max-width:250px;">{answer}</td>
            <td style="max-width:250px;">{ground_truth}</td>
            <td style="max-width:250px; font-size:0.85em;">{contexts_html}</td>
            {"".join(score_cells)}
        </tr>""")

    # Aggregate row
    agg_cells = []
    for m in METRICS:
        v = agg.get(m, float("nan"))
        if pd.isna(v):
            agg_cells.append('<td style="text-align:center;">—</td>')
        else:
            color = _score_color(v)
            agg_cells.append(
                f'<td style="text-align:center; color:{color}; font-weight:bold; font-size:1.1em;">'
                f"{v:.3f}</td>"
            )

    report_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>RAGAS Evaluation — {html_lib.escape(llm_model)}</title>
<style>
  body {{ font-family: 'Segoe UI', Tahoma, Geneva, sans-serif; margin: 2em; background: #f7f8fa; }}
  h1 {{ color: #2c3e50; }}
  .meta {{ background: #fff; border-radius: 8px; padding: 1em 1.5em; margin-bottom: 1.5em;
           box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .meta table {{ border-collapse: collapse; }}
  .meta td {{ padding: 4px 16px 4px 0; }}
  .meta td:first-child {{ font-weight: bold; color: #555; }}
  .agg {{ background: #fff; border-radius: 8px; padding: 1em 1.5em; margin-bottom: 1.5em;
          box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .agg table {{ border-collapse: collapse; width: auto; }}
  .agg th, .agg td {{ padding: 8px 20px; text-align: center; }}
  .agg th {{ background: #34495e; color: #fff; }}
  table.results {{ border-collapse: collapse; width: 100%; background: #fff;
                   box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  table.results th {{ background: #34495e; color: #fff; padding: 10px 12px;
                      text-align: left; position: sticky; top: 0; }}
  table.results td {{ padding: 8px 12px; border-bottom: 1px solid #ecf0f1;
                      vertical-align: top; word-break: break-word; }}
  table.results tr:hover {{ background: #eef2f7; }}
  .footer {{ margin-top: 2em; color: #999; font-size: 0.85em; }}
</style>
</head>
<body>
<h1>🧪 RAGAS Evaluation Report</h1>

<div class="meta">
<table>
  <tr><td>LLM model (RAG)</td><td>{html_lib.escape(llm_model)}</td></tr>
  <tr><td>Testset generator</td><td>{html_lib.escape(testset_llm or llm_model)}</td></tr>
  <tr><td>Evaluation judge</td><td>{html_lib.escape(judge_llm or llm_model)}</td></tr>
  <tr><td>Embedding model</td><td>{html_lib.escape(embed_model)}</td></tr>
  <tr><td>Questions evaluated</td><td>{n_questions}</td></tr>
  <tr><td>Test size requested</td><td>{test_size}</td></tr>
  <tr><td>Duration</td><td>{duration_sec:.1f}s ({duration_sec/60:.1f} min)</td></tr>
  <tr><td>Timestamp</td><td>{timestamp}</td></tr>
  <tr><td>Phoenix project</td><td>{html_lib.escape(phoenix_project)}</td></tr>
</table>
</div>

<h2>📊 Aggregate Scores</h2>
<div class="agg">
<table>
  <tr>{"".join(f'<th>{m}</th>' for m in METRICS)}</tr>
  <tr>{"".join(agg_cells)}</tr>
</table>
</div>

<h2>📝 Per-Question Results</h2>
<table class="results">
<tr>
  <th>#</th><th>Question</th><th>Answer</th><th>Ground Truth</th><th>Contexts</th>
  {"".join(f'<th>{m}</th>' for m in METRICS)}
</tr>
{"".join(rows_html)}
</table>

<div class="footer">
  Generated by run_ragas_test.py — {timestamp}
</div>
</body>
</html>"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(report_html, encoding="utf-8")
    print(f"   📄 HTML report → {output_path}")
    return output_path


def _parse_score_filename(stem: str) -> dict[str, str]:
    """Parse a scores parquet filename into its axis labels.

    Filename pattern: scores_{embed_tag}_ts-{testset_tag}_{model_tag}_judge-{judge_tag}
    Delimiters ``_ts-`` and ``_judge-`` are unambiguous separators.
    """
    rest = stem.removeprefix("scores_")
    # Split on _ts- (embed tag is everything before it)
    ts_marker = "_ts-"
    ts_idx = rest.index(ts_marker)
    embed_tag = rest[:ts_idx]
    after_ts = rest[ts_idx + len(ts_marker):]
    # Split on _judge- (judge tag is everything after the *last* occurrence)
    judge_marker = "_judge-"
    judge_idx = after_ts.rindex(judge_marker)
    ts_and_model = after_ts[:judge_idx]
    judge_tag = after_ts[judge_idx + len(judge_marker):]
    # Separate testset_tag from model_tag.
    # model_tag never contains "_ts-" or "_judge-", so it's the rightmost segment
    # after the last known model pattern.  Pragmatic: split on last known RAG model.
    # Fall back to splitting on the first "_" if nothing else matches.
    known_rag = ["gemma3_4b", "gemma3_12b", "gemma3_27b", "gpt-oss_20b", "gpt-oss_120b"]
    testset_tag = ts_and_model  # default: whole thing
    model_tag = ""
    for rag in known_rag:
        suffix = f"_{rag}"
        if ts_and_model.endswith(suffix):
            testset_tag = ts_and_model[: -len(suffix)]
            model_tag = rag
            break
        if ts_and_model == rag:
            testset_tag = ""
            model_tag = rag
            break
    if not model_tag:
        # Fallback: everything after the last _ that looks like a model
        parts = ts_and_model.rsplit("_", 1)
        testset_tag, model_tag = (parts[0], parts[1]) if len(parts) == 2 else ("", ts_and_model)

    return {
        "embed_tag": embed_tag,
        "testset_tag": testset_tag,
        "model_tag": model_tag,
        "judge_tag": judge_tag,
    }


def _tag_to_label(tag: str, axis: str) -> str:
    """Convert a filename tag to a human-readable label.

    Examples:
        azure                        → Azure GPT-5-mini   (for testset/judge)
        gpt-oss_20b                  → Ollama gpt-oss:20b
        azure_text-embedding-3-small → Azure text-embedding-3-small
        ollama_mxbai-embed-large_335m → Ollama mxbai-embed-large:335m
        gemma3_4b                    → gemma3:4b
    """
    if axis == "embed":
        if tag.startswith("azure_"):
            return "Azure " + tag.removeprefix("azure_")
        if tag.startswith("ollama_"):
            raw = tag.removeprefix("ollama_")
            # Restore last _ → : for model size (335m, etc.)
            parts = raw.rsplit("_", 1)
            if len(parts) == 2 and parts[1].endswith("m") and parts[1][:-1].isdigit():
                return f"Ollama {parts[0]}:{parts[1]}"
            return f"Ollama {raw}"
        return tag
    if axis in ("testset", "judge"):
        if tag == "azure":
            return "Azure GPT-5-mini"
        # Ollama model tag: gpt-oss_20b → gpt-oss:20b
        return "Ollama " + tag.replace("_", ":")
    # RAG model
    return tag.replace("_", ":")


def generate_comparison_html(results_dir: Path, output_path: str | None = None) -> str:
    """Generate a comparison HTML report across all cached model results.

    The table shows separate columns for Embeddings, Testset LLM, and Judge LLM
    (RAG LLM is constant and noted in the header).  Each metric shows mean ± SE.
    """
    import numpy as np

    result_files = sorted(results_dir.glob("scores_*.parquet"))
    if not result_files:
        # Also search one level deeper (subfolder convention)
        result_files = sorted(results_dir.rglob("scores_*.parquet"))
    if not result_files:
        print("❌ No result files found for comparison.")
        sys.exit(1)

    all_runs: list[dict[str, Any]] = []
    rag_models: set[str] = set()

    for sf in result_files:
        parsed = _parse_score_filename(sf.stem)
        embed_label = _tag_to_label(parsed["embed_tag"], "embed")
        ts_label = _tag_to_label(parsed["testset_tag"], "testset")
        judge_label = _tag_to_label(parsed["judge_tag"], "judge")
        rag_label = _tag_to_label(parsed["model_tag"], "rag")
        rag_models.add(rag_label)

        scores_df = pd.read_parquet(sf)
        agg: dict[str, float] = {}
        se: dict[str, float] = {}
        n_valid: dict[str, int] = {}
        for m in METRICS:
            if m in scores_df.columns:
                vals = scores_df[m].dropna()
                n = len(vals)
                n_valid[m] = n
                agg[m] = float(vals.mean()) if n > 0 else float("nan")
                se[m] = float(vals.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
            else:
                agg[m] = float("nan")
                se[m] = float("nan")
                n_valid[m] = 0
        all_runs.append({
            "embed": embed_label,
            "testset": ts_label,
            "judge": judge_label,
            "rag": rag_label,
            "n": len(scores_df),
            "agg": agg,
            "se": se,
            "n_valid": n_valid,
        })

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rag_note = ", ".join(sorted(rag_models)) if rag_models else "unknown"

    # ── Build HTML rows ───────────────────────────────────────────────
    rows_html = []
    for run in all_runs:
        cells = []
        for m in METRICS:
            v = run["agg"].get(m, float("nan"))
            s = run["se"].get(m, float("nan"))
            nv = run["n_valid"].get(m, 0)
            if pd.isna(v):
                cells.append('<td style="text-align:center;">—</td>')
            else:
                color = _score_color(v)
                se_str = f" ± {s:.3f}" if not pd.isna(s) else ""
                n_str = f'<span class="n-valid">({nv})</span>' if nv < run["n"] else ""
                cells.append(
                    f'<td style="text-align:center; color:{color}; font-weight:bold;">'
                    f'{v:.3f}<span class="se">{se_str}</span> {n_str}</td>'
                )
        rows_html.append(
            f'<tr>'
            f'<td>{html_lib.escape(run["embed"])}</td>'
            f'<td>{html_lib.escape(run["testset"])}</td>'
            f'<td>{html_lib.escape(run["judge"])}</td>'
            f'<td style="text-align:center;">{run["n"]}</td>'
            f'{"".join(cells)}</tr>'
        )

    metric_headers = "".join(f"<th>{m}</th>" for m in METRICS)

    compare_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>RAGAS — 8-Way Model Comparison</title>
<style>
  body {{ font-family: 'Segoe UI', Tahoma, Geneva, sans-serif; margin: 2em; background: #f7f8fa; }}
  h1 {{ color: #2c3e50; }}
  .subtitle {{ color: #7f8c8d; margin-top: -0.8em; margin-bottom: 1.5em; }}
  table {{ border-collapse: collapse; background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.1); width: auto; }}
  th {{ background: #34495e; color: #fff; padding: 10px 16px; text-align: center; white-space: nowrap; }}
  th.axis {{ background: #2c3e50; }}
  td {{ padding: 8px 14px; border-bottom: 1px solid #ecf0f1; white-space: nowrap; }}
  tr:hover {{ background: #eef2f7; }}
  .se {{ font-weight: normal; font-size: 0.85em; color: #7f8c8d; }}
  .n-valid {{ font-weight: normal; font-size: 0.75em; color: #bbb; }}
  .footer {{ margin-top: 2em; color: #999; font-size: 0.85em; }}
  .legend {{ margin-top: 1em; margin-bottom: 1.5em; color: #7f8c8d; font-size: 0.85em; }}
</style>
</head>
<body>
<h1>📊 RAGAS — 8-Way Model Comparison</h1>
<p class="subtitle">RAG LLM: <b>{html_lib.escape(rag_note)}</b> (constant across all experiments) &nbsp;·&nbsp; Requested N=10, generated N=12</p>
<p class="legend">Values shown as <b>mean ± SE</b> (standard error). Coloured by mean score:
  <span style="color:#2d8a4e; font-weight:bold;">≥0.8</span> /
  <span style="color:#c9a227; font-weight:bold;">≥0.5</span> /
  <span style="color:#c0392b; font-weight:bold;">&lt;0.5</span>.
  <span class="n-valid">(k)</span> = only k of N rows had valid (non-NaN) scores.
</p>
<table>
<tr>
  <th class="axis">Embeddings</th>
  <th class="axis">Testset LLM</th>
  <th class="axis">Judge LLM</th>
  <th>N</th>
  {metric_headers}
</tr>
{"".join(rows_html)}
</table>
<div class="footer">Generated by run_ragas_test.py --compare — {timestamp}</div>
</body>
</html>"""

    if output_path is None:
        output_path = str(results_dir / "comparison.html")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(compare_html, encoding="utf-8")
    print(f"   📄 Comparison HTML → {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Azure OpenAI helpers
# ---------------------------------------------------------------------------

def _get_azure_creds(prefix: str) -> tuple[str, str, str, str]:
    """Read Azure OpenAI credentials from env-vars.

    Returns (endpoint, api_key, deployment, api_version).
    """
    endpoint = os.environ.get(f"{prefix}_ENDPOINT")
    api_key = os.environ.get(f"{prefix}_KEY")
    deployment = os.environ.get(f"{prefix}_DEPLOYMENT")
    api_version = os.environ.get(f"{prefix}_API_VERSION", "2024-12-01-preview")

    missing = [k for k, v in {
        f"{prefix}_ENDPOINT": endpoint,
        f"{prefix}_KEY": api_key,
        f"{prefix}_DEPLOYMENT": deployment,
    }.items() if not v]
    if missing:
        raise RuntimeError(
            f"Azure OpenAI env-vars not set: {', '.join(missing)}. "
            f"Add them to {_env_file} or export directly."
        )
    return endpoint, api_key, deployment, api_version  # type: ignore[return-value]


def _make_azure_lc_llm(args: argparse.Namespace):
    """Create a LangChain AzureChatOpenAI from .env vars.

    GPT-5 family models are *reasoning* models that only accept
    ``temperature=1``.  RAGAS internally overrides the temperature
    attribute (e.g. to 0.01) via direct ``llm.temperature = 0.01``
    which triggers a 400 error.  We override ``__setattr__`` so any
    temperature write is silently clamped to 1.
    """
    from langchain_openai import AzureChatOpenAI

    class _FixedTempAzureChat(AzureChatOpenAI):  # type: ignore[misc]
        """AzureChatOpenAI that forces temperature=1 (reasoning models)."""

        def __setattr__(self, name: str, value: object) -> None:
            if name == "temperature":
                value = 1.0  # reasoning models only accept 1
            super().__setattr__(name, value)

    endpoint, api_key, deployment, api_version = _get_azure_creds(args.azure_model_prefix)
    llm = _FixedTempAzureChat(
        azure_endpoint=endpoint.rstrip("/"),
        api_key=api_key,  # type: ignore[arg-type]
        azure_deployment=deployment,
        api_version=api_version,
        temperature=1.0,
        timeout=120,
    )
    print(f"   🔑 Azure LLM (LC): {deployment} @ {endpoint[:50]}…")
    return llm


def _make_azure_li_llm(args: argparse.Namespace):
    """Create a LlamaIndex AzureOpenAI LLM for the query engine."""
    from llama_index.llms.azure_openai import AzureOpenAI as LI_AzureOpenAI

    endpoint, api_key, deployment, api_version = _get_azure_creds(args.azure_model_prefix)
    llm = LI_AzureOpenAI(
        azure_endpoint=endpoint.rstrip("/"),
        api_key=api_key,
        azure_deployment=deployment,
        api_version=api_version,
        temperature=1.0,   # reasoning models only accept temperature=1
        timeout=120,
    )
    print(f"   🔑 Azure LLM (LI): {deployment} @ {endpoint[:50]}…")
    return llm


def _make_embeddings(args: argparse.Namespace):
    """Create embedding objects for both LlamaIndex and LangChain.

    Returns ``(li_embed, lc_embed, embed_label)`` where:
    * ``li_embed`` is a LlamaIndex BaseEmbedding (for VectorStoreIndex)
    * ``lc_embed`` is a LangChain Embeddings   (for RAGAS evaluate)
    * ``embed_label`` is a human-readable string like ``ollama/mxbai-embed-large:335m``
    """
    if args.embed_provider == "azure":
        from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
        from langchain_openai import AzureOpenAIEmbeddings

        # Prefer dedicated AZURE_EMBED_* env-vars; fall back to the LLM
        # prefix vars for backwards compatibility.
        pfx = args.azure_model_prefix
        endpoint = os.environ.get("AZURE_EMBED_ENDPOINT") or os.environ.get(f"{pfx}_ENDPOINT", "")
        api_key  = os.environ.get("AZURE_EMBED_KEY") or os.environ.get(f"{pfx}_KEY", "")
        api_version = os.environ.get("AZURE_EMBED_API_VERSION") or os.environ.get(f"{pfx}_API_VERSION", "2024-12-01-preview")
        deploy = os.environ.get("AZURE_EMBED_DEPLOYMENT") or args.azure_embed_model

        li_embed = AzureOpenAIEmbedding(
            azure_endpoint=endpoint.rstrip("/"),
            api_key=api_key,
            azure_deployment=deploy,
            api_version=api_version,
        )
        lc_embed = AzureOpenAIEmbeddings(
            azure_endpoint=endpoint.rstrip("/"),
            api_key=api_key,
            azure_deployment=deploy,
            api_version=api_version,
        )
        label = f"azure/{deploy}"
        print(f"   🔑 Azure Embeddings: {deploy} @ {endpoint[:50]}…")
    else:
        from llama_index.embeddings.ollama import OllamaEmbedding
        from langchain_community.embeddings import OllamaEmbeddings

        embed_model = args.embed_model or pick_ollama_model(args.ollama_embed_url, "embed")
        li_embed = OllamaEmbedding(model_name=embed_model, base_url=args.ollama_embed_url)
        lc_embed = OllamaEmbeddings(base_url=args.ollama_embed_url, model=embed_model)
        label = f"ollama/{embed_model}"
        print(f"   Using Ollama embeddings: {embed_model}")

    return li_embed, lc_embed, label


def _make_azure_llm_and_embed(args: argparse.Namespace):
    """Create LangChain AzureChatOpenAI + embeddings (convenience wrapper).

    The embedding provider respects ``--embed-provider``.
    """
    azure_llm = _make_azure_lc_llm(args)
    _, lc_embed, _ = _make_embeddings(args)
    return azure_llm, lc_embed


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_ragas_test(args: argparse.Namespace) -> pd.DataFrame:
    """Full RAGAS evaluation pipeline. Returns the scores DataFrame."""
    t0 = time.time()
    CACHE_DIR.mkdir(exist_ok=True)

    # Allow per-run results directory (for subfolder convention)
    results_dir = Path(args.results_dir) if args.results_dir else RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.clean:
        print("🧹 Wiping caches …")
        for d in [CACHE_DIR, results_dir]:
            for p in d.iterdir():
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink()

    # ── 0. Check Ollama (always needed for at least embeddings/RAG) ─────
    llm_model, embed_model = check_ollama(
        args.ollama_llm_url, args.ollama_embed_url,
        args.llm_model, args.embed_model,
    )

    # Resolve per-stage ollama model names (testset / eval may differ from RAG)
    testset_ollama_model = args.testset_model or llm_model
    eval_ollama_model = args.eval_model or llm_model

    # Build a cache-key tag that captures *what* answered the questions.
    # When --rag-llm=azure the tag includes the Azure deployment name.
    if args.rag_llm == "azure":
        _az_ep, _az_key, _az_dep, _az_ver = _get_azure_creds(args.azure_model_prefix)
        rag_label = f"azure-{_az_dep}"
    else:
        rag_label = llm_model
    model_tag = rag_label.replace(":", "_").replace("/", "_")

    # Testset LLM label (for cache keys & display)
    if args.testset_llm == "azure":
        testset_label = f"Azure ({args.azure_model_prefix})"
        testset_tag = "azure"
    else:
        testset_label = f"Ollama ({testset_ollama_model})"
        testset_tag = testset_ollama_model.replace(":", "_").replace("/", "_")

    # Eval / judge LLM label
    if args.eval_llm == "azure":
        eval_label = f"Azure ({args.azure_model_prefix})"
        judge_tag = "azure"
    else:
        eval_label = f"Ollama ({eval_ollama_model})"
        judge_tag = eval_ollama_model.replace(":", "_").replace("/", "_")

    # ── 1. Phoenix (per-experiment project for trace grouping) ────────
    # Build a human-readable project name from the combo:
    #   e.g. "embed=azure ts=azure rag=gemma3:4b judge=azure"
    _embed_short = "azure" if args.embed_provider == "azure" else "ollama"
    _ts_short = "azure" if args.testset_llm == "azure" else testset_ollama_model
    _judge_short = "azure" if args.eval_llm == "azure" else eval_ollama_model
    _rag_short = rag_label
    experiment_project = (
        f"embed={_embed_short} ts={_ts_short} "
        f"rag={_rag_short} judge={_judge_short}"
    )
    phoenix_url = connect_phoenix(args.phoenix_port, project_name=experiment_project)
    instrument(phoenix_url, project_name=experiment_project)

    # ── 2. Dataset ───────────────────────────────────────────────────────
    download_dataset(DATA_DIR)

    # Load PDFs once with PyMuPDF heading-aware reader — replaces both
    # SimpleDirectoryReader (pypdf, loses headings) and separate LangChain loader.
    # PyMuPDF produces structured text with Markdown heading markers that
    # HeadlineSplitter can locate via text.find().
    from pymupdf_reader import load_pdfs_with_headings, load_pdfs_as_llamaindex_docs

    print(f"\n📄 Loading PDFs with heading extraction (max {args.num_files}) …")
    lc_docs = load_pdfs_with_headings(str(DATA_DIR), max_files=args.num_files)
    li_docs = load_pdfs_as_llamaindex_docs(str(DATA_DIR), max_files=args.num_files)
    for doc in lc_docs:
        nh = len(doc.metadata.get("headings", []))
        print(f"   📄 {doc.metadata['file_name'][:60]} — {len(doc.page_content)} chars, {nh} headings")
    print(f"   Loaded {len(li_docs)} LlamaIndex docs (PyMuPDF, heading-aware)")

    # ── 3. Embeddings + RAG LLM + vector index ───────────────────────────
    from llama_index.core import Settings, StorageContext, VectorStoreIndex, load_index_from_storage

    # -- embeddings (ollama or azure) --
    li_embed, lc_embed, embed_label = _make_embeddings(args)
    Settings.embed_model = li_embed
    embed_tag = embed_label.replace("/", "_").replace(":", "_")

    # -- RAG LLM (ollama or azure) --
    if args.rag_llm == "azure":
        li_llm = _make_azure_li_llm(args)
        rag_llm_label = f"Azure ({args.azure_model_prefix})"
    else:
        from llama_index.llms.ollama import Ollama
        li_llm = Ollama(model=llm_model, base_url=args.ollama_llm_url, request_timeout=600.0)
        rag_llm_label = f"Ollama ({llm_model})"
    Settings.llm = li_llm
    print(f"   RAG LLM: {rag_llm_label}")

    # Sanity-check embedding
    _ = li_embed.get_text_embedding("healthcheck")

    # Vector index keyed by embed provider so azure & ollama caches don't clash
    index_dir = CACHE_DIR / f"vector_index_{embed_tag}"
    index_meta = index_dir / "_meta.json"
    rebuild_index = True
    if index_dir.exists() and any(index_dir.iterdir()) and index_meta.exists():
        meta = json.loads(index_meta.read_text())
        if meta.get("num_docs") == len(li_docs):
            rebuild_index = False

    if not rebuild_index:
        print(f"\n♻️  Loading persisted index from {index_dir}")
        sc = StorageContext.from_defaults(persist_dir=str(index_dir))
        vector_index = load_index_from_storage(sc)
    else:
        if index_dir.exists():
            shutil.rmtree(index_dir)
        print(f"\n🧱 Building vector index ({len(li_docs)} docs → {index_dir}) …")
        vector_index = VectorStoreIndex.from_documents(li_docs, embed_model=li_embed)
        index_dir.mkdir(parents=True, exist_ok=True)
        vector_index.storage_context.persist(persist_dir=str(index_dir))
        index_meta.write_text(json.dumps({"num_docs": len(li_docs)}))

    query_engine = vector_index.as_query_engine(similarity_top_k=2, llm=li_llm)

    # Smoke test
    print("\n🔎 Smoke test …")
    resp = query_engine.query("What is prompt engineering?")
    print(f"   A: {(resp.response or '')[:200]}…")

    # ── 4. Testset generation ────────────────────────────────────────────
    testset_file = CACHE_DIR / f"ragas_testset_{testset_tag}.parquet"
    if testset_file.exists() and not args.skip_testset_cache:
        print(f"\n📂 Loading cached testset from {testset_file}")
        test_df = pd.read_parquet(testset_file)
    else:
        print(f"\n🧪 Generating synthetic testset ({args.test_size} questions) …")
        from ragas.testset import TestsetGenerator
        from ragas.testset.transforms.default import default_transforms
        from ragas.run_config import RunConfig
        from ragas.exceptions import RagasOutputParserException
        from langchain_core.exceptions import OutputParserException as LCOutputParserException

        # ── Choose LLM backend for testset generation ────────────────
        if args.testset_llm == "azure":
            testset_lc_llm, testset_lc_embed = _make_azure_llm_and_embed(args)
            generator = TestsetGenerator.from_langchain(
                llm=testset_lc_llm,
                embedding_model=testset_lc_embed,
            )
            print(f"   Testset generator LLM: Azure OpenAI ({args.azure_model_prefix})")
        else:
            # Use the dedicated testset model (may differ from the RAG LLM)
            from llama_index.llms.ollama import Ollama as _OllamaLLM
            testset_li_llm = _OllamaLLM(
                model=testset_ollama_model, base_url=args.ollama_llm_url,
                request_timeout=600.0,
            )
            generator = TestsetGenerator.from_llama_index(
                llm=testset_li_llm, embedding_model=li_embed,
            )
            print(f"   Testset generator LLM: Ollama ({testset_ollama_model})")

        # Generous timeouts — local Ollama on large papers (737K+ chars)
        # or Azure with rate-limiting.
        run_cfg = RunConfig(timeout=600, max_retries=15, max_wait=120, max_workers=4)

        # Small local LLMs sometimes produce unparseable JSON on a few KG nodes.
        # Ragas's run_async_tasks collects all exceptions and re-raises the first
        # one *after* all tasks complete — killing hours of good work.  We
        # monkeypatch it to swallow exceptions during testset generation
        # (7/696 ≈ 1% failures don't affect quality).
        import ragas.async_utils as _rau
        import ragas.testset.transforms.engine as _engine
        _orig_run_async = _rau.run_async_tasks

        def _lenient_run_async(*a, **kw):
            """Wrap run_async_tasks to swallow transform exceptions."""
            try:
                return _orig_run_async(*a, **kw)
            except (RagasOutputParserException, LCOutputParserException,
                    ValueError, KeyError) as exc:
                print(f"\n⚠️  Transform had {type(exc).__name__} on some nodes — "
                      f"continuing (non-fatal): {exc}")
                return []

        # Patch in BOTH modules (engine.py has its own import binding)
        _rau.run_async_tasks = _lenient_run_async
        _engine.run_async_tasks = _lenient_run_async
        try:
            # Re-create transforms each attempt (they may carry state).
            transforms = default_transforms(
                documents=lc_docs,
                llm=generator.llm,
                embedding_model=generator.embedding_model,
            )
            testset = generator.generate_with_langchain_docs(
                lc_docs,
                testset_size=args.test_size,
                transforms=transforms,
                with_debugging_logs=True,
                run_config=run_cfg,
            )
        finally:
            _rau.run_async_tasks = _orig_run_async
            _engine.run_async_tasks = _orig_run_async  # restore both
        test_df = testset.to_pandas()
        test_df = (
            test_df.sort_values("user_input")
            .drop_duplicates(subset=["user_input"], keep="first")
            .reset_index(drop=True)
        )
        # Atomic write: temp file → rename (crash during write won't corrupt cache)
        tmp = testset_file.with_suffix(".parquet.tmp")
        test_df.to_parquet(tmp)
        tmp.rename(testset_file)
        print(f"   💾 Saved {len(test_df)} questions → {testset_file}")

    # Normalise column names
    if "user_input" in test_df.columns and "question" not in test_df.columns:
        test_df = test_df.rename(columns={"user_input": "question"})
    if "reference" in test_df.columns and "ground_truth" not in test_df.columns:
        test_df = test_df.rename(columns={"reference": "ground_truth"})
    print(f"   📊 Testset: {len(test_df)} rows, columns: {list(test_df.columns)}")

    # ── 5. RAG answers (incremental — survives crashes) ────────────────
    eval_cache = CACHE_DIR / f"eval_dataset_{model_tag}.parquet"
    eval_partial = CACHE_DIR / f"eval_dataset_{model_tag}.partial.jsonl"
    all_questions = test_df["question"].values.tolist()
    all_ground_truths = test_df["ground_truth"].values.tolist()

    if eval_cache.exists():
        print(f"\n📂 Loading cached eval dataset from {eval_cache}")
        ragas_evals_df = pd.read_parquet(eval_cache)
    else:
        from tqdm.auto import tqdm

        # Resume from partial results if a previous run was interrupted
        done_rows: list[dict] = []
        if eval_partial.exists():
            with open(eval_partial) as f:
                for line in f:
                    if line.strip():
                        done_rows.append(json.loads(line))
            print(f"\n♻️  Resuming RAG queries — {len(done_rows)}/{len(all_questions)} already done")

        start_idx = len(done_rows)
        remaining = len(all_questions) - start_idx
        if remaining > 0:
            print(f"\n💬 Querying RAG ({rag_llm_label}) for {remaining} questions "
                  f"({start_idx} already cached) …")
            with open(eval_partial, "a") as fout:
                for i in tqdm(range(start_idx, len(all_questions)), desc="RAG queries"):
                    q = all_questions[i]
                    r = query_engine.query(q)
                    row = {
                        "question": q,
                        "answer": r.response,
                        "contexts": [c.node.get_content() for c in r.source_nodes],
                        "ground_truth": all_ground_truths[i],
                    }
                    fout.write(json.dumps(row) + "\n")
                    fout.flush()
                    done_rows.append(row)

        ragas_evals_df = pd.DataFrame(done_rows)
        ragas_evals_df.to_parquet(eval_cache)
        eval_partial.unlink(missing_ok=True)  # clean up partial file
        print(f"   💾 Saved eval dataset ({len(ragas_evals_df)} rows) → {eval_cache}")

    # ── 6. RAGAS evaluation ──────────────────────────────────────────────
    # Unique tag that captures: embed + testset + RAG + judge combination
    eval_tag = f"{embed_tag}_ts-{testset_tag}_{model_tag}_judge-{judge_tag}"
    scores_cache = results_dir / f"scores_{eval_tag}.parquet"
    if scores_cache.exists():
        print(f"\n📂 Loading cached scores from {scores_cache}")
        eval_scores_df = pd.read_parquet(scores_cache)
    else:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_correctness,
            context_precision,
            context_recall,
            faithfulness,
        )

        # ── Choose judge LLM ─────────────────────────────────────────
        if args.eval_llm == "azure":
            ragas_llm = _make_azure_lc_llm(args)
            judge_label = f"Azure OpenAI ({args.azure_model_prefix})"
        else:
            from langchain_community.chat_models import ChatOllama
            ragas_llm = ChatOllama(base_url=args.ollama_llm_url, model=eval_ollama_model, timeout=600)
            judge_label = f"Ollama ({eval_ollama_model})"
        ragas_embeddings = lc_embed  # always same embed provider as the index

        ragas_eval_dataset = Dataset.from_pandas(ragas_evals_df)

        from ragas.run_config import RunConfig
        from ragas.exceptions import RagasOutputParserException
        from langchain_core.exceptions import OutputParserException as LCOutputParserException
        eval_run_cfg = RunConfig(timeout=600, max_retries=15, max_wait=120, max_workers=4)
        print(f"\n📏 Running Ragas evaluation (judge: {judge_label}) …")
        for eval_attempt in range(1, 4):
            try:
                result = evaluate(
                    dataset=ragas_eval_dataset,
                    metrics=[faithfulness, answer_correctness, context_recall, context_precision],
                    llm=ragas_llm,
                    embeddings=ragas_embeddings,
                    run_config=eval_run_cfg,
                )
                break
            except (RagasOutputParserException, LCOutputParserException) as exc:
                if eval_attempt < 3:
                    wait = 30 * eval_attempt
                    print(f"\n⚠️  Evaluation attempt {eval_attempt}/3 failed "
                          f"({type(exc).__name__}). Retrying in {wait}s …")
                    import time as _time; _time.sleep(wait)
                else:
                    print(f"\n❌ Evaluation failed after 3 attempts: {exc}")
                    raise

        eval_scores_df = pd.DataFrame(result.scores)
        # Atomic write
        tmp = scores_cache.with_suffix(".parquet.tmp")
        eval_scores_df.to_parquet(tmp)
        tmp.rename(scores_cache)
        print(f"   💾 Saved scores → {scores_cache}")

    # ── 7. Print results ─────────────────────────────────────────────────
    duration = time.time() - t0
    judge_label_print = eval_label
    testset_label_print = testset_label
    print(f"\n{'='*60}")
    print(f"📊 RAGAS results for {rag_llm_label} ({len(eval_scores_df)} questions):")
    print(f"   {'RAG LLM':25s}: {rag_llm_label}")
    print(f"   {'Testset LLM':25s}: {testset_label_print}")
    print(f"   {'Judge LLM':25s}: {judge_label_print}")
    print(f"   {'Embeddings':25s}: {embed_label}")
    print(f"{'='*60}")
    for m in METRICS:
        if m in eval_scores_df.columns:
            print(f"   {m:25s}: {eval_scores_df[m].mean():.3f}")
    print(f"   {'Duration':25s}: {duration:.1f}s")

    # ── 8. HTML report ───────────────────────────────────────────────────
    if args.output_html and args.output_html != "auto":
        html_path = args.output_html
    else:
        html_path = str(results_dir / f"ragas_{eval_tag}_n{len(eval_scores_df)}.html")

    generate_html_report(
        eval_df=ragas_evals_df,
        scores_df=eval_scores_df,
        llm_model=rag_llm_label,
        embed_model=embed_label,
        test_size=args.test_size,
        duration_sec=duration,
        output_path=html_path,
        testset_llm=testset_label_print,
        judge_llm=judge_label_print,
        phoenix_project=experiment_project,
    )

    return eval_scores_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.compare:
        rdir = Path(args.results_dir) if args.results_dir else RESULTS_DIR
        generate_comparison_html(rdir, output_path=args.compare_output)
        return

    run_ragas_test(args)
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
