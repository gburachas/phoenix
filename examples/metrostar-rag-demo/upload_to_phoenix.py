#!/usr/bin/env python3
"""
Upload RAGAS evaluation datasets and experiment results to Phoenix.

Creates:
  - 2 Phoenix datasets (one per testset generator: azure, gpt-oss:20b)
  - 8 Phoenix experiments (one per combo) with pre-computed RAGAS scores

Usage:
    python upload_to_phoenix.py [--phoenix-url URL] [--results-dir DIR]

Requires:
    pip install arize-phoenix-client pandas
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


# ── Filename parser (same logic as run_ragas_test.py) ────────────────
def _parse_score_filename(stem: str):
    """Parse scores_<embed>_ts-<testset>_<rag>_judge-<judge> into parts."""
    s = stem.removeprefix("scores_")
    ts_idx = s.find("_ts-")
    judge_idx = s.find("_judge-")
    if ts_idx < 0 or judge_idx < 0:
        return None
    embed = s[:ts_idx]
    testset = s[ts_idx + 4 : judge_idx]
    remainder = s[judge_idx + 7 :]
    # testset is like "azure_gemma3_4b" → split at first known RAG model
    for rag in ("gemma3_4b", "gemma3_12b", "gemma3_27b", "gpt-oss_20b", "gpt-oss_120b"):
        if testset.endswith("_" + rag):
            ts_model = testset[: -(len(rag) + 1)]
            rag_model = rag
            break
    else:
        ts_model = testset
        rag_model = "unknown"
    return {
        "embed": embed,
        "testset": ts_model,
        "rag": rag_model,
        "judge": remainder,
    }


def _tag_to_label(tag: str) -> str:
    """azure → Azure, gpt-oss_20b → gpt-oss:20b, etc."""
    return tag.replace("_", ":").replace("text:embedding:3:small", "text-embedding-3-small")


def main():
    parser = argparse.ArgumentParser(description="Upload RAGAS results to Phoenix")
    parser.add_argument("--phoenix-url", default="http://127.0.0.1:6007",
                        help="Phoenix server URL")
    parser.add_argument("--results-dir",
                        default="results/8way_n10_2026-02-25",
                        help="Directory containing score parquet files")
    parser.add_argument("--testset-cache", default="notebooks/cache",
                        help="Directory containing cached testset/eval parquets")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be uploaded without doing it")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    cache_dir = Path(args.testset_cache)

    # ── Import Phoenix client ────────────────────────────────────────
    try:
        from phoenix.client import Client
    except ImportError:
        print("ERROR: phoenix-client not installed. Run: pip install arize-phoenix-client")
        sys.exit(1)

    client = Client(base_url=args.phoenix_url)
    print(f"Connected to Phoenix at {args.phoenix_url}")

    # ── 1. Discover score files ──────────────────────────────────────
    score_files = sorted(results_dir.glob("scores_*.parquet"))
    if not score_files:
        print(f"ERROR: No scores_*.parquet files in {results_dir}")
        sys.exit(1)
    print(f"Found {len(score_files)} score files\n")

    # ── 2. Load testsets & eval datasets ─────────────────────────────
    # We have 2 testset generators (azure, gpt-oss:20b) → 2 testsets
    # and 1 RAG model (gemma3:4b) → 1 eval dataset per testset
    # The eval_dataset contains Q + A + contexts + ground_truth

    testset_files = {
        "azure": cache_dir / "ragas_testset_azure.parquet",
        "gpt-oss_20b": cache_dir / "ragas_testset_gpt-oss_20b.parquet",
    }

    eval_dataset_file = cache_dir / "eval_dataset_gemma3_4b.parquet"

    # ── 3. Create Phoenix datasets ───────────────────────────────────
    # We'll create one dataset per testset generator, containing Q+A+GT+contexts
    # The eval_dataset only has 12 rows for azure testset (the first run).
    # For the full picture we join testset (questions + ground_truth) with
    # the eval_dataset (RAG answers + contexts).

    # Load eval dataset (has RAG answers)
    eval_df = pd.read_parquet(eval_dataset_file)
    print(f"Eval dataset: {len(eval_df)} rows, columns: {list(eval_df.columns)}")

    datasets = {}  # testset_tag → Phoenix Dataset object

    for ts_tag, ts_file in testset_files.items():
        if not ts_file.exists():
            print(f"  SKIP: {ts_file} not found")
            continue

        ts_df = pd.read_parquet(ts_file)
        label = _tag_to_label(ts_tag)
        ds_name = f"ragas-n12-ts-{label}"

        # The testset has: user_input, reference_contexts, reference, ...
        # Rename to match Phoenix conventions
        upload_df = pd.DataFrame({
            "question": ts_df["user_input"].values,
            "reference_answer": ts_df["reference"].values,
            "reference_contexts": ts_df["reference_contexts"].apply(
                lambda x: json.dumps(list(x)) if hasattr(x, "__iter__") else str(x)
            ).values,
            "synthesizer": ts_df["synthesizer_name"].values,
        })

        # Try to join with eval_dataset to get RAG answers
        if len(eval_df) == len(upload_df):
            upload_df["rag_answer"] = eval_df["answer"].values
            if "contexts" in eval_df.columns:
                upload_df["rag_contexts"] = eval_df["contexts"].apply(
                    lambda x: json.dumps(list(x)) if hasattr(x, "__iter__") else str(x)
                ).values

        print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Creating dataset: {ds_name}")
        print(f"  Rows: {len(upload_df)}, Columns: {list(upload_df.columns)}")

        if not args.dry_run:
            try:
                dataset = client.datasets.create_dataset(
                    name=ds_name,
                    dataframe=upload_df,
                    input_keys=["question"],
                    output_keys=["reference_answer"],
                    metadata_keys=["reference_contexts", "synthesizer",
                                   "rag_answer", "rag_contexts"],
                    dataset_description=(
                        f"RAGAS-generated testset (N=12) using {label} as testset generator. "
                        f"RAG answers from gemma3:4b. Experiment date: 2026-02-25."
                    ),
                    timeout=30,
                )
                datasets[ts_tag] = dataset
                print(f"  ✅ Created dataset '{ds_name}' (id={dataset.id})")
            except Exception as create_err:
                if "already exists" in str(create_err):
                    print(f"  ℹ️  Dataset '{ds_name}' already exists, fetching it...")
                    try:
                        ds = client.datasets.get_dataset(dataset=ds_name)
                        datasets[ts_tag] = ds
                        print(f"  ✅ Reusing dataset '{ds_name}' (id={ds.id})")
                    except Exception as fetch_err:
                        print(f"  ❌ Could not fetch existing dataset: {fetch_err}")
                else:
                    raise create_err
        else:
            print(f"  Would create dataset with {len(upload_df)} examples")

    # ── 4. Create experiments with pre-computed scores ───────────────
    print("\n" + "=" * 60)
    print("Creating experiments...")
    print("=" * 60)

    for sf in score_files:
        parsed = _parse_score_filename(sf.stem)
        if not parsed:
            print(f"  SKIP: cannot parse {sf.name}")
            continue

        scores_df = pd.read_parquet(sf)
        ts_tag = parsed["testset"]

        # Build descriptive experiment name
        embed_label = _tag_to_label(parsed["embed"])
        ts_label = _tag_to_label(parsed["testset"])
        rag_label = _tag_to_label(parsed["rag"])
        judge_label = _tag_to_label(parsed["judge"])

        exp_name = f"embed:{embed_label} | judge:{judge_label}"
        exp_desc = (
            f"8-way RAGAS evaluation. "
            f"Embeddings: {embed_label}, Testset LLM: {ts_label}, "
            f"RAG LLM: {rag_label}, Judge LLM: {judge_label}. "
            f"N=12 questions, run 2026-02-25."
        )

        # Build metadata with scores summary
        metric_cols = [c for c in scores_df.columns
                       if c in ("faithfulness", "context_precision",
                                "context_recall", "answer_correctness")]
        summary = {}
        for col in metric_cols:
            valid = scores_df[col].dropna()
            summary[col] = {
                "mean": round(float(valid.mean()), 4) if len(valid) > 0 else None,
                "n_valid": int(len(valid)),
                "n_total": int(len(scores_df)),
            }

        exp_metadata = {
            "embeddings": embed_label,
            "testset_llm": ts_label,
            "rag_llm": rag_label,
            "judge_llm": judge_label,
            "n_questions": len(scores_df),
            "score_summary": summary,
            "source": "ragas-8way-comparison",
        }

        print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Experiment: {exp_name}")
        print(f"  Dataset: ts-{ts_label}")
        print(f"  Scores: {len(scores_df)} rows, metrics: {metric_cols}")
        for col in metric_cols:
            s = summary[col]
            print(f"    {col}: mean={s['mean']}, valid={s['n_valid']}/{s['n_total']}")

        if args.dry_run:
            continue

        if ts_tag not in datasets:
            print(f"  ⚠️  Dataset for testset '{ts_tag}' not created, skipping")
            continue

        dataset = datasets[ts_tag]

        # Build a lookup from question text → scores dict.
        # The testset questions correspond 1:1 with scores rows.
        ts_file = testset_files.get(ts_tag)
        if ts_file and ts_file.exists():
            ts_df = pd.read_parquet(ts_file)
            questions = ts_df["user_input"].values.tolist()
        else:
            questions = [f"row_{i}" for i in range(len(scores_df))]

        scores_list = scores_df.to_dict("records")
        question_to_scores = {}
        for q, row in zip(questions, scores_list):
            clean_row = {k: (None if pd.isna(v) else round(float(v), 4))
                         for k, v in row.items()}
            question_to_scores[q] = clean_row

        def make_task(q2s):
            """Create a task closure that looks up pre-computed scores by question."""
            def task(input):
                question = input.get("question", "")
                scores = q2s.get(question, {})
                return scores
            return task

        # Evaluators that extract each metric from the task output.
        # For NaN/missing values (from RAGAS timeouts), we raise ValueError
        # so Phoenix records them as failed evaluations — semantically correct.
        def make_evaluator(metric_name):
            def evaluator(output):
                val = output.get(metric_name)
                if val is None:
                    raise ValueError(
                        f"{metric_name}: no score (RAGAS judge timed out)"
                    )
                return float(val)
            evaluator.__name__ = metric_name
            return evaluator

        evaluators = {col: make_evaluator(col) for col in metric_cols}

        try:
            experiment = client.experiments.run_experiment(
                dataset=dataset,
                task=make_task(question_to_scores),
                evaluators=evaluators,
                experiment_name=exp_name,
                experiment_description=exp_desc,
                experiment_metadata=exp_metadata,
                print_summary=True,
                timeout=60,
            )
            print(f"  ✅ Experiment created")
        except Exception as e:
            print(f"  ❌ Failed: {e}")

    print("\n" + "=" * 60)
    print("Done! Check Phoenix UI at %s/datasets" % args.phoenix_url)
    print("=" * 60)


if __name__ == "__main__":
    main()
