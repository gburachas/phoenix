#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# Run RAGAS evaluation for multiple LLMs sequentially.
# Fully resumable — skips models whose scores already exist.
# Designed to run under nohup so it survives SSH disconnections.
#
# Each stage backend is independently configurable:
#   TESTSET_LLM   — testset generation   (default: azure)
#   EVAL_LLM      — evaluation judge     (default: azure)
#   RAG_LLM       — RAG answering        (default: ollama)
#   EMBED_PROVIDER — embeddings          (default: ollama)
#
# Usage:
#   nohup bash run_all_models.sh > /tmp/ragas_all_models.log 2>&1 &
#   tail -f /tmp/ragas_all_models.log           # monitor progress
#   cat /tmp/ragas_all_models.pid               # get PID
#
# To force a fresh start:
#   bash run_all_models.sh --fresh
# ─────────────────────────────────────────────────────────────────────
set -uo pipefail  # no -e : we handle errors per-model

PYTHON="/data1/giedrius/anaconda3/envs/arize/bin/python"
SCRIPT="/home/giedrius/Projects/TandE/metrostar_phoenix/run_ragas_test.py"
BASE="/home/giedrius/Projects/TandE/metrostar_phoenix"
TEST_SIZE=20
NUM_FILES=10

# LLM backend for testset generation: "ollama" or "azure"
TESTSET_LLM="${TESTSET_LLM:-azure}"
# LLM backend for evaluation / judge: "ollama" or "azure"
EVAL_LLM="${EVAL_LLM:-azure}"
# LLM backend for RAG answering: "ollama" or "azure"
RAG_LLM="${RAG_LLM:-ollama}"
# Embedding provider: "ollama" or "azure"
EMBED_PROVIDER="${EMBED_PROVIDER:-ollama}"
# Azure model env-var prefix (GPT5_MINI, GPT5_CHAT, etc.)
AZURE_MODEL="${AZURE_MODEL:-GPT5_MINI}"
# Azure embedding deployment name
AZURE_EMBED_MODEL="${AZURE_EMBED_MODEL:-text-embedding-3-small}"

RESULTS="$BASE/results"
CACHE="$BASE/notebooks/cache"

# Save our PID so it's easy to check status
echo $$ > /tmp/ragas_all_models.pid

echo "============================================================"
echo "RAGAS multi-model evaluation — $(date)"
echo "  PID=$$  test_size=$TEST_SIZE  num_files=$NUM_FILES"
echo "  testset_llm=$TESTSET_LLM  eval_llm=$EVAL_LLM"
echo "  rag_llm=$RAG_LLM  embed_provider=$EMBED_PROVIDER"
echo "  azure_model=$AZURE_MODEL  azure_embed=$AZURE_EMBED_MODEL"
echo "============================================================"

# Optional --fresh flag to wipe everything
if [[ "${1:-}" == "--fresh" ]]; then
    echo "🧹 --fresh: wiping ALL caches and results …"
    rm -rf "$CACHE"/vector_index_*
    rm -f  "$CACHE/ragas_testset.parquet"
    rm -f  "$CACHE/eval_dataset_*.parquet"
    rm -f  "$CACHE/eval_dataset_*.partial.jsonl"
    rm -f  "$RESULTS/scores_*.parquet"
    rm -f  "$RESULTS/ragas_*.html"
    rm -f  "$RESULTS/comparison.html"
fi

# ── Helper: run one model (skip if scores exist) ─────────────────────
run_model() {
    local model="$1"
    local label="$2"
    local tag="${model//:/_}"     # gemma3:4b → gemma3_4b
    tag="${tag//\//_}"            # handle slashes too

    local scores_file="$RESULTS/scores_${tag}.parquet"
    if [[ -f "$scores_file" ]]; then
        echo ""
        echo "⏭️  [$label] $model — scores already exist, skipping"
        echo "     $scores_file"
        return 0
    fi

    echo ""
    echo "▶▶▶ [$label] $model — $(date) ◀◀◀"
    if $PYTHON "$SCRIPT" \
        --test-size "$TEST_SIZE" \
        --num-files "$NUM_FILES" \
        --llm-model "$model" \
        --rag-llm "$RAG_LLM" \
        --testset-llm "$TESTSET_LLM" \
        --eval-llm "$EVAL_LLM" \
        --embed-provider "$EMBED_PROVIDER" \
        --azure-model-prefix "$AZURE_MODEL" \
        --azure-embed-model "$AZURE_EMBED_MODEL" \
        --output-html; then
        echo "✅ $model done — $(date)"
    else
        echo "⚠️  $model FAILED (exit $?) — $(date)"
        echo "    Partial results preserved; rerun to resume."
    fi
}

# ── Run Ollama models ────────────────────────────────────────────────
run_model "gemma3:4b"   "1/3"
run_model "gemma3:12b"  "2/3"
run_model "gpt-oss:20b" "3/3"

# ── (Optional) Azure RAG reference baseline ──────────────────────────
# Uncomment to produce a reference run where Azure answers the questions.
# This gives a ceiling score to compare local models against.
#
# RAG_LLM=azure run_model "azure-ref" "ref"

# ── Comparison report (always regenerate) ────────────────────────────
echo ""
echo "▶▶▶ Generating comparison report — $(date) ◀◀◀"
$PYTHON "$SCRIPT" --compare || echo "⚠️  Comparison generation failed"

echo ""
echo "============================================================"
echo "ALL DONE — $(date)"
echo "  Results in: $RESULTS/"
echo "============================================================"
ls -lh "$RESULTS/" 2>/dev/null || true

# Clean up PID file
rm -f /tmp/ragas_all_models.pid
