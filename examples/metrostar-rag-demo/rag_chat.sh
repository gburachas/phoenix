#!/bin/bash
# RAG Chat Wrapper - Ensures conda environment is activated

CONDA_ENV="arize"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/rag_capability_test_gemma3.py"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda is not available"
    exit 1
fi

# Check if the environment exists
if ! conda env list | grep -q "^$CONDA_ENV "; then
    echo "❌ Error: Conda environment '$CONDA_ENV' does not exist"
    echo "Available environments:"
    conda env list
    exit 1
fi

# Check if conda environment is already activated
if [[ "$CONDA_DEFAULT_ENV" == "$CONDA_ENV" ]]; then
    echo "✓ Conda environment '$CONDA_ENV' is already active"
    echo "Running RAG chat system..."
    echo ""
    exec python "$SCRIPT_PATH" "$@"
else
    echo "ℹ️  Activating conda environment '$CONDA_ENV'..."
    echo "Running RAG chat system..."
    echo ""
    
    # Get conda base path
    CONDA_BASE=$(conda info --base)
    
    # Source conda.sh to get the conda command in this shell
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    
    # Activate the environment and run the script
    conda activate "$CONDA_ENV"
    
    # Check if activation was successful
    if [[ "$CONDA_DEFAULT_ENV" != "$CONDA_ENV" ]]; then
        echo "❌ Error: Failed to activate conda environment '$CONDA_ENV'"
        exit 1
    fi
    
    exec python "$SCRIPT_PATH" "$@"
fi
