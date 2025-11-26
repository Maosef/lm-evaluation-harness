#!/bin/bash

# Script to run lm_eval on multiple models sequentially
# Usage: ./run_lm_eval_multi_model.sh <tasks>
#
# Example:
#   ./run_lm_eval_multi_model.sh "bbh,ifeval,gsm8k,mmlu,gpqa"
#
# This script will evaluate each model with the same tasks, using:
#   VLLM_PORT=8001
#   EVAL_LIMIT=500
#
# Environment variables (optional overrides):
#   VLLM_PORT              - Port for vLLM server (default: 8001)
#   EVAL_LIMIT             - Limit samples per task (default: 500)
#   NUM_CONCURRENT         - Number of concurrent requests (default: 4)
#   GPU_MEMORY_UTILIZATION - GPU memory fraction (default: 0.85)
#   MAX_NUM_BATCHED_TOKENS - Max tokens to batch (default: 2048)

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Source library modules
source "$SCRIPT_DIR/lib/utils.sh"

# Check arguments
if [ $# -lt 1 ]; then
    cat <<EOF
Usage: $0 <tasks>

Example: $0 "ifeval,arc_challenge,gsm8k,mmlu,gpqa"

This script evaluates multiple models with the same tasks.

Environment variables:
  VLLM_PORT              - Port for vLLM server (default: 8001)
  EVAL_LIMIT             - Limit samples per task (default: 500)
  NUM_CONCURRENT         - Number of concurrent requests (default: 4)
  GPU_MEMORY_UTILIZATION - GPU memory fraction (default: 0.85)
  MAX_NUM_BATCHED_TOKENS - Max tokens to batch (default: 2048)
EOF
    exit 1
fi

TASKS="$1"

# Set defaults for multi-model evaluation
export VLLM_PORT="${VLLM_PORT:-8001}"
export EVAL_LIMIT="${EVAL_LIMIT:-500}"
export NUM_CONCURRENT="${NUM_CONCURRENT:-4}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"

# Define models to evaluate
MODELS=(
    "meta-llama/Llama-3.1-8B-Instruct"
    "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated"
    "mlabonne/Qwen3-14B-abliterated"
    "mlabonne/NeuralDaredevil-8B-abliterated"
    # "gpt-oss-20b"
    # "huihui-ai/Huihui-Qwen3-14B-abliterated-v2"
    # "Qwen3-14B"
)

# Create main results directory with timestamp
MAIN_OUTPUT_DIR="$SCRIPT_DIR/lm_eval_results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$MAIN_OUTPUT_DIR"

# Initialize overall status file
OVERALL_STATUS_FILE="$MAIN_OUTPUT_DIR/multi_model_status.txt"
echo "Multi-model evaluation started at $(date)" > "$OVERALL_STATUS_FILE"
echo "Tasks: $TASKS" >> "$OVERALL_STATUS_FILE"
echo "Models: ${MODELS[@]}" >> "$OVERALL_STATUS_FILE"
echo "VLLM_PORT: $VLLM_PORT" >> "$OVERALL_STATUS_FILE"
echo "EVAL_LIMIT: $EVAL_LIMIT" >> "$OVERALL_STATUS_FILE"
echo "" >> "$OVERALL_STATUS_FILE"

print_header "Multi-Model Evaluation"
echo "Tasks: $TASKS"
echo "Models to evaluate: ${#MODELS[@]}"
for model in "${MODELS[@]}"; do
    echo "  - $model"
done
echo ""
echo "Configuration:"
echo "  VLLM_PORT: $VLLM_PORT"
echo "  EVAL_LIMIT: $EVAL_LIMIT"
echo "  NUM_CONCURRENT: $NUM_CONCURRENT"
echo "  GPU_MEMORY_UTILIZATION: $GPU_MEMORY_UTILIZATION"
echo "  MAX_NUM_BATCHED_TOKENS: $MAX_NUM_BATCHED_TOKENS"
echo ""
echo "Results will be saved to: $MAIN_OUTPUT_DIR"
echo ""

# Track successes and failures
SUCCESSFUL_MODELS=()
FAILED_MODELS=()

# Loop through each model
for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    MODEL_NUM=$((i + 1))
    TOTAL_MODELS=${#MODELS[@]}

    print_header "Model $MODEL_NUM/$TOTAL_MODELS: $MODEL"
    echo "Started at $(date)" | tee -a "$OVERALL_STATUS_FILE"
    echo ""

    # Create model-specific output directory
    MODEL_OUTPUT_DIR="$MAIN_OUTPUT_DIR/$(echo "$MODEL" | tr '/' '_')"
    mkdir -p "$MODEL_OUTPUT_DIR"

    # Export model and output directory
    export HF_MODEL="$MODEL"
    export OUTPUT_DIR="$MODEL_OUTPUT_DIR"

    # Run evaluation for this model
    set +e  # Don't exit on error
    "$SCRIPT_DIR/run_lm_eval_vllm_docker.sh" "$TASKS"
    EXIT_CODE=$?
    set -e

    # Record result
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Model $MODEL completed successfully" | tee -a "$OVERALL_STATUS_FILE"
        SUCCESSFUL_MODELS+=("$MODEL")
    else
        echo "✗ Model $MODEL failed with exit code $EXIT_CODE" | tee -a "$OVERALL_STATUS_FILE"
        FAILED_MODELS+=("$MODEL")
    fi

    echo "Finished at $(date)" | tee -a "$OVERALL_STATUS_FILE"
    echo "" | tee -a "$OVERALL_STATUS_FILE"

    # Add spacing between models
    if [ $MODEL_NUM -lt $TOTAL_MODELS ]; then
        echo ""
        echo "Continuing to next model..."
        echo ""
        sleep 2
    fi
done

# Print final summary
print_header "Multi-Model Evaluation Complete"
echo "Finished at $(date)" | tee -a "$OVERALL_STATUS_FILE"
echo ""

echo "Summary:" | tee -a "$OVERALL_STATUS_FILE"
echo "  Total models: ${#MODELS[@]}" | tee -a "$OVERALL_STATUS_FILE"
echo "  Successful: ${#SUCCESSFUL_MODELS[@]}" | tee -a "$OVERALL_STATUS_FILE"
echo "  Failed: ${#FAILED_MODELS[@]}" | tee -a "$OVERALL_STATUS_FILE"
echo "" | tee -a "$OVERALL_STATUS_FILE"

if [ ${#SUCCESSFUL_MODELS[@]} -gt 0 ]; then
    echo "Successful models:" | tee -a "$OVERALL_STATUS_FILE"
    for model in "${SUCCESSFUL_MODELS[@]}"; do
        echo "  ✓ $model" | tee -a "$OVERALL_STATUS_FILE"
    done
    echo "" | tee -a "$OVERALL_STATUS_FILE"
fi

if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
    echo "Failed models:" | tee -a "$OVERALL_STATUS_FILE"
    for model in "${FAILED_MODELS[@]}"; do
        echo "  ✗ $model" | tee -a "$OVERALL_STATUS_FILE"
    done
    echo "" | tee -a "$OVERALL_STATUS_FILE"
fi

echo "Overall status saved to: $OVERALL_STATUS_FILE"
echo "Results directory: $MAIN_OUTPUT_DIR"
echo ""

# Exit with error if any models failed
if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
    exit 1
fi

exit 0
