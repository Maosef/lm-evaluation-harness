#!/bin/bash

# Script to run lm_eval using vLLM Docker container for faster inference with HF model
# Usage: ./run_lm_eval_vllm_docker.sh <tasks> [timeout] [num_concurrent]
#
# Example:
#   ./run_lm_eval_vllm_docker.sh "ifeval,gpqa_main_n_shot" 3600 32
#
# Model: huihui-ai/Huihui-gpt-oss-20b-mxfp4-abliterated-v2 (default, override with HF_MODEL)
#
# Environment variables:
#   HF_MODEL               - HuggingFace model to evaluate
#   VLLM_PORT              - Port for vLLM server (default: 8000)
#   NUM_CONCURRENT         - Number of concurrent requests (default: 4)
#   MAX_NUM_BATCHED_TOKENS - Max tokens to batch together (default: 2048)
#   GPU_MEMORY_UTILIZATION - GPU memory fraction for KV cache (default: 0.85)
#   EVAL_LIMIT             - Limit number of samples per task (default: none, use for testing)
#   SKIP_CONTAINER_START   - If set to 1, skip starting a new container (default: 0)

set -e

# Get script directory and load libraries
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Source library modules
source "$SCRIPT_DIR/lib/utils.sh"
source "$SCRIPT_DIR/lib/config.sh"
source "$SCRIPT_DIR/lib/docker_vllm.sh"
source "$SCRIPT_DIR/lib/lm_eval.sh"

# Setup cleanup trap
setup_cleanup_trap

# Parse command line arguments
if [ $# -lt 1 ]; then
    print_usage "$0"
    exit 1
fi

TASKS="$1"
TIMEOUT_ARG="${2:-}"
NUM_CONCURRENT_ARG="${3:-}"

# Setup configuration
setup_config

# Override timeout and num_concurrent if provided as arguments
if [ ! -z "$TIMEOUT_ARG" ]; then
    TIMEOUT="$TIMEOUT_ARG"
fi
if [ ! -z "$NUM_CONCURRENT_ARG" ]; then
    NUM_CONCURRENT="$NUM_CONCURRENT_ARG"
fi

# Validate configuration
validate_config "$TASKS" || exit 1

# Setup Python environment
setup_python_env

# Setup output directory
setup_output_dir "$SCRIPT_DIR/lm_eval_results"

# Generate output name
OUTPUT_NAME=$(get_output_name "$TASKS")

# Initialize status file
STATUS_FILE="$OUTPUT_DIR/lm_eval_vllm_status.txt"
init_status_file "$STATUS_FILE"

# Print configuration
echo "Running lm_eval with vLLM Docker acceleration"
print_config

# Print evaluation start header
print_eval_start "$HF_MODEL" "$STATUS_FILE"

# Start or connect to vLLM server
if [ "$SKIP_CONTAINER_START" == "1" ]; then
    echo "Skipping container start, using existing container on port $VLLM_PORT"
    CONTAINER_ID=""

    # Wait for existing server
    if ! wait_for_existing_vllm_server "$VLLM_PORT"; then
        log_status "$STATUS_FILE" "Failed to connect to vLLM server at $(date)"
        exit 1
    fi
else
    # Start new container
    start_vllm_container "$HF_MODEL" "$VLLM_PORT" "$GPU_MEMORY_UTILIZATION" "$MAX_NUM_BATCHED_TOKENS"
    set_cleanup_container_id "$CONTAINER_ID"

    # Wait for server to be ready
    if ! wait_for_vllm_server "$VLLM_PORT" "$CONTAINER_ID"; then
        log_status "$STATUS_FILE" "Failed to start vLLM for $HF_MODEL at $(date)"
        exit 1
    fi
fi

# Get model name from API
API_MODEL_NAME=$(get_api_model_name "$VLLM_PORT" "$HF_MODEL")
echo "API model name: $API_MODEL_NAME"

# Run lm_eval
EXIT_CODE=0
run_lm_eval "$API_MODEL_NAME" "$HF_MODEL" "$TASKS" "$VLLM_PORT" "$NUM_CONCURRENT" \
    "$OUTPUT_DIR" "$OUTPUT_NAME" "$TIMEOUT" "$EVAL_LIMIT" || EXIT_CODE=$?

# Handle exit code
if ! handle_lm_eval_exit_code "$EXIT_CODE" "$TIMEOUT"; then
    log_status "$STATUS_FILE" "Failed at $(date)"
    if [ "$SKIP_CONTAINER_START" != "1" ] && [ ! -z "$CONTAINER_ID" ]; then
        stop_vllm_container "$CONTAINER_ID"
    fi
    exit $EXIT_CODE
fi

# Save Docker logs and cleanup
if [ "$SKIP_CONTAINER_START" != "1" ] && [ ! -z "$CONTAINER_ID" ]; then
    save_docker_logs "$CONTAINER_ID" "$OUTPUT_DIR"
    stop_vllm_container "$CONTAINER_ID"
    CONTAINER_ID=""
    CLEANUP_CONTAINER_ID=""
elif [ "$SKIP_CONTAINER_START" == "1" ]; then
    echo "Skipping container cleanup (using existing container)"
    save_existing_container_logs "$VLLM_PORT" "$OUTPUT_DIR"
fi

# Print completion message
print_completion "$OUTPUT_DIR" "$OUTPUT_NAME" "$STATUS_FILE"
