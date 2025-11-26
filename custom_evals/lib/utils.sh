#!/bin/bash
# Utility functions for logging, cleanup, and status tracking

# Global variable for cleanup
CLEANUP_CONTAINER_ID=""

# Setup cleanup trap
# Note: This function should be called after CONTAINER_ID is set
setup_cleanup_trap() {
    trap cleanup_handler SIGINT SIGTERM
}

# Cleanup handler for interrupts
cleanup_handler() {
    echo ""
    echo "Caught interrupt signal. Cleaning up..."

    # Stop Docker container if running and we started it
    if [ ! -z "$CLEANUP_CONTAINER_ID" ] && [ "$SKIP_CONTAINER_START" != "1" ]; then
        echo "Stopping vLLM container..."
        docker stop $CLEANUP_CONTAINER_ID 2>/dev/null || true
        docker rm $CLEANUP_CONTAINER_ID 2>/dev/null || true
    fi

    # Kill all child processes
    pkill -P $$ 2>/dev/null || true

    exit 130
}

# Set container ID for cleanup
set_cleanup_container_id() {
    CLEANUP_CONTAINER_ID="$1"
}

# Initialize status file
# Args: $1=STATUS_FILE
init_status_file() {
    local STATUS_FILE="$1"
    echo "Started at $(date)" > "$STATUS_FILE"
}

# Log to status file
# Args: $1=STATUS_FILE, $2=MESSAGE
log_status() {
    local STATUS_FILE="$1"
    local MESSAGE="$2"
    echo "$MESSAGE" >> "$STATUS_FILE"
}

# Print section header
# Args: $1=MESSAGE
print_header() {
    local MESSAGE="$1"
    echo "================================================================================"
    echo "$MESSAGE"
    echo "================================================================================"
}

# Print usage help
print_usage() {
    local SCRIPT_NAME="$1"
    cat <<EOF
Usage: $SCRIPT_NAME <tasks> [timeout] [num_concurrent]

Example: $SCRIPT_NAME 'ifeval,gpqa_main_n_shot' 3600 32

Environment variables:
  HF_MODEL               - HuggingFace model to evaluate (required or default)
  VLLM_PORT              - Port for vLLM server (default: 8000)
  NUM_CONCURRENT         - Number of concurrent requests (default: 4)
  GPU_MEMORY_UTILIZATION - GPU memory fraction for KV cache (default: 0.85)
  MAX_NUM_BATCHED_TOKENS - Max tokens to batch together (default: 2048)
  EVAL_LIMIT             - Limit samples per task (default: none)
  SKIP_CONTAINER_START   - Set to 1 to use existing container (default: 0)
EOF
}

# Print evaluation complete message
# Args: $1=OUTPUT_DIR, $2=OUTPUT_NAME, $3=STATUS_FILE
print_completion() {
    local OUTPUT_DIR="$1"
    local OUTPUT_NAME="$2"
    local STATUS_FILE="$3"

    echo ""
    echo "✓ Completed evaluation"
    log_status "$STATUS_FILE" "Completed at $(date)"
    echo ""

    print_header "Evaluation complete!"
    log_status "$STATUS_FILE" "Finished at $(date)"
    echo "Status saved to: $STATUS_FILE"
    echo "Results saved to: $OUTPUT_DIR/${OUTPUT_NAME}_results"
}

# Print evaluation start header
# Args: $1=HF_MODEL, $2=STATUS_FILE
print_eval_start() {
    local HF_MODEL="$1"
    local STATUS_FILE="$2"

    print_header "Evaluating: $HF_MODEL"
    log_status "$STATUS_FILE" "Processing $HF_MODEL at $(date)"
}
