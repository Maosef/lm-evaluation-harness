#!/bin/bash
# Configuration and environment setup

# Set default configuration values
# Args: $1=HF_MODEL (optional override)
setup_config() {
    local MODEL_OVERRIDE="$1"

    # Model configuration
    if [ ! -z "$MODEL_OVERRIDE" ]; then
        HF_MODEL="$MODEL_OVERRIDE"
    else
        HF_MODEL="${HF_MODEL:-huihui-ai/Huihui-gpt-oss-20b-mxfp4-abliterated-v2}"
    fi

    # Port and concurrency
    VLLM_PORT="${VLLM_PORT:-8000}"
    NUM_CONCURRENT="${NUM_CONCURRENT:-4}"

    # GPU and memory settings
    MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
    GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"  # Optional, uses model default if not set

    # Evaluation settings
    EVAL_LIMIT="${EVAL_LIMIT:-}"
    SKIP_CONTAINER_START="${SKIP_CONTAINER_START:-0}"

    # Timeout (default 2 hours)
    TIMEOUT="${TIMEOUT:-7200}"

    # Export environment variables
    export HF_MODEL
    export VLLM_PORT
    export NUM_CONCURRENT
    export MAX_NUM_BATCHED_TOKENS
    export GPU_MEMORY_UTILIZATION
    export MAX_MODEL_LEN
    export EVAL_LIMIT
    export SKIP_CONTAINER_START
    export TIMEOUT
}

# Setup Python environment
setup_python_env() {
    export PYTHONPATH="/home/ec2-user/git/lm-evaluation-harness:$PYTHONPATH"
    export HF_ALLOW_CODE_EVAL="1"
}

# Setup output directory structure
# Args: $1=BASE_OUTPUT_DIR
setup_output_dir() {
    local BASE_OUTPUT_DIR="$1"
    OUTPUT_DIR="${BASE_OUTPUT_DIR:-lm_eval_results}"
    mkdir -p "$OUTPUT_DIR"
    export OUTPUT_DIR
}

# Print configuration summary
print_config() {
    echo "Configuration:"
    echo "  Model: $HF_MODEL"
    echo "  Tasks: $TASKS"
    echo "  Timeout: ${TIMEOUT}s"
    echo "  vLLM port: $VLLM_PORT"
    echo "  Concurrent requests: $NUM_CONCURRENT"
    echo "  Max batched tokens: $MAX_NUM_BATCHED_TOKENS"
    echo "  GPU memory utilization: $GPU_MEMORY_UTILIZATION"
    if [ ! -z "$MAX_MODEL_LEN" ]; then
        echo "  Max model length: $MAX_MODEL_LEN"
    fi
    echo "  Skip container start: $SKIP_CONTAINER_START"
    if [ ! -z "$EVAL_LIMIT" ]; then
        echo "  Evaluation limit: $EVAL_LIMIT samples per task"
    fi
    echo "  Output directory: $OUTPUT_DIR"
    echo ""
}

# Validate required parameters
# Args: $1=TASKS
validate_config() {
    local TASKS="$1"

    if [ -z "$TASKS" ]; then
        echo "Error: No tasks specified"
        return 1
    fi

    if [ -z "$HF_MODEL" ]; then
        echo "Error: No model specified"
        return 1
    fi

    return 0
}
