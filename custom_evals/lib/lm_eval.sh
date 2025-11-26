#!/bin/bash
# lm_eval execution logic

# Run lm_eval with vLLM backend
# Args: $1=API_MODEL_NAME, $2=HF_MODEL, $3=TASKS, $4=VLLM_PORT, $5=NUM_CONCURRENT,
#       $6=OUTPUT_DIR, $7=OUTPUT_NAME, $8=TIMEOUT, $9=EVAL_LIMIT
# Returns: Exit code from lm_eval
run_lm_eval() {
    local API_MODEL_NAME="$1"
    local HF_MODEL="$2"
    local TASKS="$3"
    local VLLM_PORT="$4"
    local NUM_CONCURRENT="$5"
    local OUTPUT_DIR="$6"
    local OUTPUT_NAME="$7"
    local TIMEOUT="$8"
    local EVAL_LIMIT="$9"

    echo "Running lm_eval..."
    echo "Note: Press Ctrl+C to interrupt (may take a moment to respond)"
    echo ""

    # Build lm_eval command
    local LM_EVAL_CMD="timeout --foreground ${TIMEOUT}s uv run python -m lm_eval \
        --model local-completions \
        --model_args model=$API_MODEL_NAME,base_url=http://localhost:$VLLM_PORT/v1/completions,num_concurrent=$NUM_CONCURRENT,max_retries=3,tokenizer=$HF_MODEL \
        --tasks $TASKS \
        --output_path $OUTPUT_DIR/${OUTPUT_NAME}_results \
        --log_samples"

    if [ ! -z "$EVAL_LIMIT" ]; then
        echo "Using evaluation limit: $EVAL_LIMIT samples per task"
        LM_EVAL_CMD="$LM_EVAL_CMD --limit $EVAL_LIMIT"
    fi

    # Run with timeout, capturing the exit code
    set +e  # Temporarily disable exit on error
    eval $LM_EVAL_CMD
    local EXIT_CODE=$?
    set -e  # Re-enable exit on error

    return $EXIT_CODE
}

# Handle lm_eval exit code
# Args: $1=EXIT_CODE, $2=TIMEOUT
# Returns: Same exit code or 0 if acceptable
handle_lm_eval_exit_code() {
    local EXIT_CODE="$1"
    local TIMEOUT="$2"

    if [ $EXIT_CODE -ne 0 ]; then
        if [ $EXIT_CODE -eq 124 ]; then
            echo "Warning: lm_eval timed out after ${TIMEOUT}s"
        elif [ $EXIT_CODE -eq 130 ]; then
            echo "Warning: lm_eval interrupted by user"
        else
            echo "Warning: lm_eval failed with exit code $EXIT_CODE"
        fi
        return $EXIT_CODE
    fi

    return 0
}

# Generate output name from tasks
# Args: $1=TASKS
# Returns: Prints sanitized output name to stdout
get_output_name() {
    local TASKS="$1"
    echo "$TASKS" | tr ',' '_'
}
