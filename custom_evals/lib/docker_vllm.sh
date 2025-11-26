#!/bin/bash
# Docker vLLM container management functions

# Start vLLM Docker container
# Args: $1=HF_MODEL, $2=VLLM_PORT, $3=GPU_MEMORY_UTILIZATION, $4=MAX_NUM_BATCHED_TOKENS
# Returns: Sets CONTAINER_ID global variable
start_vllm_container() {
    local HF_MODEL="$1"
    local VLLM_PORT="$2"
    local GPU_MEMORY_UTILIZATION="$3"
    local MAX_NUM_BATCHED_TOKENS="$4"

    # Generate container name from model (sanitize for Docker: only alphanumeric, underscore, period, hyphen)
    # Remove port number as requested
    local CONTAINER_NAME="vllm_$(echo "$HF_MODEL" | tr '/' '_' | tr '.' '_' | tr ',' '_' | sed 's/[^a-zA-Z0-9_.-]/_/g')"

    # Check if container already exists
    local EXISTING_CONTAINER=$(docker ps -a --filter "name=^${CONTAINER_NAME}$" --format "{{.ID}}" | head -1)

    if [ ! -z "$EXISTING_CONTAINER" ]; then
        echo "Found existing container: $CONTAINER_NAME"
        # Check if it's running
        if docker inspect -f '{{.State.Running}}' $EXISTING_CONTAINER 2>/dev/null | grep -q true; then
            echo "Container is already running (ID: $EXISTING_CONTAINER)"
            CONTAINER_ID="$EXISTING_CONTAINER"
            return 0
        else
            echo "Starting existing container (ID: $EXISTING_CONTAINER)"
            docker start $EXISTING_CONTAINER > /dev/null
            CONTAINER_ID="$EXISTING_CONTAINER"
            return 0
        fi
    fi

    echo "Creating new vLLM Docker container for $HF_MODEL..."

    # Build docker run command with optional max-model-len
    local DOCKER_CMD="docker run -d --name \"$CONTAINER_NAME\" --gpus all \
        -p $VLLM_PORT:8000 \
        -v \"$HOME/.cache/huggingface:/root/.cache/huggingface\" \
        --ipc=host \
        vllm/vllm-openai:latest \
        --model \"$HF_MODEL\" \
        --trust-remote-code \
        --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
        --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS"

    # Add max-model-len only if specified
    if [ ! -z "$MAX_MODEL_LEN" ]; then
        DOCKER_CMD="$DOCKER_CMD --max-model-len $MAX_MODEL_LEN"
    fi

    CONTAINER_ID=$(eval $DOCKER_CMD)

    echo "vLLM container created (Name: $CONTAINER_NAME, ID: $CONTAINER_ID)"
}

# Wait for vLLM server to be ready
# Args: $1=VLLM_PORT, $2=CONTAINER_ID (optional, for log checking)
# Returns: 0 on success, 1 on failure
wait_for_vllm_server() {
    local VLLM_PORT="$1"
    local CONTAINER_ID="$2"

    echo "Waiting for vLLM server to be ready on port $VLLM_PORT..."
    if [ ! -z "$CONTAINER_ID" ]; then
        echo "You can monitor logs with: docker logs -f $CONTAINER_ID"
        echo ""
    fi

    # Wait indefinitely (no timeout) with minimal logging
    local i=0
    while true; do
        if curl -s http://localhost:$VLLM_PORT/v1/models > /dev/null 2>&1; then
            echo "vLLM server is ready!"
            return 0
        fi

        # Only check if container crashed (every 30 seconds)
        if [ ! -z "$CONTAINER_ID" ] && [ $((i % 15)) -eq 0 ]; then
            if ! docker inspect -f '{{.State.Running}}' $CONTAINER_ID 2>/dev/null | grep -q true; then
                echo "Error: Container has stopped unexpectedly"
                echo "Container logs:"
                docker logs $CONTAINER_ID 2>&1 | tail -50
                return 1
            fi
        fi

        sleep 2
        i=$((i + 1))
    done

    return 1
}

# Wait for existing vLLM server (no container management)
# Args: $1=VLLM_PORT
# Returns: 0 on success, 1 on failure
wait_for_existing_vllm_server() {
    local VLLM_PORT="$1"

    echo "Waiting for vLLM server to be ready on port $VLLM_PORT..."
    for i in {1..60}; do
        if curl -s http://localhost:$VLLM_PORT/v1/models > /dev/null 2>&1; then
            echo "vLLM server is ready!"
            return 0
        fi
        if [ $i -eq 60 ]; then
            echo "Error: vLLM server not responding on port $VLLM_PORT"
            return 1
        fi
        echo "Still waiting... ($((i*2))s elapsed)"
        sleep 2
    done

    return 1
}

# Get model name from vLLM API
# Args: $1=VLLM_PORT, $2=FALLBACK_MODEL_NAME
# Returns: Prints model name to stdout
get_api_model_name() {
    local VLLM_PORT="$1"
    local FALLBACK_MODEL_NAME="$2"

    local API_MODEL_NAME=$(curl -s http://localhost:$VLLM_PORT/v1/models | python3 -c "import sys, json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "$FALLBACK_MODEL_NAME")
    echo "$API_MODEL_NAME"
}

# Stop vLLM container (but don't remove it for reuse)
# Args: $1=CONTAINER_ID
stop_vllm_container() {
    local CONTAINER_ID="$1"

    if [ ! -z "$CONTAINER_ID" ]; then
        echo "Stopping vLLM container (preserving for reuse)..."
        docker stop $CONTAINER_ID 2>/dev/null || true
        echo "Container stopped. It will be reused on next run."
    fi
}

# Save Docker logs (disabled)
# Args: $1=CONTAINER_ID, $2=OUTPUT_DIR
save_docker_logs() {
    local CONTAINER_ID="$1"
    local OUTPUT_DIR="$2"

    # Docker log saving is disabled
    # Use 'docker logs <container_id>' to view logs manually if needed
    return 0
}

# Save logs from existing container by port (disabled)
# Args: $1=VLLM_PORT, $2=OUTPUT_DIR
save_existing_container_logs() {
    local VLLM_PORT="$1"
    local OUTPUT_DIR="$2"

    # Docker log saving is disabled
    # Use 'docker logs <container_id>' to view logs manually if needed
    return 0
}
