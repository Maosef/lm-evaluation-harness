# Library Modules

This directory contains modular bash libraries used by the lm_eval evaluation scripts.

## Module Overview

### config.sh
Configuration and environment setup functions:
- `setup_config()` - Set default configuration values
- `setup_python_env()` - Setup Python environment variables
- `setup_output_dir()` - Create output directory structure
- `print_config()` - Display current configuration
- `validate_config()` - Validate required parameters

### docker_vllm.sh
Docker vLLM container management functions:
- `start_vllm_container()` - Start a new vLLM Docker container
- `wait_for_vllm_server()` - Wait for vLLM server to be ready (new container)
- `wait_for_existing_vllm_server()` - Wait for existing vLLM server
- `get_api_model_name()` - Get model name from vLLM API
- `stop_vllm_container()` - Stop and remove vLLM container
- `save_docker_logs()` - Save container logs to file
- `save_existing_container_logs()` - Save logs from existing container by port

### lm_eval.sh
lm_eval execution logic:
- `run_lm_eval()` - Run lm_eval with vLLM backend
- `handle_lm_eval_exit_code()` - Handle and interpret lm_eval exit codes
- `get_output_name()` - Generate sanitized output name from task list

### utils.sh
Utility functions for logging, cleanup, and status tracking:
- `setup_cleanup_trap()` - Setup interrupt signal handlers
- `cleanup_handler()` - Cleanup function for interrupts
- `set_cleanup_container_id()` - Set container ID for cleanup
- `init_status_file()` - Initialize status tracking file
- `log_status()` - Log message to status file
- `print_header()` - Print formatted section header
- `print_usage()` - Display script usage help
- `print_completion()` - Display evaluation completion message
- `print_eval_start()` - Display evaluation start header

## Usage

To use these modules in a bash script:

```bash
# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source library modules
source "$SCRIPT_DIR/lib/utils.sh"
source "$SCRIPT_DIR/lib/config.sh"
source "$SCRIPT_DIR/lib/docker_vllm.sh"
source "$SCRIPT_DIR/lib/lm_eval.sh"
```

## Benefits of Modular Design

1. **Reusability**: Functions can be used by multiple scripts
2. **Maintainability**: Changes to functionality are centralized
3. **Testability**: Individual modules can be tested independently
4. **Readability**: Main scripts are cleaner and easier to understand
5. **Extensibility**: New features can be added without modifying existing code
