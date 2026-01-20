#!/bin/bash
#
# Start vLLM server for HOI model evaluation.
#
# Usage:
#   # With defaults (merged model at outputs/qwen3vl-base-sft-merged)
#   bash scripts/start_vllm_server.sh
#
#   # With custom model path and port
#   bash scripts/start_vllm_server.sh outputs/qwen3vl-base-sft-merged 8000
#
#   # With custom model path, port, and tensor parallelism
#   bash scripts/start_vllm_server.sh outputs/qwen3vl-base-sft-merged 8000 2
#

set -e

# Default values
MODEL_PATH=${1:-"outputs/qwen3vl-base-sft-merged"}
PORT=${2:-8000}
TENSOR_PARALLEL_SIZE=${3:-1}
MAX_MODEL_LEN=${4:-4096}

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}  vLLM Server for HOI Evaluation${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""
echo "  Model:            $MODEL_PATH"
echo "  Port:             $PORT"
echo "  Tensor Parallel:  $TENSOR_PARALLEL_SIZE"
echo "  Max Model Length: $MAX_MODEL_LEN"
echo ""

# Check if model path exists
if [ ! -d "$MODEL_PATH" ]; then
    echo -e "${YELLOW}Warning: Model path '$MODEL_PATH' does not exist.${NC}"
    echo "Please run 'python scripts/merge_lora.py' first to merge LoRA adapters."
    echo ""
    echo "Example:"
    echo "  python scripts/merge_lora.py \\"
    echo "      --lora_path outputs/qwen3vl-base-sft \\"
    echo "      --output_path outputs/qwen3vl-base-sft-merged"
    exit 1
fi

# Check for required files
if [ ! -f "$MODEL_PATH/config.json" ]; then
    echo -e "${YELLOW}Warning: No config.json found in model path.${NC}"
    echo "This might be a LoRA checkpoint, not a merged model."
    echo "Please run 'python scripts/merge_lora.py' to merge LoRA into base model."
    exit 1
fi

echo -e "${GREEN}Starting vLLM server...${NC}"
echo ""

# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --port "$PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --trust-remote-code \
    --dtype bfloat16 \
    --disable-log-requests

# Note: The server runs in foreground. Use Ctrl+C to stop.
# For background execution, use: bash scripts/start_vllm_server.sh &
