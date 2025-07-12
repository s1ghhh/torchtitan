#!/bin/bash

# 源 checkpoint 根目录
INPUT_DIR="outputs_llama3_3b_profiling/checkpoint"
# 输出保存目录
OUTPUT_DIR="/mnt/weka/home/haolong.jia/opt/torchtitan/pth_checkpoint"

# 创建输出目录（如不存在）
mkdir -p "$OUTPUT_DIR"

# 遍历所有 step-* 子目录
for STEP_PATH in "$INPUT_DIR"/step-*; do
    if [[ -d "$STEP_PATH" ]]; then
        STEP_NAME=$(basename "$STEP_PATH")
        OUTPUT_FILE="$OUTPUT_DIR/${STEP_NAME}.pt"
        
        echo "Converting $STEP_NAME to $OUTPUT_FILE..."
        
        # 执行转换
        python -m torch.distributed.checkpoint.format_utils dcp_to_torch "$STEP_PATH" "$OUTPUT_FILE"
        
        if [[ $? -eq 0 ]]; then
            echo "✔️ Successfully converted $STEP_NAME"
        else
            echo "❌ Failed to convert $STEP_NAME"
        fi
    fi
done
