#!/bin/bash

# 源 tokenizer 文件目录
TOKENIZER_DIR="/mnt/weka/home/haolong.jia/opt/torchtitan/Llama-3.2-3B-Instruct"

# 输出根目录（包含多个 step 子目录）
OUTPUT_BASE_DIR="/mnt/weka/home/haolong.jia/opt/torchtitan/hf_ckpt"

# 你想复制的 tokenizer 文件（可按需修改）
FILES_TO_COPY=(
    "tokenizer.json"
    "tokenizer_config.json"
    "special_tokens_map.json"
)

# 遍历所有已存在的输出子目录
for STEP_DIR in "$OUTPUT_BASE_DIR"/step-*; do
    if [ -d "$STEP_DIR" ]; then
        echo "📦 Copying tokenizer files to $STEP_DIR"
        for FILE in "${FILES_TO_COPY[@]}"; do
            if [ -f "$TOKENIZER_DIR/$FILE" ]; then
                cp "$TOKENIZER_DIR/$FILE" "$STEP_DIR/"
            else
                echo "⚠️  Warning: $TOKENIZER_DIR/$FILE not found."
            fi
        done
    fi
done

echo "✅ Tokenizer files copied to all step directories."
