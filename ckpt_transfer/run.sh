# #!/bin/bash

# # 转换后 pt 文件所在目录
# INPUT_DIR="/mnt/weka/home/haolong.jia/opt/torchtitan/pth_checkpoint"

# # HuggingFace 输出根目录
# OUTPUT_BASE_DIR="/mnt/weka/home/haolong.jia/opt/torchtitan/hf_ckpt"

# # 固定参数
# TT_FLAVOR="3B"
# VOCAB_SIZE=128256        # <--- 根据你的模型修改
# LLAMA_VERSION=3

# # 遍历所有 .pt 文件
# for CKPT_PATH in "$INPUT_DIR"/step-*.pt; do
#     CKPT_FILE=$(basename "$CKPT_PATH")        # 如 step-100.pt
#     STEP_NUM=$(echo "$CKPT_FILE" | grep -oP '(?<=step-)\d+(?=\.pt)')
#     OUTPUT_DIR="$OUTPUT_BASE_DIR/step-$STEP_NUM"

#     echo "🔄 Converting $CKPT_FILE to Hugging Face format at $OUTPUT_DIR"

#     python p2h.py \
#         --input_dir "$INPUT_DIR" \
#         --tt_flavor "$TT_FLAVOR" \
#         --steps "$STEP_NUM" \
#         --output_dir "$OUTPUT_DIR" \
#         --vocab_size "$VOCAB_SIZE" \
#         --llama_version "$LLAMA_VERSION"

#     if [[ $? -eq 0 ]]; then
#         echo "✅ Step $STEP_NUM conversion complete!"
#     else
#         echo "❌ Step $STEP_NUM conversion failed!"
#     fi
# done


#!/bin/bash

# 转换后 pt 文件所在目录
INPUT_DIR="/mnt/weka/home/haolong.jia/opt/torchtitan/pth_checkpoint"

# HuggingFace 输出根目录
OUTPUT_BASE_DIR="/mnt/weka/home/haolong.jia/opt/torchtitan/hf_ckpt"

# 固定参数
TT_FLAVOR="3B"
VOCAB_SIZE=128256
LLAMA_VERSION=3

# 最大并发数
MAX_JOBS=2

# 线程控制函数
function wait_for_jobs {
    while [ "$(jobs -r | wc -l)" -ge "$MAX_JOBS" ]; do
        sleep 1
    done
}

# 遍历所有 .pt 文件
for CKPT_PATH in "$INPUT_DIR"/step-*.pt; do
    CKPT_FILE=$(basename "$CKPT_PATH")        # 如 step-100.pt
    STEP_NUM=$(echo "$CKPT_FILE" | grep -oP '(?<=step-)\d+(?=\.pt)')
    OUTPUT_DIR="$OUTPUT_BASE_DIR/step-$STEP_NUM"

    if [ -d "$OUTPUT_DIR" ]; then
        echo "⚠️  Output directory $OUTPUT_DIR already exists. Skipping step $STEP_NUM."
        continue
    fi

    wait_for_jobs

    echo "🔄 Converting $CKPT_FILE to Hugging Face format at $OUTPUT_DIR"

    (
        python p2h.py \
            --input_dir "$INPUT_DIR" \
            --tt_flavor "$TT_FLAVOR" \
            --steps "$STEP_NUM" \
            --output_dir "$OUTPUT_DIR" \
            --vocab_size "$VOCAB_SIZE" \
            --llama_version "$LLAMA_VERSION"

        if [[ $? -eq 0 ]]; then
            echo "✅ Step $STEP_NUM conversion complete!"
        else
            echo "❌ Step $STEP_NUM conversion failed!"
        fi
    ) &
done

# 等待所有后台任务完成
wait
echo "🎉 All conversions finished."
