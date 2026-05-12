#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

# 基础模型路径
BASE_MODEL="/mnt/bn/hjy/models/Qwen2.5-Math-7B-special-tokens"

# adapter 列表
ADAPTERS=(
    "/mnt/bn/hjy/models/correct_reasoning/hjy_correct_data_1016_normal/checkpoint-224"
    "/mnt/bn/hjy/models/correct_reasoning/hjy_correct_data_1016_normal/checkpoint-448"
    "/mnt/bn/hjy/models/correct_reasoning/hjy_correct_data_1016_normal/checkpoint-672"
    # "/mnt/bn/hjy/models/correct_reasoning/hjy_correct_data_1016_normal/checkpoint-224"
)

# 循环合并每个 adapter
for ADAPTER in "${ADAPTERS[@]}"; do
    # 导出目录就在 adapter 文件夹下的 huggingface 子目录
    EXPORT_DIR="$ADAPTER/huggingface"

    echo "Merging adapter: $ADAPTER -> $EXPORT_DIR"
    llamafactory-cli export \
        --model_name_or_path "$BASE_MODEL" \
        --adapter_name_or_path "$ADAPTER" \
        --template qwen \
        --trust_remote_code true \
        --export_dir "$EXPORT_DIR" \
        --export_device auto
done

echo "All adapters exported under their respective huggingface subfolders."

