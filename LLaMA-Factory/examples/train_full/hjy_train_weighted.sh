#!/bin/bash

# ====== 日志设置 ======
LOG_DIR="/mnt/bn/hjy/Correct-Reasoning/logs/1203"
mkdir -p ${LOG_DIR}
LOG_FILE="${LOG_DIR}/train_$(date +'%Y%m%d_%H%M%S').log"

# 让所有 stdout + stderr 同时写入日志并显示
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "日志将保存到: ${LOG_FILE}"
echo "开始时间: $(date)"
echo "----------------------------------------"

# ====== 权限设置 ======
sudo chmod -R 777 /mnt/bn/hjy/correct-reasoning/LLaMA-Factory
sudo chmod -R 777 /mnt/bn/hjy/models/correct_reasoning

# ====== 激活环境 ======
source /mnt/bn/hjy/anaconda/bin/activate
conda activate llamafac_loss


# ====== 开始训练 ======
cd /mnt/bn/hjy/correct-reasoning/LLaMA-Factory
export WANDB_PROJECT="hjy_correct_reasoning"
export WANDB_RUN_NAME="sft_Qwen2.5_7B_gen_critic_revise_weighted_0.01"
export CUDA_VISIBLE_DEVICES=4,5,6,7
set -x

MODEL_PATH=/mnt/bn/hjy/models/Qwen/Qwen2.5-7B-special-tokens

llamafactory-cli train \
    --model_name_or_path ${MODEL_PATH} \
    --trust_remote_code \
    --stage sft \
    --do_train true \
    --finetuning_type full \
    --dataset mix_all_gen_critic_revise \
    --template qwen \
    --cutoff_len 12000 \
    --max_samples 100000 \
    --overwrite_cache \
    --preprocessing_num_workers 4 \
    --dataloader_num_workers 4 \
    --output_dir /mnt/bn/hjy/models/correct_reasoning/mix_all_gen_critic_revise_0.01/sft_Qwen2.5_7B_1203_weighted_0.01 \
    --deepspeed /mnt/bn/hjy/correct-reasoning/LLaMA-Factory/examples/deepspeed/ds_z3_offload_config.json \
    --logging_steps 1 \
    --save_steps 500 \
    --plot_loss \
    --overwrite_output_dir \
    --save_only_model false \
    --report_to wandb \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1.0e-5 \
    --num_train_epochs 1.0 \
    --save_strategy epoch \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 \
    --ddp_timeout 180000000

python /mnt/bn/hjy/gpu_occ.py

echo "----------------------------------------"
echo "训练结束: $(date)"
echo "日志保存路径: ${LOG_FILE}"
