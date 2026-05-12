#!/bin/bash

# ====== 日志设置 ======
LOG_DIR="/mnt/bn/hjy/Correct-Reasoning/logs/1209"
mkdir -p ${LOG_DIR}
LOG_FILE="${LOG_DIR}/qwen3_gen_critic_revise_merge_$(date +'%Y%m%d_%H%M%S').log"

echo "日志将保存到: ${LOG_FILE}"
echo "开始时间: $(date)"
echo "----------------------------------------"

# ====== 权限设置 ======
sudo chmod -R 777 /mnt/bn/hjy/correct-reasoning/LLaMA-Factory
sudo chmod -R 777 /mnt/bn/hjy/models/correct_reasoning

# ====== 激活环境 ======
source /mnt/bn/hjy/anaconda/bin/activate
conda activate llamafac_loss

# ====== 设置环境变量 ======
cd /mnt/bn/hjy/correct-reasoning/LLaMA-Factory
export WANDB_PROJECT="hjy_correct_reasoning"
export WANDB_RUN_NAME="gen_critic_revise_merge_1209"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# MODEL_PATH=/mnt/bn/hjy/models/correct_reasoning/gen_first_eval_second_qwen3_8b_replay
MODEL_PATH="/mnt/bn/hjy/models/Qwen/Qwen3-8B-Base_special_tokens"


# ====== 后台训练 ======
nohup bash -c "
set -x

llamafactory-cli train \
    --model_name_or_path ${MODEL_PATH} \
    --trust_remote_code \
    --stage sft \
    --do_train true \
    --finetuning_type full \
    --dataset gen_critic_revise_merge_1209 \
    --template qwen3 \
    --cutoff_len 12000 \
    --max_samples 100000 \
    --overwrite_cache \
    --preprocessing_num_workers 4 \
    --dataloader_num_workers 4 \
    --output_dir /mnt/bn/hjy/models/correct_hjy/qwen3/gen_critic_revise_merge_1209_v2 \
    --deepspeed /mnt/bn/hjy/correct-reasoning/LLaMA-Factory/examples/deepspeed/ds_z3_offload_config.json \
    --logging_steps 1 \
    --save_steps 500 \
    --plot_loss \
    --overwrite_output_dir \
    --save_only_model false \
    --report_to wandb \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1.0e-6 \
    --num_train_epochs 1.0 \
    --save_strategy epoch \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 \
    --ddp_timeout 180000000

python /mnt/bn/hjy/gpu_occ.py

echo \"----------------------------------------\"
echo \"训练结束: \$(date)\"
echo \"日志保存路径: ${LOG_FILE}\"

" > "${LOG_FILE}" 2>&1 &