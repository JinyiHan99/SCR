#!/bin/bash
export WANDB_PROJECT="hjy_correct_reasoning"
export WANDB_RUN_NAME="hjy_correct_data_1016_normal"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
set -x

MODEL_PATH=/mnt/bn/hjy/models/Qwen2.5-Math-7B-special-tokens

llamafactory-cli train \
    --model_name_or_path ${MODEL_PATH} \
    --trust_remote_code \
    --stage sft \
    --do_train true\
    --finetuning_type lora \
    --lora_rank 8 \
    --lora_target all \
    --dataset hjy_correct_data_1016 \
    --template qwen \
    --cutoff_len 12000 \
    --max_samples 100000 \
    --overwrite_cache \
    --preprocessing_num_workers 4 \
    --dataloader_num_workers 4 \
    --output_dir /mnt/bn/hjy/models/correct_reasoning/hjy_correct_data_1016_normal \
    --logging_steps 1 \
    --save_steps 500 \
    --plot_loss \
    --overwrite_output_dir \
    --save_only_model false \
    --report_to wandb \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --num_train_epochs 3.0 \
    --save_strategy epoch \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 \
    --ddp_timeout 180000000
