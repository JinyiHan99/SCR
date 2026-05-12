
MODEL_PATH=Qwen2.5-7B-Instruct

llamafactory-cli train \
    --model_name_or_path ${MODEL_PATH} \
    --trust_remote_code \
    --stage sft \
    --do_train true \
    --finetuning_type full \
    --dataset SCR_SFT_Qwen \
    --template qwen \
    --cutoff_len 14000 \
    --max_samples 100000 \
    --overwrite_cache \
    --preprocessing_num_workers 4 \
    --dataloader_num_workers 4 \
    --output_dir ../ckp/SCR-SFT \
    --deepspeed /SCR/LLaMA-Factory/examples/deepspeed/ds_z3_offload_config.json \
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