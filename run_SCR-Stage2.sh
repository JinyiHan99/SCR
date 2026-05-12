MODEL_PATH=../ckp/SCR-Stage1
format_prompt=/SCR/EasyR1/examples/format_prompt/no.jinja
reward_path=/SCR/EasyR1/examples/reward_function/SCR_stage2.py:compute_score

cd EasyR1

python3 -m verl.trainer.main \
    config=rl_config.yaml \
    data.train_files=/SCR/data/training_data/SCR-Stag2_data.jsonl \
    data.format_prompt=${format_prompt} \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.n=8 \
    worker.reward.reward_function=${reward_path} \
    trainer.project_name=SCR \
    trainer.experiment_name=rl_stage2 \
    data.max_response_length=8000 \
    worker.rollout.max_num_batched_tokens=18440 \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.save_checkpoint_path=../ckp/SCR-Stage2

