


data_name="$1"
adv_estimator="$2"  # grpo for baseline, miner for our method

if [[ $adv_estimator == *"reinforce_plus_plus"* ]]; then 
    save_estimator=rpp
else 
    save_estimator=$adv_estimator
fi
model_name_or_path="$3"  # The model path of the base model


save_name="$4"
rollout_n="${5:-16}"
entry_name="${6:-baselines.baseline_main_ppo}"

NUM_GPUS=8 

python3 -u -m $entry_name \
    algorithm.adv_estimator=$adv_estimator \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=128 \
    data.max_prompt_length=1100 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$model_name_or_path \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=$rollout_n \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.n=16 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_math' \
    trainer.experiment_name="${data_name}_${save_name}" \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.max_critic_ckpt_to_keep=1 \
    trainer.test_freq=5 \
    trainer.log_val_generations=10   ${@:7} 



# scancel $job_id