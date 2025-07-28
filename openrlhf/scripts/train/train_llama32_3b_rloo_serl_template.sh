set -x
# Start time
start_time=$(date +%s)
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/path/to/your/SeRL/openrlhf", "excludes":["dataset/", "evolution_generation_data_dir", "train_eval_outputs_dir/", "train_online_filtered_data_dir/", "train_samples_dir/"]}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 8 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 8\
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 8 \
   --vllm_num_engines 8 \
   --vllm_tensor_parallel_size 1 \
   --vllm_gpu_memory_utilization 0.6 \
   --colocate_all_models \
   --advantage_estimator rloo \
   --pretrain /path/to/your/Llama-3.2-3B-Instruct \
   --remote_rm_url /path/to/your/SeRL/openrlhf/reward_utils/math_verify_maj_reward.py,/path/to/your/SeRL/openrlhf/reward_utils/math_verify_reward.py \
   --save_path /your_save_path \
   --ckpt_path //your_ckpt_path \
   --save_hf_ckpt \
   --micro_train_batch_size 1 \
   --train_batch_size 16 \
   --micro_rollout_batch_size 4 \
   --rollout_batch_size 16 \
   --n_samples_per_prompt 16 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --max_samples 100000 \
   --generate_max_len 1024 \
   --zero_stage 3 \
   --save_steps 50 \
   --eval_steps 1 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 1e-4 \
   --prompt_data /path/to/your/SeRL/openrlhf/dataset/math/0_2_0_8_train_with_idx_sample_500.jsonl \
   --input_key problem \
   --label_key answer \
   --normalize_reward \
   --adam_offload \
   --gradient_checkpointing \
   --packing_samples \
   --vllm_sync_backend nccl \
   --enforce_eager \
   --vllm_enable_sleep \
   --deepspeed_enable_sleep \
   --use_wandb <your_wandb_key> \
   --wandb_run_name <custom_run_name> \
   --eval_output_root_dir /path/to/your/SeRL/openrlhf/train_eval_outputs_dir \
   --train_samples_root_dir /path/to/your/SeRL/openrlhf/train_samples_dir \
   --filtered_data_root_dir /path/to/your/SeRL/openrlhf/train_online_filtered_data_dir \
   --eval_dataset /path/to/your/SeRL/evaluation/Math-Benchmarks/data/math_500/test_with_idx.jsonl \
   --self_reward_method bon_maj \
   --eval_n_samples_per_prompt 1 \
   --eval_temperature 0 \
   --reward_difficulty_bounds 0.2 0.8 \
   --enable_self_evolution \
   --few_shot_generation 8 \
   --evolution_generation_data_root_dir /path/to/your/SeRL/openrlhf/evolution_generation_data_dir \
   --few_shot_generation_prompt /path/to/your/SeRL/openrlhf/prompts/question_generation.json \
   --few_shot_generation_batch_size 4 \
   --num_episodes 1000000 \
   --instructions_num_per_iteration 2000

   # --apply_chat_template \
   # --reward_pretrain OpenRLHF/Llama-3-8b-rm-700k \

# You could also try
#   --use_kl_loss \
#   --use_kl_estimator_k3 \

# also supports --advantage_estimator rloo | reinforce_baseline

# End time
end_time=$(date +%s)

# Calculate and print the execution time
execution_time=$((end_time - start_time))
echo "Execution time: $execution_time seconds"
