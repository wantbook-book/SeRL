import os
import time
from abc import ABC
from datetime import timedelta

import ray
import torch
from tqdm import tqdm

from openrlhf.datasets import PromptDataset
from openrlhf.trainer.ppo_utils import AdaptiveKLController, FixedKLController
from openrlhf.trainer.ppo_utils.experience_maker import RemoteExperienceMaker
from openrlhf.trainer.ray.launcher import PPORayActorGroup
from openrlhf.utils import blending_datasets, get_tokenizer
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.remote_rm_utils import remote_rm_fn_ray
from openrlhf.utils.self_instruction_util import sample_machine_instructions, encode_prompt, post_process_response
logger = init_logger(__name__)

from evaluation.utils import save_jsonl
from math_verify import math_metric, LatexExtractionConfig, ExprExtractionConfig
import json
from rouge_score import rouge_scorer
# from multiprocessing import Pool
from ray.util.multiprocessing import Pool
from functools import partial
import numpy as np
import random
random.seed(42)

@ray.remote
class PPOTrainer(ABC):
    """
    Trainer for Proximal Policy Optimization (PPO) / REINFORCE++ / GRPO / RLOO and their variants.
    Single Controller with Multiple ActorGroups
    """

    def __init__(
        self,
        pretrain: str,
        strategy: DeepspeedStrategy,
        actor_model_group: PPORayActorGroup,
        critic_model_group: PPORayActorGroup,
        reward_model_group: PPORayActorGroup,
        reference_model_group: PPORayActorGroup,
        vllm_engines=None,
        prompt_max_len: int = 120,
        dataloader_pin_memory: bool = True,
        **generate_kwargs,
    ) -> None:
        super().__init__()

        self.strategy = strategy
        self.args = strategy.args

        self.tokenizer = get_tokenizer(pretrain, None, "left", strategy, use_fast=not self.args.disable_fast_tokenizer)
        self.actor_model_group = actor_model_group
        self.critic_model_group = critic_model_group
        self.reward_model_group = reward_model_group
        self.reference_model_group = reference_model_group
        self.dataloader_pin_memory = dataloader_pin_memory
        self.vllm_engines = vllm_engines

        self.prompt_max_len = prompt_max_len
        self.generate_kwargs = generate_kwargs

        self.max_epochs = self.args.max_epochs
        self.remote_rm_url = self.args.remote_rm_url
        self.init_kl_coef = self.args.init_kl_coef
        self.kl_target = self.args.kl_target
        self.kl_horizon = self.args.kl_horizon

        self.freezing_actor_steps = getattr(self.args, "freezing_actor_steps", -1)

        if self.kl_target:
            self.kl_ctl = AdaptiveKLController(self.init_kl_coef, self.kl_target, self.kl_horizon)
        else:
            self.kl_ctl = FixedKLController(self.init_kl_coef)

        self.experience_maker = RemoteExperienceMaker(
            self.actor_model_group,
            self.critic_model_group,
            self.reward_model_group,
            self.reference_model_group,
            self.tokenizer,
            self.prompt_max_len,
            self.kl_ctl,
            self.strategy,
            self.remote_rm_url,
            vllm_engines=self.vllm_engines,
            packing_samples=self.strategy.args.packing_samples,
        )

        self.prepare_datasets()

        # 统一日志记录器设置
        from ..utils.logger import UnifiedLogger
        self.logger = UnifiedLogger(strategy.args, self.strategy.is_rank_0())
        
        self.dynamic_sampling_cnt = 0
        
        # 时间统计相关变量
        self.step_start_time = None
        self.last_step_end_time = None

        if self.args.enable_self_evolution:
            # question generation prompt
            with open(self.args.few_shot_generation_prompt, "r") as fin:
                self.prompt_json = json.load(fin)

            # similarities = {}
            self.scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    
    def dynamic_sampling_check(self, global_step):
        checked_items = []
        append_steps = []
        for batch_index in range(len(self.replay_buffer.items) // self.strategy.args.n_samples_per_prompt):
            prompt_items = self.replay_buffer.items[batch_index * self.strategy.args.n_samples_per_prompt:(batch_index + 1) * self.strategy.args.n_samples_per_prompt]
            prompt_rewards = [item.info['reward'] for item in prompt_items]
            accuracy = sum(prompt_rewards) / self.strategy.args.n_samples_per_prompt
            if self.strategy.args.reward_difficulty_bounds[0] <= accuracy <= self.strategy.args.reward_difficulty_bounds[1]:
                checked_items.extend(prompt_items)
                append_steps.extend([global_step] * self.strategy.args.n_samples_per_prompt)
            else:
                print(f"Rank: {self.strategy.get_rank()}, accuracy: {accuracy}")

        if hasattr(self.replay_buffer, 'checked_items'):
            self.replay_buffer.checked_items.extend(checked_items)
            self.replay_buffer.append_steps.extend(append_steps)
        else:
            self.replay_buffer.checked_items = checked_items
            self.replay_buffer.append_steps = append_steps
        remove_idxs = []
        for idx, step in enumerate(self.replay_buffer.append_steps):
            if global_step - step > 1:
                remove_idxs.append(idx)
            else:
                break
        for idx in remove_idxs[::-1]:
            del self.replay_buffer.append_steps[idx]
            del self.replay_buffer.checked_items[idx]
        dist.barrier()
        required_size = self.strategy.args.n_samples_per_prompt * self.strategy.args.rollout_batch_size // self.strategy.world_size
        self.replay_buffer.items = []
        if len(self.replay_buffer.checked_items) >= required_size:
            flag_tensor = torch.tensor([1], device=torch.cuda.current_device())
        else:
            flag_tensor = torch.tensor([0], device=torch.cuda.current_device())
        dist.all_reduce(flag_tensor, op=dist.ReduceOp.SUM)

        if flag_tensor == self.strategy.world_size:
            self.replay_buffer.items = self.replay_buffer.checked_items[:required_size]
            # del self.replay_buffer.checked_items
            self.replay_buffer.checked_items = self.replay_buffer.checked_items[required_size:]
            self.replay_buffer.append_steps = self.replay_buffer.append_steps[required_size:]
            assert len(self.replay_buffer.items) == required_size, 'Replay buffer size is not enough'
            return True
        else:
            print(f'Rank: {self.strategy.get_rank()}, Replay buffer size is not enough, waiting for more samples, dynamic_sampling_cnt: {self.dynamic_sampling_cnt}')
            self.dynamic_sampling_cnt += 1
            return False

    def online_filter(self, experiences, steps):
        _experiences = []
        filtered_experiences = []
        chunk_size = self.args.n_samples_per_prompt // self.args.micro_rollout_batch_size
        for chunk_i in range(len(experiences)//chunk_size):
            n_samples_experences = experiences[chunk_i*chunk_size:(chunk_i+1)*chunk_size]
        # for experience in experiences:
            # Extract rewards and concatenate them into a single tensor
            # rewards = experience.info['reward']
            rewards = torch.cat([exp.info['reward'] for exp in n_samples_experences], dim=0)
            # The quantity does not meet the required range
            args = self.strategy.args
            acc = rewards.mean().float().item()
            assert len(rewards) == args.n_samples_per_prompt, f"len(rewards): {len(rewards)}, n_samples_per_prompt: {args.n_samples_per_prompt}"
            if args.reward_difficulty_bounds[0] <= acc <= args.reward_difficulty_bounds[1]:
                for exp in n_samples_experences:
                    exp.info['append_steps'] = steps
                    _experiences.append(exp)
            else:
                filtered_experiences.append(n_samples_experences[0])
                print(f"Rank: {self.strategy.get_rank()}, accuracy: {acc}")
            
        return _experiences, filtered_experiences
    
    def score_task(self, inst, data):
        # Assume this is the task you need to run in parallel in each process
        return self.scorer.score(inst, data)

    def fit(
        self,
    ) -> None:
        args = self.args

        # Load datasets
        num_rollouts_per_episodes = len(self.prompts_dataloader)

        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = num_rollouts_per_episodes  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        # broadcast init checkpoint to vllm
        ckpt_path = os.path.join(args.ckpt_path, "_actor")
        if args.load_checkpoint and os.path.exists(ckpt_path) and not self.vllm_engines is None:
            # vLLM wakeup when vllm_enable_sleep
            if self.strategy.args.vllm_enable_sleep:
                from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call

                batch_vllm_engine_call(self.vllm_engines, "wake_up")

            ref = self.actor_model_group.async_run_method(method_name="broadcast_to_vllm")
            ray.get(ref)

            # vLLM offload when vllm_enable_sleep
            if self.strategy.args.vllm_enable_sleep:
                batch_vllm_engine_call(self.vllm_engines, "sleep")

        # Restore step and start_epoch
        consumed_samples = ray.get(self.actor_model_group.async_run_method(method_name="get_consumed_samples"))[0]
        steps = consumed_samples // args.rollout_batch_size + 1
        start_episode = consumed_samples // args.rollout_batch_size // num_rollouts_per_episodes
        consumed_samples = consumed_samples % (num_rollouts_per_episodes * args.rollout_batch_size)
        comsumed_prompts = 0
        
        # self-instruction samples that were not filtered out after dynamic_sampling
        global_keep_data = []
        global_keep_data_idxs = []
        # After dynamic sampling, wait to accumulate enough experiences equal to rollout_batch_size * chunk_size
        global_experiences = []
        # After accumulating rollout_batch_size prompts, perform one round of dynamic_sampling
        global_batch_data = []
        global_batch_data_idxs = []
        gen_data_idx = 0
        for episode in range(start_episode, args.num_episodes):
            self.prompts_dataloader.sampler.set_epoch(
                episode, consumed_samples=0 if episode > start_episode else consumed_samples
            )
            pbar = tqdm(
                range(self.prompts_dataloader.__len__()),
                desc=f"Episode [{episode + 1}/{args.num_episodes}]",
                disable=not self.strategy.is_rank_0(),
            )
            for _, rand_prompts, labels, other_infos in self.prompts_dataloader:
                
                if args.enable_self_evolution:
                    if len(global_batch_data) < args.rollout_batch_size:
                        # vLLM wakeup when vllm_enable_sleep
                        if self.strategy.args.vllm_enable_sleep:
                            from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call

                            batch_vllm_engine_call(self.vllm_engines, "wake_up")
                        
                        # Split into multiple batches of size args.few_shot_generation_batch_size
                        batch_inputs = []
                        for batch_i in range(args.few_shot_generation_batch_size):
                            prompt_instructions = sample_machine_instructions(
                                machine_instructions=global_keep_data,
                                similarities=None,
                                n=2
                            )
                            prompt_instructions += random.sample(self.seed_data, args.few_shot_generation-len(prompt_instructions))
                            random.shuffle(prompt_instructions)
                            # start_idx = batch_i * args.few_shot_generation
                            # end_idx = min((batch_i + 1) * args.few_shot_generation, len(rand_prompts))
                            # prompt_instructions = rand_prompts[start_idx:end_idx]
                            # prompt_instructions += sample_machine_instructions( 
                            #     tokenizer=self.tokenizer,
                            #     machine_instructions=global_keep_train_data,
                            #     prompt_json=self.prompt_json,
                            # )
                            prompt = encode_prompt(
                                tokenizer=self.tokenizer,
                                prompt_instructions=prompt_instructions,
                                prompt_json=self.prompt_json,
                            )
                            batch_inputs.append(prompt)

                        # generate instructions
                        generate_kwargs = self.generate_kwargs.copy()
                        generate_kwargs["temperature"] = 0.8
                        generate_kwargs["top_p"] = 0.95
                        generate_kwargs["max_new_tokens"] = 1024
                        # generate_kwargs["frequency_penalty"] = 0
                        generate_kwargs["presence_penalty"] = 2
                        # generate_kwargs["stop"] = []
                        generate_kwargs["logprobs"] = 1
                        generate_kwargs['include_stop_str_in_output'] = False
                        generate_kwargs['n_samples_per_prompt'] = 1
                        # generate_kwargs["n"] = 1
                        # generate_kwargs["best_of"] = 1
                        # os.environ['TOKENIZERS_PARALLELISM='] = False
                        samples = self.experience_maker.generate_samples(batch_inputs, ['']*len(batch_inputs), **generate_kwargs)
                        results = self.tokenizer.batch_decode(samples.resp_sequences, skip_special_tokens=True)
                        if self.strategy.args.vllm_enable_sleep:
                            batch_vllm_engine_call(self.vllm_engines, "sleep")
                        instructions = []
                        # all_metadata = []
                        for result in results:
                            # new_instructions = post_process_gpt3_response(result["response"])
                            new_instructions = post_process_response(result)
                            # new_instructions = new_instructions[:args.max_new_instructions_per_request]
                            instructions += new_instructions
                            # all_metadata += [result] * len(new_instructions)
                        
                        # Is it enough to maintain diversity within each batch?
                        for inst in tqdm(instructions,desc="process generation instructions", disable=not self.strategy.is_rank_0(),):
                            with Pool(32) as p:
                                rouge_scores = p.map(partial(self.scorer.score, inst), self.seed_data + global_batch_data)
                            # tasks = [self.score_task.remote(inst, data) for data in (self.seed_data + global_batch_data)]
                            # rouge_scores = ray.get(tasks)
                            rouge_scores = [score["rougeL"].fmeasure for score in rouge_scores]
                            # rouge_scores = [scorer.score(inst, e_inst)["rougeL"].fmeasure for e_inst in human_instructions + machine_instructions]
                            if max(rouge_scores) > 0.7:
                                continue
                            # all_instructions = self.seed_data + global_batch_data
                            # most_similar_instructions = {
                            #         all_instructions[i] : rouge_scores[i] for i in np.argsort(rouge_scores)[-10:][::-1]
                            #     }
                            global_batch_data.append(inst)
                            global_batch_data_idxs.append(gen_data_idx)
                            gen_data_idx += 1
                            # fout.write(json.dumps({
                            #     "instruction": inst,
                            #     "most_similar": most_similar_instructions,
                            #     "avg_similarity_score": float(np.mean(rouge_scores)),
                            #     # "metadata": metadata,
                            #     "request_idx": request_idx
                            # }) + "\n")
                            # progress_bar.update(1)
                    if len(global_batch_data) < args.rollout_batch_size:
                        continue
                    else:
                        rand_prompts = global_batch_data[:args.rollout_batch_size]
                        idxs = global_batch_data_idxs[:args.rollout_batch_size]
                        other_infos = {
                            'idx': torch.tensor(idxs),
                        }
                        labels = [None] * args.rollout_batch_size
                        global_batch_data = global_batch_data[args.rollout_batch_size:]
                        global_batch_data_idxs = global_batch_data_idxs[args.rollout_batch_size:]
                    
                experiences = self.experience_maker.make_experience_list(rand_prompts, labels, other_infos, **self.generate_kwargs)
                assert args.n_samples_per_prompt % args.micro_rollout_batch_size == 0, f"n_samples_per_prompt: {args.n_samples_per_prompt}, micro_rollout_batch_size: {args.micro_rollout_batch_size}"
                chunk_size = args.n_samples_per_prompt // args.micro_rollout_batch_size
                for exp_i, experience in enumerate(experiences):
                    experience.info['idx'] = other_infos['idx'][exp_i//chunk_size].item()
                output = self.tokenizer.batch_decode(
                    experiences[0].sequences[0].unsqueeze(0), skip_special_tokens=True
                )
                experiences, filtered_experiences = self.online_filter(experiences, steps)
                if args.enable_self_evolution:
                    for chunk_i in range(len(experiences)//chunk_size):
                        experience = experiences[chunk_i*chunk_size]
                        total_length = experience.info['total_length'][0].item()
                        response_length = experience.info['response_length'][0].item()
                        prompt_answer = self.tokenizer.batch_decode(
                            experience.sequences[0].unsqueeze(0), skip_special_tokens=True
                        )[0]
                        prompt_answer_input_ids = self.tokenizer(
                            prompt_answer, add_special_tokens=False, truncation=True, max_length=10000
                        )["input_ids"]
                        prompt = self.tokenizer.decode(
                            prompt_answer_input_ids[:int(total_length - response_length)], skip_special_tokens=True
                        )
                        global_keep_data.append(prompt)
                        global_keep_data_idxs.append(experience.info['idx'])
                            
                global_experiences.extend(experiences)
                after_filter_global_experiences = []
                expired_experiences = []
                for chunk_i in range(len(global_experiences)//chunk_size):
                    n_samples_experences = global_experiences[chunk_i*chunk_size:(chunk_i+1)*chunk_size]
                # for experience in global_experiences:
                    if steps - n_samples_experences[0].info['append_steps'] < 1:
                        for exp in n_samples_experences:
                            after_filter_global_experiences.append(exp)
                    else:
                        # filtered_experiences.append(n_samples_experences[0])
                        expired_experiences.append(n_samples_experences[0])
                
                if len(filtered_experiences) > 0:
                    # save filtered_experiences instructions
                    with open(
                        os.path.join(self.strategy.args.filtered_data_dir, f"filtered_data_{steps}.jsonl"), "a", encoding="utf-8"
                    ) as f:
                        for experience in filtered_experiences:
                            total_length = experience.info['total_length'][0].item()
                            response_length = experience.info['response_length'][0].item()
                            prompt_answer = self.tokenizer.batch_decode(
                                experience.sequences[0].unsqueeze(0), skip_special_tokens=True
                            )[0]
                            prompt_answer_input_ids = self.tokenizer(
                                prompt_answer, add_special_tokens=False, truncation=True, max_length=10000
                            )["input_ids"]
                            prompt = self.tokenizer.decode(
                                prompt_answer_input_ids[:int(total_length - response_length)], skip_special_tokens=True
                            )
                            # for prompt in prompts:
                            example = {
                                "idx": experience.info['idx'],
                                "prompt": prompt,
                            }
                            json.dump(example, f, ensure_ascii=False)
                            f.write("\n")

                if len(expired_experiences) > 0:
                    # save expired_experiences instructions
                    with open(
                        os.path.join(self.strategy.args.filtered_data_dir, f"expired_data_{steps}.jsonl"), "a", encoding="utf-8"
                    ) as f:
                        for experience in expired_experiences:
                            total_length = experience.info['total_length'][0].item()
                            response_length = experience.info['response_length'][0].item()
                            prompt_answer = self.tokenizer.batch_decode(
                                experience.sequences[0].unsqueeze(0), skip_special_tokens=True
                            )[0]
                            prompt_answer_input_ids = self.tokenizer(
                                prompt_answer, add_special_tokens=False, truncation=True, max_length=10000
                            )["input_ids"]
                            prompt = self.tokenizer.decode(
                                prompt_answer_input_ids[:int(total_length - response_length)], skip_special_tokens=True
                            )
                            # for prompt in prompts:
                            example = {
                                "idx": experience.info['idx'],
                                "prompt": prompt,
                            }
                            json.dump(example, f, ensure_ascii=False)
                            f.write("\n")

                global_experiences = after_filter_global_experiences
                if len(global_experiences) < args.rollout_batch_size*chunk_size:
                    continue
                
                # 记录训练步骤开始时间
                current_time = time.time()
                if self.step_start_time is None:
                    self.step_start_time = current_time
                
                # 计算步骤间隔时间
                step_interval = current_time - self.last_step_end_time if self.last_step_end_time is not None else 0
                
                experiences = global_experiences[:args.rollout_batch_size*chunk_size]
                if args.enable_self_evolution:
                    with open(os.path.join(self.strategy.args.filtered_data_dir, f"keep_train_data_{steps}.jsonl"), "a", encoding="utf-8") as f:
                        for chunk_i in range(len(experiences)//chunk_size):
                            experience = experiences[chunk_i*chunk_size]
                            total_length = experience.info['total_length'][0].item()
                            response_length = experience.info['response_length'][0].item()
                            prompt_answer = self.tokenizer.batch_decode(
                                experience.sequences[0].unsqueeze(0), skip_special_tokens=True
                            )[0]
                            prompt_answer_input_ids = self.tokenizer(
                                prompt_answer, add_special_tokens=False, truncation=True, max_length=10000
                            )["input_ids"]
                            prompt = self.tokenizer.decode(
                                prompt_answer_input_ids[:int(total_length - response_length)], skip_special_tokens=True
                            )
                            example = {
                                "idx": experience.info['idx'],
                                "prompt": prompt,
                            }
                            json.dump(example, f, ensure_ascii=False)
                            f.write("\n")

                for experience in experiences:
                    del experience.info['idx']
                    del experience.info['append_steps']

                global_experiences = global_experiences[args.rollout_batch_size*chunk_size:]
                self.strategy.print(output)
                refs = self.actor_model_group.async_run_method_batch(method_name="append", experience=experiences, step=[steps]*len(experiences))
                if self.critic_model_group is not None:
                    refs.extend(
                        self.critic_model_group.async_run_method_batch(method_name="append", experience=experiences)
                    )
                ray.get(refs)

                status = self.ppo_train(steps)

                if "kl" in status:
                    self.kl_ctl.update(status["kl"], args.rollout_batch_size * args.n_samples_per_prompt)
                pbar.set_postfix(status)

                
                # logs/checkpoints
                client_states = {"consumed_samples": steps * args.rollout_batch_size}
                self.save_logs_and_checkpoints(args, steps, pbar, status, client_states)

                pbar.update()
                
                # 记录训练步骤结束时间并计算耗时
                step_end_time = time.time()
                step_duration = step_end_time - current_time
                
                # 保存时间统计信息到txt文件
                if self.strategy.is_rank_0():
                    timing_file_path = os.path.join(self.args.save_path, "training_timing_stats.txt")
                    os.makedirs(os.path.dirname(timing_file_path), exist_ok=True)
                    with open(timing_file_path, "a", encoding="utf-8") as f:
                        f.write(f"{steps},{step_interval:.6f},{step_duration:.6f}\n")
                
                self.last_step_end_time = step_end_time
                steps = steps + 1

                if args.enable_self_evolution:
                    global_batch_data = []
                    global_batch_data_idxs = []
                # if len(self.replay_buffer) % args.n_samples_per_prompt != 0:
                #     breakpoint()

                # self.replay_buffer.clear_rule(step=steps, n_samples_per_prompt=args.n_samples_per_prompt)

                # if len(self.replay_buffer) % args.n_samples_per_prompt != 0:
                #     breakpoint()


        
        self.logger.finish()
        # if self._tensorboard is not None and self.strategy.is_rank_0():
        #     self._tensorboard.close()



    def ppo_train(self, global_steps):
        status = {}

        # triger remote critic model training
        if self.critic_model_group is not None:
            # sync for deepspeed_enable_sleep
            if self.strategy.args.deepspeed_enable_sleep:
                ray.get(self.critic_model_group.async_run_method(method_name="reload_states"))

            critic_status_ref = self.critic_model_group.async_run_method(method_name="fit")

            if self.strategy.args.colocate_all_models or self.strategy.args.deepspeed_enable_sleep:
                status.update(ray.get(critic_status_ref)[0])
            if self.strategy.args.deepspeed_enable_sleep:
                ray.get(self.critic_model_group.async_run_method(method_name="offload_states"))

        # actor model training
        if global_steps > self.freezing_actor_steps:
            if self.strategy.args.deepspeed_enable_sleep:
                self.actor_model_group.async_run_method(method_name="reload_states")

            actor_status_ref = self.actor_model_group.async_run_method(method_name="fit", kl_ctl=self.kl_ctl.value)
            status.update(ray.get(actor_status_ref)[0])

            if self.strategy.args.deepspeed_enable_sleep:
                self.actor_model_group.async_run_method(method_name="offload_states")

            # 4. broadcast weights to vllm engines
            if self.vllm_engines is not None:
                if self.strategy.args.vllm_enable_sleep:
                    from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call

                    batch_vllm_engine_call(self.vllm_engines, "wake_up")

                ray.get(self.actor_model_group.async_run_method(method_name="broadcast_to_vllm"))

                if self.strategy.args.vllm_enable_sleep:
                    batch_vllm_engine_call(self.vllm_engines, "sleep")

        # 5. wait remote critic model training done
        if self.critic_model_group and not self.strategy.args.colocate_all_models:
            status.update(ray.get(critic_status_ref)[0])

        return status

    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}):
        if global_step % args.logging_steps == 0:
            # 统一日志记录
            if self.logger.is_available():
                logs = {"train/%s" % k: v for k, v in logs_dict.items()}
                self.logger.log(logs, step=global_step)

        # TODO: Add evaluation mechanism for PPO
        if global_step % args.eval_steps == 0 and self.eval_dataloader and len(self.eval_dataloader) > 0:
            self.evaluate(self.eval_dataloader, global_step, args.eval_temperature, args.eval_n_samples_per_prompt)
        # save ckpt
        # TODO: save best model on dev, use loss/perplexity/others on whole dev dataset as metric
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            ref = self.actor_model_group.async_run_method(
                method_name="save_checkpoint", tag=tag, client_states=client_states
            )
            if self.critic_model_group is not None:
                ref.extend(self.critic_model_group.async_run_method(method_name="save_checkpoint", tag=tag))
            ray.get(ref)

    def evaluate(self, eval_dataloader, global_step, temperature=0.6, n_samples_per_prompt=1):
        """Evaluate model performance on eval dataset.

        Args:
            eval_dataloader: DataLoader containing evaluation prompts, labels and data sources
            global_step: Current training step for logging
            n_samples_per_prompt: Number of samples to generate per prompt for pass@k calculation
        """
        start_time = time.time()
        logger.info(f"⏰ Evaluation start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # vLLM wakeup when vllm_enable_sleep
        if self.strategy.args.vllm_enable_sleep:
            from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call

            batch_vllm_engine_call(self.vllm_engines, "wake_up")

        with torch.no_grad():
            # First collect all prompts and labels
            all_prompts = []
            all_labels = []
            all_datasources = []
            all_other_infos = []

            for datasources, prompts, labels, other_infos in eval_dataloader:
                all_prompts.extend(prompts)
                all_labels.extend(labels)
                all_datasources.extend(datasources)
                other_infos = {
                   k: other_infos[k].item() for k in other_infos
                }
                all_other_infos.append(other_infos)

            # Generate samples and calculate rewards
            generate_kwargs = self.generate_kwargs.copy()
            generate_kwargs["temperature"] = temperature
            generate_kwargs["n_samples_per_prompt"] = n_samples_per_prompt
            samples = self.experience_maker.generate_samples(all_prompts, all_labels, **generate_kwargs)
            # queries_list = [self.tokenizer.batch_decode(seq, skip_special_tokens=False) for seq in samples.sequences]
            # queries_list = [self.tokenizer.batch_decode(seq, skip_special_tokens=True) for seq in samples.sequences]
            queries_list = self.tokenizer.batch_decode(samples.sequences, skip_special_tokens=True)

            # duplicate prompts and labels for each sample
            all_prompts = sum([[prompt] * n_samples_per_prompt for prompt in all_prompts], [])
            all_labels = sum([[label] * n_samples_per_prompt for label in all_labels], [])
            # Calculate rewards
            if self.experience_maker.eval_custom_reward_func:
                # Let Ray automatically distribute the workload across available resources
                batch_size = self.strategy.args.micro_rollout_batch_size
                num_chunks = (len(queries_list) + batch_size - 1) // batch_size
                r_refs = []
                for i in range(num_chunks):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, len(queries_list))
                    r = self.experience_maker.eval_custom_reward_func.remote(
                        queries_list[start_idx:end_idx],
                        all_prompts[start_idx:end_idx],
                        all_labels[start_idx:end_idx],
                    )
                    r_refs.append(r)
            else:
                # Distribute data across different remote reward function servers
                num_servers = len(self.remote_rm_url)
                batch_size = (len(queries_list) + num_servers - 1) // num_servers
                r_refs = []
                for i in range(num_servers):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, len(queries_list))
                    rm = self.remote_rm_url[i]
                    r = remote_rm_fn_ray.remote(
                        rm,
                        queries=queries_list[start_idx:end_idx],
                        prompts=all_prompts[start_idx:end_idx],
                        labels=all_labels[start_idx:end_idx],
                    )
                    r_refs.append(r)
            # Reshape rewards to (num_prompts, n_samples_per_prompt)
            rewards = ray.get(r_refs)
            rewards = torch.cat(rewards, dim=0).reshape(-1, n_samples_per_prompt)

            # Collect local statistics for each data source
            global_metrics = {}  # {datasource: {"pass{n_samples_per_prompt}": 0, "pass1": 0, "count": 0}}

            level2acc = {}
            level2cnt = {}
            for i, datasource in enumerate(all_datasources):
                if datasource not in global_metrics:
                    global_metrics[datasource] = {f"pass{n_samples_per_prompt}": 0, "pass1": 0, "count": 0}

                # Calculate pass@k and pass@1
                prompt_rewards = rewards[i]
                if n_samples_per_prompt != 1:
                    global_metrics[datasource][f"pass{n_samples_per_prompt}"] += prompt_rewards.max().float().item()
                global_metrics[datasource]["pass1"] += prompt_rewards.mean().float().item()
                global_metrics[datasource]["count"] += 1
                if all_other_infos[i]['level'] not in level2acc:
                    level2acc[all_other_infos[i]['level']] = 0
                    level2cnt[all_other_infos[i]['level']] = 0
                level2acc[all_other_infos[i]['level']] += prompt_rewards.mean().float().item()
                level2cnt[all_other_infos[i]['level']] += 1


            # Calculate global averages
            logs = {}
            for datasource, metrics in global_metrics.items():
                logs[f"eval_{datasource}_pass{n_samples_per_prompt}"] = (
                    metrics[f"pass{n_samples_per_prompt}"] / metrics["count"]
                )
                logs[f"eval_{datasource}_pass1"] = metrics["pass1"] / metrics["count"]
            for level in level2acc:
                logs[f"eval_{level}_pass1"] = level2acc[level] / level2cnt[level]
            # 统一日志记录
            if self.logger.is_available():
                eval_logs = {"eval/%s" % k: v for k, v in logs.items()}
                self.logger.log(eval_logs, step=global_step)

        if self.strategy.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "sleep")

        end_time = time.time()
        duration = end_time - start_time
        if self.strategy.is_rank_0():
            time_str = str(timedelta(seconds=duration)).split(".")[0]
            logger.info(f"✨ Evaluation completed in {time_str}")
            output_dir = os.path.join(self.strategy.args.eval_output_dir, f"step_{global_step}")
            examples = []
            for i in range(len(all_other_infos)):
                prompt_i = i*n_samples_per_prompt
                example = {
                    "idx": all_other_infos[i]['idx'],
                    "prompt": all_prompts[prompt_i],
                    "label": all_labels[prompt_i],
                    "datasource": all_datasources[prompt_i],
                    "level": all_other_infos[i]['level'],
                    "responses": queries_list[prompt_i:prompt_i+n_samples_per_prompt],
                    "reward": rewards[i].tolist(),
                }
                examples.append(example)
            # Save examples to JSONL file
            save_jsonl(examples, os.path.join(output_dir, f"step_{global_step}_qa.jsonl"))
            level2acc['overall'] = sum([level2acc[level]*level2cnt[level] for level in level2acc])/sum(level2cnt.values())
            level2cnt['overall'] = sum(level2cnt.values())
            with open(
                os.path.join(output_dir, f'step_{global_step}_acc.jsonl'), 'w', encoding='utf-8'
            ) as f:
                json.dump(level2acc, f, ensure_ascii=False, indent=4)
            with open(
                os.path.join(output_dir, f'step_{global_step}_cnt.jsonl'), 'w', encoding='utf-8'
            ) as f:
                json.dump(level2cnt, f, ensure_ascii=False, indent=4)
            logger.info(f"✨ Evaluation results saved to {output_dir}")
            
                

    def prepare_datasets(self):
        args = self.args
        strategy = self.strategy

        # prepare datasets
        train_data = blending_datasets(
            args.prompt_data,
            args.prompt_data_probs,
            strategy,
            args.seed,
            max_count=args.max_samples,
        )
        

        # Create train dataset
        train_data = train_data.select(range(min(args.max_samples, len(train_data))))
        prompts_dataset = PromptDataset(train_data, self.tokenizer, strategy, input_template=args.input_template)
        if args.enable_self_evolution:
            self.seed_data = train_data[args.input_key]
        data_batch_size = 1 if args.enable_self_evolution else args.rollout_batch_size
        prompts_dataloader = strategy.setup_dataloader(
            prompts_dataset,
            # args.rollout_batch_size,
            data_batch_size,
            True,
            True,
        )

        # Create eval dataset if eval data exists
        if getattr(args, "eval_dataset", None):
            eval_data = blending_datasets(
                args.eval_dataset,
                None,  # No probability sampling for eval datasets
                strategy,
            )
            eval_data = eval_data.select(range(min(args.max_samples, len(eval_data))))
            eval_dataset = PromptDataset(eval_data, self.tokenizer, strategy, input_template=args.input_template)
            eval_dataloader = strategy.setup_dataloader(eval_dataset, 1, True, False)
        else:
            eval_dataloader = None

        self.prompts_dataloader = prompts_dataloader
        self.eval_dataloader = eval_dataloader
        if args.enable_self_evolution:

            # Align with the scale of other experimental data: 7500 samples
            self.max_steps = args.instructions_num_per_iteration * args.n_samples_per_prompt // args.train_batch_size
        else:
            self.max_steps = len(prompts_dataset) * args.n_samples_per_prompt // args.train_batch_size

    def get_max_steps(self):
        return self.max_steps
