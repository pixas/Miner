import itertools
import os
import shutil
from typing import Dict
from numpy import positive
from enum import Enum
# from verl.trainer.ppo.ray_trainer import *
from verl.trainer.ppo.ray_trainer import *
from . import core_algos
from .core_algos import AdvantageEstimator
from verl.utils.proxy import send_feishu_message


entropy_list = []

def compute_adaptive_alpha(current_entropy, entropy_history, 
                          alpha_max=0.04, alpha_min=0.005, 
                          k=10.0, hysteresis=0.15):
    """
    基于熵历史的自适应alpha计算
    
    参数:
    current_entropy: 当前批次的平均策略熵
    entropy_history: 滑动窗口内的历史熵值列表
    alpha_max: 最大alpha值（熵最小时使用）
    alpha_min: 最小alpha值（熵最大时使用）
    k: Sigmoid函数的陡峭度参数
    hysteresis: 滞后系数，防止边界震荡
    """
    if len(entropy_history) < 5:
        # 训练初期，使用固定值
        return alpha_max * 0.5
    
    # 1. 计算熵的统计特征
    entropy_mean = np.mean(entropy_history[-20:])  # 20步滑动平均
    entropy_std = np.std(entropy_history[-20:]) + 1e-8
    
    # 2. 计算归一化的熵偏差 (z-score)
    entropy_z = (current_entropy - entropy_mean) / entropy_std
    
    # 3. 基于历史趋势的动态目标
    entropy_trend = np.polyfit(range(min(10, len(entropy_history))), 
                              entropy_history[-10:], 1)[0]  # 最近10步的趋势斜率
    
    # 4. 动态调整Sigmoid中心点（带滞后）
    if entropy_trend > 0:  # 熵在增加（探索阶段）
        target_z = -hysteresis  # 更宽松，允许熵继续增加
    else:  # 熵在减少（可能过度确定）
        target_z = hysteresis   # 更严格，提前干预
    
    # 5. Sigmoid控制函数（核心）
    sigmoid_input = k * (target_z - entropy_z)
    alpha_ratio = 1 / (1 + np.exp(sigmoid_input))
    
    # 6. 最终alpha（在最小最大值之间平滑过渡）
    alpha = alpha_min + (alpha_max - alpha_min) * alpha_ratio
    
    return alpha


def compute_log_prob_metrics(data: DataProto):
    reward_scores = data.batch['token_level_scores'].sum(-1)
    old_log_probs = data.batch['old_log_probs']
    
    response_mask = data.batch['response_mask']
    
    # we can log the sequence old-log-prob for positive/negative samples among the batch 
    # we could log the following things:
    # the mean seq-log/ the 25/75 percentile seq-log for positive/negative samples
    # the max token log-prob for positive/negative samples
    # the min token log-prob for positive/negative samples
    
    # first log about seq-log 
    seq_log = (old_log_probs * response_mask).sum(-1) / response_mask.sum(-1)  # shape: [B]
    positive_seq_log = seq_log[reward_scores > 0]
    negative_seq_log = seq_log[reward_scores <= 0]
    log_prob_metrics = {}
    
    if len(positive_seq_log) > 0:
        positive_seq_log_np = positive_seq_log.detach().cpu().numpy()
        log_prob_metrics.update({
            "log_prob/positive_seq_log_mean": positive_seq_log.mean().item(),
            "log_prob/positive_seq_log_max": positive_seq_log.max().item(),
            "log_prob/positive_seq_log_min": positive_seq_log.min().item(),
        })
        # among all correct groups, count also the mean/min/max sequence log prob
        uids = getattr(data, "non_tensor_batch", {}).get("uid", None)
        if uids is not None:
            # A group (uid) is considered correct if all its scores are 1
            reward_scores_np = reward_scores.detach().cpu().numpy()
            seq_log_all_np = seq_log.detach().cpu().numpy()
            tok_log_all_np = old_log_probs.detach().cpu().numpy()
            response_mask_np = response_mask.detach().cpu().numpy()
            uid_all_one = {}
            for uid, is_one in zip(uids, (reward_scores_np == 1)):
                uid_all_one[uid] = uid_all_one.get(uid, True) and bool(is_one)
            if uid_all_one:
                mask = [uid_all_one.get(uid, False) for uid in uids]
                if any(mask):
                    selected = seq_log_all_np[mask]
                    log_prob_metrics.update({
                        "log_prob/correct_group_seq_log_mean": float(selected.mean()),
                        "log_prob/correct_group_seq_log_max": float(selected.max()),
                        "log_prob/correct_group_seq_log_min": float(selected.min()),
                    })
                    selected_tok = tok_log_all_np[mask]
                    selected_mask = response_mask_np[mask]
                    # todo: 之前计算token级别地logp的一些数值有bug，错误地计入了pad token，现在请更新成不计入pad token的
                    
                    valid_token_mask = selected_mask.astype(bool)
                    if valid_token_mask.any():
                        valid_token_logs = selected_tok[valid_token_mask]
                        log_prob_metrics.update({
                            "log_prob/correct_group_token_log_mean": float(valid_token_logs.mean()),
                            "log_prob/correct_group_token_log_max": float(valid_token_logs.max()),
                            "log_prob/correct_group_token_log_min": float(valid_token_logs.min()),
                        })
                    else:
                        log_prob_metrics.update({
                            "log_prob/correct_group_token_log_mean": 0.0,
                            "log_prob/correct_group_token_log_max": 0.0,
                            "log_prob/correct_group_token_log_min": 0.0,
                        })
                else:
                    log_prob_metrics.update({
                        "log_prob/correct_group_seq_log_mean": 0.0,
                        "log_prob/correct_group_seq_log_max": 0.0,
                        "log_prob/correct_group_seq_log_min": 0.0,
                        "log_prob/correct_group_token_log_mean": 0.0,
                        "log_prob/correct_group_token_log_max": 0.0,
                        "log_prob/correct_group_token_log_min": 0.0,
                        
                    })
            else:
                log_prob_metrics.update({
                    "log_prob/correct_group_seq_log_mean": 0.0,
                    "log_prob/correct_group_seq_log_max": 0.0,
                    "log_prob/correct_group_seq_log_min": 0.0,
                    "log_prob/correct_group_token_log_mean": 0.0,
                    "log_prob/correct_group_token_log_max": 0.0,
                    "log_prob/correct_group_token_log_min": 0.0,
                })
        else:
            log_prob_metrics.update({
                "log_prob/correct_group_seq_log_mean": 0.0,
                "log_prob/correct_group_seq_log_max": 0.0,
                "log_prob/correct_group_seq_log_min": 0.0,
            })
    else:
        log_prob_metrics.update({
            "log_prob/positive_seq_log_mean": 0.0,
            "log_prob/positive_seq_log_max": 0.0,
            "log_prob/positive_seq_log_min": 0.0,
        })
    if len(negative_seq_log) > 0:
        negative_seq_log_np = negative_seq_log.detach().cpu().numpy()
        log_prob_metrics.update({
            "log_prob/negative_seq_log_mean": negative_seq_log.mean().item(),
            "log_prob/negative_seq_log_max": negative_seq_log.max().item(),
            "log_prob/negative_seq_log_min": negative_seq_log.min().item(),
        })
    else:
        log_prob_metrics.update({
            "log_prob/negative_seq_log_mean": 0.0,
            "log_prob/negative_seq_log_max": 0.0,
            "log_prob/negative_seq_log_min": 0.0,
        })
    
    # then log about token-log
    token_log = old_log_probs * response_mask  # shape: [B, T]
    positive_token_log = token_log[reward_scores > 0]
    negative_token_log = token_log[reward_scores <= 0]
    positive_mask = response_mask[reward_scores > 0].bool()
    negative_mask = response_mask[reward_scores <= 0].bool()
    positive_token_log = positive_token_log[positive_mask]
    negative_token_log = negative_token_log[negative_mask]
    
    if positive_token_log.numel() > 0:

        log_prob_metrics.update({
            "log_prob/positive_token_log_mean": positive_token_log.mean().item(),
            "log_prob/positive_token_log_max": positive_token_log.max().item(),
            "log_prob/positive_token_log_min": positive_token_log.min().item(),
        })
    else:
        log_prob_metrics.update({
            "log_prob/positive_token_log_mean": 0.0,
            "log_prob/positive_token_log_max": 0.0,
            "log_prob/positive_token_log_min": 0.0,
        })
    if negative_token_log.numel() > 0:

        log_prob_metrics.update({
            "log_prob/negative_token_log_mean": negative_token_log.mean().item(),
            "log_prob/negative_token_log_max": negative_token_log.max().item(),
            "log_prob/negative_token_log_min": negative_token_log.min().item(),
        })
    else:
        log_prob_metrics.update({
            "log_prob/negative_token_log_mean": 0.0,
            "log_prob/negative_token_log_max": 0.0,
            "log_prob/negative_token_log_min": 0.0,
        })
    return log_prob_metrics

def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
    metrics: Dict = None,
    current_progress=-1
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    
    elif adv_estimator == AdvantageEstimator.ENTROPY_ADV:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_ent_adv_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            token_level_entropy=data.batch["entropys"],
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO_NEG:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_neg_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            response_emb=data.batch["response_emb"],
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=config
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.SRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]
        
        ref_log_prob = data.batch['ref_log_prob'] if 'ref_log_prob' in data.batch else None
        log_prob = data.batch['old_log_probs']
        
        # TODO: compute process reward
        # p_t = \sum_{j=0}^t \beta \log \frac{\pi}{\pi_ref}
        beta = 0.05
        process_rewards = beta * torch.cumsum((log_prob - ref_log_prob), dim=-1) * grpo_calculation_mask
        
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_srpo_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            process_rewards=process_rewards,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=config
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.ABSPOS:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]
        
        # ref_log_prob = data.batch['ref_log_prob'] if 'ref_log_prob' in data.batch else None
        # log_prob = data.batch['old_log_probs']
        

        # implicit_reward = (log_prob - ref_log_prob) * grpo_calculation_mask

        
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_abspos_reward_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=config
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO_ENT:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_ent_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            token_level_entropy=data.batch["entropys"],
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=config
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns

    elif adv_estimator == AdvantageEstimator.MINER:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]
        entropy_list.append(data.meta_info['entropy'])
        additional_metrics = {}
        if not config.miner.static_alpha:
            new_alpha = compute_adaptive_alpha(data.meta_info['entropy'], entropy_list, config.miner.max_alpha,
                                               config.miner.min_alpha)
            config.miner.correct_scale = new_alpha 
        
        if current_progress <= config.miner.threshold:
            advantages, returns = core_algos.compute_grpo_outcome_advantage(
                token_level_rewards=data.batch["token_level_rewards"],
                response_mask=grpo_calculation_mask,
                index=data.non_tensor_batch["uid"],
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                config=config,
            )
        else:
            # Call compute_grpo_outcome_advantage with parameters matching its definition
            advantages, returns, additional_metrics = core_algos.compute_ent_inc_advantage(
                token_level_rewards=data.batch["token_level_rewards"],
                response_mask=grpo_calculation_mask,
                token_level_logp=data.batch['old_log_probs'],
                index=data.non_tensor_batch["uid"],
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                config=config,
            )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        metrics.update(additional_metrics)
    
    elif adv_estimator == AdvantageEstimator.GTPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_gtpo_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            token_level_entropy=data.batch["entropys"],
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=config
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    
    elif adv_estimator == AdvantageEstimator.UCAS:
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_ucas_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            token_level_logp=data.batch['old_log_probs'],
            token_logits=data.batch['token_logits'],
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=config
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    
    elif adv_estimator == AdvantageEstimator.GRPO_S:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_s_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            token_level_entropy=data.batch["entropys"],
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=config
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        
    elif adv_estimator == AdvantageEstimator.GRPO_EDGE:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_edge_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            token_level_entropy=data.batch["entropys"],
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=config
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class BaselinePPOTrainer(RayPPOTrainer):
    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        # reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys())
        uids = batch.non_tensor_batch.get("uid", None)
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )
        batch.non_tensor_batch["uid"] = uids

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _save_checkpoint(self):
        super()._save_checkpoint()
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )
        max_actor_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None)
        if max_actor_ckpt_to_keep is not None:
            try:
                max_ckpt_to_keep = int(max_actor_ckpt_to_keep)
            except (TypeError, ValueError):
                max_ckpt_to_keep = -1
            if max_ckpt_to_keep >= 0:
                base_dir = os.path.dirname(local_global_step_folder)
                prefix = "global_step_"
                checkpoint_dirs = []
                if os.path.isdir(base_dir):
                    with os.scandir(base_dir) as entries:
                        for entry in entries:
                            if entry.is_dir() and entry.name.startswith(prefix):
                                step_suffix = entry.name[len(prefix) :]
                                try:
                                    step = int(step_suffix)
                                except ValueError:
                                    continue
                                checkpoint_dirs.append((step, entry.path))
                checkpoint_dirs.sort(key=lambda item: item[0])
                excess = len(checkpoint_dirs) - max_ckpt_to_keep
                if excess > 0:
                    for _, ckpt_path in checkpoint_dirs[:excess]:
                        actor_dir = os.path.join(ckpt_path, "actor")
                        if os.path.isdir(actor_dir):
                            shutil.rmtree(actor_dir, ignore_errors=True)


        
        
        # todo: additionaly，save the entropy_list which contains a list of float numbers 
        entropy_np = np.array(entropy_list)
        np.save(os.path.join(local_global_step_folder, "entropy_list.npy"), entropy_np)
    
    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")
        
        global entropy_list
        if os.path.exists(os.path.join(global_step_folder, "entropy_list.npy")):
            entropy_list = np.load(os.path.join(global_step_folder, "entropy_list.npy")).tolist()
    
    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.gen_steps = 0
        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        self.gen_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False
        
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        timing_raw = defaultdict(float)

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                new_batch: DataProto = DataProto.from_single_dict(batch_dict)

                # add uid to batch
                # batch.non_tensor_batch["uid"] = np.array(
                #     [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                # )
                num_gen_batches += 1
                gen_batch = self._get_gen_batch(new_batch)

                # pass global_steps to trace
                # gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            new_batch = new_batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output
                    # repeat to align with repeated responses in rollout
                    
                    
                    # batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    # batch = batch.union(gen_batch_output)
                    
                    
                    new_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                    )
                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)
                    
                    
                    with marked_timer("reward", timing_raw, "yellow"):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(data=new_batch, reward_fn=self.reward_fn)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(new_batch, self.reward_fn)

                        new_batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(
                                new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(
                                kl_metrics
                            )  # TODO: This will be cleared if we use multiple genenration batches
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                    else:  # NOTE: When prompts after filtering is less than train batch size,
                        # we skip to the next generation batch
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            # Turn to numpy for easier filtering
                            new_batch.non_tensor_batch["seq_final_reward"] = (
                                new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                            )
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = (
                                new_batch.batch["token_level_scores"].sum(dim=-1).numpy()
                            )

                        # Collect the sequence reward for each trajectory
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(
                            new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name], strict=True
                        ):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        prompt_uid2metric_std = {}
                        prompt_uid2metric_mean = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)
                            prompt_uid2metric_mean[prompt_uid] = np.mean(metric_vals)
                        filter_group_cfg =self.config.algorithm.filter_groups
                        if filter_group_cfg.filter_allc and filter_group_cfg.filter_allw:
                            condition = lambda std, mean: std > 0 
                        elif filter_group_cfg.filter_allc and (not filter_group_cfg.filter_allw):
                            condition = lambda std, mean: std > 0 or (std == 0 and mean == 0)
                        elif (not filter_group_cfg.filter_allc) and filter_group_cfg.filter_allw:
                            condition = lambda std, mean: std > 0 or (std == 0 and mean == 1)
                        kept_prompt_uids = [
                            uid
                            for (uid, std), (_, mean) in zip(prompt_uid2metric_std.items(), prompt_uid2metric_mean.items()) 
                            if condition(std, mean) or len(prompt_uid2metric_vals[uid]) == 1
                        ]
                        num_prompt_in_batch += len(kept_prompt_uids)

                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)

                        new_batch = new_batch[kept_traj_idxs]
                        batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f"{num_gen_batches=}. Keep generating...")
                                self.gen_steps += 1
                                is_last_step = self.global_steps >= self.total_training_steps
                                continue
                            else:
                                raise ValueError(
                                    f"{num_gen_batches=} >= {max_num_gen_batches=}."
                                    + " Generated too many. Please check if your data are too difficult."
                                    + " You could also try set max_num_gen_batches=0 to enable endless trials."
                                )
                        else:
                            # Align the batch
                            traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                            batch = batch[:traj_bsz]

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                        
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)



                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()



                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        batch.meta_info['compute_emb'] = self.config.algorithm.neg_scale.enable
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        # batch.batch['token_logits'] = old_log_prob.batch['token_logits']
                        
                        
                        batch = batch.union(old_log_prob)
                        batch.meta_info['entropy'] = entropy_agg
                        entropy_list.append(entropy_agg)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            from verl.utils.debug.metrics import \
                                calculate_debug_metrics

                            metrics.update(calculate_debug_metrics(batch))
                        
                        metrics.update(compute_log_prob_metrics(batch))

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                            batch.batch["token_level_scores"] = reward_tensor

                            if reward_extra_infos_dict:
                                batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                            # compute rewards. apply_kl_penalty if available
                            if self.config.algorithm.use_kl_in_reward:
                                batch, kl_metrics = apply_kl_penalty(
                                    batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                                )
                                metrics.update(kl_metrics)
                            else:
                                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                            metrics=metrics,
                            current_progress=self.global_steps / self.total_training_steps
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing

                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0
                
                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                self.gen_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    pprint(f'Final validation metrics: {last_val_metrics}')
                    feishu_message = {**last_val_metrics, "exp_name": self.config.trainer.experiment_name,}
                    send_feishu_message(feishu_message)
                    progress_bar.close()
                    return 

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
