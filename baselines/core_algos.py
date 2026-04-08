# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO-like algorithms.
"""

__all__ = ["register_adv_est", "get_adv_estimator_fn", "AdvantageEstimator"]

from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

import verl.utils.torch_functional as verl_F
from verl.trainer.config import AlgoConfig
from verl.utils import as_torch_index, group_mean_std
from verl.utils.import_utils import deprecated
from verl.workers.config import ActorConfig

PolicyLossFn = Callable[
    [
        torch.Tensor,  # old_log_prob
        torch.Tensor,  # log_prob
        torch.Tensor,  # advantages
        torch.Tensor,  # response_mask
        str,  # loss_agg_mode
        Optional[DictConfig | AlgoConfig],  # config
        torch.Tensor | None,  # rollout_log_probs
    ],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
]

POLICY_LOSS_REGISTRY: dict[str, PolicyLossFn] = {}


def register_policy_loss(name: str) -> Callable[[PolicyLossFn], PolicyLossFn]:
    """Register a policy loss function with the given name.

    Args:
        name (str): The name to register the policy loss function under.

    Returns:
        function: Decorator function that registers the policy loss function.
    """

    def decorator(func: PolicyLossFn) -> PolicyLossFn:
        POLICY_LOSS_REGISTRY[name] = func
        return func

    return decorator


def get_policy_loss_fn(name):
    """Get the policy loss with a given name.

    Args:
        name: `(str)`
            The name of the policy loss.

    Returns:
        `(callable)`: The policy loss function.
    """
    loss_name = name
    if loss_name not in POLICY_LOSS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(POLICY_LOSS_REGISTRY.keys())}"
        )
    return POLICY_LOSS_REGISTRY[loss_name]


class AdvantageEstimator(str, Enum):
    """Using an enumeration class to avoid spelling errors in adv_estimator.

    Note(haibin.lin): this enum class is immutable after creation. Extending this
    enum for new estimators may not be necessary since users can always just call
    `verl.trainer.ppo.core_algos.register` with string name for a custom advantage
    estimator instead.
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    OPO = "opo"
    GRPO_PASSK = "grpo_passk"
    GPG = "gpg"
    RLOO_VECTORIZED = "rloo_vectorized"
    GRPO_VECTORIZED = "grpo_vectorized"
    SRPO = "srpo"
    GRPO_NEG = "grpo_neg"
    WREINFORCE = "wreinforce"
    GRPO_ENT = "grpo_ent"
    GTPO = "gtpo"
    GRPO_S = "grpo_s"
    GRPO_EDGE = 'grpo_edge'
    MINER = 'miner'
    ENT_INC_RPP = 'ent_inc_rpp'
    ENTROPY_ADV = 'entropy_adv'
    ABSPOS = 'abspos'
    UCAS = 'ucas'


ADV_ESTIMATOR_REGISTRY: dict[str, Any] = {}


def register_adv_est(name_or_enum: str | AdvantageEstimator) -> Any:
    """Decorator to register a advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    """

    def decorator(fn):
        name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
        if name in ADV_ESTIMATOR_REGISTRY and ADV_ESTIMATOR_REGISTRY[name] != fn:
            raise ValueError(
                f"Adv estimator {name} has already been registered: {ADV_ESTIMATOR_REGISTRY[name]} vs {fn}"
            )
        ADV_ESTIMATOR_REGISTRY[name] = fn
        return fn

    return decorator


def get_adv_estimator_fn(name_or_enum):
    """Get the advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    Returns:
        `(callable)`: The advantage estimator function.
    """
    name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
    if name not in ADV_ESTIMATOR_REGISTRY:
        raise ValueError(f"Unknown advantage estimator simply: {name}")
    return ADV_ESTIMATOR_REGISTRY[name]


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        """Update the KL coefficient based on current KL divergence.

        Args:
            current_kl (float): Current KL divergence value.
            n_steps (int): Number of steps taken.
        """
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        """Update method for fixed KL controller (no-op).

        Args:
            current_kl (float): Current KL divergence value (unused).
            n_steps (int): Number of steps taken (unused).
        """
        pass


def get_kl_controller(kl_ctrl):
    """Factory function to create appropriate KL controller based on configuration.

    Args:
        kl_ctrl: Configuration object containing KL controller settings.

    Returns:
        KL controller instance (FixedKLController or AdaptiveKLController).

    Raises:
        NotImplementedError: If controller type is not supported.
        AssertionError: If adaptive controller horizon is not positive.
    """
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert (
            kl_ctrl.horizon > 0
        ), f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(
            init_kl_coef=kl_ctrl.kl_coef,
            target_kl=kl_ctrl.target_kl,
            horizon=kl_ctrl.horizon,
        )
    else:
        raise NotImplementedError


@register_adv_est(AdvantageEstimator.GAE)  # or simply: @register_adv_est("gae")
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        values: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma is `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        nextvalues = 0
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam_ = delta + gamma * lam * lastgaelam

            # skip values and TD-error on observation tokens
            nextvalues = (
                values[:, t] * response_mask[:, t]
                + (1 - response_mask[:, t]) * nextvalues
            )
            lastgaelam = (
                lastgaelam_ * response_mask[:, t]
                + (1 - response_mask[:, t]) * lastgaelam
            )

            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
@register_adv_est(AdvantageEstimator.GRPO)  # or simply: @register_adv_est("grpo")
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        index: `(np.ndarray)`
            index array for grouping
        epsilon: `(float)`
            small value to avoid division by zero
        norm_adv_by_std_in_grpo: `(bool)`
            whether to scale the GRPO advantage
        config: `(Optional[AlgoConfig])`
            algorithm configuration object

    Note:
        If norm_adv_by_std_in_grpo is True, the advantage is scaled by the std, as in the original GRPO.
        If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (
                    id2std[index[i]] + epsilon
                )
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores

# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
@register_adv_est(AdvantageEstimator.ENTROPY_ADV)  # or simply: @register_adv_est("grpo")
def compute_grpo_ent_adv_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    token_level_entropy: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        index: `(np.ndarray)`
            index array for grouping
        epsilon: `(float)`
            small value to avoid division by zero
        norm_adv_by_std_in_grpo: `(bool)`
            whether to scale the GRPO advantage
        config: `(Optional[AlgoConfig])`
            algorithm configuration object

    Note:
        If norm_adv_by_std_in_grpo is True, the advantage is scaled by the std, as in the original GRPO.
        If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (
                    id2std[index[i]] + epsilon
                )
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask
        entropy_cfg = getattr(config, "entropy_adv", None) if config is not None else None
        if entropy_cfg is not None:
            alpha = getattr(entropy_cfg, "alpha", 0.0)
            kappa = getattr(entropy_cfg, "kappa", 1.0)
            if alpha > 0:
                entropy_term = token_level_entropy * response_mask
                adv_prime = torch.minimum(
                    scores.new_tensor(alpha) * entropy_term,
                    torch.abs(scores) / scores.new_tensor(max(kappa, epsilon)),
                )
                scores = scores + adv_prime
        scores = scores * response_mask

    return scores, scores



# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
@register_adv_est(AdvantageEstimator.ABSPOS)  # or simply: @register_adv_est("grpo")
def compute_abspos_reward_outcome_advantage( 
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        index: `(np.ndarray)`
            index array for grouping
        epsilon: `(float)`
            small value to avoid division by zero
        norm_adv_by_std_in_grpo: `(bool)`
            whether to scale the GRPO advantage
        config: `(Optional[AlgoConfig])`
            algorithm configuration object

    Note:
        If norm_adv_by_std_in_grpo is True, the advantage is scaled by the std, as in the original GRPO.
        If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            original_score = scores[i]
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (
                    id2std[index[i]] + epsilon
                )
            else:
                scores[i] = scores[i] - id2mean[index[i]]
            if original_score > 0:
                scores[i] = 0.005
            # else:
                # implicit_scaler[i] = torch.maximum(implicit_scaler[i], torch.ones_like(implicit_scaler[i]))
                # implicit_scaler[i] = torch.clamp(implicit_scaler[i], 1, 2)
        scores = scores.unsqueeze(-1) * response_mask
        # scores *= implicit_scaler
        # scores = scores * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.GTPO)
def compute_gtpo_advantage(
    token_level_rewards: torch.Tensor,
    token_level_entropy: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
):
    # scores = token_level_rewards.sum(dim=-1)
    # entropys = token_level_entropy * response_mask
    entropys = token_level_entropy
    scores = (
        token_level_rewards.sum(dim=-1, keepdim=True).repeat(1, token_level_rewards.shape[1])
        * response_mask
    )  # [each valid location would be the valid score]
    id2score = defaultdict(list)
    id2entropy = defaultdict(list)
    id2batchidx = defaultdict(list)
    scaled_reward_tensor = torch.zeros_like(token_level_rewards)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
            id2entropy[index[i]].append(entropys[i])
            id2batchidx[index[i]].append(i)
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                entropy_tensor = torch.stack(id2entropy[idx])
                batchidx_tensor = torch.tensor(id2batchidx[idx], device=scores.device)
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")

            # based on the scores_tensor and entropy_tensor, we want to compute a new reward
            # for the scores_tensor[i, t] == 1 where response_mask[i, t] == 1, compute the scaled reward as below:
            # r[i, t] = config.gtpo.alpha1 * 1 + config.gtpo.alpha2 * (entropy[i, t]) / sum(entropy[:, t] and scores[:, t] > 0 and response_mask[:, t] == 1) * (response_mask[:, t].sum())
            # symmetrically for scores_tensor[i, t] == 0 where response_mask[i, t] == 1:
            group_mask = response_mask[batchidx_tensor]
            scaled_rewards = torch.where(
                torch.logical_and(scores_tensor > 0, group_mask == 1),
                config.gtpo.alpha1 * 1
                + config.gtpo.alpha2
                * (entropy_tensor)
                / (
                    torch.sum(
                        entropy_tensor * (scores_tensor > 0).float() * group_mask, dim=0
                    )
                    + epsilon
                )
                * (
                    torch.sum(
                        torch.logical_and(group_mask, (scores_tensor > 0)), dim=0
                    ).float()
                ),
                torch.where(
                    torch.logical_and(scores_tensor <= 0, group_mask == 1),
                    -config.gtpo.alpha1
                    - config.gtpo.alpha2
                    * (1 / (entropy_tensor + epsilon))
                    / (
                        torch.sum(
                            1
                            / (entropy_tensor + epsilon)
                            * (scores_tensor <= 0).float()
                            * group_mask,
                            dim=0,
                        )
                        + epsilon
                    )
                    * (
                        torch.sum(
                            torch.logical_and(group_mask, (scores_tensor <= 0)), dim=0
                        ).float()
                    ),
                    0,
                ),
            )

            scaled_rewards = scaled_rewards * group_mask
            scaled_reward_tensor[batchidx_tensor] = scaled_rewards

        # compute the advantage based on the scaled_reward_tensor
        # for correct tokens, compute the global mean and std, for incorrect tokens, compute the global mean and std
        # separately normalize the correct and incorrect tokens
        # print("scaled_reward_tensor isnan:", scaled_reward_tensor.isnan().any())
        correct_token_mask = torch.logical_and(
            scores > 0, response_mask == 1
        )
        incorrect_token_mask = torch.logical_and(
            scores <= 0, response_mask == 1
        )

        correct_advantages = (
            verl_F.masked_whiten(
                scaled_reward_tensor * correct_token_mask.float(), correct_token_mask
            )
            * correct_token_mask.float()
        )

        incorrect_advantages = (
            verl_F.masked_whiten(
                scaled_reward_tensor * incorrect_token_mask.float(),
                incorrect_token_mask,
            )
            * incorrect_token_mask.float()
        )
        # print("correct_advantages isnan:", correct_advantages.isnan().any())
        # print("incorrect_advantages isnan:", incorrect_advantages.isnan().any())
        advantages = correct_advantages + incorrect_advantages
        advantages = advantages * response_mask
        
    return advantages, advantages

@register_adv_est(AdvantageEstimator.GRPO_S)
def compute_grpo_s_advantage(
    token_level_rewards: torch.Tensor,
    token_level_entropy: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
):
    # entropys = token_level_entropy * response_mask
    seq_entropys = (token_level_entropy * response_mask).sum(-1) / (response_mask.sum(-1))  # [B, ]
    scores = token_level_rewards.sum(-1)
    id2score = defaultdict(list)
    id2entropy = defaultdict(list)
    id2batchidx = defaultdict(list)
    advantage = torch.zeros_like(scores)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
            id2entropy[index[i]].append(seq_entropys[i])
            id2batchidx[index[i]].append(i)
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                entropy_tensor = torch.stack(id2entropy[idx])
                batchidx_tensor = torch.tensor(id2batchidx[idx], device=scores.device)
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")

            scaled_rewards = torch.where(scores_tensor > 0, 
                                         config.grpo_s.beta1 * 1 + config.grpo_s.beta2 * entropy_tensor / ((entropy_tensor * (scores_tensor > 0).float()).sum()) * ((scores_tensor > 0).float().sum()),
                                            -config.grpo_s.beta1 * 1 - config.grpo_s.beta2 * (1 / (entropy_tensor+epsilon)) / ((1 / (entropy_tensor + epsilon) * (scores_tensor <= 0).float()).sum()) * ((scores_tensor <= 0).float().sum())
                                            )
            scaled_rewards_mean = torch.mean(scaled_rewards)
            scaled_rewards_std = torch.std(scaled_rewards)
            advantage[batchidx_tensor] = (scaled_rewards - scaled_rewards_mean) / (scaled_rewards_std + epsilon)
            
        advantage = advantage.unsqueeze(-1) * response_mask

    return advantage, advantage


@register_adv_est(AdvantageEstimator.SRPO)
def compute_srpo_advantage(
    token_level_rewards: torch.Tensor,
    process_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2proc_score = defaultdict(list)
    id2mean = {}
    id2proc_mean = {}
    id2std = {}
    
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
            # keep structure consistent; process handled in a vectorized pass below
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (
                    id2std[index[i]] + epsilon
                )
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        # outcome-level advantage broadcast to tokens
        outcome_adv = scores.unsqueeze(-1) * response_mask

        # process-level advantage: subtract group-wise masked mean per timestep
        # compute per-group masked mean efficiently
        proc_adv = torch.zeros_like(process_rewards)
        # ensure tensors are on same device/dtype
        process_rewards_t = process_rewards
        response_mask_t = response_mask
        # group by index
        unique_ids = np.unique(index)
        for gid in unique_ids:
            batch_sel = np.where(index == gid)[0]
            if batch_sel.size == 0:
                continue
            br = process_rewards_t[batch_sel, :]
            bm = response_mask_t[batch_sel, :]
            # masked mean across batch for each timestep
            denom = bm.sum(dim=0, keepdim=True)
            masked_sum = (br * bm).sum(dim=0, keepdim=True)
            mean_t = masked_sum / (denom + epsilon)
            # advantage for this group and apply mask
            proc_adv[batch_sel, :] = (br - mean_t) * bm
        
        proc_adv = proc_adv * response_mask
        # final SRPO advantage: outcome advantage + process advantage
        scores = outcome_adv + proc_adv

    return scores, scores


@register_adv_est(AdvantageEstimator.WREINFORCE)
def compute_wreinforce_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
):
    scores = token_level_rewards.sum(dim=-1)

    with torch.no_grad():
        bsz = scores.shape[0]

        for i in range(bsz):
            if scores[i] > 0:
                scores[i] = config.w_reinforce.scale * scores[i]
            else:
                scores[i] = -1
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.GRPO_ENT)
def compute_grpo_ent_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    entropy: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    belong_to_all_incorrect = set()
    belong_to_all_correct = set()
    sequence_entropy = (entropy * response_mask).sum(dim=-1) / response_mask.sum(
        dim=-1
    )  # [B, ]
    id2entropy = defaultdict(list)

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
            id2entropy[index[i]].append(sequence_entropy[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")

            # check the
            if id2std[idx] == 0:
                # all incorrect
                if id2mean[idx] == 0:
                    belong_to_all_incorrect.add(idx)
                else:
                    belong_to_all_correct.add(idx)
                entropy_tensor = torch.stack(id2entropy[idx])
                id2mean[idx] = torch.mean(entropy_tensor)
                id2std[idx] = torch.std(entropy_tensor)

        for i in range(bsz):
            if index[i] in belong_to_all_incorrect:
                score = sequence_entropy[i]
                scale_factor = config.grpo_ent.incorrect_scale
            elif index[i] in belong_to_all_correct:
                score = sequence_entropy[i]
                scale_factor = config.grpo_ent.correct_scale
            else:
                score = scores[i]
                scale_factor = 1
            if norm_adv_by_std_in_grpo:
                scores[i] = (score - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = score - id2mean[index[i]]
            scores[i] *= scale_factor

        scores = scores.unsqueeze(-1) * response_mask
    return scores, scores


@register_adv_est(AdvantageEstimator.GRPO_EDGE)
def compute_grpo_edge_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    entropy: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    
    scores = token_level_rewards.sum(dim=-1)
    seq_entropy = (entropy * response_mask).sum(dim=-1) / (response_mask.sum(dim=-1) + epsilon)  # [B]
    
    id2score = defaultdict(list)
    id2entropies = defaultdict(list)
    id2mean = {}
    id2std = {}
    id2ent_mean={}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
            id2entropies[index[i]].append(seq_entropy[i])
            
        for idx in id2score:
            entropies_t = torch.stack(id2entropies[idx])
            id2ent_mean[idx] = torch.mean(entropies_t)
            
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
            
            
            
        for i in range(bsz):
            entropy_ratio = seq_entropy[i] / (id2ent_mean[index[i]] + epsilon)
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / ((
                    id2std[index[i]] + epsilon) * (entropy_ratio + epsilon ) )
                
            else:
                scores[i] = (scores[i] - id2mean[index[i]]) / (entropy_ratio + epsilon )
                
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


import torch

def masked_minmax_normalize_2d(
    x: torch.Tensor,          # shape: [B, N]
    response_mask: torch.Tensor,  # shape: [B, N], 1=valid, 0=padding
    target_range: tuple = (0.0, 1.0),
    eps: float = 1e-8
) -> torch.Tensor:
    """
    对 B×N 序列进行 mask 感知的 min-max 标准化（每行独立标准化）
    
    Args:
        x: Tensor of shape [B, N]
        response_mask: Binary mask of shape [B, N]
        target_range: 目标区间 (min, max)，默认 (0, 1)
        eps: 防止除零的微小值
    
    Returns:
        标准化后的 Tensor，shape [B, N]，padding 位置保留原值
    """
    B, N = x.shape
    assert response_mask.shape == (B, N), f"Mask shape {response_mask.shape} != input shape {x.shape}"
    
    # Step 1: 计算每行的有效 min/max（仅基于 mask=1 的位置）
    # 方法：将无效位置设为 inf/-inf，使 min/max 忽略它们
    masked_for_min = torch.where(response_mask, x, torch.tensor(float('inf'), device=x.device))
    min_val = masked_for_min.min(dim=1, keepdim=True).values  # [B, 1]
    
    masked_for_max = torch.where(response_mask, x, torch.tensor(float('-inf'), device=x.device))
    max_val = masked_for_max.max(dim=1, keepdim=True).values  # [B, 1]
    
    # Step 2: 处理常数序列（max == min）
    range_val = max_val - min_val
    range_val = torch.where(range_val < eps, torch.ones_like(range_val), range_val)
    
    # Step 3: 标准化到 [0, 1] → 映射到目标区间 [a, b]
    a, b = target_range
    x_norm = (x - min_val) / range_val * (b - a) + a  # [B, N]
    
    # Step 4: 仅保留有效位置的标准化结果，padding 位置恢复原值
    x_norm = torch.where(response_mask, x_norm, x)
    
    return x_norm

@register_adv_est(AdvantageEstimator.UCAS)
def compute_ucas_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    token_level_logp: torch.Tensor,
    token_logits: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
):
    VOCAB_SIZE = 151936
    u_p = torch.tensor(1 / VOCAB_SIZE).to(token_level_rewards)
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    id2self_confidence = defaultdict(list)
    id2mean_sc = {}
    id2std_sc = {}
    
    adv_list = []

    with torch.no_grad():
        # compute self-confidence 
        sequence_confidence = (torch.log(u_p) - token_level_logp) * response_mask
        sequence_confidence = torch.sum(sequence_confidence, dim=-1) / response_mask.sum(dim=-1)  # [B, ]

        
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
            id2self_confidence[index[i]].append(sequence_confidence[i])
            
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                sc_tensor = torch.stack(id2self_confidence[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
                id2mean_sc[idx] = torch.mean(sc_tensor)
                id2std_sc[idx] = torch.std(sc_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
            
            
        logits_shift = masked_minmax_normalize_2d(token_logits, response_mask.bool())

            
            
        for i in range(bsz):

            sign = scores[i] - id2mean[index[i]]
            cur_sc = (sequence_confidence[i] - id2mean_sc[index[i]]) / (id2std_sc[index[i]] + epsilon)
            weight = torch.exp(-0.25 * cur_sc) if sign > 0 else  torch.exp(0.25 * cur_sc) 

            
            
            temp_adv = (scores[i] - id2mean[index[i]]) / (
                id2std[index[i]] + epsilon
            )
            
            temp_adv = temp_adv * weight
            # print(temp_adv)
            adv_list.append(temp_adv)
        
        # print(adv_list)
        # print(logits_shift.shape, len(adv_list), response_mask.shape)
        advantage = (torch.stack(adv_list).unsqueeze(-1) * response_mask - 0.01 * logits_shift) * response_mask
            # scores = (temp_adv.unsqueeze(-1) * response_mask - 0.01 * logits_shift) * response_mask
                

        # scores = scores.unsqueeze(-1) * response_mask

    return advantage, advantage




@register_adv_est(AdvantageEstimator.MINER)
def compute_ent_inc_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    token_level_logp: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
):
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    belong_to_all_incorrect = set()
    belong_to_all_correct = set()
    sequence_logp = (token_level_logp * response_mask).sum(dim=-1) / response_mask.sum(
        dim=-1
    )  # [B, ]
    id2logp = defaultdict(list)
    
    advantages = torch.zeros_like(token_level_rewards)
    
    # define credit assignment method
    # uniform: all tokens enjoy the same advantage
    # inverse: low-logp enjoy higher advantage by (1-p)
    def _uniform_assn(token_logp, adv, response_mask=None):
        return (torch.ones_like(token_logp) * adv) * response_mask
        
    def _focal_assn(token_logp, adv, response_mask=None):
        # weights = torch.ones_like(token_logp)
        token_p = torch.exp(token_logp)
        focal_weight = torch.pow((1-token_p), config.miner.gamma) * response_mask 
        credit = adv * focal_weight
        return credit 
    def _inverse_assn(token_logp, adv, response_mask=None):
        # token_p = torch.exp(token_logp)
        if response_mask is None:
            valid_mask = torch.ones_like(token_logp, dtype=torch.bool)
        else:
            valid_mask = response_mask.bool()
        credit = torch.zeros_like(token_logp)
        if not torch.any(valid_mask):
            return credit

        if isinstance(adv, torch.Tensor):
            adv_value = adv.to(device=token_logp.device, dtype=token_logp.dtype)
        else:
            adv_value = torch.tensor(adv, device=token_logp.device, dtype=token_logp.dtype)

        valid_logp = token_logp[valid_mask]
        weights = torch.softmax(-config.miner.gamma * valid_logp, dim=0)


        credit[valid_mask] = weights * adv_value
        return credit
    def _rank_assn(token_logp, adv, response_mask):
        if response_mask is None:
            valid_mask = torch.ones_like(token_logp, dtype=torch.bool)
        else:
            valid_mask = response_mask.bool()
        if not torch.any(valid_mask):
            return torch.zeros_like(token_logp)

        if isinstance(adv, torch.Tensor):
            adv_value = adv.to(device=token_logp.device, dtype=token_logp.dtype)
        else:
            adv_value = torch.tensor(adv, device=token_logp.device, dtype=token_logp.dtype)

        valid_logp = token_logp[valid_mask]
        valid_count = valid_logp.shape[0]
        if valid_count == 1:
            weights = torch.ones_like(valid_logp)
        else:
            order = torch.argsort(valid_logp)
            ranks = torch.empty_like(order)
            ranks[order] = torch.arange(valid_count, device=token_logp.device, dtype=torch.long)
            weights = (valid_count - ranks.to(token_logp.dtype)) / valid_count

        credit = torch.zeros_like(token_logp)
        credit[valid_mask] = weights * adv_value
        return credit
    
    credit_func = {
        'uniform': _uniform_assn,
        'inverse': _inverse_assn,
        'rank': _rank_assn,
        'focal': _focal_assn
    }
    
    additional_advantages = []
    norm_level = config.miner.norm_level 
    additional_advantages_metrics = {}
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
            id2logp[index[i]].append(sequence_logp[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")

            # check the
            if id2std[idx] == 0:
                # all incorrect
                if id2mean[idx] == 0:
                    belong_to_all_incorrect.add(idx)
                else:
                    belong_to_all_correct.add(idx)
                logp_tensor = torch.stack(id2logp[idx])
                id2mean[idx] = torch.mean(logp_tensor)
                id2std[idx] = torch.std(logp_tensor)

        if norm_level == 'batch':
            if belong_to_all_correct:
                correct_group_logp_list = []
                for idx in belong_to_all_correct: 
                    correct_group_logp_list.extend(id2logp[idx])
                
                correct_group_logp_tensor = torch.stack(correct_group_logp_list)
                correct_group_logp_mean = torch.mean(correct_group_logp_tensor)
                for idx in belong_to_all_correct: 
                    id2mean[idx] = correct_group_logp_mean
                
        additional_adv_index = []
        for i in range(bsz):

            if (index[i] in belong_to_all_incorrect or index[i] in belong_to_all_correct):
                score = sequence_logp[i]
                normal = False 
            else:
                score = scores[i]
                normal = True
                
            if normal:
                # neither all correct nor all incorrect
                if norm_adv_by_std_in_grpo:
                    advantages[i] = (score - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
                    # scores[i] = (score - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
                else:
                    advantages[i] = score - id2mean[index[i]]
                    # scores[i] = score - id2mean[index[i]]
            else:
                # for all correct groups, we reward low logp sequence and not penalize high logp sequence (in other words, we encourage more diverse sequences)
                if index[i] in belong_to_all_correct:
                    temp_score = -(score - id2mean[index[i]]) #/ (id2std[index[i]] + epsilon)
                    temp_score = max(0, temp_score)
                    
                    credit = credit_func[config.miner.credit_method](token_level_logp[i], temp_score, response_mask[i])
                    m = credit[response_mask[i].bool()].max()
                    if config.miner.clip_high > 0 and m > config.miner.clip_high:
                        credit *= config.miner.clip_high / (m + 1e-12)
                    # if config.miner.clip_high > 0:
                    #     credit = torch.clamp(credit, 0, config.miner.clip_high)
                    addi_adv = credit * config.miner.correct_scale
                    if temp_score > 0 and config.miner.correct_scale > 0:
                        additional_adv_index.append(i)
                        additional_advantages.append(addi_adv[response_mask[i].bool()])
                    advantages[i] = addi_adv
                    # scores[i] *= config.miner.correct_scale
                elif index[i] in belong_to_all_incorrect:
                    # for all incorrect groups, we penalize high logp sequence and not reward low logp sequence (which has been discouraged sufficiently)
                    temp_score = -(score - id2mean[index[i]]) #/ (id2std[index[i]] + epsilon)
                    temp_score = min(0, temp_score)
                    credit = credit_func[config.miner.credit_method](token_level_logp[i], temp_score, response_mask[i])
                    if config.miner.clip_low > 0:
                        credit = torch.clamp(credit, -config.miner.clip_low, 0)
                    addi_adv = credit * config.miner.incorrect_scale
                    if temp_score < 0 and config.miner.incorrect_scale > 0:
                        additional_adv_index.append(i)
                        additional_advantages.append(addi_adv[response_mask[i].bool()])
                    
                    advantages[i] = addi_adv
        if config.miner.scale_by_normal > 0:
            if (
                config.miner.clip_high <= 0
                and config.miner.clip_low <= 0
                and additional_advantages
            ):
                normal_mask = torch.ones(
                    (bsz,), dtype=torch.bool, device=advantages.device
                )
                if additional_adv_index:
                    adv_idx_tensor = torch.tensor(
                        additional_adv_index,
                        dtype=torch.long,
                        device=advantages.device,
                    )
                    normal_mask[adv_idx_tensor] = False
                normal_tokens = advantages[normal_mask] * response_mask[normal_mask]
                if normal_tokens.numel() > 0:
                    normal_nonzero = normal_tokens != 0
                    normal_sum = torch.abs(normal_tokens[normal_nonzero]).sum(
                        dtype=torch.float64
                    ).item()
                else:
                    normal_sum = 0.0

                additional_sum = 0.0
                for adv in additional_advantages:
                    if adv.numel() == 0:
                        continue
                    additional_sum += torch.abs(adv).sum(dtype=torch.float64).item()

                if additional_sum > 0.0:
                    target_mean = config.miner.scale_by_normal * normal_sum
                    if target_mean <= 0.0:
                        scale_value = 0.0
                    else:
                        scale_value = min(1.0, target_mean / additional_sum)
                    additional_advantages_metrics['additional_advantages/scale_value'] = scale_value
                    if scale_value < 1.0:
                        for idx in additional_adv_index:
                            advantages[idx] = advantages[idx] * scale_value
                        for adv_idx in range(len(additional_advantages)):
                            additional_advantages[adv_idx] = (
                                additional_advantages[adv_idx] * scale_value
                            )
            else:
                # if clip_high > 0
                # scale such that the token mean of additional does not exceed that of normal
                normal_mask = torch.ones(
                    (bsz,), dtype=torch.bool, device=advantages.device
                )
                if additional_adv_index:
                    adv_idx_tensor = torch.tensor(
                        additional_adv_index,
                        dtype=torch.long,
                        device=advantages.device,
                    )
                    normal_mask[adv_idx_tensor] = False
                normal_tokens = advantages[normal_mask] * response_mask[normal_mask]
                if normal_tokens.numel() > 0:
                    normal_nonzero = normal_tokens != 0
                    normal_mean = torch.abs(normal_tokens[normal_nonzero]).mean(
                        dtype=torch.float64
                    ).item()
                else:
                    normal_mean = 0.0

                additional_sum  = 0.0
                additional_count = 0
                for adv in additional_advantages:
                    if adv.numel() == 0:
                        continue
                    additional_sum += torch.abs(adv).sum(dtype=torch.float64).item()
                    additional_count += int(adv.numel())
                
                additional_mean = additional_sum / (additional_count + 1e-7) 
                if additional_mean > 0.0:
                    target_mean = normal_mean
                    if target_mean <= 0.0:
                        scale_value = 0.0
                    else:
                        scale_value = min(1.0, target_mean / additional_mean)
                    additional_advantages_metrics['additional_advantages/scale_value'] = scale_value
                    additional_advantages_metrics["additional_advantages/normal_abs_mean"] = float(normal_mean)
                    additional_advantages_metrics["additional_advantages/additional_abs_mean_before_scale"] = float(additional_mean)

                    if scale_value < 1.0:
                        for idx in additional_adv_index:
                            advantages[idx] = advantages[idx] * scale_value
                        for adv_idx in range(len(additional_advantages)):
                            additional_advantages[adv_idx] = (
                                additional_advantages[adv_idx] * scale_value
                            )


        advantages = advantages * response_mask
    
    if additional_advantages:
        merged_additional_adv = torch.cat(additional_advantages, dim=0)
        additional_advantages_metrics.update({
            "additional_advantages/min": torch.min(merged_additional_adv).item(),
            "additional_advantages/mean": torch.mean(merged_additional_adv).item(),
            "additional_advantages/max": torch.max(merged_additional_adv).item(),
        }) 
    else:
        additional_advantages_metrics.update({"additional_advantages/min": 0.0,
            "additional_advantages/mean": 0.0,
            "additional_advantages/max": 0.0
        })

    return advantages, advantages, additional_advantages_metrics


@register_adv_est(AdvantageEstimator.GRPO_NEG)
def compute_grpo_neg_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    response_emb: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    id2emb = defaultdict(list)
    weight = torch.ones((scores.shape[0],), device=scores.device, dtype=scores.dtype)
    id2correct_index = {}
    id2sim = {}
    batchidx2groupidx = {}
    group2batchidx = {}
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
            id2emb[index[i]].append(response_emb[i])
            batchidx2groupidx[i] = len(id2score[index[i]]) - 1
            group2batchidx[(index[i], batchidx2groupidx[i])] = i
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
                if id2std[idx] > epsilon:
                    # not all correct/incorrect
                    response_emb_tensor = torch.stack(id2emb[idx])
                    x_normalized = F.normalize(response_emb_tensor, p=2, dim=1)
                    # print(x_normalized.shape)
                    sim_matrix = torch.mm(x_normalized, x_normalized.t())  # [G, G]
                    # obtain the index within the group whose scores are equal to the largest scores [may contain many]
                    correct_index = torch.where(
                        scores_tensor == torch.max(scores_tensor)
                    )[0]
                    incorrect_index = torch.where(
                        scores_tensor != torch.max(scores_tensor)
                    )[0]
                    id2correct_index[idx] = correct_index
                    id2sim[idx] = sim_matrix

                    incorrect_sim = sim_matrix[incorrect_index, :][
                        :, correct_index
                    ]  # [|incorrect_index|, |correct_index|]
                    incorrect_sim_mean = torch.max(incorrect_sim, dim=1)[
                        0
                    ]  # [|incorrect_index|]
                    # normalized_incorrect_sim_mean = (incorrect_sim_mean - incorrect_sim_mean.min()) / (incorrect_sim_mean.max() - incorrect_sim_mean.min() + epsilon)
                    if incorrect_index.shape[0] == 1:
                        group_weights = torch.tensor(
                            [1.0], device=scores.device, dtype=scores.dtype
                        )
                    else:
                        if config.neg_scale.scale_mode == "exp":
                            normalized_incorrect_sim_mean = (
                                incorrect_sim_mean - incorrect_sim_mean.min()
                            ) / (
                                incorrect_sim_mean.max()
                                - incorrect_sim_mean.min()
                                + epsilon
                            )
                            # group_weights = torch.exp(-config.neg_scale.scale * (incorrect_sim_mean - incorrect_sim_mean_mu) / (incorrect_sim_mean_std + epsilon))
                            group_weights = torch.exp(
                                -config.neg_scale.scale * normalized_incorrect_sim_mean
                            )
                        elif config.neg_scale.scale_mode == "shift":
                            softmax_incorrect_sim_mean = torch.softmax(
                                incorrect_sim_mean / config.neg_scale.scale, dim=0
                            )
                            # large sim value corresponds to 1 + sim
                            # small sim value corresponds to 1 - sim
                            # if there are odd number of sim values, the middle sim value's weight is 1
                            # first obtain the sim value's rank within the group
                            sim_ranks = torch.argsort(torch.argsort(incorrect_sim_mean))
                            # obtain the sign of the weight based on whether the rank is larger than half of the group size
                            signs = torch.where(
                                sim_ranks >= (incorrect_index.shape[0] - 1) / 2,
                                -1.0,
                                1.0,
                            )
                            group_weights = 1 + signs * softmax_incorrect_sim_mean
                            # if the number of group_weights is a odd number, the middel rank's value's weight should be 1
                            if incorrect_index.shape[0] % 2 == 1:
                                middle_rank = (incorrect_index.shape[0] - 1) // 2
                                group_weights[sim_ranks == middle_rank] = 1.0
                            pass
                        # elif config.neg_scale.scale_mode == 'linear':
                        #     group_weights = 1 - config.neg_scale.scale * (incorrect_sim_mean - incorrect_sim_mean_mu) / (incorrect_sim_mean_std + epsilon)
                        elif config.neg_scale.scale_mode == "reward":
                            group_weights = incorrect_sim_mean
                            # print(group_weights.shape)
                            pass
                        else:
                            raise NotImplementedError

                    for i, idx_within_group in enumerate(incorrect_index):
                        batch_idx = group2batchidx[(idx, idx_within_group.item())]
                        weight[batch_idx] = group_weights[i]
                        if config.neg_scale.as_reward:
                            scores[batch_idx] = (
                                scores[batch_idx]
                                + config.neg_scale.scale * group_weights[i]
                            )
                            pass

            else:
                raise ValueError(f"no score in prompt index: {idx}")

        for i in range(bsz):

            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (
                    id2std[index[i]] + epsilon
                )
            else:
                scores[i] = scores[i] - id2mean[index[i]]
            if not config.neg_scale.as_reward:
                scores[i] *= weight[i]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.GRPO_VECTORIZED)
def compute_grpo_vectorized_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized GRPO（outcome-only）:
      For each group g:
      a_i = \\frac{r_i - \\mu_g}{\\sigma_g} (or without dividing by \\sigma_g),
      then broadcast the scalar across the token dimension (multiplied by response_mask).。
    """
    with torch.no_grad():
        scores = token_level_rewards.sum(dim=-1)
        g = as_torch_index(index, device=scores.device)
        mean_g, std_g, _ = group_mean_std(scores, g, eps=epsilon)
        if norm_adv_by_std_in_grpo:
            scalars = (scores - mean_g[g]) / (std_g[g] + epsilon)
        else:
            scalars = scores - mean_g[g]
        advantages = scalars.unsqueeze(-1) * response_mask
        return advantages, advantages


@register_adv_est(
    AdvantageEstimator.GRPO_PASSK
)  # or simply: @register_adv_est("grpo_passk")
def compute_grpo_passk_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for Pass@k using a GRPO-style outcome reward formulation.
    Only the best response per group gets a non-zero advantage: r_max - r_second_max.

    Implemented as described in https://arxiv.org/abs/2503.19595.

    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs,) → group ID per sample
        epsilon: float for numerical stability
        config: (AlgoConfig) algorithm settings, which contains "norm_adv_by_std_in_grpo"

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length)
    """
    assert config is not None
    # if True, normalize advantage by std within group
    norm_adv_by_std_in_grpo = config.get("norm_adv_by_std_in_grpo", True)
    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    advantages = torch.zeros_like(scores)

    id2scores = defaultdict(list)
    id2indices = defaultdict(list)

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            idx = index[i]
            id2scores[idx].append(scores[i])
            id2indices[idx].append(i)

        for idx in id2scores:
            rewards = torch.stack(id2scores[idx])  # (k,)
            if rewards.numel() < 2:
                raise ValueError(
                    f"Pass@k requires at least 2 samples per group. Got {rewards.numel()} for group {idx}."
                )
            topk, topk_idx = torch.topk(rewards, 2)
            r_max, r_second_max = topk[0], topk[1]
            i_max = id2indices[idx][topk_idx[0].item()]
            advantage = r_max - r_second_max
            if norm_adv_by_std_in_grpo:
                std = torch.std(rewards)
                advantage = advantage / (std + epsilon)
            advantages[i_max] = advantage

    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages


@register_adv_est(
    AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE
)  # or simply: @register_adv_est("reinforce_plus_plus_baseline")
def compute_reinforce_plus_plus_baseline_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: torch.Tensor,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RF++-baseline (https://arxiv.org/abs/2501.03262), operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.RLOO)  # or simply: @register_adv_est("rloo")
def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[
                    index[i]
                ] * response_num / (response_num - 1)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.OPO)  # or simply: @register_adv_est("opo")
def compute_opo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for OPO based on https://arxiv.org/pdf/2505.23585

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = response_mask.sum(dim=-1)
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2len = defaultdict(list)
    id2bsl = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
            id2len[index[i]].append(response_length[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2bsl[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                score_tensor = torch.stack(id2score[idx])
                len_tensor = torch.stack(id2len[idx])
                id2bsl[idx] = (len_tensor * score_tensor).sum() / len_tensor.sum()
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2bsl[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(
    AdvantageEstimator.REINFORCE_PLUS_PLUS
)  # or simply: @register_adv_est("reinforce_plus_plus")
def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    assert config is not None
    gamma = config.gamma
    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * response_mask[:, t]

        advantages = verl_F.masked_whiten(returns, response_mask)
        advantages = advantages * response_mask

    return advantages, returns


@register_adv_est(AdvantageEstimator.REMAX)  # or simply: @register_adv_est("remax")
def compute_remax_outcome_advantage(
    token_level_rewards: torch.Tensor,
    reward_baselines: torch.Tensor,
    response_mask: torch.Tensor,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = (
            (token_level_rewards * response_mask)
            .flip(dims=[-1])
            .cumsum(dim=-1)
            .flip(dims=[-1])
        )
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask

    return advantages, returns


@register_adv_est(AdvantageEstimator.GPG)  # or simply: @register_adv_est("gpg")
def compute_gpg_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    f_norm: float = 1.0,
    alpha: float = 1.0,
    config=None,
    **kwargs,
):
    """
    Compute advantage for GPG, operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(np.ndarray)`
            shape: (bs,)
        epsilon: (float)
        f_norm: (float)
        alpha: (float)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        m = torch.count_nonzero(scores)
        alpha = bsz / m.clamp(min=1)

        for i in range(bsz):
            id2score[index[i]].append(scores[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = alpha * (scores[i] - id2mean[index[i]]) / (f_norm)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(
    AdvantageEstimator.RLOO_VECTORIZED
)  # or simply: @register_adv_est("rloo_vectorized")
def compute_rloo_vectorized_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    with torch.no_grad():
        inv = torch.from_numpy(np.unique(index, return_inverse=True)[1]).to(
            scores.device
        )

        c = torch.bincount(inv)[inv].to(scores.dtype)
        adv = (
            (c * scores - torch.bincount(inv, weights=scores)[inv])
            / (c - 1).clamp_min(1)
        ) * (c > 1)

        adv = adv.unsqueeze(-1) * response_mask

    return adv, adv


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    """Compute token-level rewards with KL penalty.

    Args:
        token_level_scores (torch.Tensor): Token-level reward scores.
        old_log_prob (torch.Tensor): Log probabilities from current policy.
        ref_log_prob (torch.Tensor): Log probabilities from reference policy.
        kl_ratio (float): KL penalty coefficient.

    Returns:
        torch.Tensor: Token-level rewards with KL penalty applied.
    """
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.

    Args:
        loss_mat: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_agg_mode: (str) choices:
            method to aggregate the loss matrix into a scalar.
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(
            loss_mask, dim=-1
        )  # token-mean
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]  # The divisor
        # (loss_mask.shape[-1]) should ideally be constant
        # throughout training to well-replicate the DrGRPO paper.
        # TODO: Perhaps add user-defined normalizer argument to
        # agg_loss to ensure divisor stays constant throughout.
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


@deprecated("verl.trainer.ppo.core_algos.compute_policy_loss_vanilla")
def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        clip_ratio_c (float, optional):
            Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
            Defaults to 3.0.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
    """
    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(
        torch.gt(pg_losses2, pg_losses1).float(), response_mask
    )

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode
    )

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


@register_policy_loss("vanilla")
def compute_policy_loss_vanilla(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | AlgoConfig] = None,
    rollout_log_probs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        config: `(verl.trainer.config.ActorConfig)`:
            config for the actor.
        rollout_log_probs: `(torch.Tensor)`:
            log probabilities of actions under the rollout policy, shape (batch_size, response_length).
    """

    assert config is not None
    assert not isinstance(config, AlgoConfig)
    clip_ratio = (
        config.clip_ratio
    )  # Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
    clip_ratio_low = (
        config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    )
    clip_ratio_high = (
        config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    )
    clip_ratio_c = config.get(  # Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
        "clip_ratio_c", 3.0
    )

    cliprange = clip_ratio
    cliprange_low = clip_ratio_low
    cliprange_high = clip_ratio_high

    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(
        torch.gt(pg_losses2, pg_losses1).float(), response_mask
    )

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    if config.tis_imp_ratio_cap > 0 and rollout_log_probs is not None:
        # Apply truncated importance sampling -> https://fengyao.notion.site/off-policy-rl
        tis_imp_ratio = torch.exp(old_log_prob - rollout_log_probs)
        tis_imp_ratio = torch.clamp(tis_imp_ratio, max=config.tis_imp_ratio_cap)
        pg_losses = pg_losses * tis_imp_ratio

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode
    )

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


@register_policy_loss("gspo")
def compute_policy_loss_gspo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-mean",
    config: Optional[DictConfig | ActorConfig] = None,
    rollout_log_probs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the clipped policy objective and related metrics for GSPO.

    See https://arxiv.org/pdf/2507.18071 for more details.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. For GSPO, it is recommended to use "seq-mean-token-mean".
    """

    assert config is not None
    assert isinstance(config, ActorConfig)
    clip_ratio_low = (
        config.clip_ratio_low
        if config.clip_ratio_low is not None
        else config.clip_ratio
    )
    clip_ratio_high = (
        config.clip_ratio_high
        if config.clip_ratio_high is not None
        else config.clip_ratio
    )

    negative_approx_kl = log_prob - old_log_prob

    # compute sequence-level importance ratio:
    # si(θ) = (π_θ(yi|x)/π_θold(yi|x))^(1/|yi|) =
    # exp [(1/|y_i|) * Σ_t log(π_θ(y_i,t|x,y_i,<t)/π_θold(y_i,t|x,y_i,<t))]
    seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
    negative_approx_kl_seq = (
        torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths
    )   # (bs,)

    # Combined ratio at token level:
    # s_i,t(θ) = sg[s_i(θ)] · π_θ(y_i,t|x, y_i,<t) / sg[π_θ(y_i,t|x, y_i,<t)]
    # In log space: log(s_i,t(θ)) = sg[log(s_i(θ))] + log_prob - sg[log_prob]
    log_seq_importance_ratio = (
        log_prob - log_prob.detach() + negative_approx_kl_seq.detach().unsqueeze(-1)
    )
    log_seq_importance_ratio = torch.clamp(
        log_seq_importance_ratio, max=10.0
    )  # clamp for numerical stability

    # finaly exp() to remove log
    seq_importance_ratio = torch.exp(log_seq_importance_ratio)

    pg_losses1 = -advantages * seq_importance_ratio
    pg_losses2 = -advantages * torch.clamp(
        seq_importance_ratio, 1 - clip_ratio_low, 1 + clip_ratio_high
    )
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    # for GSPO, we need to aggregate the loss at the sequence level (seq-mean-token-mean)
    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode="seq-mean-token-mean"
    )

    # For compatibility, return zero for pg_clipfrac_lower (not used in standard GSPO)
    pg_clipfrac = verl_F.masked_mean(
        torch.gt(pg_losses2, pg_losses1).float(), response_mask
    )
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)

    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


@register_policy_loss("grpo_s")
def compute_policy_loss_grpo_s(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | AlgoConfig] = None,
    rollout_log_probs: torch.Tensor | None = None,
):
    clip_ratio = (
        config.clip_ratio
    )  # Clipping parameter. See https://arxiv.org/abs/1707.06347.
    clip_ratio_low = (
        config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    )
    clip_ratio_high = (
        config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    )
    negative_approx_kl = log_prob - old_log_prob
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    # Geometric-Mean Policy Optimization
    response_mask_sum = response_mask.sum(dim=-1)
    ratio = torch.exp(negative_approx_kl * response_mask).sum(-1) / response_mask_sum
    
    pg_loss1 = -advantages[:, 0] * ratio 
    pg_loss2 = -advantages[:, 0] * torch.clamp(
        ratio, 1 - clip_ratio_low, 1 + clip_ratio_high
    )
    pg_loss = torch.maximum(pg_loss1, pg_loss2)
    pg_loss = pg_loss.mean()
    
    pg_clipfrac = verl_F.masked_mean(
        torch.gt(pg_loss2, pg_loss1).float(), response_mask[:, 0]
    )
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)
    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower
    
@register_policy_loss("aspo")
def compute_policy_loss_aspo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | AlgoConfig] = None,
    rollout_log_probs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the clipped policy objective and related metrics for ASPO.

    

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. For ASPO, it is recommended to use "seq-mean-token-mean".
    """

    assert config is not None
    assert not isinstance(config, AlgoConfig)
    clip_ratio = (
        config.clip_ratio
    )  # Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
    clip_ratio_low = (
        config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    )
    clip_ratio_high = (
        config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    )
    clip_ratio_c = config.get(  # Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
        "clip_ratio_c", 3.0
    )

    cliprange = clip_ratio
    cliprange_low = clip_ratio_low
    cliprange_high = clip_ratio_high

    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    positive_ratio = torch.exp(log_prob + old_log_prob) / (torch.exp(old_log_prob.detach()) **2 + 1e-10)
    
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
        

    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(
        torch.gt(pg_losses2, pg_losses1).float(), response_mask
    )

    # 以上是hard clip的部分,这部分是完全一样的,然后我们要计算clip_pg_losses1和剩下的部分
    # 这里涉及一个,对于advantages<0的部分,我们不改变clip_pg_losses1
    # 对于advantages>0的部分,我们只考虑没被clip的那些部分,
    # 我们把没被clip的那些部分替换成max(-A*positive_ratio, -A*clip_ratio_c * log_prob/ log_prob.detach())
    # 意在:若没触发soft clip,则用positive ratio来代替原先的importance rate
    # 若触发了soft clip,我们使用最大clip_ratio_c的幅度来限制importance ratio
    # 但是仍然用log_prob/ log_prob.detach()来保留梯度方向
    # Apply different logic for positive and negative advantages
    
    # reciprocal ppo loss 
    reciprocal_pg_losses = -advantages * positive_ratio
    # soft dual clip ppo loss 
    soft_dual_clip_pg_losses = -advantages * clip_ratio_c * (log_prob / log_prob.detach())
    # dual-soft clip for positive advantages 
    dual_soft_clip = torch.maximum(reciprocal_pg_losses, soft_dual_clip_pg_losses)
    dual_soft_clip_ratio = verl_F.masked_mean(
        torch.gt(soft_dual_clip_pg_losses, reciprocal_pg_losses).float(), response_mask
    )
    pg_losses_positive = torch.where(
        advantages > 0,
        torch.where(
            pg_losses1 == clip_pg_losses1,  # not clipped by hard clip
            dual_soft_clip,
            clip_pg_losses1  # already clipped by hard clip, keep original
        ),
        clip_pg_losses1  # negative advantages, use hard clip result
    )


    pg_losses = pg_losses_positive

    if config.tis_imp_ratio_cap > 0 and rollout_log_probs is not None:
        # Apply truncated importance sampling -> https://fengyao.notion.site/off-policy-rl
        tis_imp_ratio = torch.exp(old_log_prob - rollout_log_probs)
        tis_imp_ratio = torch.clamp(tis_imp_ratio, max=config.tis_imp_ratio_cap)
        pg_losses = pg_losses * tis_imp_ratio

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode
    )

    return pg_loss, pg_clipfrac, ppo_kl, dual_soft_clip_ratio
    

    # pg_losses3 = -advantages * clip_ratio_c * (log_prob / log_prob.detach()) # turn this to a gradient 
    # clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    # pg_clipfrac_lower = verl_F.masked_mean(
    #     torch.gt(clip_pg_losses1, pg_losses3) * (advantages > 0).float(), response_mask
    # )

    # pg_losses = torch.where(advantages < 0, clip_pg_losses1, clip_pg_losses2)
    # # pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    # if config.tis_imp_ratio_cap > 0 and rollout_log_probs is not None:
    #     # Apply truncated importance sampling -> https://fengyao.notion.site/off-policy-rl
    #     tis_imp_ratio = torch.exp(old_log_prob - rollout_log_probs)
    #     tis_imp_ratio = torch.clamp(tis_imp_ratio, max=config.tis_imp_ratio_cap)
    #     pg_losses = pg_losses * tis_imp_ratio

    # pg_loss = agg_loss(
    #     loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode
    # )

    # return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower



@register_policy_loss("gpg")
def compute_policy_loss_gpg(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | AlgoConfig] = None,
    rollout_log_probs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Adapted from
    https://github.com/AMAP-ML/GPG/blob/main/VisualThinker-R1-Zero/src/open-r1-multimodal/src/open_r1/trainer/grpo_trainer.py#L495
    Args:
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    return:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via GPG
    """
    pg_losses = -log_prob * advantages

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode
    )
    return pg_loss, torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)


@register_policy_loss("clip_cov")
def compute_policy_loss_clip_cov(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | AlgoConfig] = None,
    rollout_log_probs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the clipped policy objective and related metrics for Clip-Cov.

    Adapted from
    https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/verl/trainer/ppo/core_algos.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        clip_cvo_ratio (float, optional):
            Ratio for clipping the covariance. Defaults to 0.0002.
        clip_cov_lb (float, optional):
            Lower bound for clipping covariance. Defaults to 1.0.
        clip_cov_ub (float, optional):
            Upper bound for clipping covariance. Defaults to 5.0.
    """
    assert config is not None
    assert not isinstance(config, AlgoConfig), "passing AlgoConfig not supported yet"
    assert config.policy_loss is not None

    clip_cov_ratio = (
        config.policy_loss.clip_cov_ratio
        if config.policy_loss.clip_cov_ratio is not None
        else 0.0002
    )
    cliprange = config.clip_ratio
    cliprange_low = (
        config.clip_ratio_low if config.clip_ratio_low is not None else cliprange
    )
    cliprange_high = (
        config.clip_ratio_high if config.clip_ratio_high is not None else cliprange
    )
    clip_cov_ub = (
        config.policy_loss.clip_cov_ub
        if config.policy_loss.clip_cov_ub is not None
        else 5.0
    )
    clip_cov_lb = (
        config.policy_loss.clip_cov_lb
        if config.policy_loss.clip_cov_lb is not None
        else 1.0
    )

    assert clip_cov_ratio > 0, "clip_ratio should be larger than 0."

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio

    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    corr = torch.ones_like(advantages)
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    clip_by_origin = (pg_losses2 > pg_losses1) & (response_mask > 0)

    cov_all = (advantages - verl_F.masked_mean(advantages, response_mask)) * (
        log_prob - verl_F.masked_mean(log_prob.detach(), response_mask)
    )
    cov_all[response_mask == 0] = -torch.inf
    cov_all[clip_by_origin] = -torch.inf

    clip_num = max(int(clip_cov_ratio * response_mask.sum().item()), 1)
    top_k_idx = (cov_all < clip_cov_ub) & (cov_all > clip_cov_lb) & (response_mask > 0)
    top_k_idx = torch.nonzero(top_k_idx)

    if len(top_k_idx) > 0:
        perm = torch.randperm(len(top_k_idx))
        top_k_idx = top_k_idx[perm[: min(clip_num, len(top_k_idx))]]
    else:
        top_k_idx = torch.empty((0, 2), device=cov_all.device, dtype=torch.long)

    corr[top_k_idx[:, 0], top_k_idx[:, 1]] = 0

    pg_clipfrac = verl_F.masked_mean((corr == 0).float(), response_mask)

    pg_losses = torch.maximum(pg_losses1, pg_losses2) * corr
    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode
    )

    return pg_loss, pg_clipfrac, ppo_kl, torch.tensor(0.0)


@register_policy_loss("kl_cov")
def compute_policy_loss_kl_cov(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | AlgoConfig] = None,
    rollout_log_probs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the clipped policy objective and related metrics for Clip-Cov.

    Adapted from
    https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/verl/trainer/ppo/core_algos.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        kl_cov_ratio (float, optional):
            Ratio for selecting the top-k covariance values. Defaults to 0.0002.
        ppo_kl_coef (float, optional):
            Coefficient for the KL penalty term in the loss. Defaults to 1.
    """
    assert config is not None
    assert not isinstance(config, AlgoConfig), "passing AlgoConfig not supported yet"
    assert config.policy_loss is not None

    kl_cov_ratio = (
        config.policy_loss.kl_cov_ratio
        if config.policy_loss.kl_cov_ratio is not None
        else 0.0002
    )
    ppo_kl_coef = (
        config.policy_loss.ppo_kl_coef
        if config.policy_loss.ppo_kl_coef is not None
        else 1.0
    )

    assert kl_cov_ratio > 0, "kl_cov_ratio should be larger than 0."

    negative_approx_kl = log_prob - old_log_prob
    abs_kl = negative_approx_kl.abs()
    ratio = torch.exp(negative_approx_kl)
    ppo_kl_abs = verl_F.masked_mean(negative_approx_kl.abs(), response_mask)
    pg_losses1 = -advantages * ratio
    pg_losses_kl = -advantages * ratio + ppo_kl_coef * abs_kl
    pg_losses = pg_losses1

    all_valid = response_mask > 0
    all_valid_idx = torch.nonzero(all_valid.reshape(-1), as_tuple=True)[0]
    all_valid_adv = advantages[all_valid].detach().reshape(-1).cpu()
    all_valid_logp = log_prob[all_valid].detach().reshape(-1).cpu()

    k = min(kl_cov_ratio, len(all_valid_adv))

    if k != 0:
        cov_lst_all = (all_valid_adv - all_valid_adv.mean()) * (
            all_valid_logp - all_valid_logp.mean()
        )
        k_percent_nums = max(1, int(len(cov_lst_all) * kl_cov_ratio))
        large_cov_idxs = torch.topk(cov_lst_all, k_percent_nums, largest=True).indices

        if len(large_cov_idxs) != 0:
            large_cov_idxs = all_valid_idx[large_cov_idxs]
            pg_losses[
                large_cov_idxs // advantages.shape[1],
                large_cov_idxs % advantages.shape[1],
            ] = pg_losses_kl[
                large_cov_idxs // advantages.shape[1],
                large_cov_idxs % advantages.shape[1],
            ]

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode
    )

    return pg_loss, torch.tensor(0.0), ppo_kl_abs, torch.tensor(0.0)


@register_policy_loss("geo_mean")
def compute_policy_loss_geo_mean(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | AlgoConfig] = None,
    rollout_log_probs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the clipped policy objective and related metrics for GMPO.

    Adapted from paper https://arxiv.org/abs/2507.20673
    https://github.com/callsys/GMPO/blob/main/train_zero_math_gmpo.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            not used
    """

    assert config is not None
    assert not isinstance(config, AlgoConfig)
    clip_ratio = (
        config.clip_ratio
    )  # Clipping parameter. See https://arxiv.org/abs/1707.06347.
    clip_ratio_low = (
        config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    )
    clip_ratio_high = (
        config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    )

    cliprange = clip_ratio
    cliprange_low = clip_ratio_low
    cliprange_high = clip_ratio_high
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability (uncomment it if you like)
    # negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # Clipping at token-level & Clipping wider
    sgn_advantage = torch.sign(advantages)
    negative_approx_kl_clamp = torch.clamp(
        negative_approx_kl, -cliprange_low, cliprange_high
    )
    negative_approx_kl_min = torch.min(
        sgn_advantage * negative_approx_kl, sgn_advantage * negative_approx_kl_clamp
    )
    negative_approx_kl_min = sgn_advantage * negative_approx_kl_min

    # Geometric-Mean Policy Optimization
    response_mask_sum = response_mask.sum(dim=-1)
    ratio = torch.exp(
        (negative_approx_kl_min * response_mask).sum(dim=-1)
        / (response_mask_sum + 1e-8)        
    )
    # we only support sequence level advantage for now,
    # otherwise, below would be not consistent with the paper
    advantage = (advantages * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8)
    pg_losses = -advantage * ratio
    pg_loss = torch.mean(pg_losses)

    # higher: ratio is too large that need clamp to clip_high (when adv > 0)
    clipped = torch.ne(negative_approx_kl, negative_approx_kl_clamp)
    pg_clipfrac = verl_F.masked_mean(
        (clipped * (advantages > 0)).float(), response_mask
    )
    pg_clipfrac_lower = verl_F.masked_mean(
        (clipped * (advantages < 0)).float(), response_mask
    )

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def compute_entropy_loss(logits, response_mask, loss_agg_mode: str = "token-mean"):
    """Compute categorical entropy loss (For backward compatibility)

    Args:
        logits (torch.Tensor): shape is (bs, response_length, vocab_size)
        response_mask (torch.Tensor): shape is (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    token_entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = agg_loss(
        loss_mat=token_entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode
    )
    return entropy_loss


def compute_value_loss(
    vpreds: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange_value: float,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped value-function loss for PPO.

    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (torch.FloatTensor):
            Predicted values from the value head, shape (batch_size, response_length).
        values (torch.FloatTensor):
            Old (baseline) values from the value head, shape (batch_size, response_length).
        returns (torch.FloatTensor):
            Ground-truth returns, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the value loss calculation.
        cliprange_value (float):
            Clip range for value prediction updates.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".

    Returns:
        vf_loss (torch.FloatTensor):
            A scalar tensor containing the aggregated value-function loss.
        vf_clipfrac (float):
            Fraction of elements where the clipped loss was used.
    """
    vpredclipped = verl_F.clip_by_value(
        vpreds, values - cliprange_value, values + cliprange_value
    )
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
    vf_loss = 0.5 * agg_loss(
        loss_mat=clipped_vf_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode
    )
    vf_clipfrac = verl_F.masked_mean(
        torch.gt(vf_losses2, vf_losses1).float(), response_mask
    )
    return vf_loss, vf_clipfrac


def kl_penalty(
    logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty
) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob. Optionally using straight through to bind k2 on other
    kl penalty compute method for unbiased KL gradient estimation.
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:
        kl_estimate
    """
    forward_score = kl_penalty_forward(logprob, ref_logprob, kl_penalty)
    if not kl_penalty.endswith("+") or kl_penalty in ("mse", "k2"):
        return forward_score

    """
    The expectation of k1 and k3 estimator is the expectaed value of KL, but the expected gradient of k1 and k3
    estimator is not the expectaed gradient of KL. On the other hand k2 estimator gives right gradient estimator, 
    so we use a straight through trick here if the kl_penalty method ends with '+', .e.g., k3+. 
    """
    backward_score = 0.5 * (logprob - ref_logprob).square()

    return backward_score - backward_score.detach() + forward_score.detach()


def kl_penalty_forward(
    logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty
) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:
        kl_estimate
    """
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()
    
    if kl_penalty in ("reverse_k3"):
        kl = logprob - ref_logprob
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        # For numerical stability
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError



def compute_kl_on_mask(
    config, 
    token_level_rewards: torch.Tensor, 
    response_mask: torch.Tensor,
    ref_log_prob: torch.Tensor,
    log_prob: torch.Tensor
):

    # broadcast the token_level_rewards 
    token_level_rewards = token_level_rewards.sum(dim=-1, keepdim=True) * response_mask
    
    kl_on_mask = torch.ones_like(token_level_rewards).to(torch.bool)

    if config.skip_overlong:
        sequence_token_count = response_mask.sum(dim=-1, keepdim=True)
        max_length_reached = sequence_token_count == response_mask.shape[-1]
        overlong_sentence_mask = (token_level_rewards < 1.0) & max_length_reached
        kl_on_mask = kl_on_mask & torch.logical_not(overlong_sentence_mask)
    incorrect_mask = (response_mask > 0) & (token_level_rewards < 1.0) & (ref_log_prob > log_prob)
    kl_on_mask = kl_on_mask & torch.logical_not(incorrect_mask)
    correct_mask = (response_mask > 0) & (token_level_rewards > 0.0) & (ref_log_prob < log_prob)
    
    # need to compute KL
    kl_on_mask = kl_on_mask & torch.logical_not(correct_mask)

    
    # ratio of tokens that have been masked
    valid_token_mask = response_mask > 0
    skipped_token_mask = valid_token_mask & torch.logical_not(kl_on_mask)
    valid_token_count = valid_token_mask.sum(dtype=torch.float32).clamp_min(1.0)
    skipped_ratio = (skipped_token_mask.sum(dtype=torch.float32) / valid_token_count).item()

    return kl_on_mask.to(response_mask.dtype), skipped_ratio

def compute_pf_ppo_reweight_data(
    data,
    reweight_method: str = "pow",
    weight_pow: float = 2.0,
):
    """Reweight the data based on the token_level_scores.

    Args:
        data: DataProto object, containing batch, non_tensor_batch and meta_info
        reweight_method: str, choices: "pow", "max_min", "max_random"
        weight_pow: float, the power of the weight

    Returns:

    """

    @torch.no_grad()
    def compute_weights(
        scores: torch.Tensor, reweight_method: str, weight_pow: float
    ) -> torch.Tensor:
        """Compute importance weights for resampling based on scores.

        Args:
            scores (torch.Tensor): Tensor of scores to compute weights from.
            reweight_method (str): Method for computing weights ('pow', 'max_min', 'max_random').
            weight_pow (float): Power exponent for 'pow' method.

        Returns:
            torch.Tensor: Computed importance weights.

        Raises:
            ValueError: If reweight_method is not supported.
        """
        if reweight_method == "pow":
            weights = torch.pow(torch.abs(scores), weight_pow)
        elif reweight_method == "max_min":
            max_score = torch.max(scores)
            min_score = torch.min(scores)
            weights = torch.where(
                (scores == max_score) | (scores == min_score), 1.0, 0.0
            )
        elif reweight_method == "max_random":
            max_score = torch.max(scores)
            weights = torch.where(scores == max_score, 0.4, 0.1)
        else:
            raise ValueError(f"Unsupported reweight_method: {reweight_method}")
        return weights

    scores = data.batch["token_level_scores"].sum(dim=-1)
    weights = compute_weights(scores, reweight_method, weight_pow)
    weights = torch.clamp(weights + 1e-8, min=1e-8)

    batch_size = scores.shape[0]
    sample_indices = torch.multinomial(weights, batch_size, replacement=True)

    resampled_batch = {
        key: tensor[sample_indices] for key, tensor in data.batch.items()
    }

    sample_indices_np = sample_indices.numpy()
    resampled_non_tensor_batch = {}
    for key, array in data.non_tensor_batch.items():
        if isinstance(array, np.ndarray):
            resampled_non_tensor_batch[key] = array[sample_indices_np]
        else:
            resampled_non_tensor_batch[key] = [array[i] for i in sample_indices_np]

    resampled_meta_info = {}
    for key, value in data.meta_info.items():
        if isinstance(value, list) and len(value) == batch_size:
            resampled_meta_info[key] = [value[i] for i in sample_indices_np]
        else:
            resampled_meta_info[key] = value

    from copy import deepcopy

    resampled_data = deepcopy(data)
    resampled_data.batch = type(data.batch)(resampled_batch)
    resampled_data.batch.batch_size = data.batch.batch_size
    resampled_data.non_tensor_batch = resampled_non_tensor_batch
    resampled_data.meta_info = resampled_meta_info

    return resampled_data
