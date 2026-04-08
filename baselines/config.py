"""Configuration helpers for the BAPO recipe.

These dataclasses extend the core verl configuration objects with the
hyper-parameters that are specific to the BAPO clipping strategy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from verl.workers.config import FSDPActorConfig, PolicyLossConfig





@dataclass
class BaselineActorConfig(FSDPActorConfig):
    """Actor configuration for the BAPO recipe.

    The only behavioural change compared to the base ``FSDPActorConfig`` is the
    substitution of the default ``PolicyLossConfig`` with
    :class:`BAPOPolicyLossConfig` so that Hydra can populate the additional
    hyper-parameters.
    """
    skip_contradict_token: bool = False 
    skip_overlong: bool = True
