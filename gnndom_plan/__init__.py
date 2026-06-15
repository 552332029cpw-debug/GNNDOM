# SPDX-License-Identifier: BSD-3-Clause
"""Online VSBL planning for GNNDOM ClothDrop environments."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .actions import split_action
from .config import PlanConfig

if TYPE_CHECKING:
    from .graph import OnlineVisibleGraphBuilder
    from .planner import MPCPlanner
    from .rollout import DynamicsRollout
    from .runner import PlanRunner

__all__ = [
    "DynamicsRollout",
    "MPCPlanner",
    "OnlineVisibleGraphBuilder",
    "PlanConfig",
    "PlanRunner",
    "split_action",
]


def __getattr__(name: str):
    if name == "OnlineVisibleGraphBuilder":
        from .graph import OnlineVisibleGraphBuilder

        return OnlineVisibleGraphBuilder
    if name == "MPCPlanner":
        from .planner import MPCPlanner

        return MPCPlanner
    if name == "DynamicsRollout":
        from .rollout import DynamicsRollout

        return DynamicsRollout
    if name == "PlanRunner":
        from .runner import PlanRunner

        return PlanRunner
    raise AttributeError(name)
