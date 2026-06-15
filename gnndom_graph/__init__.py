# SPDX-License-Identifier: BSD-3-Clause
"""Graph construction utilities for GNNDOM rollout datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .builder import GraphBuildConfig, build_graphs_from_dataset

__all__ = ["GraphBuildConfig", "build_graphs_from_dataset"]


def __getattr__(name: str):
    if name in __all__:
        from .builder import GraphBuildConfig, build_graphs_from_dataset

        return {"GraphBuildConfig": GraphBuildConfig, "build_graphs_from_dataset": build_graphs_from_dataset}[name]
    raise AttributeError(name)
