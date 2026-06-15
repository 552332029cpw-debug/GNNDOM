# SPDX-License-Identifier: BSD-3-Clause
"""Action utilities for GNNDOM planning."""

from __future__ import annotations

import numpy as np


def split_action(action: np.ndarray, pred_time_interval: int) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).reshape(8)
    pred_time_interval = max(int(pred_time_interval), 1)
    if pred_time_interval == 1:
        out = action.reshape(1, 8).copy()
    else:
        out = np.zeros((pred_time_interval, 8), dtype=np.float32)
        out[:, :] = action / np.float32(pred_time_interval)
    out[:, 3] = 1.0 if action[3] > 0 else 0.0
    out[:, 7] = 1.0 if action[7] > 0 else 0.0
    return out.astype(np.float32)
