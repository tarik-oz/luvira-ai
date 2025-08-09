"""
Common HSV utilities used across color transformations.
"""

from typing import Tuple
import numpy as np


def apply_masked_channel(result_channel: np.ndarray, new_channel: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.where(mask > 0.1, new_channel, result_channel)


def shortest_hue_diff(target_hue: float, current_hue: np.ndarray) -> np.ndarray:
    hue_diff = target_hue - current_hue
    hue_diff = np.where(hue_diff > 90, hue_diff - 180, hue_diff)
    hue_diff = np.where(hue_diff < -90, hue_diff + 180, hue_diff)
    return hue_diff


def approach_target(current: np.ndarray, target: float, weight: float) -> np.ndarray:
    return current + (target - current) * weight


def apply_value_gains(
    original_value: np.ndarray,
    mask: np.ndarray,
    shadow_gain: float,
    mid_gain: float,
    highlight_gain: float,
    bounds: np.ndarray,
    alpha: float,
) -> np.ndarray:
    v = original_value
    out = np.where(
        v < 100,
        v * (1.0 + (shadow_gain - 1.0) * alpha),
        np.where(
            v > 180,
            v * (1.0 + (highlight_gain - 1.0) * alpha),
            v * (1.0 + (mid_gain - 1.0) * alpha),
        ),
    )
    out = np.clip(out, bounds[0], bounds[1])
    return apply_masked_channel(original_value, out, mask)


def anti_pink_correction(hue_channel: np.ndarray, mask: np.ndarray) -> np.ndarray:
    pink_mask = (hue_channel < 140) & (mask > 0.1)
    corrected = np.where(pink_mask, np.clip(hue_channel + 10, 140, 160), hue_channel)
    return corrected


def clip_hsv_channels(hsv: np.ndarray) -> np.ndarray:
    hsv[:, :, 0] = np.clip(hsv[:, :, 0] % 180, 0, 179)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    return hsv


