"""
HSV transformer for hair color changing operations.
"""

import numpy as np
import cv2
from typing import Optional

from color_changer.transformers.special_color_handler import SpecialColorHandler
from color_changer.utils.constants import MASK_THRESHOLD


class HsvTransformer:
    """
    Handles HSV color space transformations for hair color changing.
    """

    def __init__(self):
        self.special_color_handler = SpecialColorHandler()

    def apply_hsv_transformations(
        self,
        image_hsv: np.ndarray,
        mask_normalized: np.ndarray,
        target_hsv: np.ndarray,
        alpha: float,
        saturation_factor: float,
        brightness_adjustment: float,
        color_label: Optional[str] = None,
        tone_label: Optional[str] = None,
    ) -> np.ndarray:
        """
        Apply HSV transformations for color change. Performs a light, generic
        adjustment and then delegates to the unified special handler which is
        config-driven by color_label.
        """
        result_hsv = image_hsv.copy()

        # Slightly blur mask to avoid pixel-level artifacts on low-texture/gray hair
        try:
            smooth_mask = cv2.GaussianBlur(mask_normalized.astype(np.float32), (0, 0), 0.8)
            smooth_mask = np.clip(smooth_mask, 0.0, 1.0)
        except Exception:
            smooth_mask = mask_normalized

        # Light base shift toward target hue to initialize (stronger for tones)
        hue_diff = target_hsv[0] - image_hsv[:, :, 0]
        hue_diff = np.where(hue_diff > 90, hue_diff - 180, hue_diff)
        hue_diff = np.where(hue_diff < -90, hue_diff + 180, hue_diff)
        tone_boost = 1.0 if tone_label is not None else 0.0
        init_hue_weight = alpha * (0.5 + 0.3 * tone_boost)
        result_hsv[:, :, 0] = np.where(
            smooth_mask > 0.1, image_hsv[:, :, 0] + hue_diff * init_hue_weight, image_hsv[:, :, 0]
        )
        result_hsv[:, :, 0] = np.clip(result_hsv[:, :, 0] % 180, 0, 179)

        # Mild saturation/value preparation based on brightness comparison
        original_saturation = image_hsv[:, :, 1]
        original_value = image_hsv[:, :, 2]
        hair_pixels = (original_value * mask_normalized)[mask_normalized > MASK_THRESHOLD]
        if len(hair_pixels) > 0:
            hair_brightness = np.mean(hair_pixels) / 255.0
            target_brightness = target_hsv[2] / 255.0
            if hair_brightness < target_brightness:
                brightness_boost = 1.0 + (brightness_adjustment * 0.5 * (target_brightness - hair_brightness) / max(target_brightness, 1e-6))
                brightness_boost = np.where(original_value > 180, 1.0, brightness_boost)
                result_hsv[:, :, 2] = np.where(
                    smooth_mask > 0.1, np.clip(original_value * brightness_boost, 0, 255), original_value
                )
            elif hair_brightness > target_brightness:
                darkness_factor = 1.0 - (brightness_adjustment * 0.3 * (hair_brightness - target_brightness) / max(hair_brightness, 1e-6))
                darkness_factor = np.where(original_value < 50, 1.0, darkness_factor)
                result_hsv[:, :, 2] = np.where(
                    smooth_mask > 0.1, np.clip(original_value * darkness_factor, 0, 255), original_value
                )

        # Heuristic fallback: if no color label but target is low saturation, treat as Gray
        effective_color_label = color_label
        if effective_color_label is None and float(target_hsv[1]) < 60:
            effective_color_label = "Gray"

        # Delegate to unified handler
        result_hsv = self.special_color_handler.handle_color(
            result_hsv, image_hsv, smooth_mask, target_hsv, alpha, effective_color_label
        )

        # Optional soft lightening for very dark base -> very light target (config-driven)
        try:
            from color_changer.config.color_config import COLOR_PROFILES
            profile = COLOR_PROFILES.get(effective_color_label or 'DEFAULT', COLOR_PROFILES['DEFAULT'])
            light_cfg = profile.get('lightening', None)
            if light_cfg and light_cfg.get('enabled', False):
                # Estimate base brightness (0-1)
                base_v = image_hsv[:, :, 2]
                hair_pixels = (base_v * mask_normalized)[mask_normalized > MASK_THRESHOLD]
                if hair_pixels.size > 0:
                    base_brightness = float(np.mean(hair_pixels) / 255.0)
                else:
                    base_brightness = 1.0
                target_brightness = float(target_hsv[2] / 255.0)
                # Trigger only if base is dark and target is light
                if base_brightness < light_cfg.get('dark_thresh', 0.35) and target_brightness > light_cfg.get('light_thresh', 0.60):
                    # Region-wise gentle lift
                    v = result_hsv[:, :, 2]
                    shadow_lift = light_cfg.get('shadow', 0.30) * alpha
                    mid_lift = light_cfg.get('mid', 0.20) * alpha
                    high_lift = light_cfg.get('highlight', 0.08) * alpha
                    upper_bound = float(light_cfg.get('upper_bound', 235))

                    lift = np.where(v < 100, v * shadow_lift,
                           np.where(v > 180, v * high_lift, v * mid_lift))
                    v_new = np.clip(v + lift, 0, upper_bound)
                    result_hsv[:, :, 2] = np.where(mask_normalized > MASK_THRESHOLD, v_new, v)

                    # Mild desaturation to avoid muddy/yellow cast
                    s = result_hsv[:, :, 1]
                    desat = light_cfg.get('desat', 0.08) * alpha
                    s_new = np.clip(s * (1.0 - desat), 0, 255)
                    result_hsv[:, :, 1] = np.where(mask_normalized > MASK_THRESHOLD, s_new, s)
        except Exception:
            pass

        # Tone enhancement: when a specific tone is requested, nudge a bit more towards target
        if tone_label is not None:
            try:
                # Hue: small extra approach
                h = result_hsv[:, :, 0]
                hue_approach_w = 0.15 * alpha
                h = (h + (target_hsv[0] - h) * hue_approach_w) % 180
                result_hsv[:, :, 0] = np.where(smooth_mask > MASK_THRESHOLD, h, result_hsv[:, :, 0])

                # Saturation: increase and approach target
                s = result_hsv[:, :, 1]
                s_scale = 1.0 + 0.18 * alpha
                s = np.clip(s * s_scale, 0, 255)
                s = s + (target_hsv[1] - s) * (0.12 * alpha)
                result_hsv[:, :, 1] = np.where(smooth_mask > MASK_THRESHOLD, np.clip(s, 0, 255), result_hsv[:, :, 1])

                # Value: gentle nudge toward target to keep separability
                v = result_hsv[:, :, 2]
                v = v + (target_hsv[2] - v) * (0.08 * alpha)
                result_hsv[:, :, 2] = np.where(smooth_mask > MASK_THRESHOLD, np.clip(v, 0, 255), result_hsv[:, :, 2])
            except Exception:
                pass

        return result_hsv
