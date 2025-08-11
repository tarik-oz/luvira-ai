"""
Special color handler that applies config-driven HSV transformations.
All special colors are handled through a unified method using per-color profiles.
"""

import numpy as np
import cv2
from typing import Dict, Any, Optional

from color_changer.config.color_config import COLOR_PROFILES
from color_changer.utils.constants import MASK_THRESHOLD
from color_changer.utils.hsv_utils import (
    shortest_hue_diff,
    approach_target,
    apply_value_gains,
    anti_pink_correction,
    clip_hsv_channels,
)


class SpecialColorHandler:
    """
    Unified, config-driven color handler. Applies hue/saturation/value adjustments
    using a color profile selected by color label. Supports optional corrections
    (e.g., gray mode, anti-pink) controlled via the profile.
    The implementation is organized into small private helpers for readability.
    """

    def _get_profile(self, color_label: Optional[str]) -> Dict[str, Any]:
        if color_label and color_label in COLOR_PROFILES:
            return COLOR_PROFILES[color_label]
        return COLOR_PROFILES["DEFAULT"]

    def handle_color(
        self,
        result_hsv: np.ndarray,
        image_hsv: np.ndarray,
        mask_normalized: np.ndarray,
        target_hsv: np.ndarray,
        alpha: float,
        color_label: Optional[str] = None,
        tone_label: Optional[str] = None
    ) -> np.ndarray:
        """Apply config-driven HSV transformations for given color profile."""
        profile = self._get_profile(color_label)

        # Gray-mode early return
        if profile.get("corrections", {}).get("gray_mode", False):
            return self._apply_gray_mode(result_hsv, image_hsv, mask_normalized, target_hsv, alpha, profile)

        # Base HSV adjustments
        result_hsv = self._apply_base_hsv_adjustments(result_hsv, image_hsv, mask_normalized, target_hsv, alpha, profile)

        # Corrections in sequence (with optional tone relaxation)
        result_hsv = self._apply_corrections(result_hsv, image_hsv, mask_normalized, target_hsv, alpha, profile, tone_label)

        # Final safety clip
        result_hsv = clip_hsv_channels(result_hsv)
        return result_hsv

    # --- Helpers ---

    def _apply_gray_mode(self, result_hsv, image_hsv, mask, target_hsv, alpha, profile):
        from color_changer.utils.gray_utils import GrayUtils
        original_saturation = image_hsv[:, :, 1]
        original_value = image_hsv[:, :, 2]
        hair_pixels = (original_value * mask)[mask > MASK_THRESHOLD]
        if len(hair_pixels) == 0:
            return image_hsv.copy()
        avg_hair_brightness = np.mean(hair_pixels)
        # Saturation reduction
        sat_reduction = 0.9 if avg_hair_brightness < 50 else 0.95
        new_saturation = np.clip(original_saturation * (1 - alpha * sat_reduction), 0, 255)
        result_hsv[:, :, 1] = GrayUtils.apply_masked_channel(result_hsv[:, :, 1], new_saturation, mask)
        # Value adjustment based on target
        target_value_factor = target_hsv[2] / 255.0
        new_value = GrayUtils.get_gray_value_boost(
            avg_hair_brightness, target_value_factor, alpha, original_value, mask
        )
        result_hsv[:, :, 2] = GrayUtils.apply_masked_channel(result_hsv[:, :, 2], new_value, mask)
        # Reduce hue influence
        suppress = profile.get("hue", {}).get("suppress_factor", 0.4)
        new_hue = image_hsv[:, :, 0] * (1 - alpha * suppress)
        result_hsv[:, :, 0] = GrayUtils.apply_masked_channel(result_hsv[:, :, 0], new_hue, mask)
        return result_hsv

    def _apply_base_hsv_adjustments(self, result_hsv, image_hsv, mask, target_hsv, alpha, profile):
        current_h = image_hsv[:, :, 0]
        current_s = image_hsv[:, :, 1]
        current_v = image_hsv[:, :, 2]

        # Hue shift toward target
        hue_weight = float(profile.get("hue", {}).get("weight", 0.9)) * alpha
        hue_diff = shortest_hue_diff(target_hsv[0], current_h)
        new_h = (current_h + hue_diff * hue_weight) % 180
        result_hsv[:, :, 0] = np.where(mask > MASK_THRESHOLD, new_h, result_hsv[:, :, 0])

        # Saturation scaling and approach target
        sat_cfg = profile.get("sat", {})
        sat_scale = float(sat_cfg.get("scale", 1.2))
        sat_min = float(sat_cfg.get("min", 0))
        sat_max = float(sat_cfg.get("max", 255))
        sat_approach_w = float(sat_cfg.get("approach_target_weight", 0.4)) * alpha
        high_sat_boost = bool(sat_cfg.get("high_sat_boost", False))

        scaled_s = np.clip(current_s * (1.0 + (sat_scale - 1.0) * alpha), 0, 255)
        scaled_s = np.where(scaled_s < sat_min, sat_min, scaled_s)
        scaled_s = np.where(scaled_s > sat_max, sat_max, scaled_s)
        target_s = float(target_hsv[1])
        nudged_s = approach_target(scaled_s, target_s, sat_approach_w)
        if high_sat_boost and target_s > 200:
            nudged_s = approach_target(nudged_s, max(target_s, 220.0), 0.2 * alpha)
        result_hsv[:, :, 1] = np.where(mask > MASK_THRESHOLD, np.clip(nudged_s, 0, 255), result_hsv[:, :, 1])

        # Value (brightness) mapping using region gains
        val_cfg = profile.get("val", {})
        bounds = np.array(val_cfg.get("bounds", [0, 255]), dtype=np.float32)
        new_v = apply_value_gains(current_v, mask,
                                   float(val_cfg.get("shadow_gain", 1.1)),
                                   float(val_cfg.get("mid_gain", 1.05)),
                                   float(val_cfg.get("highlight_gain", 0.95)),
                                   bounds, alpha)
        result_hsv[:, :, 2] = new_v
        return result_hsv

    def _apply_corrections(self, result_hsv, image_hsv, mask, target_hsv, alpha, profile, tone_label: Optional[str] = None):
        corrections = profile.get("corrections", {})
        bounds = np.array(profile.get("val", {}).get("bounds", [0, 255]), dtype=np.float32)

        # Tone relaxation: when a specific tone is requested, relax hue constraints
        # and enhance approach to target saturation for more visible tonal separation.
        if tone_label is not None:
            try:
                # Relax hue band
                hue_band = corrections.get("hue_band")
                if isinstance(hue_band, (list, tuple)) and len(hue_band) == 2:
                    center = 0.5 * (float(hue_band[0]) + float(hue_band[1]))
                    width = max(6.0, float(hue_band[1]) - float(hue_band[0]))  # ensure minimum width
                    new_width = width * 2.0  # widen band for tone
                    corrections["hue_band"] = [center - new_width / 2.0, center + new_width / 2.0]

                # Reduce hue center weight
                if "hue_center_weight" in corrections:
                    corrections["hue_center_weight"] = float(corrections["hue_center_weight"]) * 0.6
                else:
                    corrections["hue_center_weight"] = 0.18

                # Increase saturation approach slightly
                sat_cfg = profile.setdefault("sat", {})
                sat_cfg["approach_target_weight"] = float(sat_cfg.get("approach_target_weight", 0.35)) * 1.25

                # Enforce smoothing on tone for clean transitions
                corrections["post_smooth"] = True
                corrections["bilateral_hue_smooth"] = True
            except Exception:
                pass

        if corrections.get("anti_pink", False):
            result_hsv[:, :, 0] = anti_pink_correction(result_hsv[:, :, 0], mask)

        # Hue band clamp
        hue_band = corrections.get("hue_band")
        if isinstance(hue_band, (list, tuple)) and len(hue_band) == 2:
            hmin, hmax = float(hue_band[0]), float(hue_band[1])
            clamped_h = np.clip(result_hsv[:, :, 0], hmin, hmax)
            result_hsv[:, :, 0] = np.where(mask > MASK_THRESHOLD, clamped_h, result_hsv[:, :, 0])

        # Hue center
        hue_center = corrections.get("hue_center")
        hue_center_weight = float(corrections.get("hue_center_weight", 0.3))
        if isinstance(hue_center, (int, float)):
            centered_h = approach_target(result_hsv[:, :, 0], float(hue_center), hue_center_weight * alpha)
            result_hsv[:, :, 0] = np.where(mask > MASK_THRESHOLD, centered_h, result_hsv[:, :, 0])

        # Value dependent hue center
        if corrections.get("value_dependent_hue_center", False):
            try:
                v = image_hsv[:, :, 2]
                if not isinstance(hue_center, (int, float)):
                    hue_center = float(target_hsv[0])
                w = np.clip((100.0 - v) / 100.0, 0.0, 1.0) * alpha * 0.5
                w = w * (mask > MASK_THRESHOLD)
                stabilized_h = result_hsv[:, :, 0] * (1 - w) + float(hue_center) * w
                result_hsv[:, :, 0] = stabilized_h
            except Exception:
                pass

        # Desaturate near pink range
        if corrections.get("desat_near_pink", False):
            try:
                h = result_hsv[:, :, 0]
                s = result_hsv[:, :, 1]
                near_pink = (h > 158) & (mask > MASK_THRESHOLD)
                s = np.where(near_pink, s * (1.0 - 0.12 * alpha), s)
                result_hsv[:, :, 1] = s
            except Exception:
                pass

        # Gaussian smoothing
        if corrections.get("post_smooth", False):
            try:
                h_blur = cv2.GaussianBlur(result_hsv[:, :, 0], (0, 0), 0.8)
                s_blur = cv2.GaussianBlur(result_hsv[:, :, 1], (0, 0), 0.6)
                v_blur = cv2.GaussianBlur(result_hsv[:, :, 2], (0, 0), 0.6)
                blend_w = np.clip(mask * 0.4, 0.0, 1.0)
                result_hsv[:, :, 0] = result_hsv[:, :, 0] * (1 - blend_w) + h_blur * blend_w
                result_hsv[:, :, 1] = result_hsv[:, :, 1] * (1 - blend_w) + s_blur * blend_w
                result_hsv[:, :, 2] = result_hsv[:, :, 2] * (1 - blend_w) + v_blur * blend_w
            except Exception:
                pass

        # Bilateral smoothing for hue
        if corrections.get("bilateral_hue_smooth", False):
            try:
                h = result_hsv[:, :, 0].astype(np.float32)
                h_b = cv2.bilateralFilter(h, d=5, sigmaColor=20, sigmaSpace=5)
                result_hsv[:, :, 0] = np.where(mask > MASK_THRESHOLD, h_b, result_hsv[:, :, 0])
            except Exception:
                pass

        # Highlight protection
        if corrections.get("highlight_protect", False):
            try:
                orig_s = image_hsv[:, :, 1]
                orig_v = image_hsv[:, :, 2]
                spec_mask = (orig_v > 220) & (mask > MASK_THRESHOLD)
                if np.any(spec_mask):
                    hp_sat_reduce = float(corrections.get("hp_sat_reduce", 0.15))
                    hp_hue_blend = float(corrections.get("hp_hue_blend", 0.40))
                    hp_v_cap = float(corrections.get("hp_v_cap", bounds[1]))
                    result_hsv[:, :, 0] = np.where(
                        spec_mask,
                        result_hsv[:, :, 0] * (1 - hp_hue_blend) + image_hsv[:, :, 0] * hp_hue_blend,
                        result_hsv[:, :, 0]
                    )
                    result_hsv[:, :, 1] = np.where(spec_mask, result_hsv[:, :, 1] * (1 - hp_sat_reduce), result_hsv[:, :, 1])
                    result_hsv[:, :, 2] = np.where(spec_mask, np.minimum(result_hsv[:, :, 2], hp_v_cap), result_hsv[:, :, 2])
            except Exception:
                pass

        # Desaturation compensation for gray/white hair
        if corrections.get("desat_comp", False):
            try:
                orig_s = image_hsv[:, :, 1]
                low_sat_mask = (orig_s < float(corrections.get("desat_thresh", 35))) & (mask > MASK_THRESHOLD)
                if np.any(low_sat_mask):
                    boost = float(corrections.get("desat_boost", 1.20))
                    extra_hue_w = float(corrections.get("desat_hue_weight_boost", 0.10)) * alpha
                    result_hsv[:, :, 0] = np.where(
                        low_sat_mask,
                        (result_hsv[:, :, 0] + (target_hsv[0] - result_hsv[:, :, 0]) * extra_hue_w) % 180,
                        result_hsv[:, :, 0]
                    )
                    result_hsv[:, :, 1] = np.where(low_sat_mask, np.clip(result_hsv[:, :, 1] * boost, 0, 255), result_hsv[:, :, 1])
            except Exception:
                pass

        return result_hsv