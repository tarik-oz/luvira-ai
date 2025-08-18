"""
Blender for natural image blending in hair color change operations.
"""

import numpy as np
import cv2

class Blender:
    """
    Handles natural blending for hair color change operations.
    """
    
    def apply_natural_blending(
        self,
        image_float: np.ndarray,
        result_rgb: np.ndarray,
        mask_3ch: np.ndarray,
        alpha: float
    ) -> np.ndarray:
        """
        Apply natural blending to preserve texture and lighting.
        
        Args:
            image_float: Original normalized image
            result_rgb: Color-changed normalized image
            mask_3ch: 3-channel mask
            alpha: Blending factor
        
        Returns:
            np.ndarray: Final blended image
        """
        # Use mask intensity to create smooth transitions
        smooth_mask = mask_3ch * alpha
        
        # Create luminance-based blending for more natural look
        original_luminance = 0.299 * image_float[:,:,0] + 0.587 * image_float[:,:,1] + 0.114 * image_float[:,:,2]
        target_luminance = 0.299 * result_rgb[:,:,0] + 0.587 * result_rgb[:,:,1] + 0.114 * result_rgb[:,:,2]
        
        # Blend based on luminance similarity - preserve natural texture
        luminance_diff = np.abs(original_luminance - target_luminance)
        luminance_factor = np.where(luminance_diff > 0.3, 0.8, 1.0)
        adaptive_blend = smooth_mask * np.stack([luminance_factor] * 3, axis=-1)
        
        # Final gentle blending
        result = image_float * (1 - adaptive_blend) + result_rgb * adaptive_blend

        # Detail preservation: add back high-frequency luminance from original
        # to improve strand definition within hair mask (reduced strength for natural look)
        try:
            # Compute luminance high-frequency from original image
            orig_lum = 0.299 * image_float[:, :, 0] + 0.587 * image_float[:, :, 1] + 0.114 * image_float[:, :, 2]
            # Small Gaussian blur to obtain base layer
            base_lum = cv2.GaussianBlur(orig_lum, (0, 0), sigmaX=1.0)
            detail_lum = orig_lum - base_lum

            # Limit detail amplitude to avoid artifacts (AGGRESSIVE SOFT TEST)
            detail_lum = np.clip(detail_lum, -0.04, 0.04)

            # Apply detail back to result only within hair mask region (reduced)
            detail_strength = 0.06
            result += (detail_lum * detail_strength)[:, :, None] * mask_3ch
            result = np.clip(result, 0.0, 1.0)
        except Exception:
            # Fail-safe: if any issue occurs, return blended result
            result = np.clip(result, 0.0, 1.0)
        
        # Global softening within hair region (strong for test)
        try:
            hair_w = np.clip(mask_3ch[:, :, 0].astype("float32"), 0.0, 1.0)
            # Bilateral filter for edge-preserving smoothing
            res_bgr = cv2.cvtColor((result * 255.0).astype("uint8"), cv2.COLOR_RGB2BGR)
            smooth_bgr = cv2.bilateralFilter(res_bgr, d=7, sigmaColor=50, sigmaSpace=9)
            smooth = cv2.cvtColor(smooth_bgr, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
            soften_strength = 0.45  # AGGRESSIVE for test
            hair_w3 = (hair_w * soften_strength)[:, :, None]
            result = result * (1.0 - hair_w3) + smooth * hair_w3
            result = np.clip(result, 0.0, 1.0)
            # Extra gentle Gaussian blur mix to further reduce crispness inside hair
            gauss = cv2.GaussianBlur((result * 255.0).astype("uint8"), (0, 0), sigmaX=0.9)
            gauss = gauss.astype("float32") / 255.0
            g_w = (hair_w * 0.12)[:, :, None]
            result = result * (1.0 - g_w) + gauss * g_w
            result = np.clip(result, 0.0, 1.0)
        except Exception:
            pass
        
        # Naturalization: recompose using original luminance/saturation (AGGRESSIVE TEST)
        try:
            hair_w = np.clip(mask_3ch[:, :, 0].astype("float32"), 0.0, 1.0)
            res_u8 = (result * 255.0).astype("uint8")
            orig_u8 = (image_float * 255.0).astype("uint8")
            res_hsv = cv2.cvtColor(res_u8, cv2.COLOR_RGB2HSV).astype("float32")
            orig_hsv = cv2.cvtColor(orig_u8, cv2.COLOR_RGB2HSV).astype("float32")
            s_blend = 0.50  # pull some saturation back from original
            v_blend = 0.60  # preserve more of original luminance
            w3 = hair_w[:, :, None]
            res_hsv[:, :, 1] = res_hsv[:, :, 1] * (1.0 - s_blend * w3[:, :, 0]) + orig_hsv[:, :, 1] * (s_blend * w3[:, :, 0])
            res_hsv[:, :, 2] = res_hsv[:, :, 2] * (1.0 - v_blend * w3[:, :, 0]) + orig_hsv[:, :, 2] * (v_blend * w3[:, :, 0])
            res_hsv = np.clip(res_hsv, [0, 0, 0], [179, 255, 255])
            result = cv2.cvtColor(res_hsv.astype("uint8"), cv2.COLOR_HSV2RGB).astype("float32") / 255.0
        except Exception:
            pass

        return result 