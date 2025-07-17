"""
Blender for natural image blending in hair color change operations.
"""

import numpy as np

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
        
        return result 