import numpy as np

class GrayUtils:
    @staticmethod
    def apply_masked_channel(result_channel: np.ndarray, new_channel: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Blend a new channel into the result channel using the mask.
        Only applies changes where mask > 0.1, preserves original elsewhere.
        Args:
            result_channel: The current channel to modify
            new_channel: The new values to blend in
            mask: Normalized mask (0-1)
        Returns:
            np.ndarray: Blended channel
        """
        return np.where(mask > 0.1, new_channel, result_channel)

    @staticmethod
    def get_gray_value_boost(avg_hair_brightness: float, target_value_factor: float, alpha: float, original_value: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Calculate the value/brightness adjustment for gray hair transformation.
        Args:
            avg_hair_brightness: Average brightness of hair pixels
            target_value_factor: Target value (0-1)
            alpha: Blending factor
            original_value: Original value channel
            mask: Normalized mask
        Returns:
            np.ndarray: Adjusted value channel
        """
        if avg_hair_brightness < 50:
            value_boost = 1.0 + (target_value_factor - 0.1) * alpha * 1.2
            new_value = original_value * value_boost
        else:
            if target_value_factor < 0.3:
                new_value = original_value * (0.3 + target_value_factor * 0.8)
            elif target_value_factor > 0.7:
                new_value = original_value * (0.7 + target_value_factor * 0.4)
            else:
                new_value = original_value * (0.5 + target_value_factor * 0.6)
        return np.clip(new_value, 0, 255)
