import cv2
import numpy as np

class HairColorChanger:
    """
    Class for changing hair color with non-binary mask, simple direct coloring and blending.
    """
    @staticmethod
    def change_hair_color(image, mask, rgb_color, alpha=0.5):
        """
        image: Original image (BGR or RGB, np.ndarray)
        mask: Non-binary mask (0-255, np.ndarray, single channel)
        rgb_color: Target hair color (e.g: [255, 0, 0] red)
        alpha: Color intensity (0-1)
        """
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = mask.astype(np.float32) / 255.0
        mask_3ch = np.stack([mask]*3, axis=-1)

        colored = np.ones_like(image, dtype=np.float32) * np.array(rgb_color, dtype=np.float32)
        
        result = image.astype(np.float32).copy()
        result = result * (1 - mask_3ch) + colored * mask_3ch

        result = cv2.addWeighted(image.astype(np.float32), 1 - alpha, result, alpha, 0)
        return np.clip(result, 0, 255).astype(np.uint8)