"""
Inference module for hair segmentation U-Net model.
Handles model prediction and result visualization.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from typing import Union, Tuple, Optional, List
import logging

from model.config import (
    TEST_IMAGES_DIR, TEST_RESULTS_DIR, 
    DATA_CONFIG, MODEL_CONFIG
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HairSegmentationPredictor:
    """
    Predictor class for hair segmentation U-Net model.
    
    Handles model inference, prediction, and result visualization.
    """
    
    def __init__(self, 
                 model,
                 test_images_dir: Path = TEST_IMAGES_DIR,
                 test_results_dir: Path = TEST_RESULTS_DIR,
                 image_size: Tuple[int, int] = DATA_CONFIG["image_size"],
                 normalization_factor: float = DATA_CONFIG["normalization_factor"],
                 mask_threshold: float = DATA_CONFIG["mask_threshold"],
                 device: str = "auto"):
        """
        Initialize the predictor.
        
        Args:
            model: Trained U-Net model
            test_images_dir: Directory containing test images
            test_results_dir: Directory to save prediction results
            image_size: Target size for images (height, width)
            normalization_factor: Factor to normalize pixel values
            mask_threshold: Threshold for binary mask conversion
            device: Device to use for inference (auto, cpu, cuda)
        """
        self.model = model
        self.test_images_dir = Path(test_images_dir)
        self.test_results_dir = Path(test_results_dir)
        self.image_size = image_size
        self.normalization_factor = normalization_factor
        self.mask_threshold = mask_threshold
        
        # Setup device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device == "cuda":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
        
        # Create results directory if it doesn't exist
        self.test_results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Using device: {self.device}")
        
    def preprocess_image(self, image_path: Union[str, Path]) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Preprocess a single image for prediction.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (preprocessed_tensor, original_size)
        """
        try:
            # Load image
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Store original size for later use
            original_size = image.shape[:2]
            
            # Resize image
            image_resized = cv2.resize(image, self.image_size)
            
            # Normalize pixel values
            image_normalized = image_resized / self.normalization_factor
            image_normalized = image_normalized.astype(np.float32)
            
            # Convert to PyTorch format (CHW)
            image_tensor = torch.FloatTensor(image_normalized).permute(2, 0, 1)
            
            # Add batch dimension
            image_batch = image_tensor.unsqueeze(0)
            
            return image_batch, original_size
            
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            return None, None
    
    def predict(self, image_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict segmentation mask for a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (original_image, predicted_mask, binary_mask)
        """
        # Preprocess image
        image_batch, original_size = self.preprocess_image(image_path)
        if image_batch is None:
            return None, None, None
        
        # Load original image for visualization
        original_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        
        # Move to device
        image_batch = image_batch.to(self.device)
        
        # Make prediction
        with torch.no_grad():
            prediction = self.model(image_batch)
            predicted_mask = prediction[0].cpu().numpy()  # Remove batch dimension
        
        # Convert to binary mask
        binary_mask = self._create_binary_mask(predicted_mask)
        
        # Create 2D version of predicted_mask for visualization
        if predicted_mask.ndim == 3 and predicted_mask.shape[0] == 1:
            predicted_mask_2d = predicted_mask[0]
        else:
            predicted_mask_2d = predicted_mask
        
        # Resize masks to original image size
        if original_size:
            predicted_mask_2d = cv2.resize(predicted_mask_2d, (original_size[1], original_size[0]))
            binary_mask = cv2.resize(binary_mask, (original_size[1], original_size[0]))
        
        return original_image, predicted_mask_2d, binary_mask
    
    def _create_binary_mask(self, predicted_mask: np.ndarray) -> np.ndarray:
        """
        Create binary mask from predicted probabilities.
        
        Args:
            predicted_mask: Predicted mask with probabilities (shape: (1, H, W) or (H, W))
            
        Returns:
            Binary mask (shape: (H, W))
        """
        # Remove channel dimension if present
        if predicted_mask.ndim == 3 and predicted_mask.shape[0] == 1:
            predicted_mask_2d = predicted_mask[0]
        else:
            predicted_mask_2d = predicted_mask
        
        # Normalize to 0-255 range
        mask_normalized = (predicted_mask_2d - predicted_mask_2d.min()) / (predicted_mask_2d.max() - predicted_mask_2d.min())
        mask_scaled = (mask_normalized * 255).astype(np.uint8)
        
        # Create binary mask using threshold
        binary_mask = np.zeros_like(mask_scaled)
        binary_mask[mask_scaled > (self.mask_threshold * 255)] = 255
        
        return binary_mask
    
    def visualize_prediction(self, 
                           original_image: np.ndarray,
                           predicted_mask: np.ndarray,
                           binary_mask: np.ndarray,
                           save_path: Optional[Path] = None,
                           show_plot: bool = True) -> None:
        """
        Visualize prediction results.
        
        Args:
            original_image: Original input image
            predicted_mask: Predicted probability mask
            binary_mask: Binary segmentation mask
            save_path: Path to save the visualization
            show_plot: Whether to display the plot
        """
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        
        # Predicted mask
        axes[1].imshow(predicted_mask, cmap='gray')
        axes[1].set_title("Predicted Mask")
        axes[1].axis("off")
        
        # Binary mask
        axes[2].imshow(binary_mask, cmap='gray')
        axes[2].set_title("Binary Mask")
        axes[2].axis("off")
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        # Show plot
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def predict_and_save(self, 
                        image_path: Union[str, Path],
                        output_name: Optional[str] = None,
                        show_visualization: bool = False) -> bool:
        """
        Predict segmentation and save results.
        
        Args:
            image_path: Path to the image file
            output_name: Name for output files (without extension)
            show_visualization: Whether to show visualization plots
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate output name if not provided
            if output_name is None:
                output_name = Path(image_path).stem
            
            # Make prediction
            original_image, predicted_mask, binary_mask = self.predict(image_path)
            
            if original_image is None:
                return False
            
            # Save binary mask
            mask_path = self.test_results_dir / f"{output_name}_binary_mask.png"
            cv2.imwrite(str(mask_path), binary_mask)
            
            # Save normal probability mask (better for hair color manipulation)
            prob_mask_path = self.test_results_dir / f"{output_name}_prob_mask.png"
            # Convert probability mask to 0-255 range for saving
            prob_mask_255 = (predicted_mask * 255).astype(np.uint8)
            cv2.imwrite(str(prob_mask_path), prob_mask_255)
            
            # Save visualization
            viz_path = self.test_results_dir / f"{output_name}_visualization.png"
            self.visualize_prediction(
                original_image, predicted_mask, binary_mask,
                save_path=viz_path, show_plot=show_visualization
            )
            
            logger.info(f"Results saved:")
            logger.info(f"  - Binary mask: {mask_path}")
            logger.info(f"  - Probability mask: {prob_mask_path}")
            logger.info(f"  - Visualization: {viz_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return False
    
    def predict_batch(self, 
                     image_paths: List[Union[str, Path]],
                     output_names: Optional[List[str]] = None,
                     show_visualization: bool = False) -> List[bool]:
        """
        Predict segmentation for multiple images.
        
        Args:
            image_paths: List of image paths
            output_names: List of output names (optional)
            show_visualization: Whether to show visualization plots
            
        Returns:
            List of success status for each image
        """
        results = []
        
        for i, image_path in enumerate(image_paths):
            output_name = output_names[i] if output_names else None
            success = self.predict_and_save(image_path, output_name, show_visualization)
            results.append(success)
        
        return results
    
    def predict_directory(self, 
                         input_dir: Optional[Path] = None,
                         file_pattern: str = "*.jpg",
                         show_visualization: bool = False) -> List[bool]:
        """
        Predict segmentation for all images in a directory.
        
        Args:
            input_dir: Input directory (uses test_images_dir if None)
            file_pattern: File pattern to match
            show_visualization: Whether to show visualization plots
            
        Returns:
            List of success status for each image
        """
        if input_dir is None:
            input_dir = self.test_images_dir
        
        # Find all matching files
        image_paths = list(input_dir.glob(file_pattern))
        
        if not image_paths:
            logger.warning(f"No images found in {input_dir} matching pattern {file_pattern}")
            return []
        
        logger.info(f"Found {len(image_paths)} images to process")
        
        # Process all images
        return self.predict_batch(image_paths, show_visualization=show_visualization)


def create_predictor(model, **kwargs) -> HairSegmentationPredictor:
    """
    Factory function to create a predictor.
    
    Args:
        model: Trained model
        **kwargs: Additional arguments for HairSegmentationPredictor
        
    Returns:
        HairSegmentationPredictor instance
    """
    return HairSegmentationPredictor(model, **kwargs) 