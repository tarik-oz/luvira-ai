"""
Utility functions for preview scripts (preview_colors.py and preview_tones.py).
Contains common functionality for mask generation and image processing.
"""

import os
import sys

# Import model predictor for automatic mask generation
try:
    from model.inference.predict import load_model
    from model.inference.predictor import create_predictor
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False


def generate_hair_masks(image_paths, model_path, device='auto'):
    """
    Generate hair masks for the given images using the segmentation model.
    
    Args:
        image_paths: List of image paths
        model_path: Path to the trained model
        device: Device to use for inference
        
    Returns:
        Dictionary mapping image paths to mask paths
    """
    if not MODEL_AVAILABLE:
        print("Error: Model inference not available. Please provide masks manually.")
        return {}
    
    print(f"Loading hair segmentation model from {model_path}...")
    try:
        # Load model
        model = load_model(model_path)
        
        # Create predictor
        predictor = create_predictor(model, device=device)
        
        # Generate masks for each image
        image_to_mask = {}
        for img_path in image_paths:
            img_name = os.path.basename(img_path)
            base_name = os.path.splitext(img_name)[0]
            
            print(f"Generating mask for {img_name}...")
            success = predictor.predict_and_save(img_path, base_name, show_visualization=False)
            
            if success:
                # The predictor saves masks to model/test_results directory
                model_results_dir = os.path.join('..', 'model', 'test_results')
                prob_mask_path = os.path.join(model_results_dir, f"{base_name}_prob_mask.png")
                if os.path.exists(prob_mask_path):
                    image_to_mask[img_path] = prob_mask_path
                    print(f"  Using mask from {prob_mask_path}")
            else:
                print(f"  Failed to generate mask for {img_name}")
        
        return image_to_mask
        
    except Exception as e:
        print(f"Error loading model or generating masks: {e}")
        return {}


def find_existing_masks(image_paths, images_dir):
    """
    Find existing masks for the given images.
    
    Args:
        image_paths: List of image paths
        images_dir: Directory containing images and masks
        
    Returns:
        Dictionary mapping image paths to mask paths
    """
    image_to_mask = {}
    model_results_dir = os.path.join('..', 'model', 'test_results')
    
    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        base_name = os.path.splitext(img_name)[0]
        
        # Try to find mask in model/test_results directory
        prob_mask_path = os.path.join(model_results_dir, f"{base_name}_prob_mask.png")
        if os.path.exists(prob_mask_path):
            image_to_mask[img_path] = prob_mask_path
            continue
            
        # If not found in model/test_results, try other locations
        mask_patterns = [
            f"{base_name}_prob_mask.png",
            f"{base_name}_mask.png",
            f"{base_name}_segmentation.png"
        ]
        
        found_mask = False
        for pattern in mask_patterns:
            mask_path = os.path.join(images_dir, pattern)
            if os.path.exists(mask_path):
                image_to_mask[img_path] = mask_path
                found_mask = True
                break
        
        if not found_mask and img_path not in image_to_mask:
            print(f"Warning: No mask found for {img_name}")
    
    return image_to_mask


def find_image_files(images_dir):
    """
    Find all valid image files in a directory, excluding masks.
    
    Args:
        images_dir: Directory to search for images
        
    Returns:
        List of image file paths
    """
    if not os.path.exists(images_dir):
        return []
    
    image_files = [f for f in os.listdir(images_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png')) 
                  and not f.endswith('_mask.png') 
                  and not 'mask' in f.lower()]
    
    return [os.path.join(images_dir, f) for f in image_files]


def validate_image_list(image_names, images_dir):
    """
    Validate a list of image names and return valid paths.
    
    Args:
        image_names: List of image file names
        images_dir: Directory containing images
        
    Returns:
        List of valid image paths
    """
    valid_images = []
    for img_name in image_names:
        img_path = os.path.join(images_dir, img_name)
        if os.path.exists(img_path):
            valid_images.append(img_path)
        else:
            print(f"Warning: Image '{img_name}' not found, ignoring.")
    
    return valid_images


def get_valid_images_with_masks(selected_images, images_dir, use_existing_masks, model_path, device):
    """
    Get valid images that have corresponding masks.
    
    Args:
        selected_images: List of selected image paths
        images_dir: Directory containing images
        use_existing_masks: Whether to use existing masks or generate new ones
        model_path: Path to the trained model (for mask generation)
        device: Device to use for mask generation
        
    Returns:
        Tuple: (valid_images, image_to_mask_dict)
    """
    if use_existing_masks:
        # Use existing masks
        image_to_mask = find_existing_masks(selected_images, images_dir)
    else:
        # Generate masks using the model
        image_to_mask = generate_hair_masks(
            selected_images, 
            model_path, 
            device=device
        )
    
    # Filter out images without masks
    valid_images = [img for img in selected_images if img in image_to_mask]
    
    return valid_images, image_to_mask
