"""
Utility functions for preview scripts (preview_colors.py and preview_tones.py).
Contains common functionality for mask generation and image processing.
"""

import os
from pathlib import Path

# Import model predictor for automatic mask generation
try:
    from model.inference.predict import load_model
    from model.inference.predictor import create_predictor
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False

from color_changer.utils.color_utils import ColorUtils
from color_changer.core.color_transformer import ColorTransformer
from color_changer.utils.image_utils import ImageUtils

def handle_list_commands(args):
    """
    Handle --list-colors and --list-tones commands for preview scripts.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        bool: True if a list command was handled (script should exit), False otherwise
    """
    if hasattr(args, 'list_colors') and args.list_colors:
        ColorUtils.list_colors()
        return True
        
    if hasattr(args, 'list_tones') and args.list_tones:
        ColorUtils.list_tones_for_color(args.list_tones)
        return True
        
    return False


def select_colors(args):
    """
    Select and validate colors from command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        List of tuples: [(rgb, name), ...] selected colors
    """
    all_colors = ColorUtils.get_available_colors()
    
    if hasattr(args, 'colors') and args.colors:
        selected_colors = []
        for color_name in args.colors:
            rgb, name = ColorUtils.find_color_by_name(color_name)
            if name:
                selected_colors.append((rgb, name))
            else:
                print(f"Warning: Color '{color_name}' not found, using all colors.")
        
        if not selected_colors:
            print("No valid colors specified, using all colors.")
            return all_colors
        return selected_colors
    else:
        return all_colors


def validate_single_color(color_name):
    """
    Validate and return a single color.
    
    Args:
        color_name: Name of the color to validate
        
    Returns:
        Tuple: (rgb, name) or (None, None) if invalid
    """
    return ColorUtils.find_color_by_name(color_name)


def select_tones_for_color(color_name, requested_tones=None):
    """
    Select tones for a specific color.
    
    Args:
        color_name: Name of the color
        requested_tones: List of requested tone names, or None for all
        
    Returns:
        List of tone names
    """
    from color_changer.config.color_config import CUSTOM_TONES
    
    if color_name not in CUSTOM_TONES:
        return []
    
    available_tones = list(CUSTOM_TONES[color_name].keys())
    
    if requested_tones:
        selected_tones = []
        for tone_name in requested_tones:
            if tone_name in available_tones:
                selected_tones.append(tone_name)
            else:
                print(f"Warning: Tone '{tone_name}' not available for {color_name}, ignoring.")
        
        if not selected_tones:
            print(f"No valid tones specified, using all available tones for {color_name}.")
            return available_tones
        return selected_tones
    else:
        return available_tones


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
                try:
                    project_root = Path(__file__).resolve().parents[2]
                    model_results_dir = str(project_root / 'model' / 'test_results')
                except Exception:
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
    try:
        project_root = Path(__file__).resolve().parents[2]
        model_results_dir = str(project_root / 'model' / 'test_results')
    except Exception:
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

def process_images_with_colors(valid_images, image_to_mask, selected_colors, results_dir):
    """
    Process images by applying multiple colors.
    
    Args:
        valid_images: List of valid image paths
        image_to_mask: Dictionary mapping image paths to mask paths
        selected_colors: List of (rgb, name) tuples
        results_dir: Directory to save results
        
    Returns:
        List of results for visualization
    """
    # Create color transformer
    transformer = ColorTransformer()
    
    # Process each image
    results = []
    for img_path in valid_images:
        img_name = os.path.basename(img_path)
        mask_path = image_to_mask[img_path]
        
        # Load image and mask
        image = ImageUtils.load_image(img_path)
        mask = ImageUtils.load_image(mask_path, grayscale=True)
        
        if image is None or mask is None:
            print(f"Failed to load {img_name} or its mask, skipping.")
            continue
        
        # Apply each color
        image_results = []
        for rgb_color, color_name in selected_colors:
            try:
                # Apply color transformation using color name
                result = transformer.change_hair_color(image, mask, color_name)
                
                # Save result
                base_name = os.path.splitext(img_name)[0]
                out_path = os.path.join(results_dir, f"{base_name}_{color_name.lower()}.png")
                ImageUtils.save_image(result, out_path, convert_to_bgr=True)
                
                image_results.append((color_name, out_path))
                print(f"Successfully applied {color_name} to {img_name}")
                
            except Exception as e:
                print(f"Failed to apply {color_name} to {img_name}: {str(e)}")
        
        if image_results:
            results.append((img_name, image_results))
    
    return results


def process_images_with_tones(valid_images, image_to_mask, base_color_name, selected_tones, results_dir):
    """
    Process images by applying base color and its tones.
    
    Args:
        valid_images: List of valid image paths
        image_to_mask: Dictionary mapping image paths to mask paths
        base_color_name: Name of base color
        selected_tones: List of tone names
        results_dir: Directory to save results
        
    Returns:
        List of results for visualization
    """
    from color_changer.core.color_transformer import ColorTransformer
    from color_changer.utils.image_utils import ImageUtils
    
    # Create color transformer
    transformer = ColorTransformer()
    
    # Process each image
    results = []
    for img_path in valid_images:
        img_name = os.path.basename(img_path)
        mask_path = image_to_mask[img_path]
        
        # Load image and mask
        image = ImageUtils.load_image(img_path)
        mask = ImageUtils.load_image(mask_path, grayscale=True)
        
        if image is None or mask is None:
            print(f"Failed to load {img_name} or its mask, skipping.")
            continue
        
        # Apply each tone
        image_results = []
        base_name = os.path.splitext(img_name)[0]
        
        # Always include base color for comparison
        try:
            # Apply base color transformation using color name
            base_result = transformer.change_hair_color(image, mask, base_color_name)
            
            out_path = os.path.join(results_dir, f"{base_name}_{base_color_name.lower()}_base.png")
            ImageUtils.save_image(base_result, out_path, convert_to_bgr=True)
            image_results.append((f"{base_color_name} (base)", out_path))
            print(f"Successfully applied base {base_color_name} to {img_name}")
        except Exception as e:
            print(f"Failed to apply base {base_color_name} to {img_name}: {str(e)}")
        
        # Apply tones
        for tone_name in selected_tones:
            try:
                # Apply tone transformation
                result = transformer.apply_color_with_tone(
                    image, mask, base_color_name, tone_name
                )
                
                # Save result
                out_path = os.path.join(results_dir, f"{base_name}_{base_color_name.lower()}_{tone_name}.png")
                ImageUtils.save_image(result, out_path, convert_to_bgr=True)
                
                image_results.append((f"{base_color_name} ({tone_name})", out_path))
                print(f"Successfully applied {base_color_name} {tone_name} tone to {img_name}")
                
            except Exception as e:
                print(f"Failed to apply {base_color_name} {tone_name} tone to {img_name}: {str(e)}")
        
        if image_results:
            results.append((img_name, image_results))
    
    return results
