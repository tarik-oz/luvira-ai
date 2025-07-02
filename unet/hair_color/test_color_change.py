import cv2
import matplotlib.pyplot as plt
from color_changer import HairColorChanger

if __name__ == "__main__":
    # Test images
    images = [
        ("../test_images/1626.jpg", "../test_results_2k/1626_prob_mask.png"),
        ("../test_images/1624.jpg", "../test_results_2k/1624_prob_mask.png")
    ]
    
    # Hair colors to test
    colors = [
        ([0, 0, 255], "Mavi"),
        ([128, 0, 128], "Mor"),
        ([139, 69, 19], "Kahverengi"),
        ([255, 192, 203], "Pembe"),
        ([128, 128, 128], "Gri")
    ]
    
    alpha = 0.4  # Color intensity

    print("Testing hair color change with multiple colors and images...")

    # Create figure with subplots
    fig, axes = plt.subplots(2, len(colors) + 1, figsize=(20, 8))
    
    for img_idx, (image_path, mask_path) in enumerate(images):
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Original image
        axes[img_idx, 0].set_title(f"Orijinal ({image_path.split('/')[-1].split('.')[0]})")
        axes[img_idx, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[img_idx, 0].axis("off")
        
        # Apply different colors
        for color_idx, (rgb_color, color_name) in enumerate(colors):
            result = HairColorChanger.change_hair_color(image, mask, rgb_color, alpha, saturation_factor=1.3)
            
            axes[img_idx, color_idx + 1].set_title(f"{color_name}")
            axes[img_idx, color_idx + 1].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            axes[img_idx, color_idx + 1].axis("off")

    plt.tight_layout()
    plt.show()