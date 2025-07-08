import cv2
import matplotlib.pyplot as plt
from color_changer import HairColorChanger

if __name__ == "__main__":
    # Test images
    images = [
        ("../model/test_images/1624.jpg", "../model/test_results_2k/1624_prob_mask.png"),
        ("../model/test_images/1626.jpg", "../model/test_results_2k/1626_prob_mask.png")
    ]
    
    # Hair colors to test (10 colors)
    colors = [
        ([255, 0, 0], "Kırmızı"),
        ([0, 255, 0], "Yeşil"),
        ([0, 0, 255], "Mavi"),
        ([255, 255, 0], "Sarı"),
        ([255, 0, 255], "Magenta"),
        ([0, 255, 255], "Cyan"),
        ([128, 0, 128], "Mor"),
        ([255, 192, 203], "Pembe"),
        ([139, 69, 19], "Kahverengi"),
        ([0, 0, 0], "Siyah")
    ]
    
    print("Testing hair color change with 10 colors on 2 images...")

    # Create figure with subplots: 2 rows x 11 columns (original + 10 colors for each image)
    fig, axes = plt.subplots(2, 11, figsize=(22, 8))
    
    for img_idx, (image_path, mask_path) in enumerate(images):
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Original image
        axes[img_idx, 0].set_title(f"Orijinal {img_idx+1}")
        axes[img_idx, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[img_idx, 0].axis("off")
        
        # Apply different colors
        for color_idx, (rgb_color, color_name) in enumerate(colors):
            result = HairColorChanger.change_hair_color(image, mask, rgb_color)
            axes[img_idx, 1 + color_idx].set_title(color_name)
            axes[img_idx, 1 + color_idx].imshow(result)  # Already in RGB format
            axes[img_idx, 1 + color_idx].axis("off")

    plt.tight_layout()
    plt.show()