import cv2
import matplotlib.pyplot as plt
import os
from color_changer import HairColorChanger
from config import COLORS

if __name__ == "__main__":
    # Test image and mask paths
    img_dir = "test_images"
    results_dir = "test_results"
    image_files = [f for f in os.listdir(img_dir) if f.endswith(".jpeg") or f.endswith(".jpg")]
    image_files.sort()  # 1.jpeg, 2.jpeg, ...

    # Mask file pattern: 1_mask.png, 2_mask.png, ...
    def mask_path_for(img_file):
        base = os.path.splitext(img_file)[0]
        mask_path = os.path.join(img_dir, f"{base}_prob_mask.png")
        if not os.path.exists(mask_path):
            print(f"UYARI: {mask_path} bulunamadı! Lütfen maskeleri uygun şekilde adlandırın.")
        return mask_path

    # CLI for user input
    print("Available colors:")
    for idx, (_, color_name) in enumerate(COLORS):
        print(f"{idx}: {color_name}")
    selected_indices = input("Enter the indices of colors to test (comma-separated): ")
    selected_indices = [int(i) for i in selected_indices.split(",") if i.isdigit()]
    selected_colors = [COLORS[i] for i in selected_indices if i < len(COLORS)]

    print(f"Testing {len(selected_colors)} different hair colors on {len(image_files)} images...")

    # Visualization with Matplotlib
    fig, axes = plt.subplots(len(image_files), len(selected_colors) + 1, figsize=(3*(len(selected_colors)+1), 3*len(image_files)))
    if len(image_files) == 1:
        axes = [axes]  # For a single image

    for img_idx, img_file in enumerate(image_files):
        image_path = os.path.join(img_dir, img_file)
        mask_path = mask_path_for(img_file)
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            print(f"{image_path} or its mask could not be loaded, skipping.")
            continue

        # Original image
        axes[img_idx][0].set_title(f"Original\n{img_file}")
        axes[img_idx][0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[img_idx][0].axis("off")

        # Result for each color
        for color_idx, (rgb_color, color_name) in enumerate(selected_colors):
            result = HairColorChanger.change_hair_color(image, mask, rgb_color)
            axes[img_idx][1 + color_idx].set_title(color_name)
            axes[img_idx][1 + color_idx].imshow(result)
            axes[img_idx][1 + color_idx].axis("off")
            # Save the result to a file
            out_path = os.path.join(results_dir, f"{os.path.splitext(img_file)[0]}_to_{color_name.lower()}.png")
            cv2.imwrite(out_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

    plt.tight_layout()
    plt.show()