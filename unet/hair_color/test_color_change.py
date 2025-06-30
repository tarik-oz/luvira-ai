import cv2
import matplotlib.pyplot as plt
from color_changer import HairColorChanger

if __name__ == "__main__":
    image_path = "../test_images/1621.jpg"
    mask_path = "../test_results/1621_prob_mask.png"
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    target_rgb = [0, 0, 255]
    alpha = 0.3  # Color intensity

    # Isolate hair area and blend with new color
    result_isolate_blend = HairColorChanger.change_hair_color(image, mask, target_rgb, alpha=alpha)
    cv2.imwrite("../test_images/1621_hair_colored_isolate_blend.png", result_isolate_blend)

    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1)
    plt.title("Orijinal")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.subplot(1,3,2)
    plt.title("Maske")
    plt.imshow(mask, cmap="gray")
    plt.axis("off")
    plt.subplot(1,3,3)
    plt.title("Saç İzole + Blend")
    plt.imshow(cv2.cvtColor(result_isolate_blend, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show() 