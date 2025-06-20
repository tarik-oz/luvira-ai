from ultralytics import YOLO
import os
import cv2
import matplotlib.pyplot as plt

model_path = "model/best.pt"

image_path = "test_images/1624.jpg"
image_name = os.path.basename(image_path).replace(".jpg", "")

img = cv2.imread(image_path)
img_height, img_width, _ = img.shape

model = YOLO(model_path)

results = model(img)

fig, axes = plt.subplots(1, len(results) + 1, figsize=(15, 5))

# Plot original image
axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original Image")
axes[0].axis("off")

for i, result in enumerate(results):
    for j, mask in enumerate(result.masks.data):

        mask = mask.cpu().numpy() * 255

        mask = cv2.resize(mask, (img_width, img_height))

        axes[i + 1].imshow(mask, cmap="gray")
        axes[i + 1].set_title(f"Mask {j+1}")
        axes[i + 1].axis("off")

        cv2.imwrite(f"{image_name}_mask.png", mask)
plt.show()
