import numpy as np
import matplotlib.pyplot as plt

# Verileri yükle
allImagesNP_loaded = np.load(
    "C:/Users/Tarik/Desktop/unetv3/data/Unet-Hair-Train-Images.npy"
)
maskImagesNP_loaded = np.load(
    "C:/Users/Tarik/Desktop/unetv3/data/Unet-Hair-Train-masks.npy"
)
allValidateImagesNP_loaded = np.load(
    "C:/Users/Tarik/Desktop/unetv3/data/Unet-Hair-Validate-Images.npy"
)
maskValidateImagesNP_loaded = np.load(
    "C:/Users/Tarik/Desktop/unetv3/data/Unet-Hair-Validate-Masks.npy"
)

# Birkaç resmi görselleştir
num_images_to_visualize = 3

# Eğitim verileri için
for i in range(num_images_to_visualize):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(allImagesNP_loaded[i])
    plt.title("Training Image")
    plt.subplot(1, 2, 2)
    plt.imshow(maskImagesNP_loaded[i], cmap="gray")
    plt.title("Training Mask")
    plt.show()

# Doğrulama verileri için
for i in range(num_images_to_visualize):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(allValidateImagesNP_loaded[i])
    plt.title("Validation Image")
    plt.subplot(1, 2, 2)
    plt.imshow(maskValidateImagesNP_loaded[i], cmap="gray")
    plt.title("Validation Mask")
    plt.show()
