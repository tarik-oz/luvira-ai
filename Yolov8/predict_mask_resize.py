from ultralytics import YOLO
import numpy as np
import cv2

model_path = "C:/Users/Tarik/Desktop/code env/comp_vision/Yolov8_conda/runs/segment/train/weights/best.pt"
image_path = "test_images/1621.jpg"

img = cv2.imread(image_path)
img_height, img_width, _ = img.shape  # Görüntünün yüksekliği ve genişliği

model = YOLO(model_path)

results = model(img)
result = results[0]  # Modelin tahmin sonuçları

# print(result)

names = model.names
# print(names)

final_mask = np.zeros((img_height, img_width), dtype=np.uint8)
predicted_classes = result.boxes.cls.cpu().numpy()
print(predicted_classes)

for j, mask in enumerate(result.masks.data):

    mask = mask.numpy() * 255
    classId = int(predicted_classes[j])

    print("Onject " + str(j) + " detected as " + str(classId) + " - " + names[classId])

    mask = cv2.resize(mask, (img_width, img_height))

    final_mask = np.maximum(final_mask, mask)

    file_name = "output" + str(j) + ".png"

    cv2.imwrite(
        "masks" + file_name,
        mask,
    )

cv2.imwrite(
    "masks/final_mask.png",
    final_mask,
)


scale_percent = 100
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
resized_mask = cv2.resize(final_mask, dim, interpolation=cv2.INTER_AREA)

cv2.imshow("img", resized)
cv2.imshow("final mask", resized_mask)
cv2.waitKey(0)
