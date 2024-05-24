from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("runs/segment/train/weights/best.pt")
    model.predict(source="test.jpg", show=True, save=True, conf=0.5, iou=0.5)
