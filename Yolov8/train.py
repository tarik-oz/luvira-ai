from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n-seg.pt")
    results = model.train(data="config.yaml", epochs=8, batch=2)
