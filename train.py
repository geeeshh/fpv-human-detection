from ultralytics import YOLO

model = YOLO("yolov8n.pt")

if __name__ == "__main__":
    model.train(
        data="data/yolo_dataset/data.yaml",
        epochs=30,
        imgsz=640,
        batch=8,
        device=0,
        workers=2,
        project="runs/detect",
        name="fpv_person",
        patience=10
    )