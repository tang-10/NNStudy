from ultralytics import YOLO


model = YOLO("./ultralytics/weights/yolov8s-cls.pt")

results = model.train(
    data="../datasets/arthrosis/Ulna",
    epochs=300,
    imgsz=224,
    batch=32,
    workers=8,
    device=0,
    patience=20,
    amp=True,
    name="arthrosis_ulna",
    exist_ok=True,
)
