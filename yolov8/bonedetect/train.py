from ultralytics import YOLO

# model = YOLO("./ultralytics/weights/yolov8s.pt")
# results = model.train(
#     data="./bonedetect/cfg/bone_data.yaml",
#     epochs=500,
#     imgsz=640,
#     batch=16,
#     workers=8,
#     device=0,
#     patience=30,
#     amp=False,
# )

model = YOLO("./ultralytics/weights/yolov8m.pt")
results = model.train(
    data="./bonedetect/cfg/bone_data.yaml",
    epochs=150,
    imgsz=960,
    batch=-1,
    workers=4,
    device=0,
    patience=30,
    amp=True,
)
