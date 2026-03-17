from ultralytics import YOLO

model = YOLO("runs/detect/train10/weights/best.pt")

model("test3.jpg", show=True)