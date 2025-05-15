from ultralytics import YOLO

model = YOLO("yolo11m.pt")  # Load your YOLO model
results = model.train(data="data_custom.yaml", imgsz=640, autoanchor=True)