from ultralytics import YOLO

model = YOLO("yolo_v11_x_run_1.pt")

model.predict(
  # source="videos/center-forward_high_break.mp4",
  source="videos\center-forward_low_break.mp4",
  show=True,
  save=True,
  # conf=0.6,
  line_width=2,
  save_crop=True,
  save_txt=True,
  show_labels=True,
  show_conf = True,
  #classes=[0,1]
)

# model.export(format="onnx")