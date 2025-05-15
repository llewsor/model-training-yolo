from ultralytics import YOLO
import time
import torch
from PIL import Image
import numpy as np

model = YOLO("best.pt")  # Replace with your YOLOv11 checkpoint

# Load and prepare a dummy image
img = Image.open("clay_whole.jpg").resize((640, 640))
img_np = np.array(img)

# Warm-up
for _ in range(5):
    _ = model.predict(img_np, device="cuda", verbose=False)

# Timing
n_runs = 100
start = time.time()
for _ in range(n_runs):
    _ = model.predict(img_np, device="cuda", verbose=False)
end = time.time()

avg_time = (end - start) / n_runs * 1000  # ms
fps = 1000 / avg_time

print(f"Inference Time: {avg_time:.2f} ms/frame")
print(f"Throughput: {fps:.2f} FPS")
