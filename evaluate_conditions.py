from ultralytics import YOLO
from pathlib import Path
import cv2
import torch
import numpy as np
from ultralytics.utils.metrics import bbox_iou

def load_labels(label_path):
    with open(label_path, "r") as f:
        lines = f.readlines()
    labels = []
    for line in lines:
        class_id, cx, cy, w, h = map(float, line.strip().split())
        # Convert YOLO (cx, cy, w, h) to (x1, y1, x2, y2)
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        labels.append([x1, y1, x2, y2])
    return torch.tensor(labels)

def compute_average_iou(model, image_dir, label_dir, img_size=640):
    image_paths = sorted(Path(image_dir).glob("*.jpg"))
    total_ious = []
    count = 0

    for img_path in image_paths:
        # Load image
        img = cv2.imread(str(img_path))
        img_resized = cv2.resize(img, (img_size, img_size))

        # Run prediction
        results = model.predict(source=img_resized, imgsz=img_size, conf=0.5, iou=0.5, verbose=False)
        pred_boxes = results[0].boxes.xyxy.cpu()

        # Load ground truth
        label_path = Path(label_dir) / (img_path.stem + ".txt")
        if not label_path.exists():
            continue
        gt_boxes = load_labels(label_path) * img_size  # scale back to pixels

        # Compute pairwise IoU
        if len(gt_boxes) > 0 and len(pred_boxes) > 0:
            ious = bbox_iou(gt_boxes, pred_boxes)
            max_ious = ious.max(dim=1).values  # best match for each GT box
            total_ious.extend(max_ious.tolist())
            count += len(max_ious)

    avg_iou = sum(total_ious) / count if count else 0
    return avg_iou

def run_yolo_evaluations():
    model = YOLO("best.pt")

    # Evaluate and print results
    print("\n🔍 Evaluating motion blur set...")
    results_blur = model.val(data="data_blur.yaml")
    avg_iou_blur = compute_average_iou(model, "C:/Projects/dataset/dataset_test_motion_blur/images", "C:/Projects/dataset/dataset_test_motion_blur/labels")
    print(f"Average IoU (motion blur): {avg_iou_blur:.3f}")

    print("\n🔍 Evaluating brightness set...")
    results_brightness = model.val(data="data_brightness.yaml")
    avg_iou_brightness = compute_average_iou(model, "C:/Projects/dataset/dataset_test_brightness/images", "C:/Projects/dataset/dataset_test_brightness/labels")
    print(f"Average IoU (brightness): {avg_iou_brightness:.3f}")

    print("\n🔍 Evaluating background complexity set...")
    results_env = model.val(data="data_env.yaml")
    avg_iou_env = compute_average_iou(model, "C:/Projects/dataset/dataset_test_background_complexity/images", "C:/Projects/dataset/dataset_test_background_complexity/labels")
    print(f"Average IoU (background complexity): {avg_iou_env:.3f}")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    run_yolo_evaluations()
