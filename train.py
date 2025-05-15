from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo11n.pt")  # Load YOLO model
    model.train(
        data="data_custom.yaml",  # Path to dataset configuration file
        imgsz=640,                # Image size
        batch=16,                  # Batch size (adjust based on memory)
        epochs=50,              # Number of epochs
        workers=10,               # Data loader workers
        device=0,                # Use GPU (device 0)
        lr0=0.001,                # Learning rate
        #lrf=0.1,                 # Final LR fraction
        #optimizer="AdamW",       # Use AdamW optimizer
        #amp=True,                # Enable mixed precision
        #augment=True,            # Enable augmentations
        #mosaic=True,             # Mosaic augmentation
        #mixup=0.1                # Mixup probability
        # optimizer="AdamW",  # Efficient optimizer
        # amp=True,           # Enable Mixed Precision for mobile
        # mosaic=False,       # Disable Mosaic (not needed for mobile)
        # flipud=0.0,         # No vertical flip (not useful for clay targets)
        # fliplr=0.3,         # Horizontal flip only sometimes
        # mixup=0.0,          # No mixup (lighter computation)
        # patience=100        # EarlyStopping(patience=100) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.
    )
    
