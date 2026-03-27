import torch
from ultralytics import YOLO
import multiprocessing

def main():
    # Verify GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    path = "data.yaml"
    model = YOLO("yolo26m.pt")

    model.train(
        data=path,
        imgsz=416,
        epochs=150,
        batch=64,
        workers=8,
        device=device,

        # Learning rate
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,

        # Warmup
        warmup_epochs=3,

        # Loss weights — boosted for small object localization
        box=7.5,

        # Early stopping & checkpointing
        patience=30,
        save_period=10,

        # Run name
        name="train4",
        exist_ok=True,
    )

if __name__ == "__main__":
    multiprocessing.freeze_support()  # Recommended on Windows
    main()