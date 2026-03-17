import torch
from ultralytics import YOLO
import multiprocessing

def main():
    path = "data.yaml"
    model = YOLO("yolo26s.pt")
    model.train(
        data=path,
        epochs=150,
        imgsz=640,
        device="cuda"
    ) 

if __name__ == "__main__":
    multiprocessing.freeze_support()  # optional but recommended on Windows
    main()