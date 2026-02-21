from ultralytics import YOLO

pt_path = r"C:\YOLO\Delta\runs\detect\train\weights\best.pt"
model = YOLO(pt_path)

model.export(
    format="onnx",
    imgsz=640,
    opset=12,
    simplify=True,   # onnxslim
    half=True,      # set True only if your target runtime supports FP16 well
    dynamic=False,
    nms=True
)