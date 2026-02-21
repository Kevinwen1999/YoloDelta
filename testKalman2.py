import socket
import threading
import time
import queue
import os
import importlib.util

import cv2
import keyboard
import mss
import numpy as np
import winsound
from pynput.mouse import Button, Listener
from ultralytics import YOLO

# ---------------- CONFIG ----------------
MODEL_PATH = r"C:\YOLO\Delta\runs\detect\train\weights\best.pt"
IMGSZ = 640
CONF = 0.2
DEVICE = "cuda"
INFER_BACKEND = "auto"   # "auto" | "ultralytics" | "onnxruntime"
ONNX_MODEL_PATH = ""     # optional explicit .onnx path; empty = derive from MODEL_PATH
AUTO_EXPORT_ONNX = True
ONNX_NMS_IOU = 0.50
ONNX_TOPK_PRE_NMS = 800
ONNX_OUTPUT_HAS_NMS = True
ONNX_FORCE_TARGET_CLASS_DECODE = True
ONNX_USE_TENSORRT_EP = True
ONNX_TRT_FP16 = True
ONNX_ENABLE_CUDA_GRAPH = False
ONNX_CUDA_DEVICE_ID = 0

SCREEN_W = 2560
SCREEN_H = 1440
CENTER = (SCREEN_W // 2, SCREEN_H // 2)
FPS = 500
SLEEP_TIME = 1 / FPS

BASE_CROP_W = 500
BASE_CROP_H = 500
LOST_CROP_W = 700
LOST_CROP_H = 700
MIN_ACTIVE_CROP_W = 420
MIN_ACTIVE_CROP_H = 420
MAX_ACTIVE_CROP_W = 860
MAX_ACTIVE_CROP_H = 860
CAPTURE_SPEED_TO_PAD = 0.20
CAPTURE_PAD_MAX = 220

TRACKER_TYPE = "MOSSE"  # "MOSSE" is fastest if available, then "KCF", then "CSRT"
YOLO_INTERVAL = 4
YOLO_INTERVAL_FAST = 1
TRACKER_MAX_STREAK = 6
FAST_TARGET_SPEED = 420.0
MAX_DETECTIONS = 10

LEAD_FRAMES = 3.0
SMOOTHING = 0.0       # base smoothing; Kalman path adds adaptive smoothing below
AIM_LEAD_EDGE_FACTOR = 0.7
AIM_EDGE_SPEED_MIN = 10.0
AIM_EDGE_TRAILING_MULT = 3.5
AIM_EDGE_NON_TRAILING_MULT = 0.45
CATCHUP_ENABLE = True
CATCHUP_MIN_SPEED = 140.0
CATCHUP_MIN_PROGRESS_PX = 1.0
CATCHUP_GAIN_UP = 0.10
CATCHUP_GAIN_DOWN = 0.08
CATCHUP_GAIN_MAX = 0.75
CATCHUP_CENTER_SCALE = 0.75
CATCHUP_MAX_EXTRA_PX = 180.0
HEAD_Y_BIAS = 0.0    # aim slightly above center for head class
USE_KALMAN_DEFAULT = True
TARGET_LOCK_MAX_JUMP = 260  # px; avoid hopping to far detections when already locked
KALMAN_MAX_SPEED_PX_S = 1800.0
KALMAN_MAX_LEAD_PX = 140.0
KALMAN_MAX_DT = 0.06
KALMAN_MIN_DT = 1.0 / 240.0
MIN_MEAS_DT = 1.0 / 120.0
KALMAN_POSITION_BLEND = 0.10  # lower blend for less perceived lag
KALMAN_LEAD_MIN_SPEED = 120.0 # px/s; ignore tiny velocity noise
MEAS_VELOCITY_BLEND = 0.32
MAX_ACCEL_PX_S2 = 14000.0
LEAD_SMOOTH_ALPHA_SLOW = 0.40
LEAD_SMOOTH_ALPHA_FAST = 0.72
KALMAN_EXTRA_SMOOTH_SLOW = 0.10
KALMAN_EXTRA_SMOOTH_FAST = 0.04
ADAPTIVE_SPEED_MIN = 120.0
ADAPTIVE_SPEED_MAX = 750.0
VELOCITY_STOP_ENTER_THRESHOLD = 90.0
VELOCITY_STOP_EXIT_THRESHOLD = 145.0
POSITION_DEADZONE_SLOW = 4.0
POSITION_DEADZONE_FAST = 2.0
EXTRA_PIPELINE_LAG_S = 0.015
NON_STATIONARY_EXTRA_LEAD_S = 0.02
PREDICTIVE_CAPTURE_LEAD_S = 0.03
MAX_FRAME_AGE_S = 0.05
SPEED_BOOST_THRESHOLD = 250.0
SPEED_BOOST_GAIN = 1.5
AIM_DEADZONE_SLOW = 1.0
AIM_DEADZONE_FAST = 0.0
DEBUG_LOG = True

# Fast-aim tuning (when enabled) - lower smoothing for snappier response
FAST_AIMING_DEFAULT = False
FAST_LEAD_SMOOTH_ALPHA_SLOW = 0.20
FAST_LEAD_SMOOTH_ALPHA_FAST = 0.50
FAST_KALMAN_EXTRA_SMOOTH_SLOW = 0.01
FAST_KALMAN_EXTRA_SMOOTH_FAST = 0.00

PIPELINE_FRAME_QUEUE = 3
PIPELINE_CMD_QUEUE = 1
DETECTION_MIN_CONF = 0.20
TRACKER_REINIT_CONF = 0.45
ASSOC_PREDICT_DT = 0.02
ASSOC_SPEED_JUMP_GAIN = 0.05
ASSOC_MAX_JUMP_PAD = 220.0
MOTION_BACKTRACK_TOL = 1400.0
TRACKER_VELOCITY_BLEND = 0.9
TRACKER_VELOCITY_REF_BLEND = 0.8
TRACKER_VELOCITY_MIN_DT = 1.0 / 300.0
TRACKER_MAX_SPEED_PX_S = 2600.0

PID_ENABLE = True
PID_KP_X = 0.36
PID_KI_X = 0.01
PID_KD_X = 0.04
PID_KP_Y = 0.34
PID_KI_Y = 0.01
PID_KD_Y = 0.035
PID_INTEGRAL_LIMIT = 700.0
PID_DERIVATIVE_ALPHA = 0.22
PID_DERIVATIVE_LIMIT = 900.0
PID_D_TERM_LIMIT = 55.0
PID_OUTPUT_MAX = 180.0
PID_MICRO_ERROR_PX = 2.2
PID_SOFT_ERROR_PX = 7.0
PID_SOFT_ZONE_GAIN = 0.70

OUTPUT_MAX_STEP_X = 220
OUTPUT_MAX_STEP_Y = 220
OUTPUT_MAX_DELTA_X = 85
OUTPUT_MAX_DELTA_Y = 85
OUTPUT_MICRO_CMD_PX = 1

MISS_HYSTERESIS_FRAMES = 3
TRACKER_MEASUREMENT_CONF = 0.68

KALMAN_PROCESS_NOISE_BASE = 0.08
KALMAN_PROCESS_NOISE_SPEED_GAIN = 0.55
KALMAN_PROCESS_NOISE_ACCEL_GAIN = 0.35
KALMAN_MEAS_NOISE_BASE = 0.50
KALMAN_MEAS_NOISE_LOW_CONF_GAIN = 2.2
KALMAN_MEAS_NOISE_MIN_CONF = 0.20

BEZIER_CURVE_ENABLED = False
BEZIER_CURVE_STRENGTH = 0.16
BEZIER_CURVE_MAX_OFFSET_PX = 55.0
BEZIER_RANDOM_MIN = 0.85
BEZIER_RANDOM_MAX = 1.15
BEZIER_ERROR_REF_PX = 180.0

LEAD_NEAR_TARGET_INNER_PX = 16.0
LEAD_NEAR_TARGET_OUTER_PX = 70.0
LEAD_CONFIDENCE_MIN_SCALE = 0.45
MEAS_JUMP_REJECT_BASE_PX = 95.0
MEAS_JUMP_REJECT_SPEED_GAIN = 0.04
MEAS_JUMP_REJECT_CONF = 0.72

MOUSE_HOST = "127.0.0.1"
MOUSE_PORT = 8080
MOUSE_CONNECT_TIMEOUT_S = 0.002
MOUSE_SEND_TIMEOUT_S = 0.002

WARMUP_CLASS = 0
TARGET_TIMEOUT_S = 0.15
# ----------------------------------------


_DLL_SEARCH_PATH_HANDLES = []
_DLL_SEARCH_PATH_DIRS = set()


def ensure_tensorrt_dll_path():
    spec = importlib.util.find_spec("tensorrt_libs")
    if not spec or not spec.submodule_search_locations:
        return False

    trt_dir = os.path.abspath(spec.submodule_search_locations[0])
    if not os.path.isdir(trt_dir):
        return False

    normalized = os.path.normcase(os.path.normpath(trt_dir))
    if normalized not in _DLL_SEARCH_PATH_DIRS:
        path_value = os.environ.get("PATH", "")
        path_parts = [p for p in path_value.split(os.pathsep) if p]
        normalized_parts = {os.path.normcase(os.path.normpath(p)) for p in path_parts}
        if normalized not in normalized_parts:
            os.environ["PATH"] = f"{trt_dir}{os.pathsep}{path_value}" if path_value else trt_dir
            print(f"[INFO] Added TensorRT DLL path: {trt_dir}")

        add_dll_directory = getattr(os, "add_dll_directory", None)
        if callable(add_dll_directory):
            try:
                _DLL_SEARCH_PATH_HANDLES.append(add_dll_directory(trt_dir))
            except OSError:
                pass
        _DLL_SEARCH_PATH_DIRS.add(normalized)
    return True


def clamp(value, low, high):
    return max(low, min(high, value))


def build_capture(center_x, center_y, width, height):
    left = int(clamp(center_x - width // 2, 0, SCREEN_W - width))
    top = int(clamp(center_y - height // 2, 0, SCREEN_H - height))
    return {"top": top, "left": left, "width": int(width), "height": int(height)}


def put_latest(q, item):
    try:
        q.put_nowait(item)
        return
    except queue.Full:
        pass

    try:
        q.get_nowait()
    except queue.Empty:
        pass

    try:
        q.put_nowait(item)
    except queue.Full:
        pass


def get_latest(q, timeout_s):
    item = q.get(timeout=timeout_s)
    while True:
        try:
            item = q.get_nowait()
        except queue.Empty:
            return item


def drain_queue(q):
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            return


def nms_xyxy(boxes, scores, iou_thresh=0.5, max_det=100):
    if boxes.size == 0:
        return np.empty((0,), dtype=np.int32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = np.argsort(scores)[::-1]
    keep = []

    while order.size > 0 and len(keep) < max_det:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[rest] - inter + 1e-6
        iou = inter / union
        order = rest[iou <= iou_thresh]

    return np.array(keep, dtype=np.int32)


class PIDController:
    def __init__(
        self,
        kp,
        ki,
        kd,
        integral_limit=PID_INTEGRAL_LIMIT,
        derivative_alpha=PID_DERIVATIVE_ALPHA,
        derivative_limit=PID_DERIVATIVE_LIMIT,
        d_term_limit=PID_D_TERM_LIMIT,
        output_limit=PID_OUTPUT_MAX,
    ):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.integral_limit = float(integral_limit)
        self.derivative_alpha = clamp(float(derivative_alpha), 0.0, 1.0)
        self.derivative_limit = float(derivative_limit)
        self.d_term_limit = float(d_term_limit)
        self.output_limit = float(output_limit) if output_limit is not None else None
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.previous_error = 0.0
        self.derivative = 0.0
        self.has_prev = False

    def _calculate_adjusted_kp(self):
        if self.kp <= 0.5:
            return self.kp
        return 0.5 + (self.kp - 0.5) * 3.0

    def update(self, error, dt):
        dt = max(1e-4, float(dt))
        error = float(error)
        self.integral += error * dt
        self.integral = clamp(self.integral, -self.integral_limit, self.integral_limit)

        if self.has_prev:
            derivative_raw = (error - self.previous_error) / dt
        else:
            derivative_raw = 0.0
            self.has_prev = True
        derivative_raw = clamp(derivative_raw, -self.derivative_limit, self.derivative_limit)
        self.derivative = self.derivative + (derivative_raw - self.derivative) * self.derivative_alpha

        adjusted_kp = self._calculate_adjusted_kp()
        d_term = clamp(self.kd * self.derivative, -self.d_term_limit, self.d_term_limit)
        output = (adjusted_kp * error) + (self.ki * self.integral) + d_term
        self.previous_error = error

        if self.output_limit is not None:
            output = clamp(output, -self.output_limit, self.output_limit)
        return output


def apply_bezier_offset(error_x, error_y, curve_sign):
    err_mag = float(np.hypot(error_x, error_y))
    if err_mag < 1e-6:
        return error_x, error_y

    error_ratio = clamp(err_mag / BEZIER_ERROR_REF_PX, 0.0, 1.0)
    rand_scale = float(np.random.uniform(BEZIER_RANDOM_MIN, BEZIER_RANDOM_MAX))
    offset_mag = err_mag * BEZIER_CURVE_STRENGTH * error_ratio * rand_scale
    offset_mag = clamp(offset_mag, 0.0, BEZIER_CURVE_MAX_OFFSET_PX)

    inv_mag = 1.0 / err_mag
    perp_x = (-error_y) * inv_mag
    perp_y = error_x * inv_mag
    offset_x = perp_x * offset_mag * curve_sign
    offset_y = perp_y * offset_mag * curve_sign
    return error_x + offset_x, error_y + offset_y


class UltralyticsBackend:
    def __init__(self, model_path, imgsz, conf, device, max_det):
        self.name = "ultralytics"
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.conf = conf
        self.device = device
        self.max_det = max_det
        self.use_half = device == "cuda"

        try:
            self.model.fuse()
        except Exception:
            pass

    def warmup(self):
        warmup_frame = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        self.predict(warmup_frame, WARMUP_CLASS)

    def predict(self, frame, target_cls):
        try:
            results = self.model.predict(
                source=frame,
                imgsz=self.imgsz,
                conf=self.conf,
                device=self.device,
                classes=[target_cls],
                verbose=False,
                half=self.use_half,
                max_det=self.max_det,
            )
        except RuntimeError as e:
            if self.use_half and "same dtype" in str(e):
                print("[WARN] FP16 failed due to dtype mismatch; falling back to FP32.")
                self.use_half = False
                results = self.model.predict(
                    source=frame,
                    imgsz=self.imgsz,
                    conf=self.conf,
                    device=self.device,
                    classes=[target_cls],
                    verbose=False,
                    half=False,
                    max_det=self.max_det,
                )
            else:
                raise

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return (
                np.empty((0, 4), dtype=np.int32),
                np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=np.float32),
            )

        xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.int32, copy=False)
        cls_ids = boxes.cls.detach().cpu().numpy().astype(np.int32, copy=False)
        confs = boxes.conf.detach().cpu().numpy().astype(np.float32, copy=False)
        return xyxy, cls_ids, confs


class OnnxRuntimeBackend:
    def __init__(self, model_path, imgsz, conf, device, max_det):
        import onnxruntime as ort

        self.name = "onnxruntime"
        self.model_path = model_path
        self.imgsz = imgsz
        self.conf = conf
        self.max_det = max_det
        self.device = device
        self.ort = ort
        self._trt_enabled = bool(ONNX_USE_TENSORRT_EP and device == "cuda")
        if self._trt_enabled:
            ensure_tensorrt_dll_path()
        self.available = ort.get_available_providers()
        self._use_cuda_graph = bool(ONNX_ENABLE_CUDA_GRAPH and device == "cuda")
        self._init_session(use_cuda_graph=self._use_cuda_graph)

    def _build_providers(self, use_cuda_graph):
        providers = []
        if self.device == "cuda":
            if self._trt_enabled and "TensorrtExecutionProvider" in self.available:
                trt_cache = os.path.join(os.path.dirname(self.model_path), "trt_cache")
                try:
                    os.makedirs(trt_cache, exist_ok=True)
                except OSError:
                    trt_cache = "."
                providers.append(
                    (
                        "TensorrtExecutionProvider",
                        {
                            "device_id": ONNX_CUDA_DEVICE_ID,
                            "trt_fp16_enable": ONNX_TRT_FP16,
                            "trt_engine_cache_enable": True,
                            "trt_engine_cache_path": trt_cache,
                        },
                    )
                )
            if "CUDAExecutionProvider" in self.available:
                providers.append(
                    (
                        "CUDAExecutionProvider",
                        {
                            "device_id": ONNX_CUDA_DEVICE_ID,
                            "arena_extend_strategy": "kNextPowerOfTwo",
                            "cudnn_conv_algo_search": "HEURISTIC",
                            "do_copy_in_default_stream": True,
                            "cudnn_conv_use_max_workspace": True,
                            "enable_cuda_graph": bool(use_cuda_graph),
                        },
                    )
                )
        providers.append("CPUExecutionProvider")
        return providers

    def _init_session(self, use_cuda_graph):
        so = self.ort.SessionOptions()
        so.graph_optimization_level = self.ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.enable_mem_pattern = True
        so.enable_cpu_mem_arena = True

        providers = self._build_providers(use_cuda_graph)
        self.session = self.ort.InferenceSession(
            self.model_path,
            sess_options=so,
            providers=providers,
        )
        self.providers = self.session.get_providers()
        if (
            self.device == "cuda"
            and self._trt_enabled
            and "CUDAExecutionProvider" in self.available
            and self.providers == ["CPUExecutionProvider"]
        ):
            print("[WARN] TensorRT EP init failed; retrying with CUDA EP only.")
            self._trt_enabled = False
            providers = self._build_providers(use_cuda_graph)
            self.session = self.ort.InferenceSession(
                self.model_path,
                sess_options=so,
                providers=providers,
            )
            self.providers = self.session.get_providers()
        self._cuda_ep_in_use = "CUDAExecutionProvider" in self.providers
        self.compute_provider = self._select_compute_provider()
        self.name = f"onnxruntime[{','.join(self.providers)}]"
        self.input_name = self.session.get_inputs()[0].name
        self.input_type = self.session.get_inputs()[0].type
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.input_dtype = np.float16 if "float16" in self.input_type else np.float32
        self._resize_bgr = np.empty((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        self._rgb = np.empty_like(self._resize_bgr)
        self._input = np.empty((1, 3, self.imgsz, self.imgsz), dtype=self.input_dtype)

    def _select_compute_provider(self):
        if "TensorrtExecutionProvider" in self.providers:
            return "tensorrt"
        if "CUDAExecutionProvider" in self.providers:
            return "cuda"
        return "cpu"

    def warmup(self):
        warmup_frame = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        self.predict(warmup_frame, WARMUP_CLASS)

    def _run_raw(self, frame):
        h, w = frame.shape[:2]
        cv2.resize(
            frame,
            (self.imgsz, self.imgsz),
            dst=self._resize_bgr,
            interpolation=cv2.INTER_LINEAR,
        )
        cv2.cvtColor(self._resize_bgr, cv2.COLOR_BGR2RGB, dst=self._rgb)
        np.multiply(
            self._rgb.transpose(2, 0, 1),
            1.0 / 255.0,
            out=self._input[0],
            casting="unsafe",
        )
        try:
            outputs = self.session.run(self.output_names, {self.input_name: self._input})
        except Exception as e:
            msg = str(e)
            cuda_graph_fail = ("cudaGraphLaunch" in msg) or ("illegal memory access" in msg)
            if self._use_cuda_graph and self._cuda_ep_in_use and cuda_graph_fail:
                print("[WARN] CUDA graph failed on this model/input; recreating ONNX session with cuda graph OFF.")
                self._use_cuda_graph = False
                self._init_session(use_cuda_graph=False)
                outputs = self.session.run(self.output_names, {self.input_name: self._input})
            else:
                raise
        return outputs, w, h

    def _decode_nms_output(self, pred, frame_w, frame_h, target_cls):
        if pred.ndim != 2 or pred.shape[1] < 6:
            return None
        confs = pred[:, 4].astype(np.float32, copy=False)
        cls_ids = pred[:, 5].astype(np.int32, copy=False)
        mask = (cls_ids == int(target_cls)) & (confs >= self.conf)
        if not np.any(mask):
            return (
                np.empty((0, 4), dtype=np.int32),
                np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=np.float32),
            )

        xyxy = pred[mask, :4].astype(np.float32, copy=False)
        confs = confs[mask]
        cls_ids = cls_ids[mask]

        coord_max = float(np.max(xyxy)) if xyxy.size else 0.0
        if coord_max <= (self.imgsz + 8):
            sx = frame_w / float(self.imgsz)
            sy = frame_h / float(self.imgsz)
            xyxy[:, [0, 2]] *= sx
            xyxy[:, [1, 3]] *= sy

        xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, frame_w - 1)
        xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, frame_h - 1)

        if ONNX_OUTPUT_HAS_NMS:
            if confs.shape[0] > self.max_det:
                sel = np.argpartition(confs, -self.max_det)[-self.max_det:]
                order = np.argsort(confs[sel])[::-1]
                sel = sel[order]
                xyxy = xyxy[sel]
                cls_ids = cls_ids[sel]
                confs = confs[sel]
            return xyxy.astype(np.int32), cls_ids, confs

        keep = nms_xyxy(xyxy, confs, iou_thresh=ONNX_NMS_IOU, max_det=self.max_det)
        if keep.size == 0:
            return (
                np.empty((0, 4), dtype=np.int32),
                np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=np.float32),
            )
        return xyxy[keep].astype(np.int32), cls_ids[keep], confs[keep]

    def _decode_raw_output(self, pred, frame_w, frame_h, target_cls):
        if pred.ndim != 2:
            return (
                np.empty((0, 4), dtype=np.int32),
                np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=np.float32),
            )
        if pred.shape[0] < pred.shape[1] and pred.shape[0] <= 128:
            pred = pred.T
        if pred.shape[1] < 6:
            return (
                np.empty((0, 4), dtype=np.int32),
                np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=np.float32),
            )

        boxes_xywh = pred[:, :4].astype(np.float32, copy=False)
        cls_scores = pred[:, 4:].astype(np.float32, copy=False)

        if ONNX_FORCE_TARGET_CLASS_DECODE and 0 <= int(target_cls) < cls_scores.shape[1]:
            confs = cls_scores[:, int(target_cls)]
            mask = confs >= self.conf
            if not np.any(mask):
                return (
                    np.empty((0, 4), dtype=np.int32),
                    np.empty((0,), dtype=np.int32),
                    np.empty((0,), dtype=np.float32),
                )
            boxes_xywh = boxes_xywh[mask]
            confs = confs[mask]
            cls_ids = np.full(confs.shape, int(target_cls), dtype=np.int32)
        else:
            cls_ids = np.argmax(cls_scores, axis=1).astype(np.int32, copy=False)
            confs = cls_scores[np.arange(cls_scores.shape[0]), cls_ids]
            mask = (cls_ids == int(target_cls)) & (confs >= self.conf)
            if not np.any(mask):
                return (
                    np.empty((0, 4), dtype=np.int32),
                    np.empty((0,), dtype=np.int32),
                    np.empty((0,), dtype=np.float32),
                )

            boxes_xywh = boxes_xywh[mask]
            confs = confs[mask]
            cls_ids = cls_ids[mask]

        if confs.shape[0] > ONNX_TOPK_PRE_NMS:
            sel = np.argpartition(confs, -ONNX_TOPK_PRE_NMS)[-ONNX_TOPK_PRE_NMS:]
            boxes_xywh = boxes_xywh[sel]
            confs = confs[sel]
            cls_ids = cls_ids[sel]

        coord_max = float(np.max(boxes_xywh)) if boxes_xywh.size else 0.0
        if coord_max <= 2.0:
            boxes_xywh *= float(self.imgsz)

        cx = boxes_xywh[:, 0]
        cy = boxes_xywh[:, 1]
        bw = np.maximum(1.0, boxes_xywh[:, 2])
        bh = np.maximum(1.0, boxes_xywh[:, 3])
        x1 = cx - (bw * 0.5)
        y1 = cy - (bh * 0.5)
        x2 = cx + (bw * 0.5)
        y2 = cy + (bh * 0.5)
        xyxy = np.stack([x1, y1, x2, y2], axis=1)

        sx = frame_w / float(self.imgsz)
        sy = frame_h / float(self.imgsz)
        xyxy[:, [0, 2]] *= sx
        xyxy[:, [1, 3]] *= sy
        xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, frame_w - 1)
        xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, frame_h - 1)

        keep = nms_xyxy(xyxy, confs, iou_thresh=ONNX_NMS_IOU, max_det=self.max_det)
        if keep.size == 0:
            return (
                np.empty((0, 4), dtype=np.int32),
                np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=np.float32),
            )
        return xyxy[keep].astype(np.int32), cls_ids[keep], confs[keep]

    def predict(self, frame, target_cls):
        outputs, frame_w, frame_h = self._run_raw(frame)
        if not outputs:
            return (
                np.empty((0, 4), dtype=np.int32),
                np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=np.float32),
            )

        if ONNX_OUTPUT_HAS_NMS:
            arr0 = np.asarray(outputs[0])
            if arr0.ndim == 3 and arr0.shape[0] == 1 and arr0.shape[-1] >= 6:
                decoded = self._decode_nms_output(arr0[0], frame_w, frame_h, target_cls)
                if decoded is not None:
                    return decoded

        for out in outputs:
            arr = np.asarray(out)
            if arr.ndim == 3 and arr.shape[0] == 1 and 6 <= arr.shape[-1] <= 8:
                decoded = self._decode_nms_output(arr[0], frame_w, frame_h, target_cls)
                if decoded is not None:
                    return decoded

        raw = np.asarray(outputs[0])
        if raw.ndim == 3 and raw.shape[0] == 1:
            raw = raw[0]
        return self._decode_raw_output(raw, frame_w, frame_h, target_cls)


def resolve_onnx_path(model_path):
    if ONNX_MODEL_PATH:
        return ONNX_MODEL_PATH
    root, _ = os.path.splitext(model_path)
    return f"{root}.onnx"


def build_inference_backend():
    backend_pref = INFER_BACKEND.lower()
    onnx_path = resolve_onnx_path(MODEL_PATH)

    if backend_pref in ("auto", "onnxruntime"):
        if not os.path.exists(onnx_path) and AUTO_EXPORT_ONNX and MODEL_PATH.lower().endswith(".pt"):
            try:
                print(f"[INFO] Exporting ONNX to {onnx_path} ...")
                export_model = YOLO(MODEL_PATH)
                export_result = export_model.export(
                    format="onnx",
                    imgsz=IMGSZ,
                    half=(DEVICE == "cuda"),
                    dynamic=False,
                    simplify=False,
                    nms=True,
                    opset=12,
                )
                if isinstance(export_result, str):
                    onnx_path = export_result
            except Exception as e:
                print(f"[WARN] ONNX export failed: {e}")

        if os.path.exists(onnx_path):
            try:
                return OnnxRuntimeBackend(
                    model_path=onnx_path,
                    imgsz=IMGSZ,
                    conf=CONF,
                    device=DEVICE,
                    max_det=MAX_DETECTIONS,
                )
            except Exception as e:
                if backend_pref == "onnxruntime":
                    raise
                print(f"[WARN] ONNX backend unavailable ({e}); using Ultralytics backend.")
        elif backend_pref == "onnxruntime":
            raise FileNotFoundError(
                f"ONNX model not found at {onnx_path}. Set ONNX_MODEL_PATH or enable AUTO_EXPORT_ONNX."
            )

    return UltralyticsBackend(
        model_path=MODEL_PATH,
        imgsz=IMGSZ,
        conf=CONF,
        device=DEVICE,
        max_det=MAX_DETECTIONS,
    )


def create_tracker():
    tracker_type = TRACKER_TYPE.upper()
    if tracker_type == "MOSSE":
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerMOSSE_create"):
            return cv2.legacy.TrackerMOSSE_create()
    if tracker_type == "KCF":
        if hasattr(cv2, "TrackerKCF_create"):
            return cv2.TrackerKCF_create()
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
            return cv2.legacy.TrackerKCF_create()
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    return None


class MouseMoveClient:
    def __init__(self, host=MOUSE_HOST, port=MOUSE_PORT):
        self.host = host
        self.port = port
        self.sock = None
        self.lock = threading.Lock()
        self.last_connect_fail_ts = 0.0

    def close(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None

    def _ensure_socket(self):
        if self.sock is not None:
            return True

        now = time.time()
        if now - self.last_connect_fail_ts < 0.1:
            return False

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(MOUSE_CONNECT_TIMEOUT_S)
            sock.connect((self.host, self.port))
            sock.settimeout(MOUSE_SEND_TIMEOUT_S)
            self.sock = sock
            return True
        except OSError:
            self.last_connect_fail_ts = now
            self.close()
            return False

    def send(self, x, y):
        message = f"{x} {y}\n".encode()
        with self.lock:
            if not self._ensure_socket():
                return False
            try:
                self.sock.sendall(message)
                return True
            except OSError:
                self.close()
                return False


def main():
    print("Initialize Start")
    backend = build_inference_backend()
    mouse_client = MouseMoveClient()

    try:
        backend.warmup()
    except Exception as e:
        print(f"[WARN] backend warmup failed: {e}")

    if isinstance(backend, OnnxRuntimeBackend):
        print(f"[INFO] Inference provider: {backend.compute_provider.upper()} ({','.join(backend.providers)})")
    elif hasattr(backend, "device"):
        print(f"[INFO] Inference provider: {str(backend.device).upper()} (Ultralytics)")
    else:
        print("[INFO] Inference provider: UNKNOWN")

    print(f"Finished initializing model (backend={backend.name})")

    mode = 0      # 0=off, 1=T only, 2=CT only, 3=T+CT heads
    aimmode = 0   # 0=head, 1=body
    use_kalman = USE_KALMAN_DEFAULT
    last_toggle = 0.0
    last_kalman_toggle = 0.0
    running = True

    mouse_button_pressed_x1 = False
    mouse_button_pressed_x2 = False

    shared_lock = threading.Lock()
    state = {
        "mode": mode,
        "aimmode": aimmode,
        "running": running,
        "last_target_full": CENTER,
        "capture_focus_full": CENTER,
        "target_speed": 0.0,
        "target_found": False,
        "aim_dx": 0,
        "aim_dy": 0,
        "target_ts": 0.0,
        "aim_seq": 0,
        "target_cls": -1,
        "use_kalman": use_kalman,
        "fast_aim": FAST_AIMING_DEFAULT,
    }

    frame_queue = queue.Queue(maxsize=PIPELINE_FRAME_QUEUE)
    control_queue = queue.Queue(maxsize=PIPELINE_CMD_QUEUE)

    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32
    )
    kalman.transitionMatrix = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=np.float32,
    )
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * KALMAN_PROCESS_NOISE_BASE
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * KALMAN_MEAS_NOISE_BASE
    kalman.errorCovPost = np.eye(4, dtype=np.float32)
    kalman_initialized = False
    last_kalman_ts = time.time()
    last_fast_toggle = 0.0

    def capture_loop():
        nonlocal running
        with mss.mss() as sct:
            while running:
                with shared_lock:
                    target_found = state["target_found"]
                    tx, ty = state["capture_focus_full"]
                    target_speed = state["target_speed"]

                if target_found:
                    pad = int(min(CAPTURE_PAD_MAX, target_speed * CAPTURE_SPEED_TO_PAD))
                    crop_w = int(clamp(BASE_CROP_W + pad, MIN_ACTIVE_CROP_W, MAX_ACTIVE_CROP_W))
                    crop_h = int(clamp(BASE_CROP_H + pad, MIN_ACTIVE_CROP_H, MAX_ACTIVE_CROP_H))
                    cap = build_capture(tx, ty, crop_w, crop_h)
                else:
                    cap = build_capture(CENTER[0], CENTER[1], LOST_CROP_W, LOST_CROP_H)

                shot = sct.grab(cap)
                frame_bgra = np.frombuffer(shot.bgra, dtype=np.uint8).reshape(
                    shot.height, shot.width, 4
                )
                frame_bgr = frame_bgra[:, :, :3].copy()
                now = time.time()

                put_latest(
                    frame_queue,
                    {"frame": frame_bgr, "capture": cap, "ts": now},
                )

                time.sleep(max(0.0005, SLEEP_TIME * 0.5))

    def inference_loop():
        nonlocal running, kalman_initialized, last_kalman_ts
        tracker = None
        tracker_active = False
        tracker_streak = 0
        inf_count = 0
        smooth_x = float(CENTER[0])
        smooth_y = float(CENTER[1])
        filtered_vx = 0.0
        filtered_vy = 0.0
        lead_filtered_x = float(CENTER[0])
        lead_filtered_y = float(CENTER[1])
        last_out_dx = 0
        last_out_dy = 0
        last_meas_ts = 0.0
        last_meas_x = float(CENTER[0])
        last_meas_y = float(CENTER[1])
        last_speed = 0.0
        catchup_gain = 0.0
        catchup_rel_prev = 0.0
        catchup_has_prev = False
        tracker_vx = 0.0
        tracker_vy = 0.0
        tracker_v_valid = False
        tracker_speed = 0.0
        last_tracker_ts = 0.0
        last_tracker_cx_full = float(CENTER[0])
        last_tracker_cy_full = float(CENTER[1])
        last_backend_err_ts = 0.0
        miss_count = 0
        bezier_curve_sign = 1.0
        last_pid_ts = time.time()
        pid_x = PIDController(PID_KP_X, PID_KI_X, PID_KD_X)
        pid_y = PIDController(PID_KP_Y, PID_KI_Y, PID_KD_Y)
        last_accel_mag = 0.0
        velocity_stop_latched = True
        kalman_q_eye = np.eye(4, dtype=np.float32)
        kalman_r_eye = np.eye(2, dtype=np.float32)
        measurement = np.zeros((2, 1), dtype=np.float32)

        while running:
            with shared_lock:
                local_mode = state["mode"]
                local_aimmode = state["aimmode"]
                local_use_kalman = state["use_kalman"]
                local_fast_aim = state.get("fast_aim", False)
                prev_target_found = state["target_found"]
                prev_tx, prev_ty = state["last_target_full"]
                prev_target_cls = state["target_cls"]

            if local_mode == 0:
                with shared_lock:
                    state["target_found"] = False
                    state["target_speed"] = 0.0
                    state["capture_focus_full"] = CENTER
                    state["target_cls"] = -1
                kalman_initialized = False
                tracker_active = False
                tracker_streak = 0
                filtered_vx = 0.0
                filtered_vy = 0.0
                catchup_gain = 0.0
                catchup_rel_prev = 0.0
                catchup_has_prev = False
                tracker_vx = 0.0
                tracker_vy = 0.0
                tracker_v_valid = False
                tracker_speed = 0.0
                last_tracker_ts = 0.0
                lead_filtered_x = float(CENTER[0])
                lead_filtered_y = float(CENTER[1])
                smooth_x = float(CENTER[0])
                smooth_y = float(CENTER[1])
                miss_count = 0
                bezier_curve_sign = 1.0
                last_accel_mag = 0.0
                last_out_dx = 0
                last_out_dy = 0
                last_pid_ts = time.time()
                pid_x.reset()
                pid_y.reset()
                velocity_stop_latched = True
                time.sleep(0.002)
                continue

            try:
                packet = get_latest(frame_queue, 0.01)
            except queue.Empty:
                time.sleep(0.001)
                continue
            frame = packet["frame"]
            cap = packet["capture"]
            frame_ts = packet["ts"]

            frame_age = max(0.0, time.time() - frame_ts)
            if frame_age > MAX_FRAME_AGE_S:
                time.sleep(0.0005)
                continue

            # Match class mapping from testKalman.py: classes are directly mapped by aimmode (0=head, 1=body)
            target_cls = int(local_aimmode)
            target_classes = np.array([target_cls], dtype=np.int32)
            selected_cls = int(prev_target_cls) if int(prev_target_cls) in target_classes else target_cls

            bbox = None
            used_tracker = False
            best_conf = 0.0

            effective_interval = YOLO_INTERVAL_FAST if last_speed >= FAST_TARGET_SPEED else YOLO_INTERVAL
            use_tracker_step = (
                tracker_active
                and tracker is not None
                and tracker_streak < TRACKER_MAX_STREAK
                and (inf_count % effective_interval != 0)
            )

            if use_tracker_step:
                ok, tracked = tracker.update(frame)
                if ok:
                    x, y, w, h = tracked
                    w_i = int(max(1, w))
                    h_i = int(max(1, h))
                    cand_bbox = (int(x), int(y), int(x + w_i), int(y + h_i))
                    cand_cx_full = cap["left"] + (cand_bbox[0] + cand_bbox[2]) // 2
                    cand_cy_full = cap["top"] + (cand_bbox[1] + cand_bbox[3]) // 2
                    tracker_consistent = True

                    if prev_target_found:
                        jump_pad = min(ASSOC_MAX_JUMP_PAD, abs(last_speed) * ASSOC_SPEED_JUMP_GAIN)
                        max_jump_sq = (TARGET_LOCK_MAX_JUMP + jump_pad) ** 2
                        dx_track = cand_cx_full - prev_tx
                        dy_track = cand_cy_full - prev_ty
                        dist_track = (dx_track * dx_track) + (dy_track * dy_track)
                        tracker_consistent = dist_track <= max_jump_sq

                    if tracker_consistent:
                        now_track = time.time()
                        dt_track = (
                            max(TRACKER_VELOCITY_MIN_DT, now_track - last_tracker_ts)
                            if last_tracker_ts > 0.0
                            else 0.0
                        )
                        if dt_track > 0.0:
                            tvx = (cand_cx_full - last_tracker_cx_full) / dt_track
                            tvy = (cand_cy_full - last_tracker_cy_full) / dt_track
                            tspeed = float(np.hypot(tvx, tvy))
                            if tspeed > TRACKER_MAX_SPEED_PX_S and tspeed > 1e-6:
                                tscale = TRACKER_MAX_SPEED_PX_S / tspeed
                                tvx *= tscale
                                tvy *= tscale
                                tspeed = TRACKER_MAX_SPEED_PX_S
                            tracker_vx = tvx
                            tracker_vy = tvy
                            tracker_speed = tspeed
                            tracker_v_valid = True
                        else:
                            tracker_v_valid = False
                            tracker_speed = 0.0
                        last_tracker_ts = now_track
                        last_tracker_cx_full = float(cand_cx_full)
                        last_tracker_cy_full = float(cand_cy_full)

                        bbox = cand_bbox
                        best_conf = TRACKER_MEASUREMENT_CONF
                        used_tracker = True
                        tracker_streak += 1
                    else:
                        tracker_active = False
                        tracker_streak = 0
                        tracker_v_valid = False
                        tracker_speed = 0.0
                else:
                    tracker_active = False
                    tracker_streak = 0
                    tracker_v_valid = False
                    tracker_speed = 0.0

            if bbox is None:
                tracker_streak = 0
                best_conf = 0.0
                try:
                    if target_classes.size == 1:
                        xyxy, cls_ids, confs = backend.predict(frame, int(target_classes[0]))
                    else:
                        boxes_parts = []
                        cls_parts = []
                        conf_parts = []
                        for cls_target in target_classes:
                            part_xyxy, part_cls_ids, part_confs = backend.predict(frame, int(cls_target))
                            if part_xyxy.size == 0:
                                continue
                            boxes_parts.append(part_xyxy)
                            cls_parts.append(part_cls_ids)
                            conf_parts.append(part_confs)
                        if boxes_parts:
                            xyxy = np.concatenate(boxes_parts, axis=0)
                            cls_ids = np.concatenate(cls_parts, axis=0)
                            confs = np.concatenate(conf_parts, axis=0)
                        else:
                            xyxy = np.empty((0, 4), dtype=np.int32)
                            cls_ids = np.empty((0,), dtype=np.int32)
                            confs = np.empty((0,), dtype=np.float32)
                except Exception as e:
                    now_err = time.time()
                    if now_err - last_backend_err_ts > 1.0:
                        print(f"[WARN] backend inference failed: {e}")
                        last_backend_err_ts = now_err
                    continue
                if xyxy.size > 0:
                    if cls_ids.size == confs.size and np.all(np.isin(cls_ids, target_classes)):
                        class_mask = confs >= DETECTION_MIN_CONF
                    else:
                        class_mask = np.isin(cls_ids, target_classes) & (confs >= DETECTION_MIN_CONF)
                    if np.any(class_mask):
                        candidates = xyxy[class_mask]
                        cand_cls = cls_ids[class_mask]
                        cand_confs = confs[class_mask]
                        cx = (candidates[:, 0] + candidates[:, 2]) // 2
                        cy = (candidates[:, 1] + candidates[:, 3]) // 2
                        cx_full = cap["left"] + cx
                        cy_full = cap["top"] + cy

                        if prev_target_found:
                            ref_vx = filtered_vx
                            ref_vy = filtered_vy
                            if tracker_v_valid:
                                ref_vx = (1.0 - TRACKER_VELOCITY_REF_BLEND) * ref_vx + (
                                    TRACKER_VELOCITY_REF_BLEND * tracker_vx
                                )
                                ref_vy = (1.0 - TRACKER_VELOCITY_REF_BLEND) * ref_vy + (
                                    TRACKER_VELOCITY_REF_BLEND * tracker_vy
                                )
                            ref_x = prev_tx + (ref_vx * ASSOC_PREDICT_DT)
                            ref_y = prev_ty + (ref_vy * ASSOC_PREDICT_DT)
                            dx_lock = cx_full - ref_x
                            dy_lock = cy_full - ref_y
                            dist = (dx_lock * dx_lock) + (dy_lock * dy_lock)
                            jump_pad = min(ASSOC_MAX_JUMP_PAD, abs(last_speed) * ASSOC_SPEED_JUMP_GAIN)
                            max_jump_sq = (TARGET_LOCK_MAX_JUMP + jump_pad) ** 2
                            jump_mask = dist <= max_jump_sq

                            if abs(filtered_vx) + abs(filtered_vy) > 1e-3:
                                step_x = cx_full - prev_tx
                                step_y = cy_full - prev_ty
                                dot_motion = (step_x * filtered_vx) + (step_y * filtered_vy)
                                motion_mask = dot_motion >= -MOTION_BACKTRACK_TOL
                            else:
                                motion_mask = np.ones_like(jump_mask, dtype=bool)

                            assoc_mask = jump_mask & motion_mask
                            if np.any(assoc_mask):
                                gated = candidates[assoc_mask]
                                gated_cls = cand_cls[assoc_mask]
                                gated_conf = cand_confs[assoc_mask]
                                dist = dist[assoc_mask]
                                score = dist / np.maximum(0.05, gated_conf)
                                best_idx = int(np.argmin(score))
                                bbox = tuple(map(int, gated[best_idx]))
                                selected_cls = int(gated_cls[best_idx])
                                best_conf = float(gated_conf[best_idx])
                        else:
                            crop_cx = cap["width"] // 2
                            crop_cy = cap["height"] // 2
                            dx_center = cx - crop_cx
                            dy_center = cy - crop_cy
                            dist = (dx_center * dx_center) + (dy_center * dy_center)
                            score = dist / np.maximum(0.05, cand_confs)
                            best_idx = int(np.argmin(score))
                            bbox = tuple(map(int, candidates[best_idx]))
                            selected_cls = int(cand_cls[best_idx])
                            best_conf = float(cand_confs[best_idx])

                if bbox is not None:
                    if (not tracker_active) or (best_conf >= TRACKER_REINIT_CONF):
                        tracker = create_tracker()
                        tracker_active = False
                        if tracker is not None:
                            x1, y1, x2, y2 = bbox
                            w = max(1, x2 - x1)
                            h = max(1, y2 - y1)
                            tracker_active = tracker.init(frame, (x1, y1, w, h))

            inf_count += 1

            if bbox is None:
                miss_count += 1
                if prev_target_found and miss_count < MISS_HYSTERESIS_FRAMES:
                    hold_focus_x = clamp(
                        last_meas_x + (filtered_vx * PREDICTIVE_CAPTURE_LEAD_S),
                        0,
                        SCREEN_W - 1,
                    )
                    hold_focus_y = clamp(
                        last_meas_y + (filtered_vy * PREDICTIVE_CAPTURE_LEAD_S),
                        0,
                        SCREEN_H - 1,
                    )
                    with shared_lock:
                        state["capture_focus_full"] = (int(hold_focus_x), int(hold_focus_y))
                        state["target_speed"] = max(0.0, last_speed * 0.85)
                        state["target_ts"] = time.time()
                    continue

                with shared_lock:
                    state["target_found"] = False
                    state["target_speed"] = 0.0
                    state["capture_focus_full"] = CENTER
                    state["target_cls"] = -1
                kalman_initialized = False
                tracker_active = False
                last_speed = 0.0
                filtered_vx = 0.0
                filtered_vy = 0.0
                catchup_gain = 0.0
                catchup_rel_prev = 0.0
                catchup_has_prev = False
                tracker_v_valid = False
                tracker_speed = 0.0
                last_accel_mag = 0.0
                lead_filtered_x = float(CENTER[0])
                lead_filtered_y = float(CENTER[1])
                smooth_x = float(CENTER[0])
                smooth_y = float(CENTER[1])
                last_out_dx = 0
                last_out_dy = 0
                miss_count = 0
                last_pid_ts = time.time()
                pid_x.reset()
                pid_y.reset()
                bezier_curve_sign = 1.0 if np.random.rand() >= 0.5 else -1.0
                velocity_stop_latched = True
                continue

            x1, y1, x2, y2 = bbox
            miss_count = 0
            box_h = max(1, y2 - y1)

            cx_crop = (x1 + x2) // 2
            cy_crop = (y1 + y2) // 2
            cx_full = cap["left"] + cx_crop
            cy_full = cap["top"] + cy_crop

            if prev_target_found and (not used_tracker):
                jump_dist = float(np.hypot(float(cx_full - prev_tx), float(cy_full - prev_ty)))
                jump_limit = MEAS_JUMP_REJECT_BASE_PX + min(140.0, last_speed * MEAS_JUMP_REJECT_SPEED_GAIN)
                if (jump_dist > jump_limit) and (best_conf < MEAS_JUMP_REJECT_CONF):
                    hold_focus_x = clamp(last_meas_x, 0, SCREEN_W - 1)
                    hold_focus_y = clamp(last_meas_y, 0, SCREEN_H - 1)
                    with shared_lock:
                        state["capture_focus_full"] = (int(hold_focus_x), int(hold_focus_y))
                        state["target_speed"] = max(0.0, last_speed * 0.80)
                        state["target_ts"] = time.time()
                    continue

            measurement[0, 0] = np.float32(cx_full)
            measurement[1, 0] = np.float32(cy_full)

            now_k = time.time()
            if used_tracker and best_conf <= 0.0:
                measurement_conf = TRACKER_MEASUREMENT_CONF
            else:
                measurement_conf = max(DETECTION_MIN_CONF, best_conf)
            if not kalman_initialized:
                kalman.statePost = np.array([[cx_full], [cy_full], [0], [0]], dtype=np.float32)
                kalman_initialized = True
                last_kalman_ts = now_k
                last_meas_ts = now_k
                last_meas_x = float(cx_full)
                last_meas_y = float(cy_full)
                filtered_vx = 0.0
                filtered_vy = 0.0
                last_tracker_ts = now_k
                last_tracker_cx_full = float(cx_full)
                last_tracker_cy_full = float(cy_full)
                lead_filtered_x = float(cx_full)
                lead_filtered_y = float(cy_full)
                last_pid_ts = now_k
                pid_x.reset()
                pid_y.reset()
                velocity_stop_latched = True

            if local_use_kalman:
                dt = float(np.clip(now_k - last_kalman_ts, KALMAN_MIN_DT, KALMAN_MAX_DT))
                last_kalman_ts = now_k
                kalman.transitionMatrix[0, 2] = dt
                kalman.transitionMatrix[1, 3] = dt

                speed_ratio_q = clamp(last_speed / max(1e-6, ADAPTIVE_SPEED_MAX), 0.0, 1.0)
                accel_ratio_q = clamp(last_accel_mag / max(1e-6, MAX_ACCEL_PX_S2), 0.0, 1.0)
                process_noise = KALMAN_PROCESS_NOISE_BASE * (
                    1.0
                    + (KALMAN_PROCESS_NOISE_SPEED_GAIN * speed_ratio_q)
                    + (KALMAN_PROCESS_NOISE_ACCEL_GAIN * accel_ratio_q)
                )
                conf_norm = clamp(
                    (measurement_conf - KALMAN_MEAS_NOISE_MIN_CONF)
                    / max(1e-6, (1.0 - KALMAN_MEAS_NOISE_MIN_CONF)),
                    0.0,
                    1.0,
                )
                meas_noise = KALMAN_MEAS_NOISE_BASE * (
                    1.0 + ((1.0 - conf_norm) * KALMAN_MEAS_NOISE_LOW_CONF_GAIN)
                )
                if used_tracker:
                    meas_noise *= 1.15
                kalman.processNoiseCov[:] = kalman_q_eye * np.float32(process_noise)
                kalman.measurementNoiseCov[:] = kalman_r_eye * np.float32(meas_noise)

                kalman.predict()
                kalman.correct(measurement)

                state_post = kalman.statePost
                state_x = float(state_post[0])
                state_y = float(state_post[1])
                vx = float(state_post[2])
                vy = float(state_post[3])

                meas_dt = max(MIN_MEAS_DT, now_k - last_meas_ts) if last_meas_ts > 0.0 else dt
                meas_vx = (float(cx_full) - last_meas_x) / meas_dt
                meas_vy = (float(cy_full) - last_meas_y) / meas_dt
                target_vx = (1.0 - MEAS_VELOCITY_BLEND) * vx + (MEAS_VELOCITY_BLEND * meas_vx)
                target_vy = (1.0 - MEAS_VELOCITY_BLEND) * vy + (MEAS_VELOCITY_BLEND * meas_vy)
                if used_tracker and tracker_v_valid:
                    target_vx = (1.0 - TRACKER_VELOCITY_BLEND) * target_vx + (
                        TRACKER_VELOCITY_BLEND * tracker_vx
                    )
                    target_vy = (1.0 - TRACKER_VELOCITY_BLEND) * target_vy + (
                        TRACKER_VELOCITY_BLEND * tracker_vy
                    )

                meas_speed = float(np.hypot(meas_vx, meas_vy))
                if velocity_stop_latched:
                    if meas_speed >= VELOCITY_STOP_EXIT_THRESHOLD:
                        velocity_stop_latched = False
                else:
                    if meas_speed <= VELOCITY_STOP_ENTER_THRESHOLD:
                        velocity_stop_latched = True

                if velocity_stop_latched:
                    # Hold zero velocity while near-stop to avoid rapid 0/non-zero toggling.
                    filtered_vx = 0.0
                    filtered_vy = 0.0
                    last_accel_mag *= 0.6
                else:
                    dot_motion = (meas_vx * filtered_vx) + (meas_vy * filtered_vy)
                    if (abs(filtered_vx) + abs(filtered_vy) > 1e-3) and (dot_motion < 0.0):
                        # Zero-lag response on direction flip (ADAD / abrupt reversal).
                        filtered_vx = meas_vx
                        filtered_vy = meas_vy

                    accel_mag = float(
                        np.hypot(target_vx - filtered_vx, target_vy - filtered_vy) / max(dt, 1e-6)
                    )
                    last_accel_mag = accel_mag

                    max_dv = MAX_ACCEL_PX_S2 * dt
                    dvx = target_vx - filtered_vx
                    dvy = target_vy - filtered_vy
                    dv = float(np.hypot(dvx, dvy))
                    if dv > max_dv and dv > 1e-6:
                        scale = max_dv / dv
                        dvx *= scale
                        dvy *= scale
                    filtered_vx += dvx
                    filtered_vy += dvy
                vx = filtered_vx
                vy = filtered_vy

                speed = float(np.hypot(vx, vy))
                if speed > KALMAN_MAX_SPEED_PX_S and speed > 1e-6:
                    scale = KALMAN_MAX_SPEED_PX_S / speed
                    vx *= scale
                    vy *= scale
                    filtered_vx = vx
                    filtered_vy = vy
                    speed = KALMAN_MAX_SPEED_PX_S

                if speed < KALMAN_LEAD_MIN_SPEED:
                    vx = 0.0
                    vy = 0.0
                    lead_dx = 0.0
                    lead_dy = 0.0
                    speed_ratio = 0.0
                else:
                    speed_ratio = (speed - ADAPTIVE_SPEED_MIN) / max(
                        1e-6, (ADAPTIVE_SPEED_MAX - ADAPTIVE_SPEED_MIN)
                    )
                    speed_ratio = clamp(speed_ratio, 0.0, 1.0)
                    lead_time = frame_age + EXTRA_PIPELINE_LAG_S + (LEAD_FRAMES * dt)
                    if speed >= SPEED_BOOST_THRESHOLD:
                        lead_time *= SPEED_BOOST_GAIN
                    if speed >= FAST_TARGET_SPEED:
                        lead_time += NON_STATIONARY_EXTRA_LEAD_S
                    lead_dx = float(np.clip(vx * lead_time, -KALMAN_MAX_LEAD_PX, KALMAN_MAX_LEAD_PX))
                    lead_dy = float(np.clip(vy * lead_time, -KALMAN_MAX_LEAD_PX, KALMAN_MAX_LEAD_PX))

                    center_error = float(
                        np.hypot(float(cx_full) - CENTER[0], float(cy_full) - CENTER[1])
                    )
                    near_scale = clamp(
                        (center_error - LEAD_NEAR_TARGET_INNER_PX)
                        / max(1e-6, (LEAD_NEAR_TARGET_OUTER_PX - LEAD_NEAR_TARGET_INNER_PX)),
                        0.0,
                        1.0,
                    )
                    conf_norm_lead = clamp(
                        (measurement_conf - DETECTION_MIN_CONF) / max(1e-6, (1.0 - DETECTION_MIN_CONF)),
                        0.0,
                        1.0,
                    )
                    conf_scale = LEAD_CONFIDENCE_MIN_SCALE + (
                        (1.0 - LEAD_CONFIDENCE_MIN_SCALE) * conf_norm_lead
                    )
                    lead_scale = near_scale * conf_scale
                    lead_dx *= lead_scale
                    lead_dy *= lead_scale

                base_x = (float(cx_full) * (1.0 - KALMAN_POSITION_BLEND)) + (state_x * KALMAN_POSITION_BLEND)
                base_y = (float(cy_full) * (1.0 - KALMAN_POSITION_BLEND)) + (state_y * KALMAN_POSITION_BLEND)
                lead_x_raw = base_x + lead_dx
                lead_y_raw = base_y + lead_dy
                if local_fast_aim:
                    lead_alpha = FAST_LEAD_SMOOTH_ALPHA_SLOW + (
                        (FAST_LEAD_SMOOTH_ALPHA_FAST - FAST_LEAD_SMOOTH_ALPHA_SLOW) * speed_ratio
                    )
                else:
                    lead_alpha = LEAD_SMOOTH_ALPHA_SLOW + (
                        (LEAD_SMOOTH_ALPHA_FAST - LEAD_SMOOTH_ALPHA_SLOW) * speed_ratio
                    )
                lead_filtered_x = lead_filtered_x + (lead_x_raw - lead_filtered_x) * lead_alpha
                lead_filtered_y = lead_filtered_y + (lead_y_raw - lead_filtered_y) * lead_alpha
                lead_x = lead_filtered_x
                lead_y = lead_filtered_y
            else:
                lead_x = float(cx_full)
                lead_y = float(cy_full)
                vx = 0.0
                vy = 0.0
                speed = 0.0
                speed_ratio = 1.0
                filtered_vx = 0.0
                filtered_vy = 0.0
                lead_filtered_x = lead_x
                lead_filtered_y = lead_y
                last_accel_mag = 0.0

            if used_tracker and tracker_v_valid:
                last_speed = max(speed, tracker_speed)
            else:
                last_speed = speed
            last_meas_ts = now_k
            last_meas_x = float(cx_full)
            last_meas_y = float(cy_full)

            if abs(vx) >= AIM_EDGE_SPEED_MIN and AIM_LEAD_EDGE_FACTOR > 0.0:
                box_left_full = float(cap["left"] + x1)
                box_right_full = float(cap["left"] + x2)
                edge_x = box_left_full if vx < 0.0 else box_right_full

                # If target is moving away from center, push harder to the leading edge.
                rel_x = lead_x - CENTER[0]
                trailing = (vx * rel_x) > 0.0
                edge_blend = AIM_LEAD_EDGE_FACTOR * (
                    AIM_EDGE_TRAILING_MULT if trailing else AIM_EDGE_NON_TRAILING_MULT
                )
                edge_blend = clamp(edge_blend, 0.0, 1.0)
                aim_x = (lead_x * (1.0 - edge_blend)) + (edge_x * edge_blend)
            else:
                aim_x = lead_x

            if CATCHUP_ENABLE and abs(vx) >= CATCHUP_MIN_SPEED:
                motion_sign_x = -1.0 if vx < 0.0 else 1.0
                rel_along = (aim_x - CENTER[0]) * motion_sign_x
                if rel_along > 0.0:
                    if catchup_has_prev:
                        progress = catchup_rel_prev - rel_along
                        if progress < CATCHUP_MIN_PROGRESS_PX:
                            catchup_gain = min(CATCHUP_GAIN_MAX, catchup_gain + CATCHUP_GAIN_UP)
                        else:
                            catchup_gain = max(0.0, catchup_gain - CATCHUP_GAIN_DOWN)
                    catchup_rel_prev = rel_along
                    catchup_has_prev = True
                else:
                    catchup_gain = max(0.0, catchup_gain - CATCHUP_GAIN_DOWN)
                    catchup_rel_prev = 0.0
                    catchup_has_prev = False
            else:
                catchup_gain = max(0.0, catchup_gain - CATCHUP_GAIN_DOWN)
                catchup_rel_prev = 0.0
                catchup_has_prev = False

            if catchup_gain > 0.0:
                base_dx = aim_x - CENTER[0]
                extra_dx = base_dx * (catchup_gain * CATCHUP_CENTER_SCALE)
                extra_dx = clamp(extra_dx, -CATCHUP_MAX_EXTRA_PX, CATCHUP_MAX_EXTRA_PX)
                aim_x += extra_dx

            aim_y = lead_y - (box_h * HEAD_Y_BIAS if local_aimmode == 0 else 0)

            if local_use_kalman:
                if local_fast_aim:
                    kalman_extra_smooth = FAST_KALMAN_EXTRA_SMOOTH_SLOW + (
                        (FAST_KALMAN_EXTRA_SMOOTH_FAST - FAST_KALMAN_EXTRA_SMOOTH_SLOW) * speed_ratio
                    )
                else:
                    kalman_extra_smooth = KALMAN_EXTRA_SMOOTH_SLOW + (
                        (KALMAN_EXTRA_SMOOTH_FAST - KALMAN_EXTRA_SMOOTH_SLOW) * speed_ratio
                    )
            else:
                kalman_extra_smooth = 0.0

            effective_smoothing = SMOOTHING + kalman_extra_smooth
            effective_smoothing = clamp(effective_smoothing, 0.0, 0.92)
            smooth_x = smooth_x + (aim_x - smooth_x) * (1.0 - effective_smoothing)
            smooth_y = smooth_y + (aim_y - smooth_y) * (1.0 - effective_smoothing)

            error_x = smooth_x - CENTER[0]
            error_y = smooth_y - CENTER[1]
            error_mag = float(np.hypot(error_x, error_y))
            if BEZIER_CURVE_ENABLED:
                error_x, error_y = apply_bezier_offset(error_x, error_y, bezier_curve_sign)

            position_deadzone = (
                POSITION_DEADZONE_SLOW + ((POSITION_DEADZONE_FAST - POSITION_DEADZONE_SLOW) * speed_ratio)
                if local_use_kalman
                else POSITION_DEADZONE_SLOW
            )
            pid_dt = float(np.clip(now_k - last_pid_ts, KALMAN_MIN_DT, KALMAN_MAX_DT))
            last_pid_ts = now_k

            # Instant snapping: compute raw aim offset and send directly (no output smoothing)
            raw_dx = aim_x - CENTER[0]
            raw_dy = aim_y - CENTER[1]
            dx = int(round(clamp(raw_dx, -OUTPUT_MAX_STEP_X, OUTPUT_MAX_STEP_X)))
            dy = int(round(clamp(raw_dy, -OUTPUT_MAX_STEP_Y, OUTPUT_MAX_STEP_Y)))
            # apply deadzone when using Kalman to avoid tiny jitter
            deadzone = (
                AIM_DEADZONE_SLOW + ((AIM_DEADZONE_FAST - AIM_DEADZONE_SLOW) * speed_ratio)
                if local_use_kalman
                else 0.0
            )
            if abs(dx) <= deadzone:
                dx = 0
            if abs(dy) <= deadzone:
                dy = 0
            last_out_dx = dx
            last_out_dy = dy

            focus_x = clamp(lead_x + (vx * PREDICTIVE_CAPTURE_LEAD_S), 0, SCREEN_W - 1)
            focus_y = clamp(lead_y + (vy * PREDICTIVE_CAPTURE_LEAD_S), 0, SCREEN_H - 1)

            with shared_lock:
                state["target_found"] = True
                state["last_target_full"] = (int(cx_full), int(cy_full))
                state["capture_focus_full"] = (int(focus_x), int(focus_y))
                state["target_speed"] = speed
                state["target_cls"] = int(selected_cls)
                state["aim_dx"] = dx
                state["aim_dy"] = dy
                state["target_ts"] = now_k
                state["aim_seq"] += 1

            put_latest(
                control_queue,
                {"dx": dx, "dy": dy, "ts": now_k},
            )

            if DEBUG_LOG and (not used_tracker):
                print(
                    f"[DETECTED] cls={selected_cls} at ({cx_full},{cy_full}) "
                    f"pred=({int(lead_x)},{int(lead_y)}) v=({int(vx)},{int(vy)}) s={int(speed)}"
                )

    def control_loop():
        nonlocal running
        while running:
            try:
                cmd = get_latest(control_queue, 0.01)
            except queue.Empty:
                continue

            now = time.time()
            if now - cmd["ts"] > TARGET_TIMEOUT_S:
                continue

            with shared_lock:
                local_mode = state["mode"]
            if local_mode == 0:
                continue

            mouse_client.send(cmd["dx"], cmd["dy"])

    def on_click(x, y, button, pressed):
        nonlocal mouse_button_pressed_x1, mouse_button_pressed_x2
        if button == Button.x2:
            mouse_button_pressed_x2 = pressed
        if button == Button.x1:
            mouse_button_pressed_x1 = pressed

    capture_thread = threading.Thread(target=capture_loop, daemon=True)
    inference_thread = threading.Thread(target=inference_loop, daemon=True)
    control_thread = threading.Thread(target=control_loop, daemon=True)
    capture_thread.start()
    inference_thread.start()
    control_thread.start()

    with Listener(on_click=on_click):
        while running:
            try:
                if keyboard.is_pressed("insert"):
                    running = False
                    break

                now = time.time()
                if mouse_button_pressed_x2 and now - last_toggle > 0.2:
                    mode = (mode + 1) % 2
                    with shared_lock:
                        state["mode"] = mode
                    last_toggle = now
                    freq = 500
                    if mode == 1:
                        freq = 1000
                    if mode == 0:
                        drain_queue(control_queue)
                        with shared_lock:
                            state["target_found"] = False
                            state["target_cls"] = -1
                    if mode == 0:
                        mode_label = "OFF"
                    else:
                        mode_label = "T"
                    print(f"Mode: {mode} ({mode_label})")
                    winsound.Beep(freq, 100)

                if mouse_button_pressed_x1 and now - last_toggle > 0.2:
                    aimmode = (aimmode + 1) % 2
                    with shared_lock:
                        state["aimmode"] = aimmode
                    last_toggle = now
                    freq = 1200 if aimmode == 0 else 600
                    print(f"AimMode: {aimmode}")
                    winsound.Beep(freq, 100)

                if keyboard.is_pressed("f7") and now - last_kalman_toggle > 0.25:
                    use_kalman = not use_kalman
                    with shared_lock:
                        state["use_kalman"] = use_kalman
                    last_kalman_toggle = now
                    print(f"Kalman: {'ON' if use_kalman else 'OFF'}")
                    winsound.Beep(1500 if use_kalman else 700, 100)

                if keyboard.is_pressed("f9") and now - last_fast_toggle > 0.25:
                    with shared_lock:
                        new_fast = not state.get("fast_aim", False)
                        state["fast_aim"] = new_fast
                    last_fast_toggle = now
                    print(f"FastAim: {'ON' if new_fast else 'OFF'}")
                    winsound.Beep(1800 if new_fast else 700, 100)

                time.sleep(SLEEP_TIME)
            except KeyboardInterrupt:
                running = False
                break

    with shared_lock:
        state["running"] = False
    mouse_client.close()


if __name__ == "__main__":
    main()
