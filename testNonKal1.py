import socket
import threading
import time
import queue
import os
import importlib.util
import struct
import ctypes
import json
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import keyboard
import mss
import numpy as np
import winsound
from ultralytics import YOLO

try:
    from pynput.mouse import Button, Listener
except ImportError:
    Button = None
    Listener = None

try:
    import pydirectinput
except ImportError:
    pydirectinput = None

try:
    import mouse as ghub_mouse
except ImportError:
    ghub_mouse = None

try:
    import dxcam
except ImportError:
    dxcam = None

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    try:
        # Throughput-oriented CUDA settings reduce warmup and steady-state latency.
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

# ---------------- CONFIG ----------------
MODEL_PATH = r"C:\YOLO\Delta\runs\detect\train2\weights\best.pt"
IMGSZ = 640  # latency-first input size; fixed-shape ONNX models may override at runtime
CONF = 0.2
DEVICE = "cuda"
INFER_BACKEND = "auto"   # "auto" | "ultralytics" | "onnxruntime"
ONNX_MODEL_PATH = ""     # optional explicit .onnx path; empty = derive from MODEL_PATH
AUTO_EXPORT_ONNX = False
ONNX_NMS_IOU = 0.50
ONNX_TOPK_PRE_NMS = 500
ONNX_OUTPUT_HAS_NMS = True
ONNX_FORCE_TARGET_CLASS_DECODE = True
ONNX_USE_TENSORRT_EP = True
ONNX_TRT_FP16 = True
ONNX_TRT_MAX_WORKSPACE_SIZE = 0  # bytes; 0 = ORT default
ONNX_TRT_TIMING_CACHE_ENABLE = True
ONNX_TRT_FORCE_TIMING_CACHE = False
ONNX_TRT_TIMING_CACHE_PATH = ""  # empty = use TRT engine cache directory
ONNX_TRT_OP_TYPES_TO_EXCLUDE = ""  # e.g. "NonMaxSuppression,NonZero,RoiAlign"
ONNX_TRT_CUDA_GRAPH_ENABLE = False
ONNX_ENABLE_CUDA_GRAPH = False
ONNX_CUDA_DEVICE_ID = 0
ONNX_CUDNN_CONV_ALGO_SEARCH = "EXHAUSTIVE"  # "DEFAULT" | "HEURISTIC" | "EXHAUSTIVE"
ONNX_DISABLE_ORT_CPU_FALLBACK = True

SCREEN_W = 2560
SCREEN_H = 1440
CENTER = (SCREEN_W // 2, SCREEN_H // 2)
FPS = 10000
SLEEP_TIME = 1 / FPS
CAPTURE_BACKEND = "auto"  # "auto" | "dxcam" | "dxcam_sync" | "mss"
CAPTURE_DEVICE_IDX = 0
CAPTURE_OUTPUT_IDX = 0
CAPTURE_AUTO_BENCHMARK = True
CAPTURE_AUTO_BENCHMARK_SECONDS = 0.75
CAPTURE_AUTO_MAX_NONE_RATIO = 0.10
CAPTURE_AUTO_BENCHMARK_DYNAMIC = True
CAPTURE_AUTO_BENCHMARK_MOTION_PX = 48
CAPTURE_BENCHMARK_ON_START = False
CAPTURE_BENCHMARK_SECONDS = 2.0
CAPTURE_BENCHMARK_W = 700
CAPTURE_BENCHMARK_H = 700
CAPTURE_YIELD_EACH_LOOP = True
CAPTURE_NONE_BACKOFF_S = 0.0
CAPTURE_THREAD_PRIORITY = "highest"  # "normal" | "above_normal" | "highest"
INFERENCE_THREAD_PRIORITY = "highest"  # "normal" | "above_normal" | "highest"
CONTROL_THREAD_PRIORITY = "highest"  # "normal" | "above_normal" | "highest"
MSS_USE_RAW_BUFFER = True
CAPTURE_DXGI_TARGET_FPS = 0
CAPTURE_DXGI_VIDEO_MODE = True
CAPTURE_DXGI_ANCHOR_PAD = 128
CAPTURE_DXGI_EDGE_MARGIN = 32
CAPTURE_DXGI_FRAME_SHAPE_RETRIES = 2
CAPTURE_DXGI_FORCE_CONTIGUOUS = True
CAPTURE_FORCE_INPUT_SIZE_WHEN_LOCKED = True
ONNX_SKIP_RESIZE_IF_MATCH = True
ONNX_RESIZE_INTERPOLATION = "nearest"  # "nearest" | "linear"

BASE_CROP_W = IMGSZ
BASE_CROP_H = IMGSZ
LOST_CROP_W = IMGSZ
LOST_CROP_H = IMGSZ
MIN_ACTIVE_CROP_W = IMGSZ
MIN_ACTIVE_CROP_H = IMGSZ
MAX_ACTIVE_CROP_W = IMGSZ
MAX_ACTIVE_CROP_H = IMGSZ
CAPTURE_SPEED_TO_PAD = 0.14
CAPTURE_PAD_MAX = 160

TRACKER_TYPE = "MIL"  # "MOSSE"/"KCF"/"CSRT" if available; fallback to "MIL"
TRACKER_ENABLE = False
TRACKER_FORCE_FIXED_CAPTURE_SIZE = True
TRACKER_REQUIRE_SAME_FRAME_SHAPE = True
TRACKER_UPDATE_ERROR_LOG_COOLDOWN_S = 1.0
YOLO_INTERVAL = 1
YOLO_INTERVAL_FAST = 1
TRACKER_MIN_INTERVAL = 3
TRACKER_MAX_STREAK = 6
FAST_TARGET_SPEED = 220.0
MAX_DETECTIONS = 6

LEAD_FRAMES = 3.0
SMOOTHING = 0.0       # base smoothing; Kalman path adds adaptive smoothing below
AIM_LEAD_EDGE_FACTOR = 0.0
AIM_EDGE_SPEED_MIN = 10.0
AIM_EDGE_TRAILING_MULT = 3.5
AIM_EDGE_NON_TRAILING_MULT = 0.45
CATCHUP_ENABLE = False
CATCHUP_MIN_SPEED = 140.0
CATCHUP_MIN_PROGRESS_PX = 1.0
CATCHUP_GAIN_UP = 0.10
CATCHUP_GAIN_DOWN = 0.08
CATCHUP_GAIN_MAX = 0.75
CATCHUP_CENTER_SCALE = 0.75
CATCHUP_MAX_EXTRA_PX = 180.0
HEAD_Y_BIAS = 0.0    # aim slightly above center for head class
USE_KALMAN_DEFAULT = False
TRACKING_STRATEGY_RAW = "raw"
TRACKING_STRATEGY_KALMAN = "kalman"
TRACKING_STRATEGY_EMA = "ema"
TRACKING_STRATEGY_DEMA = "dema"
TRACKING_STRATEGY_RAW_DELTA = "raw_delta"
TRACKING_STRATEGY_DEFAULT = (
    TRACKING_STRATEGY_KALMAN if USE_KALMAN_DEFAULT else TRACKING_STRATEGY_RAW_DELTA
)
TRACKING_POSITION_ALPHA = 0.42
TRACKING_VELOCITY_ALPHA = 1.0
STICKY_BIAS_PX = 80.0
TARGET_MAX_LOST_FRAMES = 8
TARGET_LOCK_MAX_JUMP = 260  # kept for capture/tracking sanity when reacquiring
PREDICTION_TIME = 0.001
KALMAN_MAX_DT = 0.06
KALMAN_MIN_DT = 1.0 / 240.0
MIN_MEAS_DT = 1.0 / 120.0
KALMAN_MAX_SPEED_PX_S = 1800.0
FEEDFORWARD_MIN_UPDATES = 6
FEEDFORWARD_RAMP_UPDATES = 12
DELAY_COMPENSATION_ENABLE = False
DELAY_COMP_EMA_ALPHA = 0.22
DELAY_COMP_INPUT_APPLY_S = 0.0025
DELAY_COMP_MAX_S = 0.04
DELAY_COMP_MIN_SPEED = 80.0
EGO_MOTION_COMP_ENABLE = True
EGO_MOTION_COMP_ALPHA = 0.30
EGO_MOTION_COMP_GAIN_X = 0.70
EGO_MOTION_COMP_GAIN_Y = 0.65
EGO_MOTION_COMP_MAX_PX_S = 3200.0
EGO_MOTION_COMP_DECAY = 0.92
EGO_MOTION_ERROR_GATE_ENABLE = True
EGO_MOTION_ERROR_GATE_PX = 70
EGO_MOTION_ERROR_GATE_NORMALIZE_BY_BOX = False
EGO_MOTION_ERROR_GATE_NORM_THRESHOLD = 2.0
EGO_MOTION_RESET_ON_SWITCH = True
KALMAN_USE_CA_MODEL = False
MAX_FRAME_AGE_S = 0.05
DEBUG_LOG = False
PERF_LOG_ENABLE = True
PERF_LOG_INTERVAL_S = 1.0
PERF_LOG_WHEN_MODE_OFF = False

PIPELINE_FRAME_QUEUE = 1
PIPELINE_CMD_QUEUE = 1
DETECTION_MIN_CONF = 0.20
TRACKER_REINIT_CONF = 0.45
TRACKER_REINIT_MIN_IOU = 0.35
ASSOC_PREDICT_DT = 0.02
ASSOC_SPEED_JUMP_GAIN = 0.05
ASSOC_MAX_JUMP_PAD = 220.0
MOTION_BACKTRACK_TOL = 1400.0
TRACKER_VELOCITY_BLEND = 0.9
TRACKER_VELOCITY_REF_BLEND = 0.8
TRACKER_VELOCITY_MIN_DT = 1.0 / 300.0
TRACKER_MAX_SPEED_PX_S = 2600.0

PID_ENABLE = True
PID_KP = 0.50
PID_KI = 0.0
PID_KD = 0.008
PID_INTEGRAL_LIMIT = 20.0
PID_ANTI_WINDUP_GAIN = 1.0
PID_DERIVATIVE_ALPHA = 0.2
PID_OUTPUT_MAX = 350.0
PID_MICRO_ERROR_PX = 0
PID_SOFT_ERROR_PX = 0
PID_SOFT_ZONE_GAIN = 0

PID_FRONTEND_ENABLE = True
PID_FRONTEND_HOST = "127.0.0.1"
PID_FRONTEND_PORT = 8765
PID_FRONTEND_PORT_RETRIES = 10

OUTPUT_MAX_STEP_X = 220
OUTPUT_MAX_STEP_Y = 220
OUTPUT_MICRO_CMD_PX = 1

# Raw output stays direct; only hard step caps remain.
RAW_MAX_STEP_X = 280
RAW_MAX_STEP_Y = 280
MOVING_LAG_BIAS_ENABLE = False
MOVING_LAG_BIAS_MIN_SPEED = 45.0
MOVING_LAG_BIAS_MIN_ERROR_PX = 0.8
MOVING_LAG_BIAS_KI_X = 1.10
MOVING_LAG_BIAS_KI_Y = 1.10
MOVING_LAG_BIAS_MAX_X = 120.0
MOVING_LAG_BIAS_MAX_Y = 120.0
MOVING_LAG_BIAS_NEAR_ERROR_PX = 16.0
MOVING_LAG_BIAS_NEAR_CAP_PX = 10.0
MOVING_LAG_BIAS_FAR_CAP_PX = 30.0
MOVING_LAG_BIAS_SIGN_RESET_ERROR_PX = 20.0
MOVING_LAG_BIAS_SIGN_RESET_DECAY = 0.35
MOVING_LAG_BIAS_DECAY = 0.88
MOVING_LAG_BIAS_ALIGN_ONLY = True

MISS_HYSTERESIS_FRAMES = 3
TRACKER_MEASUREMENT_CONF = 0.68

# Kalman tuning: raise Q for faster direction changes, raise R for more smoothing.
KALMAN_INITIAL_UNCERTAINTY = 1000.0
KALMAN_PROCESS_NOISE_POSITION = 0.01
KALMAN_PROCESS_NOISE_BASE = 1.5
KALMAN_MEAS_NOISE_BASE = 16.0

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
MOUSE_SEND_BACKEND = "ghub"  # "ghub" | "pydirectinput" | "socket"
MOUSE_SOCKET_BINARY = True
PYDIRECTINPUT_GAIN_X = 1.0
PYDIRECTINPUT_GAIN_Y = 1.0
GHUB_GAIN_X = 1.0
GHUB_GAIN_Y = 1.0
GHUB_MAX_STEP = 127
RECOIL_CONTROL_ENABLE = True
RECOIL_COMPENSATION_Y_PX = 12
TRIGGERBOT_ENABLE_DEFAULT = False
TRIGGERBOT_BOX_PERCENT_DEFAULT = 100.0
TRIGGERBOT_CLICK_HOLD_S = 0.001
TRIGGERBOT_CLICK_COOLDOWN_S = 0.001
TRIGGERBOT_TOGGLE_KEY = "f8"
TRIGGERBOT_TOGGLE_COOLDOWN_S = 0.2
RECOIL_TUNE_FALLBACK_DEFAULT = False
RECOIL_TUNE_FALLBACK_IGNORE_MODE_CHECK_DEFAULT = False
RECOIL_TUNE_FALLBACK_TOGGLE_KEY = "f7"
RECOIL_TUNE_FALLBACK_TOGGLE_COOLDOWN_S = 0.2
WARMUP_CLASS = 0
LEFT_HOLD_ENGAGE_DEFAULT = False
LEFT_HOLD_ENGAGE_TOGGLE_KEY = "f6"
LEFT_HOLD_ENGAGE_TOGGLE_COOLDOWN_S = 0.2
VK_LBUTTON = 0x01
VK_RBUTTON = 0x02
VK_XBUTTON1 = 0x05
VK_XBUTTON2 = 0x06
TARGET_TIMEOUT_S = 0.08
# ----------------------------------------


_DLL_SEARCH_PATH_HANDLES = []
_DLL_SEARCH_PATH_DIRS = set()


def patch_dxcam_runtime():
    if dxcam is None:
        return

    dxcamera_cls = getattr(dxcam, "DXCamera", None)
    if dxcamera_cls is None:
        return
    if getattr(dxcamera_cls, "_codex_runtime_patch", False):
        return

    original_start = dxcamera_cls.start
    original_grab = dxcamera_cls._grab
    original_capture = getattr(dxcamera_cls, "_DXCamera__capture", None)

    def patched_start(self, region=None, target_fps=10000, video_mode=False, delay=0):
        if region is not None:
            region = tuple(int(v) for v in region)
            self.region = region
            self._region_set_by_user = True
        return original_start(
            self,
            region=region,
            target_fps=target_fps,
            video_mode=video_mode,
            delay=delay,
        )

    def patched_stop(self):
        thread = getattr(self, "_DXCamera__thread", None)
        frame_available = getattr(self, "_DXCamera__frame_available", None)
        stop_capture = getattr(self, "_DXCamera__stop_capture", None)

        if self.is_capturing:
            if frame_available is not None:
                frame_available.set()
            if stop_capture is not None:
                stop_capture.set()
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=10)

        self.is_capturing = False
        self._DXCamera__thread = None
        self._DXCamera__frame_buffer = None
        self._DXCamera__frame_count = 0
        self._DXCamera__head = 0
        self._DXCamera__tail = 0
        self._DXCamera__full = False
        if frame_available is not None:
            frame_available.clear()
        if stop_capture is not None:
            stop_capture.clear()

    def patched_get_latest_frame(self):
        frame_available = getattr(self, "_DXCamera__frame_available", None)
        frame_buffer = getattr(self, "_DXCamera__frame_buffer", None)
        lock = getattr(self, "_DXCamera__lock", None)
        if frame_available is None or lock is None or frame_buffer is None:
            return None
        if not frame_available.wait(timeout=0.05):
            return None
        try:
            with lock:
                frame_buffer = getattr(self, "_DXCamera__frame_buffer", None)
                if frame_buffer is None:
                    return None
                head = int(getattr(self, "_DXCamera__head", 0))
                frame = frame_buffer[(head - 1) % self.max_buffer_len]
                frame_available.clear()
            return np.array(frame)
        except Exception:
            return None

    def patched_grab(self, region):
        try:
            return original_grab(self, region)
        except Exception as exc:
            setattr(self, "_codex_last_capture_error", exc)
            try:
                self._on_output_change()
            except Exception:
                pass
            return None

    def patched_capture(self, region, target_fps=60, video_mode=False):
        if original_capture is None:
            return None
        try:
            return original_capture(self, region, target_fps, video_mode)
        except Exception as exc:
            setattr(self, "_codex_last_capture_error", exc)
            return None

    dxcamera_cls.start = patched_start
    dxcamera_cls.stop = patched_stop
    dxcamera_cls.get_latest_frame = patched_get_latest_frame
    dxcamera_cls._grab = patched_grab
    if original_capture is not None:
        setattr(dxcamera_cls, "_DXCamera__capture", patched_capture)
    dxcamera_cls._codex_runtime_patch = True


patch_dxcam_runtime()


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


def ema_update(prev, sample, alpha):
    alpha = clamp(float(alpha), 0.0, 1.0)
    sample = float(sample)
    prev = float(prev)
    if alpha <= 0.0:
        return prev
    if not np.isfinite(prev) or abs(prev) <= 1e-12:
        return sample
    return prev + ((sample - prev) * alpha)


def ema_update_signed(prev, sample, alpha):
    alpha = clamp(float(alpha), 0.0, 1.0)
    sample = float(sample)
    prev = float(prev)
    if alpha <= 0.0:
        return prev
    return prev + ((sample - prev) * alpha)


def bbox_iou_xyxy(a, b):
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter
    if union <= 1e-6:
        return 0.0
    return inter / union


def set_current_thread_priority(level):
    if os.name != "nt":
        return
    level = str(level).strip().lower()
    priority_map = {
        "normal": 0,
        "above_normal": 1,
        "highest": 2,
    }
    priority = priority_map.get(level)
    if priority is None:
        return

    try:
        handle = ctypes.windll.kernel32.GetCurrentThread()
        ctypes.windll.kernel32.SetThreadPriority(handle, int(priority))
    except Exception:
        pass


def is_vk_pressed(vk_code):
    if os.name != "nt":
        return False
    try:
        return bool(ctypes.windll.user32.GetAsyncKeyState(int(vk_code)) & 0x8000)
    except Exception:
        return False


def build_capture(center_x, center_y, width, height):
    left = int(clamp(center_x - width // 2, 0, SCREEN_W - width))
    top = int(clamp(center_y - height // 2, 0, SCREEN_H - height))
    return {"top": top, "left": left, "width": int(width), "height": int(height)}


def cap_to_region_tuple(cap):
    left = int(cap["left"])
    top = int(cap["top"])
    right = left + int(cap["width"])
    bottom = top + int(cap["height"])
    return (left, top, right, bottom)


class MSSCaptureSource:
    def __init__(self):
        self.name = "mss"
        self._sct = mss.mss()

    def grab(self, cap):
        shot = self._sct.grab(cap)
        raw = shot.raw if MSS_USE_RAW_BUFFER else shot.bgra
        frame_bgra = np.frombuffer(raw, dtype=np.uint8).reshape(
            shot.height, shot.width, 4
        )
        return cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)

    def close(self):
        if self._sct is not None:
            try:
                self._sct.close()
            except Exception:
                pass
            self._sct = None


def create_dxcam_capture():
    post_convert = None
    try:
        cam = dxcam.create(
            device_idx=CAPTURE_DEVICE_IDX,
            output_idx=CAPTURE_OUTPUT_IDX,
            output_color="BGRA",
        )
    except TypeError:
        try:
            cam = dxcam.create(
                device_idx=CAPTURE_DEVICE_IDX,
                output_idx=CAPTURE_OUTPUT_IDX,
                output_color="BGR",
            )
        except TypeError:
            # Older dxcam versions may not accept output_color.
            cam = dxcam.create(
                device_idx=CAPTURE_DEVICE_IDX,
                output_idx=CAPTURE_OUTPUT_IDX,
            )
            post_convert = "rgb_to_bgr"
        else:
            post_convert = None
    else:
        post_convert = "bgra_to_bgr"
    return cam, post_convert


def stop_dxcam_capture(cam):
    if cam is None:
        return
    stop = getattr(cam, "stop", None)
    if callable(stop):
        try:
            stop()
        except RuntimeError:
            pass
        except Exception:
            pass


class DXGISyncCaptureSource:
    def __init__(self):
        if dxcam is None:
            raise RuntimeError("dxcam not installed")
        self._cam, self._post_convert = create_dxcam_capture()
        self.name = "dxcam(dxgi-sync)"

    def grab(self, cap):
        try:
            frame = self._cam.grab(region=cap_to_region_tuple(cap))
        except Exception as exc:
            print(f"[WARN] {self.name} runtime failure ({exc})")
            self.close()
            self._cam, self._post_convert = create_dxcam_capture()
            return None
        if frame is None:
            return None
        if self._post_convert == "rgb_to_bgr":
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if self._post_convert == "bgra_to_bgr":
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame

    def close(self):
        if self._cam is not None:
            stop_dxcam_capture(self._cam)
            self._cam = None


class DXGIAsyncCaptureSource:
    def __init__(self):
        if dxcam is None:
            raise RuntimeError("dxcam not installed")
        self._cam = None
        self._post_convert = None
        self.name = "dxcam(dxgi-async-anchor)"
        self._anchor_cap = None
        self._expected_shape = None
        self._camera_shape = None
        self._restart_count = 0
        self._last_runtime_error_ts = 0.0
        self._ensure_camera()

    def _ensure_camera(self):
        if self._cam is None:
            self._cam, self._post_convert = create_dxcam_capture()

    def _dispose_camera(self):
        if self._cam is not None:
            stop_dxcam_capture(self._cam)
            self._cam = None

    def _reset_runtime_state(self):
        self._dispose_camera()
        self._anchor_cap = None
        self._expected_shape = None
        self._camera_shape = None

    def _handle_runtime_error(self, exc):
        now = time.time()
        if now - self._last_runtime_error_ts > 1.0:
            print(f"[WARN] {self.name} runtime failure; restarting capture ({exc})")
            self._last_runtime_error_ts = now
        self._reset_runtime_state()

    def _compute_anchor_cap(self, cap):
        pad = max(0, int(CAPTURE_DXGI_ANCHOR_PAD))
        width = min(SCREEN_W, int(cap["width"]) + (pad * 2))
        height = min(SCREEN_H, int(cap["height"]) + (pad * 2))
        center_x = int(cap["left"]) + (int(cap["width"]) // 2)
        center_y = int(cap["top"]) + (int(cap["height"]) // 2)
        return build_capture(center_x, center_y, width, height)

    def _cap_inside_anchor(self, cap):
        if self._anchor_cap is None:
            return False

        anchor = self._anchor_cap
        anchor_left = int(anchor["left"])
        anchor_top = int(anchor["top"])
        anchor_right = anchor_left + int(anchor["width"])
        anchor_bottom = anchor_top + int(anchor["height"])

        cap_left = int(cap["left"])
        cap_top = int(cap["top"])
        cap_right = cap_left + int(cap["width"])
        cap_bottom = cap_top + int(cap["height"])

        margin_x = min(
            max(0, int(CAPTURE_DXGI_EDGE_MARGIN)),
            max(0, (int(anchor["width"]) - int(cap["width"])) // 2),
        )
        margin_y = min(
            max(0, int(CAPTURE_DXGI_EDGE_MARGIN)),
            max(0, (int(anchor["height"]) - int(cap["height"])) // 2),
        )

        return (
            (cap_left >= (anchor_left + margin_x))
            and (cap_top >= (anchor_top + margin_y))
            and (cap_right <= (anchor_right - margin_x))
            and (cap_bottom <= (anchor_bottom - margin_y))
        )

    def _restart_capture(self, cap):
        anchor_cap = self._compute_anchor_cap(cap)
        anchor_shape = (int(anchor_cap["height"]), int(anchor_cap["width"]))

        # dxcam binds its internal frame buffer to the start() region size.
        # Recreate the camera when the anchor size changes so stale worker
        # threads cannot write differently shaped frames into a reused buffer.
        if self._camera_shape != anchor_shape:
            self._dispose_camera()
        else:
            stop_dxcam_capture(self._cam)

        self._ensure_camera()
        self._cam.start(
            region=cap_to_region_tuple(anchor_cap),
            target_fps=int(CAPTURE_DXGI_TARGET_FPS),
            video_mode=bool(CAPTURE_DXGI_VIDEO_MODE),
        )
        self._anchor_cap = anchor_cap
        self._expected_shape = anchor_shape
        self._camera_shape = anchor_shape
        self._restart_count += 1

    def _get_anchor_frame(self):
        frame = self._cam.get_latest_frame()
        if frame is None:
            return None
        if self._expected_shape is None:
            return frame
        if frame.shape[:2] == self._expected_shape:
            return frame

        retries = max(0, int(CAPTURE_DXGI_FRAME_SHAPE_RETRIES))
        for _ in range(retries):
            frame = self._cam.get_latest_frame()
            if frame is None:
                continue
            if frame.shape[:2] == self._expected_shape:
                return frame
        return None

    def _crop_from_anchor(self, frame, cap):
        anchor = self._anchor_cap
        if anchor is None:
            return None

        x0 = int(cap["left"]) - int(anchor["left"])
        y0 = int(cap["top"]) - int(anchor["top"])
        x1 = x0 + int(cap["width"])
        y1 = y0 + int(cap["height"])
        crop = frame[y0:y1, x0:x1]
        if crop.shape[:2] != (int(cap["height"]), int(cap["width"])):
            return None

        if self._post_convert == "rgb_to_bgr":
            return cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        if self._post_convert == "bgra_to_bgr":
            return cv2.cvtColor(crop, cv2.COLOR_BGRA2BGR)
        if CAPTURE_DXGI_FORCE_CONTIGUOUS and (not crop.flags.c_contiguous):
            return np.ascontiguousarray(crop)
        return crop

    def grab(self, cap):
        try:
            if (self._anchor_cap is None) or (not self._cap_inside_anchor(cap)):
                self._restart_capture(cap)

            frame = self._get_anchor_frame()
            if frame is None:
                self._restart_capture(cap)
                frame = self._get_anchor_frame()
                if frame is None:
                    return None

            crop = self._crop_from_anchor(frame, cap)
            if crop is not None:
                return crop

            self._restart_capture(cap)
            frame = self._get_anchor_frame()
            if frame is None:
                return None
            return self._crop_from_anchor(frame, cap)
        except Exception as exc:
            self._handle_runtime_error(exc)
            return None

    def close(self):
        if self._cam is not None:
            self._dispose_camera()


def _capture_backend_names():
    names = ["mss"]
    if dxcam is not None:
        names = ["dxcam", "dxcam_sync"] + names
    return names


def _capture_auto_benchmark_region():
    width = int(clamp(max(IMGSZ, BASE_CROP_W, MIN_ACTIVE_CROP_W), 160, SCREEN_W))
    height = int(clamp(max(IMGSZ, BASE_CROP_H, MIN_ACTIVE_CROP_H), 160, SCREEN_H))
    return build_capture(CENTER[0], CENTER[1], width, height)


def _capture_auto_benchmark_regions():
    base = _capture_auto_benchmark_region()
    if not CAPTURE_AUTO_BENCHMARK_DYNAMIC:
        return base

    motion = max(0, int(CAPTURE_AUTO_BENCHMARK_MOTION_PX))
    offsets = (
        (-motion, 0),
        (-motion // 2, motion // 2),
        (0, motion),
        (motion // 2, motion // 2),
        (motion, 0),
        (motion // 2, -motion // 2),
        (0, -motion),
        (-motion // 2, -motion // 2),
    )
    regions = []
    for dx, dy in offsets:
        center_x = CENTER[0] + int(dx)
        center_y = CENTER[1] + int(dy)
        regions.append(build_capture(center_x, center_y, base["width"], base["height"]))
    return regions


def choose_best_capture_backend(seconds, capture_region):
    best = None
    candidates = _capture_backend_names()
    for backend_name in candidates:
        try:
            result = benchmark_capture_backend(backend_name, seconds, capture_region)
        except Exception as e:
            print(f"[CAPTURE-AUTO] {backend_name}: unavailable ({e})")
            continue

        none_ratio = (
            float(result["none"]) / float(result["frames"] + result["none"])
            if (result["frames"] + result["none"]) > 0
            else 1.0
        )
        status = (
            f"fps={result['fps']:.1f} grab={result['grab_ms']:.2f}ms "
            f"none={result['none']} ({none_ratio * 100.0:.1f}%)"
        )
        print(f"[CAPTURE-AUTO] {backend_name}: {status}")
        if none_ratio > float(CAPTURE_AUTO_MAX_NONE_RATIO):
            continue
        if (best is None) or (result["grab_ms"] < best["grab_ms"]):
            best = {
                "backend": backend_name,
                "grab_ms": float(result["grab_ms"]),
            }

    if best is not None:
        return best["backend"]
    return None


def build_capture_source():
    backend = str(CAPTURE_BACKEND).strip().lower()
    if backend not in ("auto", "dxcam", "dxcam_sync", "mss"):
        print(f"[WARN] Invalid CAPTURE_BACKEND='{CAPTURE_BACKEND}', using auto.")
        backend = "auto"

    if backend == "auto" and CAPTURE_AUTO_BENCHMARK:
        bench_region = _capture_auto_benchmark_regions()
        selected = choose_best_capture_backend(
            seconds=CAPTURE_AUTO_BENCHMARK_SECONDS,
            capture_region=bench_region,
        )
        if selected is not None:
            backend = selected
            print(f"[INFO] Capture auto-selected backend: {backend}")

    if backend in ("auto", "dxcam"):
        try:
            return DXGIAsyncCaptureSource()
        except Exception as e:
            if backend == "dxcam":
                print(f"[WARN] DXGI capture init failed ({e}); falling back to MSS.")
            else:
                print(f"[INFO] DXGI capture unavailable ({e}); using MSS.")

    if backend == "dxcam_sync":
        try:
            return DXGISyncCaptureSource()
        except Exception as e:
            print(f"[WARN] DXGI sync capture init failed ({e}); falling back to MSS.")

    return MSSCaptureSource()


def benchmark_capture_backend(backend_name, seconds, capture_region):
    backend_name = str(backend_name).strip().lower()
    source = None
    try:
        if backend_name == "dxcam":
            source = DXGIAsyncCaptureSource()
        elif backend_name == "dxcam_sync":
            source = DXGISyncCaptureSource()
        elif backend_name == "mss":
            source = MSSCaptureSource()
        else:
            raise ValueError(f"Unsupported backend '{backend_name}'")

        frames = 0
        none_count = 0
        grab_total_s = 0.0
        regions = None
        if isinstance(capture_region, (list, tuple)) and capture_region:
            if isinstance(capture_region[0], dict):
                regions = list(capture_region)
        start = time.perf_counter()
        region_idx = 0
        while (time.perf_counter() - start) < float(seconds):
            if regions is None:
                active_region = capture_region
            else:
                active_region = regions[region_idx % len(regions)]
                region_idx += 1
            t0 = time.perf_counter()
            frame = source.grab(active_region)
            dt = time.perf_counter() - t0
            if frame is None:
                none_count += 1
                continue
            frames += 1
            grab_total_s += dt

        elapsed = max(1e-6, time.perf_counter() - start)
        return {
            "backend": backend_name,
            "frames": frames,
            "fps": frames / elapsed,
            "grab_ms": (grab_total_s * 1000.0 / frames) if frames else 0.0,
            "none": none_count,
        }
    finally:
        if source is not None:
            source.close()


def run_capture_benchmark():
    seconds = max(0.5, float(CAPTURE_BENCHMARK_SECONDS))
    w = int(clamp(CAPTURE_BENCHMARK_W, 160, SCREEN_W))
    h = int(clamp(CAPTURE_BENCHMARK_H, 160, SCREEN_H))
    cap = build_capture(CENTER[0], CENTER[1], w, h)
    dynamic_caps = _capture_auto_benchmark_regions()
    print(f"[INFO] Running capture benchmark ({seconds:.1f}s, {w}x{h}) ...")
    for backend_name in _capture_backend_names():
        try:
            result = benchmark_capture_backend(backend_name, seconds, cap)
            print(
                f"[CAPTURE-BENCH] {result['backend']}: "
                f"{result['fps']:.1f} fps, grab={result['grab_ms']:.2f} ms, none={result['none']}"
            )
        except Exception as e:
            print(f"[CAPTURE-BENCH] {backend_name}: unavailable ({e})")
        try:
            result = benchmark_capture_backend(backend_name, seconds, dynamic_caps)
            print(
                f"[CAPTURE-BENCH-DYN] {result['backend']}: "
                f"{result['fps']:.1f} fps, grab={result['grab_ms']:.2f} ms, none={result['none']}"
            )
        except Exception as e:
            print(f"[CAPTURE-BENCH-DYN] {backend_name}: unavailable ({e})")


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


class RuntimePerf:
    def __init__(self):
        self.lock = threading.Lock()
        self._reset_locked(time.time())

    def _reset_locked(self, now):
        self.window_start = float(now)
        self.capture_frames = 0
        self.capture_none = 0
        self.capture_grab_s = 0.0

        self.infer_frames = 0
        self.infer_stale = 0
        self.infer_found = 0
        self.infer_loop_s = 0.0
        self.infer_frame_age_s = 0.0
        self.infer_frame_age_max_s = 0.0
        self.infer_yolo_calls = 0
        self.infer_yolo_s = 0.0
        self.infer_tracker_calls = 0
        self.infer_tracker_s = 0.0
        self.infer_backend_samples = 0
        self.infer_backend_pre_s = 0.0
        self.infer_backend_exec_s = 0.0
        self.infer_backend_post_s = 0.0
        self.infer_cmd_latency_s = 0.0
        self.infer_cmd_samples = 0

        self.control_cmds = 0
        self.control_sent = 0
        self.control_stale_drop = 0
        self.control_mode_drop = 0
        self.control_send_s = 0.0
        self.control_cmd_age_s = 0.0
        self.control_total_latency_s = 0.0
        self.control_latency_samples = 0
        self.control_total_latency_full_s = 0.0
        self.control_latency_full_samples = 0
        self.control_total_apply_latency_s = 0.0
        self.control_apply_latency_samples = 0
        self.control_total_apply_latency_full_s = 0.0
        self.control_apply_latency_full_samples = 0

    def record_capture(self, grab_s, is_none=False):
        with self.lock:
            if is_none:
                self.capture_none += 1
                return
            self.capture_frames += 1
            self.capture_grab_s += max(0.0, float(grab_s))

    def record_inference(
        self,
        frame_age_s,
        loop_s,
        stale_drop=False,
        target_found=False,
        yolo_s=0.0,
        tracker_s=0.0,
        backend_pre_s=None,
        backend_exec_s=None,
        backend_post_s=None,
        cmd_latency_s=None,
    ):
        with self.lock:
            self.infer_frames += 1
            self.infer_loop_s += max(0.0, float(loop_s))

            frame_age_s = max(0.0, float(frame_age_s))
            self.infer_frame_age_s += frame_age_s
            if frame_age_s > self.infer_frame_age_max_s:
                self.infer_frame_age_max_s = frame_age_s

            if stale_drop:
                self.infer_stale += 1
            if target_found:
                self.infer_found += 1

            if yolo_s > 0.0:
                self.infer_yolo_calls += 1
                self.infer_yolo_s += float(yolo_s)
            if tracker_s > 0.0:
                self.infer_tracker_calls += 1
                self.infer_tracker_s += float(tracker_s)

            if (
                (backend_pre_s is not None)
                or (backend_exec_s is not None)
                or (backend_post_s is not None)
            ):
                self.infer_backend_samples += 1
                self.infer_backend_pre_s += max(0.0, float(backend_pre_s or 0.0))
                self.infer_backend_exec_s += max(0.0, float(backend_exec_s or 0.0))
                self.infer_backend_post_s += max(0.0, float(backend_post_s or 0.0))

            if cmd_latency_s is not None:
                self.infer_cmd_samples += 1
                self.infer_cmd_latency_s += max(0.0, float(cmd_latency_s))

    def record_control(
        self,
        cmd_age_s,
        sent=False,
        send_s=0.0,
        stale_drop=False,
        mode_drop=False,
        total_latency_s=None,
        total_latency_full_s=None,
        total_apply_latency_s=None,
        total_apply_latency_full_s=None,
    ):
        with self.lock:
            self.control_cmds += 1
            if stale_drop:
                self.control_stale_drop += 1
            if mode_drop:
                self.control_mode_drop += 1
            if not sent:
                return

            self.control_sent += 1
            self.control_send_s += max(0.0, float(send_s))
            self.control_cmd_age_s += max(0.0, float(cmd_age_s))

            if total_latency_s is not None:
                self.control_latency_samples += 1
                self.control_total_latency_s += max(0.0, float(total_latency_s))
            if total_latency_full_s is not None:
                self.control_latency_full_samples += 1
                self.control_total_latency_full_s += max(0.0, float(total_latency_full_s))
            if total_apply_latency_s is not None:
                self.control_apply_latency_samples += 1
                self.control_total_apply_latency_s += max(0.0, float(total_apply_latency_s))
            if total_apply_latency_full_s is not None:
                self.control_apply_latency_full_samples += 1
                self.control_total_apply_latency_full_s += max(0.0, float(total_apply_latency_full_s))

    def snapshot(self, min_interval_s=PERF_LOG_INTERVAL_S):
        now = time.time()
        with self.lock:
            elapsed = now - self.window_start
            if elapsed < float(min_interval_s):
                return None

            cap_fps = self.capture_frames / elapsed if elapsed > 0.0 else 0.0
            cap_grab_ms = (self.capture_grab_s * 1000.0 / self.capture_frames) if self.capture_frames else 0.0

            infer_fps = self.infer_frames / elapsed if elapsed > 0.0 else 0.0
            infer_loop_ms = (self.infer_loop_s * 1000.0 / self.infer_frames) if self.infer_frames else 0.0
            infer_age_ms = (self.infer_frame_age_s * 1000.0 / self.infer_frames) if self.infer_frames else 0.0
            infer_age_max_ms = self.infer_frame_age_max_s * 1000.0
            infer_lock_rate = (self.infer_found / self.infer_frames) if self.infer_frames else 0.0

            yolo_hz = self.infer_yolo_calls / elapsed if elapsed > 0.0 else 0.0
            yolo_ms = (self.infer_yolo_s * 1000.0 / self.infer_yolo_calls) if self.infer_yolo_calls else 0.0
            tracker_hz = self.infer_tracker_calls / elapsed if elapsed > 0.0 else 0.0
            tracker_ms = (self.infer_tracker_s * 1000.0 / self.infer_tracker_calls) if self.infer_tracker_calls else 0.0
            infer_backend_pre_ms = (
                self.infer_backend_pre_s * 1000.0 / self.infer_backend_samples
                if self.infer_backend_samples
                else 0.0
            )
            infer_backend_exec_ms = (
                self.infer_backend_exec_s * 1000.0 / self.infer_backend_samples
                if self.infer_backend_samples
                else 0.0
            )
            infer_backend_post_ms = (
                self.infer_backend_post_s * 1000.0 / self.infer_backend_samples
                if self.infer_backend_samples
                else 0.0
            )
            infer_cmd_ms = (
                self.infer_cmd_latency_s * 1000.0 / self.infer_cmd_samples
                if self.infer_cmd_samples
                else 0.0
            )

            control_send_hz = self.control_sent / elapsed if elapsed > 0.0 else 0.0
            control_send_ms = (
                self.control_send_s * 1000.0 / self.control_sent if self.control_sent else 0.0
            )
            control_cmd_age_ms = (
                self.control_cmd_age_s * 1000.0 / self.control_sent if self.control_sent else 0.0
            )
            control_total_latency_ms = (
                self.control_total_latency_s * 1000.0 / self.control_latency_samples
                if self.control_latency_samples
                else 0.0
            )
            control_total_latency_full_ms = (
                self.control_total_latency_full_s * 1000.0 / self.control_latency_full_samples
                if self.control_latency_full_samples
                else 0.0
            )
            control_total_apply_latency_ms = (
                self.control_total_apply_latency_s * 1000.0 / self.control_apply_latency_samples
                if self.control_apply_latency_samples
                else 0.0
            )
            control_total_apply_latency_full_ms = (
                self.control_total_apply_latency_full_s * 1000.0 / self.control_apply_latency_full_samples
                if self.control_apply_latency_full_samples
                else 0.0
            )

            snapshot = {
                "elapsed_s": elapsed,
                "cap_fps": cap_fps,
                "cap_grab_ms": cap_grab_ms,
                "cap_none": self.capture_none,
                "infer_fps": infer_fps,
                "infer_loop_ms": infer_loop_ms,
                "infer_age_ms": infer_age_ms,
                "infer_age_max_ms": infer_age_max_ms,
                "infer_stale": self.infer_stale,
                "infer_lock_rate": infer_lock_rate,
                "yolo_hz": yolo_hz,
                "yolo_ms": yolo_ms,
                "tracker_hz": tracker_hz,
                "tracker_ms": tracker_ms,
                "infer_backend_samples": self.infer_backend_samples,
                "infer_backend_pre_ms": infer_backend_pre_ms,
                "infer_backend_exec_ms": infer_backend_exec_ms,
                "infer_backend_post_ms": infer_backend_post_ms,
                "infer_cmd_ms": infer_cmd_ms,
                "control_send_hz": control_send_hz,
                "control_send_ms": control_send_ms,
                "control_cmd_age_ms": control_cmd_age_ms,
                "control_total_latency_ms": control_total_latency_ms,
                "control_total_latency_full_ms": control_total_latency_full_ms,
                "control_total_apply_latency_ms": control_total_apply_latency_ms,
                "control_total_apply_latency_full_ms": control_total_apply_latency_full_ms,
                "control_stale_drop": self.control_stale_drop,
                "control_mode_drop": self.control_mode_drop,
            }
            self._reset_locked(now)
            return snapshot


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
        anti_windup_gain=PID_ANTI_WINDUP_GAIN,
        derivative_alpha=PID_DERIVATIVE_ALPHA,
        output_limit=PID_OUTPUT_MAX,
    ):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.integral_limit = float(integral_limit)
        self.anti_windup_gain = clamp(float(anti_windup_gain), 0.0, 1.0)
        self.derivative_alpha = clamp(float(derivative_alpha), 0.0, 1.0)
        self.output_limit = float(output_limit) if output_limit is not None else None
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.previous_measurement = 0.0
        self.derivative = 0.0
        self.has_prev_measurement = False

    def configure(
        self,
        *,
        kp=None,
        ki=None,
        kd=None,
        integral_limit=None,
        anti_windup_gain=None,
        derivative_alpha=None,
        output_limit=None,
    ):
        if kp is not None:
            self.kp = float(kp)
        if ki is not None:
            self.ki = float(ki)
        if kd is not None:
            self.kd = float(kd)
        if integral_limit is not None:
            self.integral_limit = float(integral_limit)
        if anti_windup_gain is not None:
            self.anti_windup_gain = clamp(float(anti_windup_gain), 0.0, 1.0)
        if derivative_alpha is not None:
            self.derivative_alpha = clamp(float(derivative_alpha), 0.0, 1.0)
        if output_limit is not None:
            self.output_limit = float(output_limit)

    def update(self, target, measurement, dt, *, integrate=True, integral_decay=1.0):
        dt = max(1e-4, float(dt))
        target = float(target)
        measurement = float(measurement)
        error = target - measurement
        integral_decay = clamp(float(integral_decay), 0.0, 1.0)

        p_term = self.kp * error
        integral_term = float(self.integral)
        if integral_decay < 1.0:
            integral_term *= integral_decay
            if abs(integral_term) < 1e-6:
                integral_term = 0.0
        if integrate:
            integral_term = clamp(
                integral_term + (self.ki * error * dt),
                -self.integral_limit,
                self.integral_limit,
            )

        if self.has_prev_measurement:
            derivative_raw = (measurement - self.previous_measurement) / dt
            self.derivative = (
                (self.derivative_alpha * self.derivative)
                + ((1.0 - self.derivative_alpha) * derivative_raw)
            )
        else:
            self.derivative = 0.0
            self.has_prev_measurement = True

        d_term = -self.kd * self.derivative
        output = p_term + integral_term + d_term
        if self.output_limit is not None:
            clamped_output = clamp(output, -self.output_limit, self.output_limit)
            if self.anti_windup_gain > 0.0:
                # Bleed off hidden integral whenever the actuator is saturated.
                integral_term += (clamped_output - output) * self.anti_windup_gain
                integral_term = clamp(integral_term, -self.integral_limit, self.integral_limit)
                output = p_term + integral_term + d_term
                clamped_output = clamp(output, -self.output_limit, self.output_limit)
            output = clamped_output
        self.integral = integral_term
        self.previous_measurement = measurement
        return output


class KalmanTargetTracker:
    def __init__(
        self,
        *,
        initial_uncertainty=KALMAN_INITIAL_UNCERTAINTY,
        process_noise=KALMAN_PROCESS_NOISE_BASE,
        measurement_noise=KALMAN_MEAS_NOISE_BASE,
    ):
        self.initial_uncertainty = float(initial_uncertainty)
        self.h = np.zeros((2, 4), dtype=np.float32)
        self.h[0, 0] = 1.0
        self.h[1, 1] = 1.0
        self.identity = np.eye(4, dtype=np.float32)
        self.q = np.diag(
            [
                np.float32(KALMAN_PROCESS_NOISE_POSITION),
                np.float32(KALMAN_PROCESS_NOISE_POSITION),
                np.float32(max(1e-6, float(process_noise))),
                np.float32(max(1e-6, float(process_noise))),
            ]
        ).astype(np.float32)
        self.r = np.eye(2, dtype=np.float32) * np.float32(max(1e-6, float(measurement_noise)))
        self.reset()

    def configure(
        self,
        *,
        process_noise=None,
        measurement_noise=None,
        smoothing_alpha=None,
        velocity_alpha=None,
    ):
        if process_noise is not None:
            process_noise = max(1e-6, float(process_noise))
            self.q = np.diag(
                [
                    np.float32(KALMAN_PROCESS_NOISE_POSITION),
                    np.float32(KALMAN_PROCESS_NOISE_POSITION),
                    np.float32(process_noise),
                    np.float32(process_noise),
                ]
            ).astype(np.float32)
        if measurement_noise is not None:
            measurement_noise = max(1e-6, float(measurement_noise))
            self.r = np.eye(2, dtype=np.float32) * np.float32(measurement_noise)

    def reset(self):
        self.x = np.zeros((4, 1), dtype=np.float32)
        self.p = np.eye(4, dtype=np.float32) * np.float32(self.initial_uncertainty)
        self.initialized = False
        self.measurement_updates = 0

    def initialize(self, x, y):
        self.x[:] = 0.0
        self.x[0, 0] = np.float32(x)
        self.x[1, 0] = np.float32(y)
        self.p = np.eye(4, dtype=np.float32) * np.float32(self.initial_uncertainty)
        self.initialized = True
        self.measurement_updates = 1

    def _transition(self, dt):
        dt = float(np.clip(dt, KALMAN_MIN_DT, KALMAN_MAX_DT))
        return np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def predict(self, dt):
        if not self.initialized:
            return self.x
        f = self._transition(dt)
        self.x = f @ self.x
        self.p = (f @ self.p @ f.T) + self.q
        return self.x

    def update(self, mx, my):
        if not self.initialized:
            self.initialize(mx, my)
            return self.x

        z = np.array([[mx], [my]], dtype=np.float32)
        innovation = z - (self.h @ self.x)
        innovation_cov = (self.h @ self.p @ self.h.T) + self.r
        kalman_gain = self.p @ self.h.T @ np.linalg.inv(innovation_cov)
        self.x = self.x + (kalman_gain @ innovation)
        self.p = (self.identity - (kalman_gain @ self.h)) @ self.p
        self.measurement_updates += 1
        return self.x

    def get_state(self):
        x, y, vx, vy = self.x.reshape(-1)
        return (
            float(x),
            float(y),
            float(vx),
            float(vy),
            0.0,
            0.0,
        )

    def predict_ahead(self, lead_time):
        x, y, vx, vy, _, _ = self.get_state()
        lead_time = max(0.0, float(lead_time))
        return (
            x + (vx * lead_time),
            y + (vy * lead_time),
        )

    def feedforward_scale(self, *, min_updates=FEEDFORWARD_MIN_UPDATES, ramp_updates=FEEDFORWARD_RAMP_UPDATES):
        updates = max(0, int(self.measurement_updates) - int(min_updates))
        if updates <= 0:
            return 0.0
        return clamp(updates / max(1.0, float(ramp_updates)), 0.0, 1.0)


TRACKING_STRATEGY_OPTIONS = (
    {"value": TRACKING_STRATEGY_RAW, "label": "Raw Detection"},
    {"value": TRACKING_STRATEGY_KALMAN, "label": "Kalman"},
    {"value": TRACKING_STRATEGY_EMA, "label": "EMA + Velocity"},
    {"value": TRACKING_STRATEGY_DEMA, "label": "DEMA"},
    {"value": TRACKING_STRATEGY_RAW_DELTA, "label": "Raw + Vel EMA"},
)
TRACKING_STRATEGY_LABELS = {
    item["value"]: item["label"] for item in TRACKING_STRATEGY_OPTIONS
}


def normalize_tracking_strategy(value):
    normalized = str(value).strip().lower()
    if normalized in TRACKING_STRATEGY_LABELS:
        return normalized
    return TRACKING_STRATEGY_DEFAULT


class ObservedMotionTracker:
    def __init__(
        self,
        mode=TRACKING_STRATEGY_EMA,
        *,
        smoothing_alpha=TRACKING_POSITION_ALPHA,
        velocity_alpha=TRACKING_VELOCITY_ALPHA,
        max_speed=KALMAN_MAX_SPEED_PX_S,
    ):
        self.mode = normalize_tracking_strategy(mode)
        self.max_speed = float(max_speed)
        self.smoothing_alpha = clamp(float(smoothing_alpha), 1e-3, 1.0)
        self.velocity_alpha = clamp(float(velocity_alpha), 0.0, 1.0)
        self.reset()

    def configure(
        self,
        *,
        process_noise=None,
        measurement_noise=None,
        smoothing_alpha=None,
        velocity_alpha=None,
    ):
        if smoothing_alpha is not None:
            self.smoothing_alpha = clamp(float(smoothing_alpha), 1e-3, 1.0)
        if velocity_alpha is not None:
            self.velocity_alpha = clamp(float(velocity_alpha), 0.0, 1.0)

    def reset(self):
        self.x = 0.0
        self.y = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.raw_x = 0.0
        self.raw_y = 0.0
        self.ema1_x = 0.0
        self.ema1_y = 0.0
        self.ema2_x = 0.0
        self.ema2_y = 0.0
        self.last_dt = float(KALMAN_MIN_DT)
        self.initialized = False
        self.measurement_updates = 0

    def initialize(self, x, y):
        x = float(x)
        y = float(y)
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.raw_x = x
        self.raw_y = y
        self.ema1_x = x
        self.ema1_y = y
        self.ema2_x = x
        self.ema2_y = y
        self.initialized = True
        self.measurement_updates = 1

    def predict(self, dt):
        self.last_dt = float(np.clip(dt, KALMAN_MIN_DT, KALMAN_MAX_DT))
        return self.get_state()

    def update(self, mx, my):
        if not self.initialized:
            self.initialize(mx, my)
            return self.get_state()

        dt = max(KALMAN_MIN_DT, float(self.last_dt))
        mx = float(mx)
        my = float(my)

        if self.mode == TRACKING_STRATEGY_RAW:
            next_x = mx
            next_y = my
            raw_vx = 0.0
            raw_vy = 0.0
        elif self.mode == TRACKING_STRATEGY_RAW_DELTA:
            next_x = mx
            next_y = my
            raw_vx = (mx - float(self.raw_x)) / dt
            raw_vy = (my - float(self.raw_y)) / dt
        elif self.mode == TRACKING_STRATEGY_DEMA:
            alpha = self.smoothing_alpha
            self.ema1_x += alpha * (mx - self.ema1_x)
            self.ema1_y += alpha * (my - self.ema1_y)
            self.ema2_x += alpha * (self.ema1_x - self.ema2_x)
            self.ema2_y += alpha * (self.ema1_y - self.ema2_y)
            next_x = (2.0 * self.ema1_x) - self.ema2_x
            next_y = (2.0 * self.ema1_y) - self.ema2_y
            raw_vx = (mx - float(self.raw_x)) / dt
            raw_vy = (my - float(self.raw_y)) / dt
        else:
            alpha = self.smoothing_alpha
            self.ema1_x += alpha * (mx - self.ema1_x)
            self.ema1_y += alpha * (my - self.ema1_y)
            next_x = self.ema1_x
            next_y = self.ema1_y
            raw_vx = (mx - float(self.raw_x)) / dt
            raw_vy = (my - float(self.raw_y)) / dt

        raw_vx = clamp(float(raw_vx), -self.max_speed, self.max_speed)
        raw_vy = clamp(float(raw_vy), -self.max_speed, self.max_speed)
        if self.mode == TRACKING_STRATEGY_RAW:
            self.vx = 0.0
            self.vy = 0.0
        elif self.measurement_updates <= 1:
            self.vx = raw_vx
            self.vy = raw_vy
        else:
            blend = self.velocity_alpha
            self.vx = ((1.0 - blend) * float(self.vx)) + (blend * raw_vx)
            self.vy = ((1.0 - blend) * float(self.vy)) + (blend * raw_vy)

        self.x = float(next_x)
        self.y = float(next_y)
        self.raw_x = mx
        self.raw_y = my
        self.measurement_updates += 1
        return self.get_state()

    def get_state(self):
        return (
            float(self.x),
            float(self.y),
            float(self.vx),
            float(self.vy),
            0.0,
            0.0,
        )

    def predict_ahead(self, lead_time):
        x, y, vx, vy, _, _ = self.get_state()
        lead_time = max(0.0, float(lead_time))
        return (
            x + (vx * lead_time),
            y + (vy * lead_time),
        )

    def feedforward_scale(self, *, min_updates=FEEDFORWARD_MIN_UPDATES, ramp_updates=FEEDFORWARD_RAMP_UPDATES):
        if self.mode == TRACKING_STRATEGY_RAW:
            return 0.0
        if self.mode == TRACKING_STRATEGY_RAW_DELTA:
            warm_min = 1
            warm_ramp = 3
        else:
            warm_min = max(1, min(2, int(min_updates)))
            warm_ramp = max(2, min(6, int(ramp_updates)))
        updates = max(0, int(self.measurement_updates) - warm_min)
        if updates <= 0:
            return 0.0
        return clamp(updates / max(1.0, float(warm_ramp)), 0.0, 1.0)


def build_motion_tracker(
    strategy_name,
    *,
    process_noise=KALMAN_PROCESS_NOISE_BASE,
    measurement_noise=KALMAN_MEAS_NOISE_BASE,
    smoothing_alpha=TRACKING_POSITION_ALPHA,
    velocity_alpha=TRACKING_VELOCITY_ALPHA,
):
    strategy_name = normalize_tracking_strategy(strategy_name)
    if strategy_name == TRACKING_STRATEGY_KALMAN:
        return KalmanTargetTracker(
            process_noise=process_noise,
            measurement_noise=measurement_noise,
        )
    return ObservedMotionTracker(
        mode=strategy_name,
        smoothing_alpha=smoothing_alpha,
        velocity_alpha=velocity_alpha,
    )


def pick_sticky_target(detections, screen_center, locked_point=None, *, sticky_bias=STICKY_BIAS_PX):
    if not detections:
        return None, {"locked_idx": None, "selected_idx": None, "switched": False}

    locked_idx = None
    if locked_point is not None:
        lock_x, lock_y = locked_point
        locked_distances = [
            float(np.hypot(det["x"] - lock_x, det["y"] - lock_y)) for det in detections
        ]
        locked_idx = int(np.argmin(locked_distances))

    center_x, center_y = screen_center
    best_det = None
    best_idx = None
    best_score = float("inf")
    best_conf = -1.0
    for idx, det in enumerate(detections):
        score = float(np.hypot(det["x"] - center_x, det["y"] - center_y))
        if locked_idx is not None and idx == locked_idx:
            score -= float(sticky_bias)
        if score < best_score or (abs(score - best_score) <= 1e-6 and det["conf"] > best_conf):
            best_det = det
            best_idx = idx
            best_score = score
            best_conf = float(det["conf"])
    return best_det, {
        "locked_idx": locked_idx,
        "selected_idx": best_idx,
        "switched": (locked_idx is not None and best_idx is not None and best_idx != locked_idx),
    }


def reset_ego_motion_state(shared_lock, state):
    with shared_lock:
        state["ctrl_sent_vx_ema"] = 0.0
        state["ctrl_sent_vy_ema"] = 0.0
        state["ctrl_last_send_tick"] = 0.0


PID_RUNTIME_PARAM_SPECS = (
    {"key": "enable", "label": "PID Enabled", "type": "bool"},
    {
        "key": "tracking_strategy",
        "label": "Tracking Strategy",
        "type": "select",
        "options": list(TRACKING_STRATEGY_OPTIONS),
    },
    {"key": "tracking_alpha", "label": "Position Alpha", "type": "number", "step": 0.001},
    {"key": "tracking_velocity_alpha", "label": "Velocity Beta", "type": "number", "step": 0.001},
    {"key": "kp", "label": "Kp (X/Y)", "type": "number", "step": 0.001},
    {"key": "ki", "label": "Ki (X/Y)", "type": "number", "step": 0.001},
    {"key": "kd", "label": "Kd (X/Y)", "type": "number", "step": 0.001},
    {"key": "integral_limit", "label": "Integral Limit", "type": "number", "step": 1.0},
    {"key": "anti_windup_gain", "label": "Anti-Windup Gain", "type": "number", "step": 0.001},
    {"key": "derivative_alpha", "label": "Derivative Alpha", "type": "number", "step": 0.001},
    {"key": "output_limit", "label": "PID Output Limit", "type": "number", "step": 1.0},
    {"key": "sticky_bias_px", "label": "Sticky Bias (px)", "type": "number", "step": 1.0},
    {"key": "prediction_time", "label": "Lead Time (s)", "type": "number", "step": 0.001},
    {"key": "target_max_lost_frames", "label": "Max Lost Frames", "type": "number", "step": 1.0},
    {"key": "kalman_process_noise", "label": "Kalman Q (vel)", "type": "number", "step": 0.001},
    {"key": "kalman_measurement_noise", "label": "Kalman R", "type": "number", "step": 0.001},
    {"key": "model_conf", "label": "Model Conf", "type": "number", "step": 0.001},
    {"key": "detection_min_conf", "label": "Detection Min Conf", "type": "number", "step": 0.001},
    {"key": "triggerbot_enable", "label": "Trigger Bot", "type": "bool"},
    {"key": "triggerbot_box_percent", "label": "Trigger Box (%)", "type": "number", "step": 1.0},
    {"key": "triggerbot_click_hold_s", "label": "Trigger Hold (s)", "type": "number", "step": 0.001},
    {"key": "triggerbot_click_cooldown_s", "label": "Trigger Cooldown (s)", "type": "number", "step": 0.001},
    {"key": "recoil_compensation_y_px", "label": "Recoil Comp Y (px)", "type": "number", "step": 1.0},
    {"key": "ghub_max_step", "label": "GHUB Max Step", "type": "number", "step": 1.0},
    {
        "key": "recoil_tune_fallback_ignore_mode_check",
        "label": "F7 Ignore Mode Check",
        "type": "bool",
    },
    {"key": "ego_motion_comp_enable", "label": "Ego Motion Comp", "type": "bool"},
    {"key": "ego_motion_comp_gain_x", "label": "Ego Comp Gain X", "type": "number", "step": 0.001},
    {"key": "ego_motion_comp_gain_y", "label": "Ego Comp Gain Y", "type": "number", "step": 0.001},
    {"key": "ego_motion_error_gate_enable", "label": "Ego Error Gate", "type": "bool"},
    {"key": "ego_motion_error_gate_px", "label": "Ego Gate Error (px)", "type": "number", "step": 1.0},
    {"key": "ego_motion_error_gate_normalize_by_box", "label": "Ego Gate Normalize By Box", "type": "bool"},
    {"key": "ego_motion_error_gate_norm_threshold", "label": "Ego Gate Norm Threshold", "type": "number", "step": 0.01},
    {"key": "ego_motion_reset_on_switch", "label": "Reset Ego On Switch", "type": "bool"},
)


def build_pid_runtime_defaults():
    return {
        "enable": bool(PID_ENABLE),
        "tracking_strategy": str(TRACKING_STRATEGY_DEFAULT),
        "tracking_alpha": float(TRACKING_POSITION_ALPHA),
        "tracking_velocity_alpha": float(TRACKING_VELOCITY_ALPHA),
        "kp": float(PID_KP),
        "ki": float(PID_KI),
        "kd": float(PID_KD),
        "integral_limit": float(PID_INTEGRAL_LIMIT),
        "anti_windup_gain": float(PID_ANTI_WINDUP_GAIN),
        "derivative_alpha": float(PID_DERIVATIVE_ALPHA),
        "output_limit": float(PID_OUTPUT_MAX),
        "sticky_bias_px": float(STICKY_BIAS_PX),
        "prediction_time": float(PREDICTION_TIME),
        "target_max_lost_frames": float(TARGET_MAX_LOST_FRAMES),
        "kalman_process_noise": float(KALMAN_PROCESS_NOISE_BASE),
        "kalman_measurement_noise": float(KALMAN_MEAS_NOISE_BASE),
        "model_conf": float(CONF),
        "detection_min_conf": float(DETECTION_MIN_CONF),
        "triggerbot_enable": bool(TRIGGERBOT_ENABLE_DEFAULT),
        "triggerbot_box_percent": float(TRIGGERBOT_BOX_PERCENT_DEFAULT),
        "triggerbot_click_hold_s": float(TRIGGERBOT_CLICK_HOLD_S),
        "triggerbot_click_cooldown_s": float(TRIGGERBOT_CLICK_COOLDOWN_S),
        "recoil_compensation_y_px": float(RECOIL_COMPENSATION_Y_PX),
        "ghub_max_step": float(GHUB_MAX_STEP),
        "recoil_tune_fallback_ignore_mode_check": bool(
            RECOIL_TUNE_FALLBACK_IGNORE_MODE_CHECK_DEFAULT
        ),
        "ego_motion_comp_enable": bool(EGO_MOTION_COMP_ENABLE),
        "ego_motion_comp_gain_x": float(EGO_MOTION_COMP_GAIN_X),
        "ego_motion_comp_gain_y": float(EGO_MOTION_COMP_GAIN_Y),
        "ego_motion_error_gate_enable": bool(EGO_MOTION_ERROR_GATE_ENABLE),
        "ego_motion_error_gate_px": float(EGO_MOTION_ERROR_GATE_PX),
        "ego_motion_error_gate_normalize_by_box": bool(EGO_MOTION_ERROR_GATE_NORMALIZE_BY_BOX),
        "ego_motion_error_gate_norm_threshold": float(EGO_MOTION_ERROR_GATE_NORM_THRESHOLD),
        "ego_motion_reset_on_switch": bool(EGO_MOTION_RESET_ON_SWITCH),
    }


PID_RESET_RUNTIME_KEYS = {
    "enable",
    "kp",
    "ki",
    "kd",
    "integral_limit",
    "anti_windup_gain",
    "derivative_alpha",
    "output_limit",
}


class RuntimePIDConfig:
    def __init__(self):
        self._lock = threading.Lock()
        self._values = build_pid_runtime_defaults()
        self._version = 0
        self._reset_token = 0

    def snapshot(self):
        with self._lock:
            snapshot = dict(self._values)
            snapshot["tracking_strategy"] = normalize_tracking_strategy(
                snapshot.get("tracking_strategy", TRACKING_STRATEGY_DEFAULT)
            )
            snapshot["use_kalman"] = snapshot["tracking_strategy"] == TRACKING_STRATEGY_KALMAN
            snapshot["version"] = int(self._version)
            snapshot["reset_token"] = int(self._reset_token)
            return snapshot

    def update(self, updates, *, reset_pid=True):
        if not isinstance(updates, dict):
            raise ValueError("Runtime update payload must be a JSON object.")

        normalized = {}
        spec_by_key = {item["key"]: item for item in PID_RUNTIME_PARAM_SPECS}
        valid_keys = set(spec_by_key)
        for key, value in updates.items():
            if key == "use_kalman":
                if isinstance(value, str):
                    lowered = value.strip().lower()
                    if lowered in ("1", "true", "yes", "on"):
                        value = True
                    elif lowered in ("0", "false", "no", "off"):
                        value = False
                    else:
                        raise ValueError(f"Invalid boolean value for '{key}'.")
                normalized["tracking_strategy"] = (
                    TRACKING_STRATEGY_KALMAN if bool(value) else TRACKING_STRATEGY_RAW
                )
                continue
            if key not in valid_keys:
                raise ValueError(f"Unknown runtime parameter '{key}'.")
            field_type = spec_by_key[key].get("type")
            if field_type == "bool":
                if isinstance(value, str):
                    lowered = value.strip().lower()
                    if lowered in ("1", "true", "yes", "on"):
                        normalized[key] = True
                    elif lowered in ("0", "false", "no", "off"):
                        normalized[key] = False
                    else:
                        raise ValueError(f"Invalid boolean value for '{key}'.")
                else:
                    normalized[key] = bool(value)
                continue
            if field_type == "select":
                valid_values = {str(item["value"]) for item in spec_by_key[key].get("options", ())}
                normalized_value = str(value).strip().lower()
                if normalized_value not in valid_values:
                    raise ValueError(f"Invalid value for '{key}'.")
                normalized[key] = normalized_value
                continue
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid numeric value for '{key}'.") from exc
            if not np.isfinite(numeric_value):
                raise ValueError(f"Non-finite value is not allowed for '{key}'.")
            if key in (
                "derivative_alpha",
                "tracking_alpha",
                "tracking_velocity_alpha",
                "anti_windup_gain",
                "model_conf",
                "detection_min_conf",
            ):
                numeric_value = clamp(numeric_value, 0.0, 1.0)
            elif key in (
                "sticky_bias_px",
                "prediction_time",
                "kalman_process_noise",
                "kalman_measurement_noise",
                "ego_motion_error_gate_px",
                "ego_motion_error_gate_norm_threshold",
            ):
                numeric_value = max(1e-6, numeric_value)
            elif key in ("triggerbot_click_hold_s", "triggerbot_click_cooldown_s"):
                numeric_value = max(0.0, numeric_value)
            elif key in ("ego_motion_comp_gain_x", "ego_motion_comp_gain_y"):
                numeric_value = max(0.0, numeric_value)
            elif key in ("target_max_lost_frames", "ghub_max_step"):
                numeric_value = float(max(1, int(round(numeric_value))))
            elif key == "triggerbot_box_percent":
                numeric_value = clamp(numeric_value, 1.0, 100.0)
            normalized[key] = numeric_value

        with self._lock:
            changed = False
            changed_keys = set()
            for key, value in normalized.items():
                current = self._values.get(key)
                if isinstance(value, float) and isinstance(current, float):
                    if abs(value - current) <= 1e-12:
                        continue
                elif value == current:
                    continue
                self._values[key] = value
                changed = True
                changed_keys.add(key)
            if changed:
                self._version += 1
            if reset_pid and changed_keys.intersection(PID_RESET_RUNTIME_KEYS):
                self._reset_token += 1
            snapshot = dict(self._values)
            snapshot["version"] = int(self._version)
            snapshot["reset_token"] = int(self._reset_token)
            return snapshot

    def request_reset(self):
        with self._lock:
            self._reset_token += 1
            return self._reset_token


def apply_pid_runtime_to_controllers(pid_x, pid_y, snapshot):
    common_args = {
        "integral_limit": snapshot["integral_limit"],
        "anti_windup_gain": snapshot["anti_windup_gain"],
        "derivative_alpha": snapshot["derivative_alpha"],
        "output_limit": snapshot["output_limit"],
    }
    pid_x.configure(
        kp=snapshot["kp"],
        ki=snapshot["ki"],
        kd=snapshot["kd"],
        **common_args,
    )
    pid_y.configure(
        kp=snapshot["kp"],
        ki=snapshot["ki"],
        kd=snapshot["kd"],
        **common_args,
    )


def build_pid_status_snapshot(shared_lock, state):
    mode_labels = {
        0: "OFF",
        1: "ACTIVE",
    }
    with shared_lock:
        mode = int(state.get("mode", 0))
        aimmode = int(state.get("aimmode", 0))
        target_ts = float(state.get("target_ts", 0.0))
        age_ms = 0.0
        if target_ts > 0.0:
            age_ms = max(0.0, (time.time() - target_ts) * 1000.0)
        return {
            "running": bool(state.get("running", False)),
            "mode": mode,
            "mode_label": mode_labels.get(mode, str(mode)),
            "aimmode": aimmode,
            "aimmode_label": "HEAD" if aimmode == 0 else "BODY",
            "tracking_strategy": normalize_tracking_strategy(
                state.get("tracking_strategy", TRACKING_STRATEGY_DEFAULT)
            ),
            "tracking_strategy_label": TRACKING_STRATEGY_LABELS.get(
                normalize_tracking_strategy(state.get("tracking_strategy", TRACKING_STRATEGY_DEFAULT)),
                "Unknown",
            ),
            "use_kalman": bool(state.get("use_kalman", USE_KALMAN_DEFAULT)),
            "target_found": bool(state.get("target_found", False)),
            "target_speed": float(state.get("target_speed", 0.0)),
            "target_cls": int(state.get("target_cls", -1)),
            "aim_dx": int(state.get("aim_dx", 0)),
            "aim_dy": int(state.get("aim_dy", 0)),
            "target_age_ms": age_ms,
        }


def build_pid_frontend_html():
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Runtime Tuning</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0d1418;
      --panel: rgba(15, 27, 33, 0.94);
      --panel-alt: rgba(23, 38, 47, 0.92);
      --border: rgba(162, 189, 204, 0.22);
      --text: #f2f7fb;
      --muted: #97aab8;
      --accent: #efb366;
      --accent-strong: #ff8b5e;
      --good: #87d68d;
      --bad: #ff8b78;
      --shadow: rgba(0, 0, 0, 0.28);
      font-family: "Segoe UI", "Trebuchet MS", sans-serif;
    }
    * {
      box-sizing: border-box;
    }
    body {
      margin: 0;
      min-height: 100vh;
      background:
        radial-gradient(circle at top left, rgba(239, 179, 102, 0.18), transparent 34%),
        radial-gradient(circle at bottom right, rgba(255, 139, 94, 0.16), transparent 28%),
        linear-gradient(135deg, #091015 0%, #101a20 48%, #0c1216 100%);
      color: var(--text);
      display: flex;
      align-items: stretch;
      justify-content: center;
      padding: 20px;
    }
    .panel {
      width: min(960px, 100%);
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 24px;
      box-shadow: 0 18px 48px var(--shadow);
      overflow: hidden;
      backdrop-filter: blur(14px);
    }
    .hero {
      padding: 24px 24px 16px;
      background:
        linear-gradient(120deg, rgba(239, 179, 102, 0.10), transparent 55%),
        linear-gradient(180deg, rgba(255, 139, 94, 0.08), rgba(23, 38, 47, 0.04));
      border-bottom: 1px solid var(--border);
    }
    .hero h1 {
      margin: 0 0 6px;
      font-size: clamp(1.6rem, 2.6vw, 2.1rem);
      letter-spacing: 0.03em;
    }
    .hero p {
      margin: 0;
      color: var(--muted);
      max-width: 780px;
      line-height: 1.45;
    }
    .status {
      display: grid;
      gap: 12px;
      padding: 20px 24px 0;
    }
    .status-bar {
      padding: 14px 16px;
      border-radius: 16px;
      background: var(--panel-alt);
      border: 1px solid var(--border);
      color: var(--muted);
      font-size: 0.95rem;
    }
    .status-bar strong {
      color: var(--text);
    }
    .status-bar.good {
      border-color: rgba(135, 214, 141, 0.35);
    }
    .status-bar.bad {
      border-color: rgba(255, 139, 120, 0.35);
    }
    form {
      padding: 20px 24px 24px;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 14px;
    }
    .field {
      display: grid;
      gap: 8px;
      padding: 14px;
      border-radius: 16px;
      background: rgba(17, 29, 35, 0.82);
      border: 1px solid rgba(162, 189, 204, 0.12);
    }
    .field label {
      font-size: 0.9rem;
      color: var(--muted);
    }
    .field input[type="number"],
    .field select {
      width: 100%;
      border: 1px solid rgba(162, 189, 204, 0.18);
      background: rgba(9, 16, 20, 0.85);
      color: var(--text);
      border-radius: 12px;
      padding: 11px 12px;
      font-size: 1rem;
      outline: none;
    }
    .field input[type="number"]:focus,
    .field select:focus {
      border-color: rgba(239, 179, 102, 0.62);
      box-shadow: 0 0 0 3px rgba(239, 179, 102, 0.14);
    }
    .toggle {
      display: flex;
      align-items: center;
      justify-content: space-between;
      min-height: 74px;
    }
    .toggle input {
      width: 24px;
      height: 24px;
      accent-color: var(--accent-strong);
    }
    .actions {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 18px;
    }
    button {
      border: 0;
      border-radius: 999px;
      padding: 12px 18px;
      font-size: 0.95rem;
      cursor: pointer;
      transition: transform 0.12s ease, opacity 0.12s ease;
    }
    button:hover {
      transform: translateY(-1px);
      opacity: 0.96;
    }
    .primary {
      background: linear-gradient(135deg, var(--accent), var(--accent-strong));
      color: #14110d;
      font-weight: 700;
    }
    .secondary {
      background: rgba(162, 189, 204, 0.14);
      color: var(--text);
      border: 1px solid rgba(162, 189, 204, 0.18);
    }
    .footnote {
      margin-top: 14px;
      color: var(--muted);
      font-size: 0.9rem;
      line-height: 1.45;
    }
    @media (max-width: 640px) {
      body {
        padding: 12px;
      }
      .hero,
      .status,
      form {
        padding-left: 16px;
        padding-right: 16px;
      }
      .grid {
        grid-template-columns: 1fr;
      }
      .actions {
        flex-direction: column;
      }
      button {
        width: 100%;
      }
    }
  </style>
</head>
<body>
  <main class="panel">
    <section class="hero">
      <h1>Runtime Tuning</h1>
      <p>Adjust PID and tracking parameters from the browser while the script is running. PID gain changes are applied to both X and Y controllers, while sticky selection, prediction horizon, persistence, and Kalman Q/R update live in the tracking loop.</p>
    </section>
    <section class="status">
      <div id="runtime-status" class="status-bar">Connecting to runtime...</div>
      <div id="message-status" class="status-bar">Loading current runtime values...</div>
    </section>
    <form id="pid-form">
      <div id="field-grid" class="grid"></div>
      <div class="actions">
        <button class="primary" type="submit">Apply Changes</button>
        <button class="secondary" type="button" id="reload-btn">Reload</button>
        <button class="secondary" type="button" id="reset-btn">Reset PID State</button>
      </div>
      <div class="footnote">
        Runtime status refreshes automatically. Click <strong>Apply Changes</strong> to push PID or tracking changes into the running loop.
      </div>
    </form>
  </main>
  <script>
    const fields = __PID_FIELDS__;
    const form = document.getElementById("pid-form");
    const fieldGrid = document.getElementById("field-grid");
    const runtimeStatus = document.getElementById("runtime-status");
    const messageStatus = document.getElementById("message-status");

    function setMessage(text, tone = "good") {
      messageStatus.textContent = text;
      messageStatus.className = `status-bar ${tone}`;
    }

    function renderStatus(status) {
      const targetText = status.target_found
        ? `locked cls=${status.target_cls} age=${status.target_age_ms.toFixed(1)}ms`
        : "not locked";
      runtimeStatus.textContent =
        `Runtime ${status.running ? "running" : "stopped"} | mode ${status.mode_label} | aim ${status.aimmode_label} | track ${status.tracking_strategy_label} | target ${targetText} | speed ${status.target_speed.toFixed(1)} px/s | cmd (${status.aim_dx}, ${status.aim_dy})`;
      runtimeStatus.className = `status-bar ${status.running ? "good" : "bad"}`;
    }

    function buildFields() {
      for (const field of fields) {
        const wrapper = document.createElement("div");
        wrapper.className = field.type === "bool" ? "field toggle" : "field";

        const label = document.createElement("label");
        label.setAttribute("for", field.key);
        label.textContent = field.label;
        wrapper.appendChild(label);

        let input;
        if (field.type === "select") {
          input = document.createElement("select");
          for (const option of field.options || []) {
            const optionNode = document.createElement("option");
            optionNode.value = option.value;
            optionNode.textContent = option.label;
            input.appendChild(optionNode);
          }
        } else {
          input = document.createElement("input");
        }
        input.id = field.key;
        input.name = field.key;
        if (field.type === "bool") {
          input.type = "checkbox";
        } else if (field.type !== "select") {
          input.type = "number";
          input.step = field.step ?? "any";
          input.inputMode = "decimal";
        }
        wrapper.appendChild(input);
        fieldGrid.appendChild(wrapper);
      }
    }

    function applyConfigToForm(config) {
      for (const field of fields) {
        const input = document.getElementById(field.key);
        if (!input) {
          continue;
        }
        if (field.type === "bool") {
          input.checked = Boolean(config[field.key]);
        } else if (field.type === "select") {
          input.value = String(config[field.key] ?? "");
        } else if (typeof config[field.key] !== "undefined") {
          input.value = Number(config[field.key]).toFixed(field.step === 1 || field.step === 1.0 ? 0 : 3);
        }
      }
    }

    function collectPayload() {
      const payload = {};
      for (const field of fields) {
        const input = document.getElementById(field.key);
        if (!input) {
          continue;
        }
        if (field.type === "bool") {
          payload[field.key] = Boolean(input.checked);
        } else if (field.type === "select") {
          payload[field.key] = String(input.value);
        } else {
          payload[field.key] = Number(input.value);
        }
      }
      return payload;
    }

    async function requestJson(path, options = {}) {
      const response = await fetch(path, options);
      let data = {};
      try {
        data = await response.json();
      } catch (error) {
        data = {};
      }
      if (!response.ok) {
        throw new Error(data.error || `Request failed with ${response.status}`);
      }
      return data;
    }

    async function loadConfig() {
      const payload = await requestJson("/api/pid");
      applyConfigToForm(payload.config);
      renderStatus(payload.status);
      setMessage("Loaded current runtime values from the running script.", "good");
    }

    async function refreshStatus() {
      try {
        const payload = await requestJson("/api/pid/status");
        renderStatus(payload.status);
      } catch (error) {
        runtimeStatus.textContent = error.message;
        runtimeStatus.className = "status-bar bad";
      }
    }

    async function submitChanges(event) {
      event.preventDefault();
      try {
        const payload = collectPayload();
        const result = await requestJson("/api/pid", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        applyConfigToForm(result.config);
        renderStatus(result.status);
        setMessage("Applied runtime changes.", "good");
      } catch (error) {
        setMessage(error.message, "bad");
      }
    }

    async function resetPidState() {
      try {
        const result = await requestJson("/api/pid/reset", { method: "POST" });
        renderStatus(result.status);
        setMessage("PID state reset requested.", "good");
      } catch (error) {
        setMessage(error.message, "bad");
      }
    }

    buildFields();
    form.addEventListener("submit", submitChanges);
    document.getElementById("reload-btn").addEventListener("click", loadConfig);
    document.getElementById("reset-btn").addEventListener("click", resetPidState);

    loadConfig().catch((error) => {
      setMessage(error.message, "bad");
      runtimeStatus.textContent = "Unable to connect to the PID tuning backend.";
      runtimeStatus.className = "status-bar bad";
    });
    window.setInterval(refreshStatus, 1000);
  </script>
</body>
</html>
"""
    return html.replace("__PID_FIELDS__", json.dumps(PID_RUNTIME_PARAM_SPECS))


def start_pid_frontend_server(pid_runtime, shared_lock, state):
    if not PID_FRONTEND_ENABLE:
        return None, None

    html = build_pid_frontend_html().encode("utf-8")

    class PIDFrontendHandler(BaseHTTPRequestHandler):
        def _send_bytes(self, body, *, content_type, status=200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, payload, status=200):
            body = json.dumps(payload).encode("utf-8")
            self._send_bytes(body, content_type="application/json; charset=utf-8", status=status)

        def _read_json(self):
            content_length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
            try:
                return json.loads(raw.decode("utf-8") or "{}")
            except json.JSONDecodeError as exc:
                raise ValueError("Request body must be valid JSON.") from exc

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/":
                self._send_bytes(html, content_type="text/html; charset=utf-8")
                return
            if path in ("/api/pid", "/api/pidf"):
                self._send_json(
                    {
                        "config": pid_runtime.snapshot(),
                        "status": build_pid_status_snapshot(shared_lock, state),
                    }
                )
                return
            if path in ("/api/pid/status", "/api/pidf/status"):
                self._send_json({"status": build_pid_status_snapshot(shared_lock, state)})
                return
            self._send_json({"error": "Not found."}, status=404)

        def do_POST(self):
            path = self.path.split("?", 1)[0]
            try:
                if path in ("/api/pid", "/api/pidf"):
                    payload = self._read_json()
                    snapshot = pid_runtime.update(payload, reset_pid=True)
                    self._send_json(
                        {
                            "config": snapshot,
                            "status": build_pid_status_snapshot(shared_lock, state),
                        }
                    )
                    return
                if path in ("/api/pid/reset", "/api/pidf/reset"):
                    pid_runtime.request_reset()
                    self._send_json(
                        {
                            "config": pid_runtime.snapshot(),
                            "status": build_pid_status_snapshot(shared_lock, state),
                        }
                    )
                    return
                self._send_json({"error": "Not found."}, status=404)
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=400)

        def log_message(self, fmt, *args):
            return

    server = None
    server_port = None
    for port in range(PID_FRONTEND_PORT, PID_FRONTEND_PORT + max(1, PID_FRONTEND_PORT_RETRIES)):
        try:
            server = ThreadingHTTPServer((PID_FRONTEND_HOST, port), PIDFrontendHandler)
            server.daemon_threads = True
            server_port = port
            break
        except OSError:
            continue

    if server is None:
        print(
            "[WARN] runtime tuning frontend failed to start; "
            f"ports {PID_FRONTEND_PORT}-{PID_FRONTEND_PORT + max(1, PID_FRONTEND_PORT_RETRIES) - 1} are unavailable."
        )
        return None, None

    thread = threading.Thread(target=server.serve_forever, daemon=True, name="pid-frontend")
    thread.start()
    print(f"[INFO] Runtime tuning UI: http://{PID_FRONTEND_HOST}:{server_port}/")
    return server, server_port


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
        self.last_preprocess_s = None
        self.last_infer_s = None
        self.last_postprocess_s = None

        try:
            self.model.fuse()
        except Exception:
            pass

    def warmup(self):
        warmup_frame = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        self.predict(warmup_frame, WARMUP_CLASS)

    def predict(self, frame, target_cls):
        self.last_preprocess_s = None
        self.last_infer_s = None
        self.last_postprocess_s = None
        if np.isscalar(target_cls):
            target_classes = [int(target_cls)]
        else:
            target_classes = [int(c) for c in np.asarray(target_cls).reshape(-1)]

        try:
            results = self.model.predict(
                source=frame,
                imgsz=self.imgsz,
                conf=self.conf,
                device=self.device,
                classes=target_classes,
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
                    classes=target_classes,
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
        self.last_preprocess_s = None
        self.last_infer_s = None
        self.last_postprocess_s = None
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
                trt_options = {
                    "device_id": ONNX_CUDA_DEVICE_ID,
                    "trt_fp16_enable": ONNX_TRT_FP16,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": trt_cache,
                }
                if int(ONNX_TRT_MAX_WORKSPACE_SIZE) > 0:
                    trt_options["trt_max_workspace_size"] = int(ONNX_TRT_MAX_WORKSPACE_SIZE)
                if ONNX_TRT_TIMING_CACHE_ENABLE:
                    trt_options["trt_timing_cache_enable"] = True
                    trt_options["trt_timing_cache_path"] = (
                        str(ONNX_TRT_TIMING_CACHE_PATH).strip() or trt_cache
                    )
                    if ONNX_TRT_FORCE_TIMING_CACHE:
                        trt_options["trt_force_timing_cache"] = True
                if ONNX_TRT_CUDA_GRAPH_ENABLE:
                    trt_options["trt_cuda_graph_enable"] = True
                op_exclude = str(ONNX_TRT_OP_TYPES_TO_EXCLUDE).strip()
                if op_exclude:
                    trt_options["trt_op_types_to_exclude"] = op_exclude
                providers.append(
                    (
                        "TensorrtExecutionProvider",
                        trt_options,
                    )
                )
            if "CUDAExecutionProvider" in self.available:
                conv_algo = str(ONNX_CUDNN_CONV_ALGO_SEARCH).strip().upper()
                if conv_algo not in ("DEFAULT", "HEURISTIC", "EXHAUSTIVE"):
                    conv_algo = "HEURISTIC"
                providers.append(
                    (
                        "CUDAExecutionProvider",
                        {
                            "device_id": ONNX_CUDA_DEVICE_ID,
                            "arena_extend_strategy": "kNextPowerOfTwo",
                            "cudnn_conv_algo_search": conv_algo,
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
        if ONNX_DISABLE_ORT_CPU_FALLBACK:
            disable_fallback = getattr(self.session, "disable_fallback", None)
            if callable(disable_fallback):
                try:
                    disable_fallback()
                except Exception:
                    pass
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
            if ONNX_DISABLE_ORT_CPU_FALLBACK:
                disable_fallback = getattr(self.session, "disable_fallback", None)
                if callable(disable_fallback):
                    try:
                        disable_fallback()
                    except Exception:
                        pass
        self._cuda_ep_in_use = "CUDAExecutionProvider" in self.providers
        self.compute_provider = self._select_compute_provider()
        self.name = f"onnxruntime[{','.join(self.providers)}]"
        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name
        self.input_type = input_meta.type
        self.input_h, self.input_w = self._resolve_input_hw(input_meta.shape)
        if (self.input_h != self.imgsz) or (self.input_w != self.imgsz):
            print(
                f"[INFO] ONNX model input is fixed at {self.input_w}x{self.input_h}; "
                f"using that instead of IMGSZ={self.imgsz}."
            )
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.input_dtype = np.float16 if "float16" in self.input_type else np.float32
        self._resize_bgr = np.empty((self.input_h, self.input_w, 3), dtype=np.uint8)
        self._rgb = np.empty_like(self._resize_bgr)
        self._input = np.empty((1, 3, self.input_h, self.input_w), dtype=self.input_dtype)
        resize_mode = str(ONNX_RESIZE_INTERPOLATION).strip().lower()
        if resize_mode == "nearest":
            self._resize_interp = cv2.INTER_NEAREST
        else:
            self._resize_interp = cv2.INTER_LINEAR

    def _resolve_input_hw(self, input_shape):
        if isinstance(input_shape, (list, tuple)) and len(input_shape) >= 4:
            h_dim = input_shape[2]
            w_dim = input_shape[3]
            h_fixed = isinstance(h_dim, (int, np.integer)) and int(h_dim) > 0
            w_fixed = isinstance(w_dim, (int, np.integer)) and int(w_dim) > 0
            if h_fixed and w_fixed:
                return int(h_dim), int(w_dim)
        return int(self.imgsz), int(self.imgsz)

    def _select_compute_provider(self):
        if "TensorrtExecutionProvider" in self.providers:
            return "tensorrt"
        if "CUDAExecutionProvider" in self.providers:
            return "cuda"
        return "cpu"

    def _normalize_target_classes(self, target_cls):
        if np.isscalar(target_cls):
            arr = np.array([int(target_cls)], dtype=np.int32)
        else:
            arr = np.asarray(target_cls, dtype=np.int32).reshape(-1)
        if arr.size == 0:
            return np.empty((0,), dtype=np.int32)
        return np.unique(arr)

    def warmup(self):
        warmup_frame = np.zeros((self.input_h, self.input_w, 3), dtype=np.uint8)
        self.predict(warmup_frame, WARMUP_CLASS)

    def _run_raw(self, frame):
        h, w = frame.shape[:2]
        preprocess_start = time.perf_counter()
        if (
            ONNX_SKIP_RESIZE_IF_MATCH
            and (w == self.input_w)
            and (h == self.input_h)
        ):
            preprocess_bgr = frame
        else:
            cv2.resize(
                frame,
                (self.input_w, self.input_h),
                dst=self._resize_bgr,
                interpolation=self._resize_interp,
            )
            preprocess_bgr = self._resize_bgr
        cv2.cvtColor(preprocess_bgr, cv2.COLOR_BGR2RGB, dst=self._rgb)
        np.multiply(
            self._rgb.transpose(2, 0, 1),
            1.0 / 255.0,
            out=self._input[0],
            casting="unsafe",
        )
        self.last_preprocess_s = time.perf_counter() - preprocess_start
        exec_start = time.perf_counter()
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
        self.last_infer_s = time.perf_counter() - exec_start
        return outputs, w, h

    def _decode_nms_output(self, pred, frame_w, frame_h, target_classes):
        if pred.ndim != 2 or pred.shape[1] < 6:
            return None
        confs = pred[:, 4].astype(np.float32, copy=False)
        cls_ids = pred[:, 5].astype(np.int32, copy=False)
        if target_classes.size == 0:
            return (
                np.empty((0, 4), dtype=np.int32),
                np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=np.float32),
            )
        mask = np.isin(cls_ids, target_classes) & (confs >= self.conf)
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
        input_ref = float(max(self.input_w, self.input_h))
        if coord_max <= (input_ref + 8.0):
            sx = frame_w / float(self.input_w)
            sy = frame_h / float(self.input_h)
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

    def _decode_raw_output(self, pred, frame_w, frame_h, target_classes):
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

        valid_targets = target_classes[
            (target_classes >= 0) & (target_classes < cls_scores.shape[1])
        ]
        if ONNX_FORCE_TARGET_CLASS_DECODE and valid_targets.size > 0:
            selected_scores = cls_scores[:, valid_targets]
            best_rel = np.argmax(selected_scores, axis=1)
            confs = selected_scores[np.arange(selected_scores.shape[0]), best_rel]
            cls_ids = valid_targets[best_rel]
            mask = confs >= self.conf
            if not np.any(mask):
                return (
                    np.empty((0, 4), dtype=np.int32),
                    np.empty((0,), dtype=np.int32),
                    np.empty((0,), dtype=np.float32),
                )
            boxes_xywh = boxes_xywh[mask]
            confs = confs[mask]
            cls_ids = cls_ids[mask]
        else:
            cls_ids = np.argmax(cls_scores, axis=1).astype(np.int32, copy=False)
            confs = cls_scores[np.arange(cls_scores.shape[0]), cls_ids]
            if valid_targets.size > 0:
                mask = np.isin(cls_ids, valid_targets) & (confs >= self.conf)
            else:
                mask = confs >= self.conf
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
            boxes_xywh[:, [0, 2]] *= float(self.input_w)
            boxes_xywh[:, [1, 3]] *= float(self.input_h)

        cx = boxes_xywh[:, 0]
        cy = boxes_xywh[:, 1]
        bw = np.maximum(1.0, boxes_xywh[:, 2])
        bh = np.maximum(1.0, boxes_xywh[:, 3])
        x1 = cx - (bw * 0.5)
        y1 = cy - (bh * 0.5)
        x2 = cx + (bw * 0.5)
        y2 = cy + (bh * 0.5)
        xyxy = np.stack([x1, y1, x2, y2], axis=1)

        sx = frame_w / float(self.input_w)
        sy = frame_h / float(self.input_h)
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
        target_classes = self._normalize_target_classes(target_cls)
        outputs, frame_w, frame_h = self._run_raw(frame)
        post_start = time.perf_counter()
        if not outputs:
            result = (
                np.empty((0, 4), dtype=np.int32),
                np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=np.float32),
            )
            self.last_postprocess_s = time.perf_counter() - post_start
            return result

        if ONNX_OUTPUT_HAS_NMS:
            arr0 = np.asarray(outputs[0])
            if arr0.ndim == 3 and arr0.shape[0] == 1 and arr0.shape[-1] >= 6:
                decoded = self._decode_nms_output(arr0[0], frame_w, frame_h, target_classes)
                if decoded is not None:
                    self.last_postprocess_s = time.perf_counter() - post_start
                    return decoded

        for out in outputs:
            arr = np.asarray(out)
            if arr.ndim == 3 and arr.shape[0] == 1 and 6 <= arr.shape[-1] <= 8:
                decoded = self._decode_nms_output(arr[0], frame_w, frame_h, target_classes)
                if decoded is not None:
                    self.last_postprocess_s = time.perf_counter() - post_start
                    return decoded

        raw = np.asarray(outputs[0])
        if raw.ndim == 3 and raw.shape[0] == 1:
            raw = raw[0]
        result = self._decode_raw_output(raw, frame_w, frame_h, target_classes)
        self.last_postprocess_s = time.perf_counter() - post_start
        return result


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
                    simplify=True,
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
    # Build a list so requested tracker is attempted first, then robust fallbacks.
    order = [tracker_type]
    for fallback in ("MOSSE", "KCF", "CSRT", "MIL"):
        if fallback not in order:
            order.append(fallback)

    for name in order:
        if name == "MOSSE":
            if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerMOSSE_create"):
                return cv2.legacy.TrackerMOSSE_create()
        elif name == "KCF":
            if hasattr(cv2, "TrackerKCF_create"):
                return cv2.TrackerKCF_create()
            if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
                return cv2.legacy.TrackerKCF_create()
        elif name == "CSRT":
            if hasattr(cv2, "TrackerCSRT_create"):
                return cv2.TrackerCSRT_create()
            if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
                return cv2.legacy.TrackerCSRT_create()
        elif name == "MIL":
            if hasattr(cv2, "TrackerMIL_create"):
                return cv2.TrackerMIL_create()
            if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerMIL_create"):
                return cv2.legacy.TrackerMIL_create()
    return None


class MouseMoveClient:
    _SIO_LOOPBACK_FAST_PATH = 0x98000010
    _MOUSEEVENTF_LEFTDOWN = 0x0002
    _MOUSEEVENTF_LEFTUP = 0x0004

    def __init__(self, host=MOUSE_HOST, port=MOUSE_PORT, send_backend=MOUSE_SEND_BACKEND):
        self.host = host
        self.port = port
        self.sock = None
        self.lock = threading.Lock()
        self.last_connect_fail_ts = 0.0
        self.send_backend = str(send_backend).lower()
        self._ghub_ready = False
        self._ghub_frac_x = 0.0
        self._ghub_frac_y = 0.0
        self._pdi_frac_x = 0.0
        self._pdi_frac_y = 0.0
        self._send_buf = bytearray(8)
        self._send_view = memoryview(self._send_buf)
        self._binary_mode = bool(MOUSE_SOCKET_BINARY)
        self._ghub_max_step = max(1, int(GHUB_MAX_STEP))

        if self.send_backend == "ghub":
            if ghub_mouse is None:
                print("[WARN] local mouse.py GHUB backend not available; falling back to socket mouse backend.")
                self.send_backend = "socket"
            else:
                try:
                    self._ghub_ready = bool(ghub_mouse.mouse_open())
                except Exception as e:
                    print(f"[WARN] GHUB mouse init failed ({e}); falling back to socket mouse backend.")
                    self.send_backend = "socket"
                    self._ghub_ready = False

        if self.send_backend == "pydirectinput":
            if pydirectinput is None:
                print("[WARN] pydirectinput not installed; falling back to socket mouse backend.")
                self.send_backend = "socket"
            else:
                pydirectinput.FAILSAFE = False
                pydirectinput.PAUSE = 0

    def configure(self, *, ghub_max_step=None):
        with self.lock:
            if ghub_max_step is not None:
                self._ghub_max_step = max(1, int(round(float(ghub_max_step))))

    def close(self):
        if ghub_mouse is not None and self._ghub_ready:
            try:
                ghub_mouse.mouse_close()
            except Exception:
                pass
            self._ghub_ready = False
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
            try:
                sock.ioctl(self._SIO_LOOPBACK_FAST_PATH, 1)
            except (AttributeError, OSError):
                pass
            sock.settimeout(MOUSE_CONNECT_TIMEOUT_S)
            sock.connect((self.host, self.port))
            sock.settimeout(MOUSE_SEND_TIMEOUT_S)
            self.sock = sock
            return True
        except OSError:
            self.last_connect_fail_ts = now
            self.close()
            return False

    def _send_socket(self, x, y):
        ix = int(x)
        iy = int(y)
        if self._binary_mode:
            struct.pack_into("<ii", self._send_buf, 0, ix, iy)
            message = self._send_view
        else:
            message = f"{ix} {iy}\n".encode()
        if not self._ensure_socket():
            return False
        try:
            self.sock.sendall(message)
            return True
        except OSError:
            self.close()
            return False

    def send_input_ghub(self, x, y):
        if ghub_mouse is None:
            return False
        if not self._ghub_ready:
            try:
                self._ghub_ready = bool(ghub_mouse.mouse_open())
            except Exception:
                self._ghub_ready = False
        if not self._ghub_ready:
            return False

        try:
            move_x = (float(x) * GHUB_GAIN_X) + self._ghub_frac_x
            move_y = (float(y) * GHUB_GAIN_Y) + self._ghub_frac_y
            send_x = int(round(move_x))
            send_y = int(round(move_y))
            self._ghub_frac_x = move_x - send_x
            self._ghub_frac_y = move_y - send_y

            remaining_x = send_x
            remaining_y = send_y
            step_limit = self._ghub_max_step
            while remaining_x != 0 or remaining_y != 0:
                step_x = int(clamp(remaining_x, -step_limit, step_limit))
                step_y = int(clamp(remaining_y, -step_limit, step_limit))
                ghub_mouse.mouse_move(0, step_x, step_y, 0)
                remaining_x -= step_x
                remaining_y -= step_y
            return True
        except Exception:
            try:
                ghub_mouse.mouse_close()
            except Exception:
                pass
            self._ghub_ready = False
            return False

    def send_input_pydirectinput(self, x, y):
        if pydirectinput is None:
            return False
        try:
            move_x = (float(x) * PYDIRECTINPUT_GAIN_X) + self._pdi_frac_x
            move_y = (float(y) * PYDIRECTINPUT_GAIN_Y) + self._pdi_frac_y
            send_x = int(round(move_x))
            send_y = int(round(move_y))
            self._pdi_frac_x = move_x - send_x
            self._pdi_frac_y = move_y - send_y
            pydirectinput.moveRel(send_x, send_y, relative=True, _pause=False)
            return True
        except Exception:
            return False

    def send_input(self, x, y):
        with self.lock:
            if self.send_backend == "ghub":
                return self.send_input_ghub(x, y)
            if self.send_backend == "pydirectinput":
                return self.send_input_pydirectinput(x, y)
            return self._send_socket(x, y)

    def click_left(self, hold_s=TRIGGERBOT_CLICK_HOLD_S):
        hold_s = max(0.0, float(hold_s))
        with self.lock:
            try:
                ctypes.windll.user32.mouse_event(self._MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                if hold_s > 0.0:
                    time.sleep(hold_s)
                ctypes.windll.user32.mouse_event(self._MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                return True
            except Exception:
                return False

    def send(self, x, y):
        return self.send_input(x, y)


def main():
    print("Initialize Start")
    if CAPTURE_BENCHMARK_ON_START:
        run_capture_benchmark()
    tracker_capture_w = int(clamp(BASE_CROP_W, MIN_ACTIVE_CROP_W, MAX_ACTIVE_CROP_W))
    tracker_capture_h = int(clamp(BASE_CROP_H, MIN_ACTIVE_CROP_H, MAX_ACTIVE_CROP_H))
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
    if TRACKER_ENABLE and YOLO_INTERVAL <= 1 and YOLO_INTERVAL_FAST <= 1:
        print(
            "[INFO] TRACKER_ENABLE is ON with YOLO interval=1; "
            f"using TRACKER_MIN_INTERVAL={TRACKER_MIN_INTERVAL} for tracker steps."
        )
    if TRACKER_ENABLE and TRACKER_FORCE_FIXED_CAPTURE_SIZE:
        print(
            "[INFO] TRACKER_FORCE_FIXED_CAPTURE_SIZE is ON; "
            f"using stable crop {tracker_capture_w}x{tracker_capture_h} while target is active."
        )

    mode = 0      # 0=off, 1=active
    aimmode = 1   # 0=head, 1=body
    use_kalman = TRACKING_STRATEGY_DEFAULT == TRACKING_STRATEGY_KALMAN
    last_toggle = 0.0
    left_hold_engage = LEFT_HOLD_ENGAGE_DEFAULT
    last_left_hold_toggle = 0.0
    recoil_tune_fallback = RECOIL_TUNE_FALLBACK_DEFAULT
    last_recoil_tune_toggle = 0.0
    last_triggerbot_toggle = 0.0
    running = True
    mouse_button_pressed_x1 = False
    mouse_button_pressed_x2 = False
    last_polled_left_pressed = False
    last_polled_right_pressed = False

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
        "tracking_strategy": str(TRACKING_STRATEGY_DEFAULT),
        "use_kalman": use_kalman,
        "left_hold_engage": left_hold_engage,
        "recoil_tune_fallback": recoil_tune_fallback,
        "ctrl_cmd_age_ema": 0.0,
        "ctrl_send_ema": 0.0,
        "ctrl_sent_vx_ema": 0.0,
        "ctrl_sent_vy_ema": 0.0,
        "ctrl_last_send_tick": 0.0,
        "left_pressed": False,
        "right_pressed": False,
    }
    pid_runtime = RuntimePIDConfig()
    perf = RuntimePerf()
    pid_frontend_server, _ = start_pid_frontend_server(
        pid_runtime,
        shared_lock,
        state,
    )

    frame_queue = queue.Queue(maxsize=PIPELINE_FRAME_QUEUE)
    control_queue = queue.Queue(maxsize=PIPELINE_CMD_QUEUE)

    kalman_state_dim = 6 if KALMAN_USE_CA_MODEL else 4
    kalman = cv2.KalmanFilter(kalman_state_dim, 2)
    if kalman_state_dim == 6:
        # State = [x, y, vx, vy, ax, ay], measurement = [x, y].
        kalman.measurementMatrix = np.array(
            [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]], dtype=np.float32
        )
        kalman.transitionMatrix = np.array(
            [
                [1, 0, 1, 0, 0.5, 0],
                [0, 1, 0, 1, 0, 0.5],
                [0, 0, 1, 0, 1, 0],
                [0, 0, 0, 1, 0, 1],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
    else:
        kalman.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32
        )
        kalman.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
    kalman.processNoiseCov = np.eye(kalman_state_dim, dtype=np.float32) * KALMAN_PROCESS_NOISE_BASE
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * KALMAN_MEAS_NOISE_BASE
    kalman.errorCovPost = np.eye(kalman_state_dim, dtype=np.float32)
    kalman_initialized = False
    last_kalman_ts = time.time()

    def capture_loop():
        nonlocal running
        set_current_thread_priority(CAPTURE_THREAD_PRIORITY)
        capture_source = build_capture_source()
        last_capture_error_ts = 0.0
        print(f"[INFO] Capture backend: {capture_source.name}")
        try:
            while running:
                with shared_lock:
                    target_found = state["target_found"]
                    tx, ty = state["capture_focus_full"]
                    target_speed = state["target_speed"]

                if target_found:
                    if TRACKER_ENABLE and TRACKER_FORCE_FIXED_CAPTURE_SIZE:
                        crop_w = tracker_capture_w
                        crop_h = tracker_capture_h
                    elif CAPTURE_FORCE_INPUT_SIZE_WHEN_LOCKED:
                        crop_w = int(IMGSZ)
                        crop_h = int(IMGSZ)
                    else:
                        pad = int(min(CAPTURE_PAD_MAX, target_speed * CAPTURE_SPEED_TO_PAD))
                        crop_w = int(clamp(BASE_CROP_W + pad, MIN_ACTIVE_CROP_W, MAX_ACTIVE_CROP_W))
                        crop_h = int(clamp(BASE_CROP_H + pad, MIN_ACTIVE_CROP_H, MAX_ACTIVE_CROP_H))
                    cap = build_capture(tx, ty, crop_w, crop_h)
                else:
                    cap = build_capture(CENTER[0], CENTER[1], LOST_CROP_W, LOST_CROP_H)

                capture_ts = time.time()
                grab_start = time.perf_counter()
                try:
                    frame_bgr = capture_source.grab(cap)
                except Exception as e:
                    grab_elapsed = time.perf_counter() - grab_start
                    perf.record_capture(grab_elapsed, is_none=True)
                    now_err = time.time()
                    if now_err - last_capture_error_ts > 1.0:
                        print(f"[WARN] capture backend '{capture_source.name}' crashed ({e}); switching to MSS.")
                        last_capture_error_ts = now_err
                    try:
                        capture_source.close()
                    except Exception:
                        pass
                    capture_source = MSSCaptureSource()
                    time.sleep(0.01)
                    continue
                grab_elapsed = time.perf_counter() - grab_start
                if frame_bgr is None:
                    perf.record_capture(grab_elapsed, is_none=True)
                    if CAPTURE_NONE_BACKOFF_S > 0.0:
                        time.sleep(CAPTURE_NONE_BACKOFF_S)
                    elif CAPTURE_YIELD_EACH_LOOP:
                        time.sleep(0)
                    continue
                now = time.time()
                perf.record_capture(grab_elapsed, is_none=False)

                put_latest(
                    frame_queue,
                    {
                        "frame": frame_bgr,
                        "capture": cap,
                        "ts": now,
                        "capture_ts": capture_ts,
                    },
                )

                if CAPTURE_YIELD_EACH_LOOP:
                    time.sleep(0)
        finally:
            capture_source.close()

    def inference_loop_simple():
        nonlocal running
        set_current_thread_priority(INFERENCE_THREAD_PRIORITY)
        last_backend_err_ts = 0.0
        last_pid_tick = time.perf_counter()
        last_track_tick = 0.0
        last_box_w = 0.0
        last_box_h = 0.0
        lost_frames = 0
        active_target_cls = -1

        pid_snapshot = pid_runtime.snapshot()
        pid_x = PIDController(
            pid_snapshot["kp"],
            pid_snapshot["ki"],
            pid_snapshot["kd"],
            integral_limit=pid_snapshot["integral_limit"],
            anti_windup_gain=pid_snapshot["anti_windup_gain"],
            derivative_alpha=pid_snapshot["derivative_alpha"],
            output_limit=pid_snapshot["output_limit"],
        )
        pid_y = PIDController(
            pid_snapshot["kp"],
            pid_snapshot["ki"],
            pid_snapshot["kd"],
            integral_limit=pid_snapshot["integral_limit"],
            anti_windup_gain=pid_snapshot["anti_windup_gain"],
            derivative_alpha=pid_snapshot["derivative_alpha"],
            output_limit=pid_snapshot["output_limit"],
        )
        last_pid_version = int(pid_snapshot["version"])
        last_pid_reset_token = int(pid_snapshot["reset_token"])
        last_runtime_tracking_strategy = normalize_tracking_strategy(
            pid_snapshot.get("tracking_strategy", TRACKING_STRATEGY_DEFAULT)
        )
        target_tracker = build_motion_tracker(
            last_runtime_tracking_strategy,
            process_noise=pid_snapshot["kalman_process_noise"],
            measurement_noise=pid_snapshot["kalman_measurement_noise"],
            smoothing_alpha=pid_snapshot["tracking_alpha"],
            velocity_alpha=pid_snapshot["tracking_velocity_alpha"],
        )
        with shared_lock:
            state["tracking_strategy"] = last_runtime_tracking_strategy
            state["use_kalman"] = last_runtime_tracking_strategy == TRACKING_STRATEGY_KALMAN

        while running:
            with shared_lock:
                local_mode = state["mode"]
                local_aimmode = state["aimmode"]
                local_tracking_strategy = normalize_tracking_strategy(
                    state.get("tracking_strategy", TRACKING_STRATEGY_DEFAULT)
                )
                local_left_hold_engage = bool(state.get("left_hold_engage", False))
                local_right_pressed = bool(state.get("right_pressed", False))
                prev_target_found = state["target_found"]
                prev_tx, prev_ty = state["last_target_full"]
                prev_target_cls = state["target_cls"]
                prev_target_ts = float(state.get("target_ts", 0.0))
                ctrl_sent_vx_ema = float(state.get("ctrl_sent_vx_ema", 0.0))
                ctrl_sent_vy_ema = float(state.get("ctrl_sent_vy_ema", 0.0))

            pid_snapshot = pid_runtime.snapshot()
            pid_version = int(pid_snapshot["version"])
            pid_reset_token = int(pid_snapshot["reset_token"])
            if pid_version != last_pid_version:
                apply_pid_runtime_to_controllers(pid_x, pid_y, pid_snapshot)
                last_pid_version = pid_version
            if pid_reset_token != last_pid_reset_token:
                pid_x.reset()
                pid_y.reset()
                last_pid_reset_token = pid_reset_token
            pid_enable = bool(pid_snapshot["enable"])
            tracking_strategy = normalize_tracking_strategy(
                pid_snapshot.get("tracking_strategy", TRACKING_STRATEGY_DEFAULT)
            )
            use_kalman = tracking_strategy == TRACKING_STRATEGY_KALMAN
            tracking_alpha = float(pid_snapshot["tracking_alpha"])
            tracking_velocity_alpha = float(pid_snapshot["tracking_velocity_alpha"])
            pid_integrate = abs(float(pid_snapshot["ki"])) > 1e-12
            sticky_bias_px = float(pid_snapshot["sticky_bias_px"])
            prediction_time = float(pid_snapshot["prediction_time"])
            target_max_lost_frames = max(1, int(round(pid_snapshot["target_max_lost_frames"])))
            model_conf = float(pid_snapshot.get("model_conf", CONF))
            detection_min_conf = float(pid_snapshot.get("detection_min_conf", DETECTION_MIN_CONF))
            triggerbot_enable = bool(pid_snapshot.get("triggerbot_enable", TRIGGERBOT_ENABLE_DEFAULT))
            triggerbot_box_percent = float(
                pid_snapshot.get("triggerbot_box_percent", TRIGGERBOT_BOX_PERCENT_DEFAULT)
            )
            if hasattr(backend, "conf"):
                backend.conf = model_conf
            ego_motion_comp_enable = bool(pid_snapshot["ego_motion_comp_enable"])
            ego_motion_comp_gain_x = float(pid_snapshot["ego_motion_comp_gain_x"])
            ego_motion_comp_gain_y = float(pid_snapshot["ego_motion_comp_gain_y"])
            ego_motion_error_gate_enable = bool(
                pid_snapshot.get("ego_motion_error_gate_enable", EGO_MOTION_ERROR_GATE_ENABLE)
            )
            ego_motion_error_gate_px = float(
                pid_snapshot.get("ego_motion_error_gate_px", EGO_MOTION_ERROR_GATE_PX)
            )
            ego_motion_error_gate_normalize_by_box = bool(
                pid_snapshot.get(
                    "ego_motion_error_gate_normalize_by_box",
                    EGO_MOTION_ERROR_GATE_NORMALIZE_BY_BOX,
                )
            )
            ego_motion_error_gate_norm_threshold = float(
                pid_snapshot.get(
                    "ego_motion_error_gate_norm_threshold",
                    EGO_MOTION_ERROR_GATE_NORM_THRESHOLD,
                )
            )
            ego_motion_reset_on_switch = bool(
                pid_snapshot.get("ego_motion_reset_on_switch", EGO_MOTION_RESET_ON_SWITCH)
            )
            target_tracker.configure(
                process_noise=pid_snapshot["kalman_process_noise"],
                measurement_noise=pid_snapshot["kalman_measurement_noise"],
                smoothing_alpha=tracking_alpha,
                velocity_alpha=tracking_velocity_alpha,
            )
            if tracking_strategy != last_runtime_tracking_strategy:
                target_tracker = build_motion_tracker(
                    tracking_strategy,
                    process_noise=pid_snapshot["kalman_process_noise"],
                    measurement_noise=pid_snapshot["kalman_measurement_noise"],
                    smoothing_alpha=tracking_alpha,
                    velocity_alpha=tracking_velocity_alpha,
                )
                pid_x.reset()
                pid_y.reset()
                active_target_cls = -1
                lost_frames = 0
                last_box_w = 0.0
                last_box_h = 0.0
                last_track_tick = 0.0
                last_pid_tick = time.perf_counter()
                drain_queue(control_queue)
                with shared_lock:
                    state["tracking_strategy"] = tracking_strategy
                    state["use_kalman"] = use_kalman
                    state["target_found"] = False
                    state["last_target_full"] = CENTER
                    state["capture_focus_full"] = CENTER
                    state["target_speed"] = 0.0
                    state["target_cls"] = -1
                    state["aim_dx"] = 0
                    state["aim_dy"] = 0
                last_runtime_tracking_strategy = tracking_strategy
                continue
            if tracking_strategy != local_tracking_strategy:
                with shared_lock:
                    state["tracking_strategy"] = tracking_strategy
                    state["use_kalman"] = use_kalman

            engage_active = (local_mode != 0) and ((not local_left_hold_engage) or local_right_pressed)
            triggerbot_monitor_active = (local_mode != 0) and triggerbot_enable
            if not (engage_active or triggerbot_monitor_active):
                with shared_lock:
                    state["target_found"] = False
                    state["last_target_full"] = CENTER
                    state["capture_focus_full"] = CENTER
                    state["target_speed"] = 0.0
                    state["target_cls"] = -1
                    state["aim_dx"] = 0
                    state["aim_dy"] = 0
                target_tracker.reset()
                pid_x.reset()
                pid_y.reset()
                active_target_cls = -1
                lost_frames = 0
                last_box_w = 0.0
                last_box_h = 0.0
                last_track_tick = 0.0
                last_pid_tick = time.perf_counter()
                if local_left_hold_engage and (not local_right_pressed):
                    drain_queue(control_queue)
                time.sleep(0.002)
                continue

            try:
                packet = get_latest(frame_queue, 0.01)
            except queue.Empty:
                # time.sleep(0.001)
                continue

            frame = packet["frame"]
            cap = packet["capture"]
            frame_ts = packet["ts"]
            capture_ts = packet.get("capture_ts", frame_ts)
            loop_perf_start = time.perf_counter()
            yolo_elapsed_s = 0.0
            tracker_elapsed_s = 0.0
            backend_pre_s = None
            backend_exec_s = None
            backend_post_s = None

            frame_age = max(0.0, time.time() - frame_ts)
            if frame_age > MAX_FRAME_AGE_S:
                perf.record_inference(
                    frame_age_s=frame_age,
                    loop_s=time.perf_counter() - loop_perf_start,
                    stale_drop=True,
                    target_found=False,
                    yolo_s=yolo_elapsed_s,
                    tracker_s=tracker_elapsed_s,
                )
                # time.sleep(0.0005)
                continue

            # Model classes map directly from aimmode: 0=head, 1=body.
            target_cls = int(local_aimmode)
            target_classes = np.array([target_cls], dtype=np.int32)

            if active_target_cls != -1 and active_target_cls not in target_classes:
                target_tracker.reset()
                pid_x.reset()
                pid_y.reset()
                active_target_cls = -1
                lost_frames = 0
                last_box_w = 0.0
                last_box_h = 0.0
                last_track_tick = 0.0
                last_pid_tick = time.perf_counter()

            selected_detection = None
            selected_meta = {"locked_idx": None, "selected_idx": None, "switched": False}
            try:
                yolo_t0 = time.perf_counter()
                query_classes = target_classes if target_classes.size > 1 else int(target_classes[0])
                xyxy, cls_ids, confs = backend.predict(frame, query_classes)
                yolo_elapsed_s += time.perf_counter() - yolo_t0
                backend_pre_s = getattr(backend, "last_preprocess_s", None)
                backend_exec_s = getattr(backend, "last_infer_s", None)
                backend_post_s = getattr(backend, "last_postprocess_s", None)
            except Exception as e:
                now_err = time.time()
                if now_err - last_backend_err_ts > 1.0:
                    print(f"[WARN] backend inference failed: {e}")
                    last_backend_err_ts = now_err
                perf.record_inference(
                    frame_age_s=frame_age,
                    loop_s=time.perf_counter() - loop_perf_start,
                    stale_drop=False,
                    target_found=False,
                    yolo_s=yolo_elapsed_s,
                    tracker_s=tracker_elapsed_s,
                    backend_pre_s=backend_pre_s,
                    backend_exec_s=backend_exec_s,
                    backend_post_s=backend_post_s,
                )
                continue

            if xyxy.size > 0:
                if cls_ids.size == confs.size and np.all(np.isin(cls_ids, target_classes)):
                    class_mask = confs >= detection_min_conf
                else:
                    class_mask = np.isin(cls_ids, target_classes) & (confs >= detection_min_conf)
                if np.any(class_mask):
                    candidates = xyxy[class_mask]
                    candidate_classes = cls_ids[class_mask]
                    candidate_confs = confs[class_mask]
                    detections = []
                    for bbox_raw, cls_id, conf in zip(candidates, candidate_classes, candidate_confs):
                        x1, y1, x2, y2 = map(int, bbox_raw)
                        detections.append(
                            {
                                "bbox": (x1, y1, x2, y2),
                                "x": float(cap["left"] + ((x1 + x2) * 0.5)),
                                "y": float(cap["top"] + ((y1 + y2) * 0.5)),
                                "cls": int(cls_id),
                                "conf": float(conf),
                            }
                        )
                    locked_point = None
                    if target_tracker.initialized:
                        locked_point = target_tracker.get_state()[:2]
                    else:
                        target_age_s = (
                            max(0.0, time.time() - prev_target_ts) if prev_target_ts > 0.0 else float("inf")
                        )
                        if (
                            prev_target_found
                            and prev_target_cls in target_classes
                            and target_age_s <= TARGET_TIMEOUT_S
                        ):
                            locked_point = (float(prev_tx), float(prev_ty))
                    selected_detection, selected_meta = pick_sticky_target(
                        detections,
                        CENTER,
                        locked_point,
                        sticky_bias=sticky_bias_px,
                    )

            now_k = time.time()
            now_tick = time.perf_counter()
            dt = float(
                np.clip(
                    now_tick - last_track_tick if last_track_tick > 0.0 else KALMAN_MIN_DT,
                    KALMAN_MIN_DT,
                    KALMAN_MAX_DT,
                )
            )
            last_track_tick = now_tick
            target_tracker.predict(dt)

            raw_meas_x = None
            raw_meas_y = None
            target_switched = False
            trigger_fire = False
            if selected_detection is not None:
                raw_meas_x = float(selected_detection["x"])
                raw_meas_y = float(selected_detection["y"])
                target_switched = bool(selected_meta.get("switched", False))
                if target_switched and ego_motion_reset_on_switch:
                    reset_ego_motion_state(shared_lock, state)
                    ctrl_sent_vx_ema = 0.0
                    ctrl_sent_vy_ema = 0.0
                target_tracker.update(raw_meas_x, raw_meas_y)
                active_target_cls = int(selected_detection["cls"])
                lost_frames = 0
                x1, y1, x2, y2 = selected_detection["bbox"]
                last_box_w = float(max(1, x2 - x1))
                last_box_h = float(max(1, y2 - y1))
                if triggerbot_monitor_active:
                    trigger_scale = clamp(triggerbot_box_percent / 100.0, 0.01, 1.0)
                    trigger_half_w = max(1.0, (last_box_w * 0.5) * trigger_scale)
                    trigger_half_h = max(1.0, (last_box_h * 0.5) * trigger_scale)
                    trigger_fire = (
                        abs(raw_meas_x - CENTER[0]) <= trigger_half_w
                        and abs(raw_meas_y - CENTER[1]) <= trigger_half_h
                    )
            elif not target_tracker.initialized:
                with shared_lock:
                    state["target_found"] = False
                    state["last_target_full"] = CENTER
                    state["capture_focus_full"] = CENTER
                    state["target_speed"] = 0.0
                    state["target_cls"] = -1
                    state["aim_dx"] = 0
                    state["aim_dy"] = 0
                perf.record_inference(
                    frame_age_s=frame_age,
                    loop_s=time.perf_counter() - loop_perf_start,
                    stale_drop=False,
                    target_found=False,
                    yolo_s=yolo_elapsed_s,
                    tracker_s=tracker_elapsed_s,
                    backend_pre_s=backend_pre_s,
                    backend_exec_s=backend_exec_s,
                    backend_post_s=backend_post_s,
                )
                continue
            else:
                lost_frames += 1
                if lost_frames > target_max_lost_frames:
                    target_tracker.reset()
                    pid_x.reset()
                    pid_y.reset()
                    active_target_cls = -1
                    lost_frames = 0
                    last_box_w = 0.0
                    last_box_h = 0.0
                    last_pid_tick = now_tick
                    drain_queue(control_queue)
                    with shared_lock:
                        state["target_found"] = False
                        state["last_target_full"] = CENTER
                        state["capture_focus_full"] = CENTER
                        state["target_speed"] = 0.0
                        state["target_cls"] = -1
                        state["aim_dx"] = 0
                        state["aim_dy"] = 0
                    perf.record_inference(
                        frame_age_s=frame_age,
                        loop_s=time.perf_counter() - loop_perf_start,
                        stale_drop=False,
                        target_found=False,
                        yolo_s=yolo_elapsed_s,
                        tracker_s=tracker_elapsed_s,
                        backend_pre_s=backend_pre_s,
                        backend_exec_s=backend_exec_s,
                    backend_post_s=backend_post_s,
                )
                continue

            ego_vx = 0.0
            ego_vy = 0.0
            ego_gate_x = 1.0
            ego_gate_y = 1.0
            track_x, track_y, vx, vy, _, _ = target_tracker.get_state()
            ff_scale = target_tracker.feedforward_scale()
            lead_time_offset = (dt * ff_scale) if use_kalman else 0.0
            prediction_lead_time = max(0.0, float(prediction_time) - lead_time_offset)
            base_predicted_x = track_x + (vx * prediction_lead_time)
            base_predicted_y = track_y + (vy * prediction_lead_time)
            base_aim_y = base_predicted_y - (last_box_h * HEAD_Y_BIAS if local_aimmode == 0 else 0.0)
            if ego_motion_error_gate_enable:
                if ego_motion_error_gate_normalize_by_box and ego_motion_error_gate_norm_threshold > 1e-6:
                    norm_box_w = max(1.0, float(last_box_w))
                    norm_box_h = max(1.0, float(last_box_h))
                    normalized_error_x = abs(base_predicted_x - CENTER[0]) / norm_box_w
                    normalized_error_y = abs(base_aim_y - CENTER[1]) / norm_box_h
                    ego_gate_x = clamp(
                        1.0 - (normalized_error_x / ego_motion_error_gate_norm_threshold),
                        0.0,
                        1.0,
                    )
                    ego_gate_y = clamp(
                        1.0 - (normalized_error_y / ego_motion_error_gate_norm_threshold),
                        0.0,
                        1.0,
                    )
                elif ego_motion_error_gate_px > 1e-6:
                    ego_gate_x = clamp(
                        1.0 - (abs(base_predicted_x - CENTER[0]) / ego_motion_error_gate_px),
                        0.0,
                        1.0,
                    )
                    ego_gate_y = clamp(
                        1.0 - (abs(base_aim_y - CENTER[1]) / ego_motion_error_gate_px),
                        0.0,
                        1.0,
                    )
            if ego_motion_comp_enable and ff_scale > 0.0:
                ego_vx = clamp(
                    ctrl_sent_vx_ema * ego_motion_comp_gain_x * ff_scale * ego_gate_x,
                    -EGO_MOTION_COMP_MAX_PX_S,
                    EGO_MOTION_COMP_MAX_PX_S,
                )
                ego_vy = clamp(
                    ctrl_sent_vy_ema * ego_motion_comp_gain_y * ff_scale * ego_gate_y,
                    -EGO_MOTION_COMP_MAX_PX_S,
                    EGO_MOTION_COMP_MAX_PX_S,
                )
                # Camera motion makes stationary targets appear to move opposite the
                # mouse command. Remove that apparent motion from the velocity used
                # for prediction and feed-forward while keeping the raw position.
                vx += ego_vx
                vy += ego_vy
            speed = float(np.hypot(vx, vy))
            if speed > KALMAN_MAX_SPEED_PX_S and speed > 1e-6:
                speed_scale = KALMAN_MAX_SPEED_PX_S / speed
                vx *= speed_scale
                vy *= speed_scale
                speed = KALMAN_MAX_SPEED_PX_S

            predicted_x = track_x + (vx * prediction_lead_time)
            predicted_y = track_y + (vy * prediction_lead_time)
            aim_y = predicted_y - (last_box_h * HEAD_Y_BIAS if local_aimmode == 0 else 0.0)
            error_x = predicted_x - CENTER[0]
            error_y = aim_y - CENTER[1]

            pid_dt = float(np.clip(now_tick - last_pid_tick, KALMAN_MIN_DT, KALMAN_MAX_DT))
            last_pid_tick = now_tick
            if pid_enable:
                pid_term_x = pid_x.update(0.0, CENTER[0] - predicted_x, pid_dt, integrate=pid_integrate)
                pid_term_y = pid_y.update(0.0, CENTER[1] - aim_y, pid_dt, integrate=pid_integrate)
            else:
                pid_term_x = error_x
                pid_term_y = error_y

            ff_x = (vx * dt) * ff_scale
            ff_y = (vy * dt) * ff_scale
            desired_x = pid_term_x + ff_x
            desired_y = pid_term_y + ff_y
            if engage_active:
                dx = int(round(clamp(desired_x, -RAW_MAX_STEP_X, RAW_MAX_STEP_X)))
                dy = int(round(clamp(desired_y, -RAW_MAX_STEP_Y, RAW_MAX_STEP_Y)))
            else:
                dx = 0
                dy = 0

            focus_x = int(clamp(track_x, 0, SCREEN_W - 1))
            focus_y = int(clamp(track_y, 0, SCREEN_H - 1))
            selected_cls = int(active_target_cls if active_target_cls != -1 else target_cls)

            with shared_lock:
                state["target_found"] = True
                state["last_target_full"] = (focus_x, focus_y)
                state["capture_focus_full"] = (focus_x, focus_y)
                state["target_speed"] = speed
                state["target_cls"] = selected_cls
                state["aim_dx"] = dx
                state["aim_dy"] = dy
                state["target_ts"] = now_k
                state["aim_seq"] += 1

            if engage_active or trigger_fire:
                put_latest(
                    control_queue,
                    {
                        "dx": dx,
                        "dy": dy,
                        "ts": now_k,
                        "frame_ts": frame_ts,
                        "capture_ts": capture_ts,
                        "trigger_fire": trigger_fire,
                    },
                )
            perf.record_inference(
                frame_age_s=frame_age,
                loop_s=time.perf_counter() - loop_perf_start,
                stale_drop=False,
                target_found=True,
                yolo_s=yolo_elapsed_s,
                tracker_s=tracker_elapsed_s,
                backend_pre_s=backend_pre_s,
                backend_exec_s=backend_exec_s,
                backend_post_s=backend_post_s,
                cmd_latency_s=max(0.0, now_k - frame_ts),
            )

            if DEBUG_LOG:
                debug_source = "det" if selected_detection is not None else f"pred/{lost_frames}"
                meas_info = ""
                if raw_meas_x is not None and raw_meas_y is not None:
                    meas_info = (
                        f" raw=({int(raw_meas_x)},{int(raw_meas_y)})"
                        f" cmdV=({int(ctrl_sent_vx_ema)},{int(ctrl_sent_vy_ema)})"
                        f" egoVel=({int(ego_vx)},{int(ego_vy)})"
                        f" egoGate=({ego_gate_x:.2f},{ego_gate_y:.2f})"
                        f" lead={prediction_lead_time:.3f}s"
                    )
                print(
                    f"[TRACK] trk={tracking_strategy} src={debug_source} cls={selected_cls} "
                    f"lock=({focus_x},{focus_y}) pred=({int(predicted_x)},{int(aim_y)}) "
                    f"v=({int(vx)},{int(vy)}) "
                    f"pid=({pid_term_x:.2f},{pid_term_y:.2f}) ff=({ff_x:.2f},{ff_y:.2f}) "
                    f"dt={pid_dt:.4f} ffScale={ff_scale:.2f}{meas_info}"
                )

    def control_loop():
        nonlocal running
        set_current_thread_priority(CONTROL_THREAD_PRIORITY)
        last_trigger_click_ts = 0.0
        while running:
            try:
                cmd = get_latest(control_queue, 0.01)
            except queue.Empty:
                runtime_snapshot = pid_runtime.snapshot()
                with shared_lock:
                    local_mode = state["mode"]
                    left_pressed = bool(state.get("left_pressed", False))
                    left_hold_engage = bool(state.get("left_hold_engage", False))
                    right_pressed = bool(state.get("right_pressed", False))
                    recoil_tune_fallback = bool(state.get("recoil_tune_fallback", False))
                recoil_tune_fallback_ignore_mode_check = bool(
                    runtime_snapshot.get(
                        "recoil_tune_fallback_ignore_mode_check",
                        RECOIL_TUNE_FALLBACK_IGNORE_MODE_CHECK_DEFAULT,
                    )
                )
                mode_ok = recoil_tune_fallback_ignore_mode_check or (local_mode != 0)
                engage_ok = (not left_hold_engage) or right_pressed
                if recoil_tune_fallback and left_pressed and mode_ok and engage_ok:
                    now = time.time()
                    cmd = {
                        "dx": 0,
                        "dy": 0,
                        "ts": now,
                        "frame_ts": now,
                        "capture_ts": now,
                        "synthetic_recoil": True,
                    }
                else:
                    with shared_lock:
                        state["ctrl_sent_vx_ema"] = (
                            float(state.get("ctrl_sent_vx_ema", 0.0)) * EGO_MOTION_COMP_DECAY
                        )
                        state["ctrl_sent_vy_ema"] = (
                            float(state.get("ctrl_sent_vy_ema", 0.0)) * EGO_MOTION_COMP_DECAY
                        )
                    continue

            synthetic_recoil = bool(cmd.get("synthetic_recoil", False))
            trigger_fire = bool(cmd.get("trigger_fire", False))
            if synthetic_recoil:
                left_pressed = True
                local_mode = 1
                left_hold_engage = False
                engage_active = True
            else:
                with shared_lock:
                    local_mode = state["mode"]
                    left_pressed = bool(state.get("left_pressed", False))
                    left_hold_engage = bool(state.get("left_hold_engage", False))
                    right_pressed = bool(state.get("right_pressed", False))
                engage_active = (local_mode != 0) and ((not left_hold_engage) or right_pressed)

            now = time.time()
            cmd_age = now - cmd["ts"]
            frame_ts_cmd = cmd.get("frame_ts", cmd["ts"])
            capture_ts_cmd = cmd.get("capture_ts", frame_ts_cmd)
            total_latency = now - frame_ts_cmd
            total_latency_full = now - capture_ts_cmd
            if cmd_age > TARGET_TIMEOUT_S:
                with shared_lock:
                    state["ctrl_sent_vx_ema"] = float(state.get("ctrl_sent_vx_ema", 0.0)) * EGO_MOTION_COMP_DECAY
                    state["ctrl_sent_vy_ema"] = float(state.get("ctrl_sent_vy_ema", 0.0)) * EGO_MOTION_COMP_DECAY
                perf.record_control(
                    cmd_age_s=cmd_age,
                    sent=False,
                    stale_drop=True,
                    total_latency_s=total_latency,
                    total_latency_full_s=total_latency_full,
                )
                continue

            if not (engage_active or trigger_fire):
                with shared_lock:
                    state["ctrl_sent_vx_ema"] = float(state.get("ctrl_sent_vx_ema", 0.0)) * EGO_MOTION_COMP_DECAY
                    state["ctrl_sent_vy_ema"] = float(state.get("ctrl_sent_vy_ema", 0.0)) * EGO_MOTION_COMP_DECAY
                perf.record_control(
                    cmd_age_s=cmd_age,
                    sent=False,
                    mode_drop=True,
                    total_latency_s=total_latency,
                    total_latency_full_s=total_latency_full,
                )
                continue

            with shared_lock:
                state["ctrl_cmd_age_ema"] = ema_update(
                    float(state.get("ctrl_cmd_age_ema", 0.0)),
                    max(0.0, cmd_age),
                    DELAY_COMP_EMA_ALPHA,
                )

            dx = int(cmd["dx"])
            dy = int(cmd["dy"])
            control_runtime = pid_runtime.snapshot()
            recoil_compensation_y_px = float(
                control_runtime.get("recoil_compensation_y_px", RECOIL_COMPENSATION_Y_PX)
            )
            triggerbot_click_hold_s = float(
                control_runtime.get("triggerbot_click_hold_s", TRIGGERBOT_CLICK_HOLD_S)
            )
            triggerbot_click_cooldown_s = float(
                control_runtime.get("triggerbot_click_cooldown_s", TRIGGERBOT_CLICK_COOLDOWN_S)
            )
            mouse_client.configure(
                ghub_max_step=control_runtime.get("ghub_max_step", GHUB_MAX_STEP)
            )
            trigger_sent_ok = False
            trigger_click_now = time.time()
            trigger_will_click = (
                trigger_fire
                and (not left_pressed)
                and ((trigger_click_now - last_trigger_click_ts) >= triggerbot_click_cooldown_s)
            )
            if RECOIL_CONTROL_ENABLE and (left_pressed or trigger_will_click):
                dy = int(clamp(dy + recoil_compensation_y_px, -RAW_MAX_STEP_Y, RAW_MAX_STEP_Y))
            movement_sent_ok = False
            send_elapsed = 0.0
            send_end_tick = time.perf_counter()
            if dx != 0 or dy != 0:
                send_start = time.perf_counter()
                movement_sent_ok = mouse_client.send_input(dx, dy)
                send_end_tick = time.perf_counter()
                send_elapsed = send_end_tick - send_start

            if trigger_will_click:
                trigger_sent_ok = mouse_client.click_left(triggerbot_click_hold_s)
                if trigger_sent_ok:
                    last_trigger_click_ts = trigger_click_now

            if dx == 0 and dy == 0 and not trigger_sent_ok:
                with shared_lock:
                    state["ctrl_sent_vx_ema"] = float(state.get("ctrl_sent_vx_ema", 0.0)) * EGO_MOTION_COMP_DECAY
                    state["ctrl_sent_vy_ema"] = float(state.get("ctrl_sent_vy_ema", 0.0)) * EGO_MOTION_COMP_DECAY
                perf.record_control(
                    cmd_age_s=cmd_age,
                    sent=False,
                    total_latency_s=total_latency,
                    total_latency_full_s=total_latency_full,
                )
                continue

            sent_ok = movement_sent_ok or trigger_sent_ok
            send_end = time.time()
            total_apply_latency = send_end - frame_ts_cmd
            total_apply_latency_full = send_end - capture_ts_cmd
            if (dx != 0 or dy != 0) and movement_sent_ok:
                with shared_lock:
                    state["ctrl_send_ema"] = ema_update(
                        float(state.get("ctrl_send_ema", 0.0)),
                        max(0.0, send_elapsed),
                        DELAY_COMP_EMA_ALPHA,
                    )
                    last_send_tick = float(state.get("ctrl_last_send_tick", 0.0))
                    if last_send_tick > 0.0:
                        send_dt = max(1e-4, send_end_tick - last_send_tick)
                        sent_vx = clamp(float(dx) / send_dt, -EGO_MOTION_COMP_MAX_PX_S, EGO_MOTION_COMP_MAX_PX_S)
                        sent_vy = clamp(float(dy) / send_dt, -EGO_MOTION_COMP_MAX_PX_S, EGO_MOTION_COMP_MAX_PX_S)
                        state["ctrl_sent_vx_ema"] = ema_update_signed(
                            float(state.get("ctrl_sent_vx_ema", 0.0)),
                            sent_vx,
                            EGO_MOTION_COMP_ALPHA,
                        )
                        state["ctrl_sent_vy_ema"] = ema_update_signed(
                            float(state.get("ctrl_sent_vy_ema", 0.0)),
                            sent_vy,
                            EGO_MOTION_COMP_ALPHA,
                        )
                    state["ctrl_last_send_tick"] = send_end_tick
            elif not trigger_sent_ok:
                with shared_lock:
                    state["ctrl_sent_vx_ema"] = float(state.get("ctrl_sent_vx_ema", 0.0)) * EGO_MOTION_COMP_DECAY
                    state["ctrl_sent_vy_ema"] = float(state.get("ctrl_sent_vy_ema", 0.0)) * EGO_MOTION_COMP_DECAY
            perf.record_control(
                cmd_age_s=cmd_age,
                sent=sent_ok,
                send_s=send_elapsed if sent_ok else 0.0,
                total_latency_s=total_latency,
                total_latency_full_s=total_latency_full,
                total_apply_latency_s=total_apply_latency if sent_ok else None,
                total_apply_latency_full_s=total_apply_latency_full if sent_ok else None,
            )

    def perf_loop():
        nonlocal running
        while running:
            time.sleep(0.05)
            snapshot = perf.snapshot(PERF_LOG_INTERVAL_S)
            if snapshot is None:
                continue
            with shared_lock:
                perf_mode = state["mode"]
                left_hold_engage = bool(state.get("left_hold_engage", False))
                right_pressed = bool(state.get("right_pressed", False))
            perf_engaged = (perf_mode != 0) and ((not left_hold_engage) or right_pressed)
            if (not PERF_LOG_WHEN_MODE_OFF) and (not perf_engaged):
                continue

            backend_breakdown = ""
            if snapshot["infer_backend_samples"]:
                backend_breakdown = (
                    f" onnx(pre/exec/post)="
                    f"{snapshot['infer_backend_pre_ms']:.2f}/"
                    f"{snapshot['infer_backend_exec_ms']:.2f}/"
                    f"{snapshot['infer_backend_post_ms']:.2f}ms"
                )

            print(
                "[PERF] "
                f"cap={snapshot['cap_fps']:.1f}fps grab={snapshot['cap_grab_ms']:.2f}ms none={snapshot['cap_none']} | "
                f"inf={snapshot['infer_fps']:.1f}fps loop={snapshot['infer_loop_ms']:.2f}ms "
                f"age={snapshot['infer_age_ms']:.2f}/{snapshot['infer_age_max_ms']:.2f}ms stale={snapshot['infer_stale']} "
                f"lock={snapshot['infer_lock_rate'] * 100.0:.0f}% | "
                f"yolo={snapshot['yolo_hz']:.1f}Hz@{snapshot['yolo_ms']:.2f}ms "
                f"trk={snapshot['tracker_hz']:.1f}Hz@{snapshot['tracker_ms']:.2f}ms"
                f"{backend_breakdown} | "
                f"ctl={snapshot['control_send_hz']:.1f}Hz send={snapshot['control_send_ms']:.3f}ms "
                f"cmdAge={snapshot['control_cmd_age_ms']:.2f}ms "
                f"e2e={snapshot['control_total_latency_ms']:.2f}ms "
                f"e2eIn={snapshot['control_total_apply_latency_ms']:.2f}ms "
                f"e2eFull={snapshot['control_total_latency_full_ms']:.2f}ms "
                f"e2eFullIn={snapshot['control_total_apply_latency_full_ms']:.2f}ms "
                f"drop(stale/mode)={snapshot['control_stale_drop']}/{snapshot['control_mode_drop']} "
                f"aimPipe={snapshot['infer_cmd_ms']:.2f}ms"
            )

    def update_left_pressed(pressed):
        pressed = bool(pressed)
        with shared_lock:
            state["left_pressed"] = pressed

    def update_right_pressed(pressed):
        pressed = bool(pressed)
        with shared_lock:
            state["right_pressed"] = pressed
            local_left_hold_engage = bool(state.get("left_hold_engage", False))
        if (not pressed) and local_left_hold_engage:
            drain_queue(control_queue)

    def on_click(x, y, button, pressed):
        nonlocal mouse_button_pressed_x1, mouse_button_pressed_x2
        if Button is None:
            return
        if button == Button.x2:
            mouse_button_pressed_x2 = bool(pressed)
        if button == Button.x1:
            mouse_button_pressed_x1 = bool(pressed)
        if button == Button.left:
            update_left_pressed(pressed)
        if button == Button.right:
            update_right_pressed(pressed)

    capture_thread = threading.Thread(target=capture_loop, daemon=True)
    inference_thread = threading.Thread(target=inference_loop_simple, daemon=True)
    control_thread = threading.Thread(target=control_loop, daemon=True)
    perf_thread = None
    if PERF_LOG_ENABLE:
        perf_thread = threading.Thread(target=perf_loop, daemon=True)
    capture_thread.start()
    inference_thread.start()
    control_thread.start()
    if perf_thread is not None:
        perf_thread.start()

    listener_context = Listener(on_click=on_click) if Listener is not None else nullcontext()
    with listener_context:
        while running:
            try:
                if keyboard.is_pressed("insert"):
                    running = False
                    break

                if Listener is None:
                    mouse_button_pressed_x2 = is_vk_pressed(VK_XBUTTON2)
                    mouse_button_pressed_x1 = is_vk_pressed(VK_XBUTTON1)
                    polled_left_pressed = is_vk_pressed(VK_LBUTTON)
                    if polled_left_pressed != last_polled_left_pressed:
                        update_left_pressed(polled_left_pressed)
                        last_polled_left_pressed = polled_left_pressed
                    polled_right_pressed = is_vk_pressed(VK_RBUTTON)
                    if polled_right_pressed != last_polled_right_pressed:
                        update_right_pressed(polled_right_pressed)
                        last_polled_right_pressed = polled_right_pressed

                now = time.time()
                if keyboard.is_pressed(LEFT_HOLD_ENGAGE_TOGGLE_KEY) and (
                    now - last_left_hold_toggle > LEFT_HOLD_ENGAGE_TOGGLE_COOLDOWN_S
                ):
                    left_hold_engage = not left_hold_engage
                    with shared_lock:
                        state["left_hold_engage"] = left_hold_engage
                        right_pressed = bool(state.get("right_pressed", False))
                    last_left_hold_toggle = now
                    if left_hold_engage and (not right_pressed):
                        drain_queue(control_queue)
                        with shared_lock:
                            state["target_found"] = False
                            state["target_speed"] = 0.0
                            state["capture_focus_full"] = CENTER
                            state["target_cls"] = -1
                            state["aim_dx"] = 0
                            state["aim_dy"] = 0
                    print(
                        f"RightHoldEngage: {'ON' if left_hold_engage else 'OFF'} "
                        f"({LEFT_HOLD_ENGAGE_TOGGLE_KEY})"
                    )
                    winsound.Beep(1400 if left_hold_engage else 700, 100)

                if keyboard.is_pressed(RECOIL_TUNE_FALLBACK_TOGGLE_KEY) and (
                    now - last_recoil_tune_toggle > RECOIL_TUNE_FALLBACK_TOGGLE_COOLDOWN_S
                ):
                    recoil_tune_fallback = not recoil_tune_fallback
                    with shared_lock:
                        state["recoil_tune_fallback"] = recoil_tune_fallback
                    last_recoil_tune_toggle = now
                    print(
                        f"RecoilTuneFallback: {'ON' if recoil_tune_fallback else 'OFF'} "
                        f"({RECOIL_TUNE_FALLBACK_TOGGLE_KEY})"
                    )
                    winsound.Beep(1500 if recoil_tune_fallback else 800, 100)

                if keyboard.is_pressed(TRIGGERBOT_TOGGLE_KEY) and (
                    now - last_triggerbot_toggle > TRIGGERBOT_TOGGLE_COOLDOWN_S
                ):
                    triggerbot_enabled = bool(
                        pid_runtime.snapshot().get("triggerbot_enable", TRIGGERBOT_ENABLE_DEFAULT)
                    )
                    triggerbot_enabled = not triggerbot_enabled
                    pid_runtime.update(
                        {"triggerbot_enable": triggerbot_enabled},
                        reset_pid=False,
                    )
                    last_triggerbot_toggle = now
                    print(
                        f"TriggerBot: {'ON' if triggerbot_enabled else 'OFF'} "
                        f"({TRIGGERBOT_TOGGLE_KEY})"
                    )
                    winsound.Beep(1600 if triggerbot_enabled else 900, 100)

                if mouse_button_pressed_x2 and now - last_toggle > 0.2:
                    mode = (mode + 1) % 2
                    with shared_lock:
                        state["mode"] = mode
                    last_toggle = now
                    freq = 1000 if mode == 1 else 500
                    if mode == 0:
                        drain_queue(control_queue)
                        with shared_lock:
                            state["target_found"] = False
                            state["target_speed"] = 0.0
                            state["capture_focus_full"] = CENTER
                            state["target_cls"] = -1
                            state["aim_dx"] = 0
                            state["aim_dy"] = 0
                    mode_label = "ACTIVE" if mode == 1 else "OFF"
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

                time.sleep(SLEEP_TIME)
            except KeyboardInterrupt:
                running = False
                break

    with shared_lock:
        state["running"] = False
    if pid_frontend_server is not None:
        pid_frontend_server.shutdown()
        pid_frontend_server.server_close()
    mouse_client.close()


if __name__ == "__main__":
    main()
