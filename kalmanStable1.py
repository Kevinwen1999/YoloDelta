import socket
import threading
import time
import queue
import os
import importlib.util
import struct
import ctypes

import cv2
import keyboard
import mss
import numpy as np
import winsound
from pynput.mouse import Button, Listener
from ultralytics import YOLO

try:
    import pydirectinput
except ImportError:
    pydirectinput = None

try:
    import dxcam
except ImportError:
    dxcam = None

# ---------------- CONFIG ----------------
MODEL_PATH = r"C:\YOLO\Delta\runs\detect\train\weights\best.pt"
IMGSZ = 416  # requested input size; fixed-shape ONNX models may override at runtime
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
ONNX_ENABLE_CUDA_GRAPH = False
ONNX_CUDA_DEVICE_ID = 0

SCREEN_W = 2560
SCREEN_H = 1440
CENTER = (SCREEN_W // 2, SCREEN_H // 2)
FPS = 10000
SLEEP_TIME = 1 / FPS
CAPTURE_BACKEND = "mss"  # "auto" | "dxcam" | "mss"
CAPTURE_DEVICE_IDX = 0
CAPTURE_OUTPUT_IDX = 0
CAPTURE_BENCHMARK_ON_START = False
CAPTURE_BENCHMARK_SECONDS = 2.0
CAPTURE_BENCHMARK_W = 700
CAPTURE_BENCHMARK_H = 700
CAPTURE_YIELD_EACH_LOOP = True
CAPTURE_NONE_BACKOFF_S = 0.0
CAPTURE_THREAD_PRIORITY = "highest"  # "normal" | "above_normal" | "highest"
MSS_USE_RAW_BUFFER = True

BASE_CROP_W = 300
BASE_CROP_H = 300
LOST_CROP_W = 500
LOST_CROP_H = 500
MIN_ACTIVE_CROP_W = 420
MIN_ACTIVE_CROP_H = 420
MAX_ACTIVE_CROP_W = 860
MAX_ACTIVE_CROP_H = 860
CAPTURE_SPEED_TO_PAD = 0.20
CAPTURE_PAD_MAX = 220

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
USE_KALMAN_DEFAULT = True
TARGET_LOCK_MAX_JUMP = 260  # px; avoid hopping to far detections when already locked
KALMAN_MAX_SPEED_PX_S = 1800.0
KALMAN_MAX_LEAD_PX = 500.0
KALMAN_MAX_DT = 0.06
KALMAN_MIN_DT = 1.0 / 240.0
MIN_MEAS_DT = 1.0 / 120.0
KALMAN_POSITION_BLEND = 0.10  # lower blend for less perceived lag
KALMAN_LEAD_MIN_SPEED = 35.0 # px/s; ignore tiny velocity noise
MEAS_VELOCITY_BLEND = 0.32
MAX_ACCEL_PX_S2 = 14000.0
LEAD_SMOOTH_ALPHA_SLOW = 0.40
LEAD_SMOOTH_ALPHA_FAST = 0.72
KALMAN_EXTRA_SMOOTH_SLOW = 0.06
KALMAN_EXTRA_SMOOTH_FAST = 0.02
ADAPTIVE_SPEED_MIN = 120.0
ADAPTIVE_SPEED_MAX = 750.0
VELOCITY_STOP_ENTER_THRESHOLD = 35.0
VELOCITY_STOP_EXIT_THRESHOLD = 60.0
VELOCITY_STOP_HOLD_DECAY = 0.65
VELOCITY_STOP_MIN_KEEP_SPEED = 12.0
POSITION_DEADZONE_SLOW = 4.0
POSITION_DEADZONE_FAST = 2.0
EXTRA_PIPELINE_LAG_S = 0.015
NON_STATIONARY_EXTRA_LEAD_S = 0.02
PREDICTIVE_CAPTURE_LEAD_S = 0.03
DELAY_COMPENSATION_ENABLE = True
DELAY_COMP_EMA_ALPHA = 0.22
DELAY_COMP_INPUT_APPLY_S = 0.0025
DELAY_COMP_MAX_S = 0.04
DELAY_COMP_MIN_SPEED = 80.0
EGO_MOTION_COMP_ENABLE = True
EGO_MOTION_COMP_ALPHA = 0.30
EGO_MOTION_COMP_GAIN_X = 0.95
EGO_MOTION_COMP_GAIN_Y = 0.95
EGO_MOTION_COMP_MAX_PX_S = 3200.0
EGO_MOTION_COMP_DECAY = 0.92
PREDICTION_PIPELINE_AGE_USE_CURRENT = True
PREDICTION_LEAD_GAIN_BASE = 1.15
PREDICTION_LEAD_GAIN_FAST = 1.70
PREDICTION_NEAR_SCALE_FLOOR_FAST = 0.35
PREDICTION_CONF_SCALE_FLOOR_FAST = 0.75
KALMAN_USE_CA_MODEL = True
PREDICTION_HORIZON_USE_CAPTURE_AGE = True
PREDICTION_ACCEL_GAIN = 0.70
PREDICTION_MAX_ACCEL_PX_S2 = 22000.0
PREDICTION_MOVING_LEAD_SCALE_FLOOR = 0.55
MAX_FRAME_AGE_S = 0.05
SPEED_BOOST_THRESHOLD = 250.0
SPEED_BOOST_GAIN = 1.5
AIM_DEADZONE_SLOW = 1.0
AIM_DEADZONE_FAST = 0.0
DEBUG_LOG = False
PERF_LOG_ENABLE = True
PERF_LOG_INTERVAL_S = 1.0
PERF_LOG_WHEN_MODE_OFF = False

PIPELINE_FRAME_QUEUE = 3
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

PID_ENABLE = False
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
PID_OUTPUT_MAX = 320.0
PID_MICRO_ERROR_PX = 2.2
PID_SOFT_ERROR_PX = 7.0
PID_SOFT_ZONE_GAIN = 0.70

# PIDF controller (PID + velocity feed-forward from tracker/Kalman velocity).
PIDF_ENABLE = True
PIDF_KP_X = 0.45
PIDF_KI_X = 0.0
PIDF_KD_X = 0.03
PIDF_KP_Y = 0.45
PIDF_KI_Y = 0.0
PIDF_KD_Y = 0.028
PIDF_FF_GAIN_X = 1.30
PIDF_FF_GAIN_Y = 1.30

OUTPUT_MAX_STEP_X = 220
OUTPUT_MAX_STEP_Y = 220
OUTPUT_MAX_DELTA_X = 85
OUTPUT_MAX_DELTA_Y = 85
OUTPUT_MICRO_CMD_PX = 1

# Raw-output anti-oscillation tuning (keeps far movement instant, damps only near center).
RAW_AIM_DEADZONE_PX = 1.0
RAW_NEAR_ERROR_PX = 18.0
RAW_NEAR_GAIN = 0.30
RAW_FAR_GAIN = 0.60
RAW_MAX_STEP_X = 280
RAW_MAX_STEP_Y = 280
RAW_SIGN_FLIP_STOP_PX = 8.0
RAW_SIGN_FLIP_COOLDOWN_FRAMES = 2
MOVING_TRACK_SPEED_THRESHOLD = 0.0
MOVING_DEADZONE_MIN_SCALE = 0.15
MOVING_SIGN_FLIP_DISABLE_SPEED = FAST_TARGET_SPEED
MOVING_SIGN_FLIP_BYPASS_ERROR_PX = 38.0
MOVING_SIGN_FLIP_BYPASS_ERROR_PX_FAST = 70.0
MOVING_SIGN_FLIP_STOP_PX_FAST_MULT = 1.6
MOVING_RESIDUAL_DEADZONE_PX = 0.9
MOVING_RESIDUAL_CMD_PX = 1
MOVING_FF_NEAR_BOOST = 1.95
MOVING_FF_BOOST_ERROR_PX = 14.0
MOVING_FF_MISALIGN_SCALE = 0.30
MOVING_FF_NEAR_CAP_PX = 5.0
MOVING_FF_FAR_CAP_PX = 42.0
MOVING_FF_CAP_ERROR_PX = 28.0
MOVING_FF_MIN_CAP_FAST = 8.0
MOVING_LAG_BIAS_ENABLE = True
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
MOVING_STICK_BLEND_INNER_PX = 4.0
MOVING_STICK_BLEND_OUTER_PX = 24.0
MOVING_STICK_PID_WEIGHT = 0.45
MOVING_STICK_LAG_WEIGHT = 0.25
# Close-target oscillation damping (targets with large on-screen box are "close").
CLOSE_TARGET_BOX_H_ENTER_PX = 50.0
CLOSE_TARGET_BOX_H_FULL_PX = 260.0
CLOSE_DEADZONE_ADD_PX = 1.4
CLOSE_SIGN_FLIP_EXTRA_MULT = 1.4
CLOSE_BYPASS_ERROR_EXTRA_PX = 106.0
CLOSE_SLEW_EXTRA_DAMP = 0.85
CLOSE_LAG_CAP_REDUCE = 0.55
CLOSE_FF_CAP_REDUCE = 0.45
MOVING_AHEAD_HOLD_ENABLE = True
MOVING_AHEAD_HOLD_MIN_SPEED = 110.0
MOVING_AHEAD_HOLD_ERROR_PX = 8.0
MOVING_AHEAD_HOLD_CLOSE_EXTRA_PX = 10.0
MOVING_AHEAD_HOLD_LAG_DECAY = 0.60
MOVING_AHEAD_HOLD_OPPOSITE_SCALE = 0.35
CONTROL_OUTPUT_RATE_NORMALIZE = True
CONTROL_REFERENCE_HZ = 65.0
CONTROL_RATE_SCALE_MIN = 0.25
CONTROL_RATE_SCALE_MAX = 1.35
CONTROL_RATE_BLEND_INNER_PX = 10.0
CONTROL_RATE_BLEND_OUTER_PX = 220.0
CONTROL_RATE_MOVING_SCALE_FLOOR = 1.0
SLEW_LIMITER_FACTOR = 0.25

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
MOUSE_SEND_BACKEND = "socket"  # "pydirectinput" | "socket"
MOUSE_SOCKET_BINARY = True
PYDIRECTINPUT_GAIN_X = 1.0
PYDIRECTINPUT_GAIN_Y = 1.0
RECOIL_CONTROL_ENABLE = True
RECOIL_COMPENSATION_Y_PX = 6  # positive Y nudges aim down while firing

WARMUP_CLASS = 0
TARGET_TIMEOUT_S = 0.08
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


def ema_update(prev, sample, alpha):
    alpha = clamp(float(alpha), 0.0, 1.0)
    sample = float(sample)
    if alpha <= 0.0:
        return prev
    if prev <= 0.0:
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


class DXGICaptureSource:
    def __init__(self):
        if dxcam is None:
            raise RuntimeError("dxcam not installed")
        self._needs_rgb_to_bgr = False
        try:
            self._cam = dxcam.create(
                device_idx=CAPTURE_DEVICE_IDX,
                output_idx=CAPTURE_OUTPUT_IDX,
                output_color="BGR",
            )
        except TypeError:
            # Older dxcam versions may not accept output_color.
            self._cam = dxcam.create(
                device_idx=CAPTURE_DEVICE_IDX,
                output_idx=CAPTURE_OUTPUT_IDX,
            )
            self._needs_rgb_to_bgr = True
        self.name = "dxcam(dxgi)"

    def grab(self, cap):
        frame = self._cam.grab(region=cap_to_region_tuple(cap))
        if frame is None:
            return None
        if self._needs_rgb_to_bgr:
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame

    def close(self):
        if self._cam is not None:
            stop = getattr(self._cam, "stop", None)
            if callable(stop):
                try:
                    stop()
                except Exception:
                    pass
            self._cam = None


def build_capture_source():
    backend = str(CAPTURE_BACKEND).strip().lower()
    if backend not in ("auto", "dxcam", "mss"):
        print(f"[WARN] Invalid CAPTURE_BACKEND='{CAPTURE_BACKEND}', using auto.")
        backend = "auto"

    if backend in ("auto", "dxcam"):
        try:
            return DXGICaptureSource()
        except Exception as e:
            if backend == "dxcam":
                print(f"[WARN] DXGI capture init failed ({e}); falling back to MSS.")
            else:
                print(f"[INFO] DXGI capture unavailable ({e}); using MSS.")

    return MSSCaptureSource()


def benchmark_capture_backend(backend_name, seconds, capture_region):
    backend_name = str(backend_name).strip().lower()
    source = None
    try:
        if backend_name == "dxcam":
            source = DXGICaptureSource()
        elif backend_name == "mss":
            source = MSSCaptureSource()
        else:
            raise ValueError(f"Unsupported backend '{backend_name}'")

        frames = 0
        none_count = 0
        grab_total_s = 0.0
        start = time.perf_counter()
        while (time.perf_counter() - start) < float(seconds):
            t0 = time.perf_counter()
            frame = source.grab(capture_region)
            dt = time.perf_counter() - t0
            if frame is None:
                none_count += 1
                continue
            frames += 1
            grab_total_s += dt

        elapsed = max(1e-6, time.perf_counter() - start)
        return {
            "backend": backend_name,
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
    print(f"[INFO] Running capture benchmark ({seconds:.1f}s, {w}x{h}) ...")
    for backend_name in ("dxcam", "mss"):
        try:
            result = benchmark_capture_backend(backend_name, seconds, cap)
            print(
                f"[CAPTURE-BENCH] {result['backend']}: "
                f"{result['fps']:.1f} fps, grab={result['grab_ms']:.2f} ms, none={result['none']}"
            )
        except Exception as e:
            print(f"[CAPTURE-BENCH] {backend_name}: unavailable ({e})")


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
                "infer_cmd_ms": infer_cmd_ms,
                "control_send_hz": control_send_hz,
                "control_send_ms": control_send_ms,
                "control_cmd_age_ms": control_cmd_age_ms,
                "control_total_latency_ms": control_total_latency_ms,
                "control_total_latency_full_ms": control_total_latency_full_ms,
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
        cv2.resize(
            frame,
            (self.input_w, self.input_h),
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
        if not outputs:
            return (
                np.empty((0, 4), dtype=np.int32),
                np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=np.float32),
            )

        if ONNX_OUTPUT_HAS_NMS:
            arr0 = np.asarray(outputs[0])
            if arr0.ndim == 3 and arr0.shape[0] == 1 and arr0.shape[-1] >= 6:
                decoded = self._decode_nms_output(arr0[0], frame_w, frame_h, target_classes)
                if decoded is not None:
                    return decoded

        for out in outputs:
            arr = np.asarray(out)
            if arr.ndim == 3 and arr.shape[0] == 1 and 6 <= arr.shape[-1] <= 8:
                decoded = self._decode_nms_output(arr[0], frame_w, frame_h, target_classes)
                if decoded is not None:
                    return decoded

        raw = np.asarray(outputs[0])
        if raw.ndim == 3 and raw.shape[0] == 1:
            raw = raw[0]
        return self._decode_raw_output(raw, frame_w, frame_h, target_classes)


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

    def __init__(self, host=MOUSE_HOST, port=MOUSE_PORT, send_backend=MOUSE_SEND_BACKEND):
        self.host = host
        self.port = port
        self.sock = None
        self.lock = threading.Lock()
        self.last_connect_fail_ts = 0.0
        self.send_backend = str(send_backend).lower()
        self._pdi_frac_x = 0.0
        self._pdi_frac_y = 0.0
        self._send_buf = bytearray(8)
        self._send_view = memoryview(self._send_buf)
        self._binary_mode = bool(MOUSE_SOCKET_BINARY)

        if self.send_backend == "pydirectinput":
            if pydirectinput is None:
                print("[WARN] pydirectinput not installed; falling back to socket mouse backend.")
                self.send_backend = "socket"
            else:
                pydirectinput.FAILSAFE = False
                pydirectinput.PAUSE = 0

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
            if self.send_backend == "pydirectinput":
                return self.send_input_pydirectinput(x, y)
            return self._send_socket(x, y)

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
        "ctrl_cmd_age_ema": 0.0,
        "ctrl_send_ema": 0.0,
        "ctrl_sent_vx_ema": 0.0,
        "ctrl_sent_vy_ema": 0.0,
        "ctrl_last_send_ts": 0.0,
        "left_pressed": False,
    }
    perf = RuntimePerf()

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
                    else:
                        pad = int(min(CAPTURE_PAD_MAX, target_speed * CAPTURE_SPEED_TO_PAD))
                        crop_w = int(clamp(BASE_CROP_W + pad, MIN_ACTIVE_CROP_W, MAX_ACTIVE_CROP_W))
                        crop_h = int(clamp(BASE_CROP_H + pad, MIN_ACTIVE_CROP_H, MAX_ACTIVE_CROP_H))
                    cap = build_capture(tx, ty, crop_w, crop_h)
                else:
                    cap = build_capture(CENTER[0], CENTER[1], LOST_CROP_W, LOST_CROP_H)

                capture_ts = time.time()
                grab_start = time.perf_counter()
                frame_bgr = capture_source.grab(cap)
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

    def inference_loop():
        nonlocal running, kalman_initialized, last_kalman_ts
        tracker = None
        tracker_active = False
        tracker_warned_unavailable = False
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
        flip_cooldown_x = 0
        flip_cooldown_y = 0
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
        tracker_box_local = None
        tracker_frame_shape = None
        last_tracker_update_err_ts = 0.0
        last_backend_err_ts = 0.0
        miss_count = 0
        bezier_curve_sign = 1.0
        last_pid_ts = time.time()
        pid_x = PIDController(PIDF_KP_X, PIDF_KI_X, PIDF_KD_X)
        pid_y = PIDController(PIDF_KP_Y, PIDF_KI_Y, PIDF_KD_Y)
        last_accel_mag = 0.0
        velocity_stop_latched = True
        moving_lag_bias_x = 0.0
        moving_lag_bias_y = 0.0
        kalman_q_eye = np.eye(kalman_state_dim, dtype=np.float32)
        kalman_r_eye = np.eye(2, dtype=np.float32)
        measurement = np.zeros((2, 1), dtype=np.float32)

        while running:
            with shared_lock:
                local_mode = state["mode"]
                local_aimmode = state["aimmode"]
                local_use_kalman = state["use_kalman"]
                prev_target_found = state["target_found"]
                prev_tx, prev_ty = state["last_target_full"]
                prev_target_cls = state["target_cls"]
                ctrl_cmd_age_ema = float(state.get("ctrl_cmd_age_ema", 0.0))
                ctrl_send_ema = float(state.get("ctrl_send_ema", 0.0))
                ctrl_sent_vx_ema = float(state.get("ctrl_sent_vx_ema", 0.0))
                ctrl_sent_vy_ema = float(state.get("ctrl_sent_vy_ema", 0.0))
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
                tracker_box_local = None
                tracker_frame_shape = None
                lead_filtered_x = float(CENTER[0])
                lead_filtered_y = float(CENTER[1])
                smooth_x = float(CENTER[0])
                smooth_y = float(CENTER[1])
                miss_count = 0
                bezier_curve_sign = 1.0
                last_accel_mag = 0.0
                last_out_dx = 0
                last_out_dy = 0
                flip_cooldown_x = 0
                flip_cooldown_y = 0
                last_pid_ts = time.time()
                pid_x.reset()
                pid_y.reset()
                velocity_stop_latched = True
                moving_lag_bias_x = 0.0
                moving_lag_bias_y = 0.0
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
            capture_ts = packet.get("capture_ts", frame_ts)
            loop_perf_start = time.perf_counter()
            yolo_elapsed_s = 0.0
            tracker_elapsed_s = 0.0

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
                time.sleep(0.0005)
                continue

            # Match class mapping from testKalman2.py: classes are directly mapped by aimmode (0=head, 1=body)
            target_cls = int(local_aimmode)
            target_classes = np.array([target_cls], dtype=np.int32)
            selected_cls = int(prev_target_cls) if int(prev_target_cls) in target_classes else target_cls

            bbox = None
            used_tracker = False
            best_conf = 0.0

            effective_interval = YOLO_INTERVAL_FAST if last_speed >= FAST_TARGET_SPEED else YOLO_INTERVAL
            tracker_interval = max(1, int(effective_interval))
            if TRACKER_ENABLE and tracker_interval <= 1:
                tracker_interval = max(2, int(TRACKER_MIN_INTERVAL))
            current_frame_shape = tuple(frame.shape[:2])
            if (
                TRACKER_ENABLE
                and TRACKER_REQUIRE_SAME_FRAME_SHAPE
                and tracker_active
                and (tracker_frame_shape is not None)
                and (current_frame_shape != tracker_frame_shape)
            ):
                tracker_active = False
                tracker_streak = 0
                tracker_v_valid = False
                tracker_speed = 0.0
                last_tracker_ts = 0.0
                tracker_box_local = None
                tracker_frame_shape = None
            use_tracker_step = (
                TRACKER_ENABLE
                and
                tracker_active
                and tracker is not None
                and tracker_streak < TRACKER_MAX_STREAK
                and (inf_count % tracker_interval != 0)
            )

            if use_tracker_step:
                tracker_t0 = time.perf_counter()
                ok = False
                tracked = None
                try:
                    ok, tracked = tracker.update(frame)
                except cv2.error as e:
                    now_track_err = time.time()
                    if now_track_err - last_tracker_update_err_ts > TRACKER_UPDATE_ERROR_LOG_COOLDOWN_S:
                        print(f"[WARN] tracker.update failed; disabling tracker this cycle ({e})")
                        last_tracker_update_err_ts = now_track_err
                tracker_elapsed_s += time.perf_counter() - tracker_t0
                if ok:
                    tracker_frame_shape = current_frame_shape
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
                        tracker_box_local = cand_bbox
                        best_conf = TRACKER_MEASUREMENT_CONF
                        used_tracker = True
                        tracker_streak += 1
                    else:
                        tracker_active = False
                        tracker_streak = 0
                        tracker_v_valid = False
                        tracker_speed = 0.0
                        last_tracker_ts = 0.0
                        tracker_box_local = None
                        tracker_frame_shape = None
                else:
                    tracker_active = False
                    tracker_streak = 0
                    tracker_v_valid = False
                    tracker_speed = 0.0
                    last_tracker_ts = 0.0
                    tracker_box_local = None
                    tracker_frame_shape = None

            if bbox is None:
                tracker_streak = 0
                best_conf = 0.0
                try:
                    yolo_t0 = time.perf_counter()
                    query_classes = target_classes if target_classes.size > 1 else int(target_classes[0])
                    xyxy, cls_ids, confs = backend.predict(frame, query_classes)
                    yolo_elapsed_s += time.perf_counter() - yolo_t0
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
                    )
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
                    if TRACKER_ENABLE:
                        should_init_tracker = (not tracker_active) or (tracker is None)
                        if (not should_init_tracker) and (best_conf >= TRACKER_REINIT_CONF):
                            overlap = bbox_iou_xyxy(bbox, tracker_box_local)
                            if overlap < TRACKER_REINIT_MIN_IOU:
                                should_init_tracker = True

                        if should_init_tracker:
                            tracker = create_tracker()
                            if tracker is None and not tracker_warned_unavailable:
                                print(
                                    "[WARN] No OpenCV tracker backend available "
                                    "(MOSSE/KCF/CSRT/MIL). Running YOLO-only."
                                )
                                tracker_warned_unavailable = True
                            tracker_active = False
                            if tracker is not None:
                                x1, y1, x2, y2 = bbox
                                w = max(1, x2 - x1)
                                h = max(1, y2 - y1)
                                try:
                                    tracker_init_result = tracker.init(frame, (x1, y1, w, h))
                                except cv2.error as e:
                                    tracker_init_result = False
                                    now_track_err = time.time()
                                    if now_track_err - last_tracker_update_err_ts > TRACKER_UPDATE_ERROR_LOG_COOLDOWN_S:
                                        print(f"[WARN] tracker.init failed ({e})")
                                        last_tracker_update_err_ts = now_track_err
                                tracker_active = True if tracker_init_result is None else bool(tracker_init_result)
                                if tracker_active:
                                    tracker_box_local = bbox
                                    tracker_frame_shape = current_frame_shape
                                    tracker_streak = 0
                                    tracker_v_valid = False
                                    tracker_speed = 0.0
                                    last_tracker_ts = 0.0
                                else:
                                    tracker_box_local = None
                                    tracker_frame_shape = None

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
                    perf.record_inference(
                        frame_age_s=frame_age,
                        loop_s=time.perf_counter() - loop_perf_start,
                        stale_drop=False,
                        target_found=True,
                        yolo_s=yolo_elapsed_s,
                        tracker_s=tracker_elapsed_s,
                    )
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
                tracker_box_local = None
                tracker_frame_shape = None
                last_accel_mag = 0.0
                lead_filtered_x = float(CENTER[0])
                lead_filtered_y = float(CENTER[1])
                smooth_x = float(CENTER[0])
                smooth_y = float(CENTER[1])
                last_out_dx = 0
                last_out_dy = 0
                flip_cooldown_x = 0
                flip_cooldown_y = 0
                miss_count = 0
                last_pid_ts = time.time()
                pid_x.reset()
                pid_y.reset()
                bezier_curve_sign = 1.0 if np.random.rand() >= 0.5 else -1.0
                velocity_stop_latched = True
                moving_lag_bias_x = 0.0
                moving_lag_bias_y = 0.0
                perf.record_inference(
                    frame_age_s=frame_age,
                    loop_s=time.perf_counter() - loop_perf_start,
                    stale_drop=False,
                    target_found=False,
                    yolo_s=yolo_elapsed_s,
                    tracker_s=tracker_elapsed_s,
                )
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
                    perf.record_inference(
                        frame_age_s=frame_age,
                        loop_s=time.perf_counter() - loop_perf_start,
                        stale_drop=False,
                        target_found=True,
                        yolo_s=yolo_elapsed_s,
                        tracker_s=tracker_elapsed_s,
                    )
                    continue

            measurement[0, 0] = np.float32(cx_full)
            measurement[1, 0] = np.float32(cy_full)

            now_k = time.time()
            if used_tracker and best_conf <= 0.0:
                measurement_conf = TRACKER_MEASUREMENT_CONF
            else:
                measurement_conf = max(DETECTION_MIN_CONF, best_conf)
            if not kalman_initialized:
                if kalman_state_dim == 6:
                    kalman.statePost = np.array(
                        [[cx_full], [cy_full], [0], [0], [0], [0]],
                        dtype=np.float32,
                    )
                else:
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
                if kalman_state_dim == 6:
                    dt2_half = 0.5 * dt * dt
                    kalman.transitionMatrix[0, 4] = dt2_half
                    kalman.transitionMatrix[1, 5] = dt2_half
                    kalman.transitionMatrix[2, 4] = dt
                    kalman.transitionMatrix[3, 5] = dt

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
                state_x = float(state_post[0, 0])
                state_y = float(state_post[1, 0])
                vx = float(state_post[2, 0])
                vy = float(state_post[3, 0])
                if kalman_state_dim == 6:
                    ax = float(state_post[4, 0])
                    ay = float(state_post[5, 0])
                else:
                    ax = 0.0
                    ay = 0.0

                meas_dt = max(MIN_MEAS_DT, now_k - last_meas_ts) if last_meas_ts > 0.0 else dt
                meas_vx = (float(cx_full) - last_meas_x) / meas_dt
                meas_vy = (float(cy_full) - last_meas_y) / meas_dt
                if EGO_MOTION_COMP_ENABLE:
                    # Compensate apparent target velocity by adding back camera motion induced by our own mouse output.
                    cam_vx = clamp(
                        ctrl_sent_vx_ema * EGO_MOTION_COMP_GAIN_X,
                        -EGO_MOTION_COMP_MAX_PX_S,
                        EGO_MOTION_COMP_MAX_PX_S,
                    )
                    cam_vy = clamp(
                        ctrl_sent_vy_ema * EGO_MOTION_COMP_GAIN_Y,
                        -EGO_MOTION_COMP_MAX_PX_S,
                        EGO_MOTION_COMP_MAX_PX_S,
                    )
                    meas_vx += cam_vx
                    meas_vy += cam_vy
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
                    # Decay velocity near-stop instead of hard-zeroing to preserve moving-target compensation.
                    filtered_vx *= VELOCITY_STOP_HOLD_DECAY
                    filtered_vy *= VELOCITY_STOP_HOLD_DECAY
                    if np.hypot(filtered_vx, filtered_vy) < VELOCITY_STOP_MIN_KEEP_SPEED:
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
                    ax = 0.0
                    ay = 0.0
                    lead_dx = 0.0
                    lead_dy = 0.0
                    speed_ratio = 0.0
                else:
                    speed_ratio = (speed - ADAPTIVE_SPEED_MIN) / max(
                        1e-6, (ADAPTIVE_SPEED_MAX - ADAPTIVE_SPEED_MIN)
                    )
                    speed_ratio = clamp(speed_ratio, 0.0, 1.0)
                    delay_comp_s = 0.0
                    if DELAY_COMPENSATION_ENABLE and speed >= DELAY_COMP_MIN_SPEED:
                        delay_comp_s = (
                            max(0.0, ctrl_cmd_age_ema)
                            + max(0.0, ctrl_send_ema)
                            + max(0.0, DELAY_COMP_INPUT_APPLY_S)
                        )
                        delay_comp_s = clamp(delay_comp_s, 0.0, DELAY_COMP_MAX_S)

                    pipeline_age_s = frame_age
                    if PREDICTION_PIPELINE_AGE_USE_CURRENT:
                        pipeline_age_s = max(pipeline_age_s, max(0.0, now_k - frame_ts))
                    if PREDICTION_HORIZON_USE_CAPTURE_AGE:
                        pipeline_age_s = max(pipeline_age_s, max(0.0, now_k - capture_ts))

                    lead_time = pipeline_age_s + EXTRA_PIPELINE_LAG_S + delay_comp_s + (LEAD_FRAMES * dt)
                    if speed >= SPEED_BOOST_THRESHOLD:
                        lead_time *= SPEED_BOOST_GAIN
                    if speed >= FAST_TARGET_SPEED:
                        lead_time += NON_STATIONARY_EXTRA_LEAD_S

                    lead_gain = PREDICTION_LEAD_GAIN_BASE + (
                        (PREDICTION_LEAD_GAIN_FAST - PREDICTION_LEAD_GAIN_BASE) * speed_ratio
                    )
                    pred_ax = float(np.clip(ax, -PREDICTION_MAX_ACCEL_PX_S2, PREDICTION_MAX_ACCEL_PX_S2))
                    pred_ay = float(np.clip(ay, -PREDICTION_MAX_ACCEL_PX_S2, PREDICTION_MAX_ACCEL_PX_S2))
                    lead_dx = float(
                        np.clip(
                            (vx * lead_time * lead_gain)
                            + (0.5 * pred_ax * lead_time * lead_time * PREDICTION_ACCEL_GAIN),
                            -KALMAN_MAX_LEAD_PX,
                            KALMAN_MAX_LEAD_PX,
                        )
                    )
                    lead_dy = float(
                        np.clip(
                            (vy * lead_time * lead_gain)
                            + (0.5 * pred_ay * lead_time * lead_time * PREDICTION_ACCEL_GAIN),
                            -KALMAN_MAX_LEAD_PX,
                            KALMAN_MAX_LEAD_PX,
                        )
                    )

                    center_error = float(
                        np.hypot(float(cx_full) - CENTER[0], float(cy_full) - CENTER[1])
                    )
                    near_scale = clamp(
                        (center_error - LEAD_NEAR_TARGET_INNER_PX)
                        / max(1e-6, (LEAD_NEAR_TARGET_OUTER_PX - LEAD_NEAR_TARGET_INNER_PX)),
                        0.0,
                        1.0,
                    )
                    near_scale = max(near_scale, PREDICTION_NEAR_SCALE_FLOOR_FAST * speed_ratio)
                    conf_norm_lead = clamp(
                        (measurement_conf - DETECTION_MIN_CONF) / max(1e-6, (1.0 - DETECTION_MIN_CONF)),
                        0.0,
                        1.0,
                    )
                    conf_scale = LEAD_CONFIDENCE_MIN_SCALE + (
                        (1.0 - LEAD_CONFIDENCE_MIN_SCALE) * conf_norm_lead
                    )
                    conf_floor = LEAD_CONFIDENCE_MIN_SCALE + (
                        (PREDICTION_CONF_SCALE_FLOOR_FAST - LEAD_CONFIDENCE_MIN_SCALE) * speed_ratio
                    )
                    conf_scale = max(conf_scale, conf_floor)
                    lead_scale = near_scale * conf_scale
                    lead_scale = max(lead_scale, PREDICTION_MOVING_LEAD_SCALE_FLOOR * speed_ratio)
                    lead_dx *= lead_scale
                    lead_dy *= lead_scale

                base_x = (float(cx_full) * (1.0 - KALMAN_POSITION_BLEND)) + (state_x * KALMAN_POSITION_BLEND)
                base_y = (float(cy_full) * (1.0 - KALMAN_POSITION_BLEND)) + (state_y * KALMAN_POSITION_BLEND)
                lead_x_raw = base_x + lead_dx
                lead_y_raw = base_y + lead_dy
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
                ax = 0.0
                ay = 0.0
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

            # PIDF output: PID on aim error + velocity feed-forward for moving targets.
            error_x = aim_x - CENTER[0]
            error_y = aim_y - CENTER[1]
            error_mag = float(np.hypot(error_x, error_y))
            if BEZIER_CURVE_ENABLED:
                error_x, error_y = apply_bezier_offset(error_x, error_y, bezier_curve_sign)

            pid_dt = float(np.clip(now_k - last_pid_ts, KALMAN_MIN_DT, KALMAN_MAX_DT))
            last_pid_ts = now_k

            moving_ratio = clamp(
                (speed - MOVING_TRACK_SPEED_THRESHOLD) / max(1e-6, (ADAPTIVE_SPEED_MAX - MOVING_TRACK_SPEED_THRESHOLD)),
                0.0,
                1.0,
            )
            close_ratio = clamp(
                (float(box_h) - CLOSE_TARGET_BOX_H_ENTER_PX)
                / max(1e-6, (CLOSE_TARGET_BOX_H_FULL_PX - CLOSE_TARGET_BOX_H_ENTER_PX)),
                0.0,
                1.0,
            )
            close_move_ratio = close_ratio * moving_ratio
            lag_bias_term_x = 0.0
            lag_bias_term_y = 0.0
            moving_bias_active = False
            if MOVING_LAG_BIAS_ENABLE:
                moving_bias_active = speed >= MOVING_LAG_BIAS_MIN_SPEED

                align_x = (error_x * vx) > 0.0
                align_y = (error_y * vy) > 0.0
                use_x = moving_bias_active and (abs(error_x) >= MOVING_LAG_BIAS_MIN_ERROR_PX)
                use_y = moving_bias_active and (abs(error_y) >= MOVING_LAG_BIAS_MIN_ERROR_PX)
                if MOVING_LAG_BIAS_ALIGN_ONLY:
                    use_x = use_x and align_x
                    use_y = use_y and align_y

                if use_x:
                    if (
                        (moving_lag_bias_x * error_x) < 0.0
                        and abs(error_x) <= MOVING_LAG_BIAS_SIGN_RESET_ERROR_PX
                    ):
                        moving_lag_bias_x *= MOVING_LAG_BIAS_SIGN_RESET_DECAY
                    moving_lag_bias_x += error_x * pid_dt * MOVING_LAG_BIAS_KI_X
                    moving_lag_bias_x = clamp(
                        moving_lag_bias_x,
                        -MOVING_LAG_BIAS_MAX_X,
                        MOVING_LAG_BIAS_MAX_X,
                    )
                else:
                    moving_lag_bias_x *= MOVING_LAG_BIAS_DECAY
                    if abs(moving_lag_bias_x) < 1e-4:
                        moving_lag_bias_x = 0.0

                if use_y:
                    if (
                        (moving_lag_bias_y * error_y) < 0.0
                        and abs(error_y) <= MOVING_LAG_BIAS_SIGN_RESET_ERROR_PX
                    ):
                        moving_lag_bias_y *= MOVING_LAG_BIAS_SIGN_RESET_DECAY
                    moving_lag_bias_y += error_y * pid_dt * MOVING_LAG_BIAS_KI_Y
                    moving_lag_bias_y = clamp(
                        moving_lag_bias_y,
                        -MOVING_LAG_BIAS_MAX_Y,
                        MOVING_LAG_BIAS_MAX_Y,
                    )
                else:
                    moving_lag_bias_y *= MOVING_LAG_BIAS_DECAY
                    if abs(moving_lag_bias_y) < 1e-4:
                        moving_lag_bias_y = 0.0

                lag_bias_term_x = moving_lag_bias_x
                lag_bias_term_y = moving_lag_bias_y
                if moving_bias_active:
                    # Avoid near-center ping-pong from a large accumulated lag-bias term.
                    lag_cap_w = clamp(
                        error_mag / max(1e-6, MOVING_LAG_BIAS_NEAR_ERROR_PX),
                        0.0,
                        1.0,
                    )
                    lag_cap = MOVING_LAG_BIAS_NEAR_CAP_PX + (
                        (MOVING_LAG_BIAS_FAR_CAP_PX - MOVING_LAG_BIAS_NEAR_CAP_PX) * lag_cap_w
                    )
                    lag_cap *= (1.0 - (CLOSE_LAG_CAP_REDUCE * close_move_ratio))
                    lag_cap = max(1.0, lag_cap)
                    lag_bias_term_x = clamp(lag_bias_term_x, -lag_cap, lag_cap)
                    lag_bias_term_y = clamp(lag_bias_term_y, -lag_cap, lag_cap)

            adaptive_deadzone_px = RAW_AIM_DEADZONE_PX * (
                1.0 - ((1.0 - MOVING_DEADZONE_MIN_SCALE) * moving_ratio)
            )
            adaptive_deadzone_px += CLOSE_DEADZONE_ADD_PX * close_move_ratio
            adaptive_deadzone_px = max(0.0, adaptive_deadzone_px)

            if error_mag <= adaptive_deadzone_px and moving_ratio <= 0.0:
                dx = 0
                dy = 0
                pid_x.reset()
                pid_y.reset()
            else:
                pid_term_x = 0.0
                pid_term_y = 0.0
                ff_term_x = 0.0
                ff_term_y = 0.0
                if PIDF_ENABLE:
                    pid_out_x = pid_x.update(error_x, pid_dt)
                    pid_out_y = pid_y.update(error_y, pid_dt)
                    ff_x = vx * pid_dt * PIDF_FF_GAIN_X
                    ff_y = vy * pid_dt * PIDF_FF_GAIN_Y
                    if error_mag <= MOVING_FF_BOOST_ERROR_PX and moving_ratio > 0.0:
                        ff_boost = 1.0 + ((MOVING_FF_NEAR_BOOST - 1.0) * moving_ratio)
                        ff_x *= ff_boost
                        ff_y *= ff_boost
                    if moving_ratio > 0.0:
                        # If predicted velocity disagrees with current error direction, reduce FF to avoid overshoot.
                        if (error_x * vx) < 0.0:
                            ff_x *= MOVING_FF_MISALIGN_SCALE
                        if (error_y * vy) < 0.0:
                            ff_y *= MOVING_FF_MISALIGN_SCALE

                        ff_cap_w = clamp(error_mag / max(1e-6, MOVING_FF_CAP_ERROR_PX), 0.0, 1.0)
                        ff_cap = MOVING_FF_NEAR_CAP_PX + (
                            (MOVING_FF_FAR_CAP_PX - MOVING_FF_NEAR_CAP_PX) * ff_cap_w
                        )
                        ff_cap = max(ff_cap, MOVING_FF_MIN_CAP_FAST * moving_ratio)
                        ff_cap *= (1.0 - (CLOSE_FF_CAP_REDUCE * close_move_ratio))
                        ff_cap = max(1.0, ff_cap)
                        ff_x = clamp(ff_x, -ff_cap, ff_cap)
                        ff_y = clamp(ff_y, -ff_cap, ff_cap)
                    pid_term_x = pid_out_x
                    pid_term_y = pid_out_y
                    ff_term_x = ff_x
                    ff_term_y = ff_y
                    desired_x = pid_term_x + ff_term_x + lag_bias_term_x
                    desired_y = pid_term_y + ff_term_y + lag_bias_term_y
                else:
                    near_scale = clamp(error_mag / max(1e-6, RAW_NEAR_ERROR_PX), 0.0, 1.0)
                    raw_gain = RAW_NEAR_GAIN + ((RAW_FAR_GAIN - RAW_NEAR_GAIN) * near_scale)
                    desired_x = (error_x * raw_gain) + lag_bias_term_x
                    desired_y = (error_y * raw_gain) + lag_bias_term_y

                rate_scale = 1.0
                rate_scale_base = 1.0
                if CONTROL_OUTPUT_RATE_NORMALIZE:
                    ref_dt = 1.0 / max(1e-6, float(CONTROL_REFERENCE_HZ))
                    rate_scale_base = clamp(
                        pid_dt / ref_dt,
                        CONTROL_RATE_SCALE_MIN,
                        CONTROL_RATE_SCALE_MAX,
                    )
                    far_weight = clamp(
                        (error_mag - CONTROL_RATE_BLEND_INNER_PX)
                        / max(1e-6, (CONTROL_RATE_BLEND_OUTER_PX - CONTROL_RATE_BLEND_INNER_PX)),
                        0.0,
                        1.0,
                    )
                    # Keep strong damping near center, but restore full-speed corrections for far errors.
                    rate_scale = rate_scale_base + ((1.0 - rate_scale_base) * far_weight)
                    # Moving targets need stronger near-center correction to avoid steady chase offset.
                    rate_scale = max(rate_scale, CONTROL_RATE_MOVING_SCALE_FLOOR * moving_ratio)
                    if PIDF_ENABLE:
                        # Dampen the error-correction term at high-Hz, but keep velocity feed-forward responsive.
                        desired_x = (pid_term_x * rate_scale) + ff_term_x + lag_bias_term_x
                        desired_y = (pid_term_y * rate_scale) + ff_term_y + lag_bias_term_y
                    else:
                        desired_x *= rate_scale
                        desired_y *= rate_scale

                if PIDF_ENABLE and moving_ratio > 0.0:
                    # Near lock on fast targets, bias toward smoother FF tracking and downweight aggressive correction.
                    stick_blend = clamp(
                        (MOVING_STICK_BLEND_OUTER_PX - error_mag)
                        / max(1e-6, (MOVING_STICK_BLEND_OUTER_PX - MOVING_STICK_BLEND_INNER_PX)),
                        0.0,
                        1.0,
                    ) * moving_ratio
                    if stick_blend > 0.0:
                        stick_x = (
                            ff_term_x
                            + (pid_term_x * MOVING_STICK_PID_WEIGHT)
                            + (lag_bias_term_x * MOVING_STICK_LAG_WEIGHT)
                        )
                        stick_y = (
                            ff_term_y
                            + (pid_term_y * MOVING_STICK_PID_WEIGHT)
                            + (lag_bias_term_y * MOVING_STICK_LAG_WEIGHT)
                        )
                        desired_x = (desired_x * (1.0 - stick_blend)) + (stick_x * stick_blend)
                        desired_y = (desired_y * (1.0 - stick_blend)) + (stick_y * stick_blend)

                dx = int(round(clamp(desired_x, -RAW_MAX_STEP_X, RAW_MAX_STEP_X)))
                dy = int(round(clamp(desired_y, -RAW_MAX_STEP_Y, RAW_MAX_STEP_Y)))

                # Adaptive slew limiter: strong damping only near lock to reduce high-speed oscillation.
                slew_w = clamp(
                    error_mag / max(1e-6, RAW_SIGN_FLIP_STOP_PX * 6.0),
                    0.0,
                    1.0,
                )
                max_delta_x = int(round(max(1.0, OUTPUT_MAX_DELTA_X * (SLEW_LIMITER_FACTOR + (0.65 * slew_w)))))
                max_delta_y = int(round(max(1.0, OUTPUT_MAX_DELTA_Y * (SLEW_LIMITER_FACTOR + (0.65 * slew_w)))))
                close_slew_scale = 1.0 - (CLOSE_SLEW_EXTRA_DAMP * close_move_ratio)
                max_delta_x = int(round(max(1.0, max_delta_x * close_slew_scale)))
                max_delta_y = int(round(max(1.0, max_delta_y * close_slew_scale)))
                dx = int(clamp(dx, last_out_dx - max_delta_x, last_out_dx + max_delta_x))
                dy = int(clamp(dy, last_out_dy - max_delta_y, last_out_dy + max_delta_y))

                if abs(error_x) <= adaptive_deadzone_px and moving_ratio <= 0.0:
                    dx = 0
                if abs(error_y) <= adaptive_deadzone_px and moving_ratio <= 0.0:
                    dy = 0

                ahead_hold_active_x = False
                ahead_hold_active_y = False
                if (
                    MOVING_AHEAD_HOLD_ENABLE
                    and moving_ratio > 0.0
                    and speed >= MOVING_AHEAD_HOLD_MIN_SPEED
                ):
                    ahead_hold_px = MOVING_AHEAD_HOLD_ERROR_PX + (
                        MOVING_AHEAD_HOLD_CLOSE_EXTRA_PX * close_move_ratio
                    )

                    # If we are slightly ahead of target motion, suppress pull-back reversals to avoid ping-pong.
                    if (error_x * vx) < 0.0:
                        ahead_hold_active_x = abs(error_x) <= (ahead_hold_px * 2.0)
                        if abs(error_x) <= ahead_hold_px and (dx * vx) < 0.0:
                            dx = 0
                            moving_lag_bias_x *= MOVING_AHEAD_HOLD_LAG_DECAY
                        elif (dx * vx) < 0.0:
                            dx = int(round(dx * MOVING_AHEAD_HOLD_OPPOSITE_SCALE))
                            moving_lag_bias_x *= MOVING_AHEAD_HOLD_LAG_DECAY
                    if (error_y * vy) < 0.0:
                        ahead_hold_active_y = abs(error_y) <= (ahead_hold_px * 2.0)
                        if abs(error_y) <= ahead_hold_px and (dy * vy) < 0.0:
                            dy = 0
                            moving_lag_bias_y *= MOVING_AHEAD_HOLD_LAG_DECAY
                        elif (dy * vy) < 0.0:
                            dy = int(round(dy * MOVING_AHEAD_HOLD_OPPOSITE_SCALE))
                            moving_lag_bias_y *= MOVING_AHEAD_HOLD_LAG_DECAY
                ahead_hold_any = ahead_hold_active_x or ahead_hold_active_y

                flip_cooldown_frames = RAW_SIGN_FLIP_COOLDOWN_FRAMES
                if CONTROL_OUTPUT_RATE_NORMALIZE:
                    flip_cooldown_frames = int(max(
                        RAW_SIGN_FLIP_COOLDOWN_FRAMES,
                        round(RAW_SIGN_FLIP_COOLDOWN_FRAMES / max(1e-6, rate_scale_base)),
                    ))

                sign_flip_stop_px = RAW_SIGN_FLIP_STOP_PX
                if speed >= MOVING_SIGN_FLIP_DISABLE_SPEED:
                    sign_flip_stop_px *= MOVING_SIGN_FLIP_STOP_PX_FAST_MULT
                sign_flip_stop_px *= (1.0 + (CLOSE_SIGN_FLIP_EXTRA_MULT * close_move_ratio))
                bypass_error_px = MOVING_SIGN_FLIP_BYPASS_ERROR_PX + (
                    (MOVING_SIGN_FLIP_BYPASS_ERROR_PX_FAST - MOVING_SIGN_FLIP_BYPASS_ERROR_PX)
                    * moving_ratio
                )
                bypass_error_px += CLOSE_BYPASS_ERROR_EXTRA_PX * close_move_ratio
                fast_motion_bypass = (
                    speed >= MOVING_SIGN_FLIP_DISABLE_SPEED
                    and error_mag > bypass_error_px
                    and (not ahead_hold_any)
                )
                if fast_motion_bypass:
                    flip_cooldown_x = 0
                    flip_cooldown_y = 0
                else:
                    # Prevent ping-pong when command flips sign near the target.
                    if (dx * last_out_dx) < 0 and abs(error_x) <= sign_flip_stop_px:
                        flip_cooldown_x = flip_cooldown_frames
                        dx = 0
                    if (dy * last_out_dy) < 0 and abs(error_y) <= sign_flip_stop_px:
                        flip_cooldown_y = flip_cooldown_frames
                        dy = 0
                    if flip_cooldown_x > 0 and abs(error_x) <= sign_flip_stop_px:
                        dx = 0
                        flip_cooldown_x -= 1
                    if flip_cooldown_y > 0 and abs(error_y) <= sign_flip_stop_px:
                        dy = 0
                        flip_cooldown_y -= 1

                # Suppress 1px moving-target chatter without affecting catch-up movement.
                if moving_ratio > 0.0:
                    if abs(error_x) <= MOVING_RESIDUAL_DEADZONE_PX and abs(dx) <= MOVING_RESIDUAL_CMD_PX:
                        dx = 0
                    if abs(error_y) <= MOVING_RESIDUAL_DEADZONE_PX and abs(dy) <= MOVING_RESIDUAL_CMD_PX:
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
                {
                    "dx": dx,
                    "dy": dy,
                    "ts": now_k,
                    "frame_ts": frame_ts,
                    "capture_ts": capture_ts,
                },
            )
            perf.record_inference(
                frame_age_s=frame_age,
                loop_s=time.perf_counter() - loop_perf_start,
                stale_drop=False,
                target_found=True,
                yolo_s=yolo_elapsed_s,
                tracker_s=tracker_elapsed_s,
                cmd_latency_s=max(0.0, now_k - frame_ts),
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
                with shared_lock:
                    state["ctrl_sent_vx_ema"] = float(state.get("ctrl_sent_vx_ema", 0.0)) * EGO_MOTION_COMP_DECAY
                    state["ctrl_sent_vy_ema"] = float(state.get("ctrl_sent_vy_ema", 0.0)) * EGO_MOTION_COMP_DECAY
                continue

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

            with shared_lock:
                local_mode = state["mode"]
                left_pressed = bool(state.get("left_pressed", False))
            if local_mode == 0:
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
            if RECOIL_CONTROL_ENABLE and left_pressed:
                dy = int(clamp(dy + RECOIL_COMPENSATION_Y_PX, -RAW_MAX_STEP_Y, RAW_MAX_STEP_Y))
            if dx == 0 and dy == 0:
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

            send_start = time.perf_counter()
            sent_ok = mouse_client.send_input(dx, dy)
            send_elapsed = time.perf_counter() - send_start
            if sent_ok:
                with shared_lock:
                    state["ctrl_send_ema"] = ema_update(
                        float(state.get("ctrl_send_ema", 0.0)),
                        max(0.0, send_elapsed),
                        DELAY_COMP_EMA_ALPHA,
                    )
                    last_send_ts = float(state.get("ctrl_last_send_ts", 0.0))
                    if last_send_ts > 0.0:
                        send_dt = max(1e-4, now - last_send_ts)
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
                    state["ctrl_last_send_ts"] = now
            else:
                with shared_lock:
                    state["ctrl_sent_vx_ema"] = float(state.get("ctrl_sent_vx_ema", 0.0)) * EGO_MOTION_COMP_DECAY
                    state["ctrl_sent_vy_ema"] = float(state.get("ctrl_sent_vy_ema", 0.0)) * EGO_MOTION_COMP_DECAY
            perf.record_control(
                cmd_age_s=cmd_age,
                sent=sent_ok,
                send_s=send_elapsed if sent_ok else 0.0,
                total_latency_s=total_latency,
                total_latency_full_s=total_latency_full,
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
            if (not PERF_LOG_WHEN_MODE_OFF) and perf_mode == 0:
                continue

            print(
                "[PERF] "
                f"cap={snapshot['cap_fps']:.1f}fps grab={snapshot['cap_grab_ms']:.2f}ms none={snapshot['cap_none']} | "
                f"inf={snapshot['infer_fps']:.1f}fps loop={snapshot['infer_loop_ms']:.2f}ms "
                f"age={snapshot['infer_age_ms']:.2f}/{snapshot['infer_age_max_ms']:.2f}ms stale={snapshot['infer_stale']} "
                f"lock={snapshot['infer_lock_rate'] * 100.0:.0f}% | "
                f"yolo={snapshot['yolo_hz']:.1f}Hz@{snapshot['yolo_ms']:.2f}ms "
                f"trk={snapshot['tracker_hz']:.1f}Hz@{snapshot['tracker_ms']:.2f}ms | "
                f"ctl={snapshot['control_send_hz']:.1f}Hz send={snapshot['control_send_ms']:.3f}ms "
                f"cmdAge={snapshot['control_cmd_age_ms']:.2f}ms "
                f"e2e={snapshot['control_total_latency_ms']:.2f}ms "
                f"e2eFull={snapshot['control_total_latency_full_ms']:.2f}ms "
                f"drop(stale/mode)={snapshot['control_stale_drop']}/{snapshot['control_mode_drop']} "
                f"aimPipe={snapshot['infer_cmd_ms']:.2f}ms"
            )

    def on_click(x, y, button, pressed):
        nonlocal mouse_button_pressed_x1, mouse_button_pressed_x2
        if button == Button.x2:
            mouse_button_pressed_x2 = pressed
        if button == Button.x1:
            mouse_button_pressed_x1 = pressed
        if button == Button.left:
            with shared_lock:
                state["left_pressed"] = bool(pressed)

    capture_thread = threading.Thread(target=capture_loop, daemon=True)
    inference_thread = threading.Thread(target=inference_loop, daemon=True)
    control_thread = threading.Thread(target=control_loop, daemon=True)
    perf_thread = None
    if PERF_LOG_ENABLE:
        perf_thread = threading.Thread(target=perf_loop, daemon=True)
    capture_thread.start()
    inference_thread.start()
    control_thread.start()
    if perf_thread is not None:
        perf_thread.start()

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

                time.sleep(SLEEP_TIME)
            except KeyboardInterrupt:
                running = False
                break

    with shared_lock:
        state["running"] = False
    mouse_client.close()


if __name__ == "__main__":
    main()
