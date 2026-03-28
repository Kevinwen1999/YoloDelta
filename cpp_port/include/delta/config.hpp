#pragma once

#include <cstdint>
#include <string>

#include "delta/recoil_types.hpp"

namespace delta {

enum class TrackingStrategy {
    Raw,
    Kalman,
    Ema,
    Dema,
    RawDelta,
};

enum class LeftHoldEngageButton {
    Left,
    Right,
    X1,
    Both,
};

struct StaticConfig {
    std::string model_path = R"(C:\YOLO\Delta\runs\detect\train4\weights\best.onnx)";
    std::string onnxruntime_root;
    std::string cuda_root;
    std::string tensorrt_root;
    std::string tensorrt_cache_dir;
    std::string recoil_profiles_dir = R"(C:\YOLO\Delta\cpp_port\runtime\recoil_profiles)";
    int screen_w = 2560;
    int screen_h = 1440;
    int imgsz = 416;
    int capture_crop_size = 800;
    float conf = 0.30F;
    int max_detections = 6;
    std::string inference_device = "cuda";
    std::string onnx_provider = "auto";
    std::string onnx_resize_interpolation = "nearest";
    float onnx_nms_iou = 0.50F;
    int onnx_topk_pre_nms = 500;
    int onnx_cuda_device_id = 0;
    bool onnx_output_has_nms = true;
    bool onnx_force_target_class_decode = true;
    bool onnx_use_tensorrt = true;
    bool onnx_require_gpu = true;
    bool onnx_trt_fp16 = true;
    bool onnx_skip_resize_if_match = true;
    bool onnx_enable_cuda_graph = true;
    bool onnx_trt_cuda_graph_enable = true;
    int capture_device_idx = 0;
    int capture_output_idx = 0;
    int capture_timeout_ms = 1;
    bool capture_video_mode = true;
    bool tracker_enable = false;
    bool force_input_size_when_locked = true;
    bool perf_log_enable = true;
    bool debug_log = true;
    std::uint16_t frontend_port = 8765;
    std::string frontend_host = "127.0.0.1";
};

inline int effectiveCaptureCropSize(const StaticConfig& config) {
    return config.capture_crop_size > 0 ? config.capture_crop_size : config.imgsz;
}

struct RuntimeConfig {
    bool pid_enable = true;
    bool tracking_enabled = true;
    bool debug_preview_enable = true;
    int capture_cached_timeout_ms = 0;
    float body_y_ratio = 0.15F;
    float head_y_ratio = 0.50F;
    TrackingStrategy tracking_strategy = TrackingStrategy::RawDelta;
    float tracking_alpha = 0.42F;
    float tracking_velocity_alpha = 0.5F;
    float kp = 0.40F;
    float ki = 0.7F;
    float kd = 0.009F;
    float integral_limit = 2000.0F;
    float anti_windup_gain = 1.0F;
    float derivative_alpha = 0.2F;
    float output_limit = 3000.0F;
    float sticky_bias_px = 800.0F;
    float prediction_time = 0.000F;
    int target_max_lost_frames = 8;
    float model_conf = 0.30F;
    float detection_min_conf = 0.30F;
    float kalman_process_noise = 1.5F;
    float kalman_measurement_noise = 16.0F;
    bool ego_motion_comp_enable = true;
    float ego_motion_comp_gain_x = 0.3F;
    float ego_motion_comp_gain_y = 0.3F;
    bool ego_motion_error_gate_enable = false;
    float ego_motion_error_gate_px = 500.0F;
    bool ego_motion_error_gate_normalize_by_box = false;
    float ego_motion_error_gate_norm_threshold = 2.0F;
    bool ego_motion_reset_on_switch = true;
    RecoilMode recoil_mode = RecoilMode::Legacy;
    std::string selected_recoil_profile_id;
    float recoil_compensation_y_rate_px_s = 0.0F;
    float recoil_compensation_y_px = 4.0F;
    LeftHoldEngageButton left_hold_engage_button = LeftHoldEngageButton::Both;
    bool recoil_tune_fallback_ignore_mode_check = false;
    bool triggerbot_enable = false;
    float triggerbot_click_hold_s = 0.001F;
    float triggerbot_click_cooldown_s = 0.001F;
    bool side_button_key_sequence_use_key3 = true;
    int side_button_key_sequence_key3_press_time_ms = 0;
    bool side_button_key_sequence_use_key1 = true;
    int side_button_key_sequence_key1_press_time_ms = 0;
    bool side_button_key_sequence_use_right_click = true;
    int side_button_key_sequence_right_click_hold_ms = 1;
    bool side_button_key_sequence_use_left_click = true;
    int side_button_key_sequence_left_click_hold_ms = 0;
    int side_button_key_sequence_loop_delay_ms = 8;
    float sendinput_gain_x = 1.0F;
    float sendinput_gain_y = 1.0F;
    int sendinput_max_step = 1270;
    int raw_max_step_x = 280;
    int raw_max_step_y = 2800;
};

struct ToggleState {
    int mode = 0;
    int aimmode = 0;
    bool left_hold_engage = false;
    bool recoil_tune_fallback = false;
    bool left_pressed = false;
    bool right_pressed = false;
    bool x1_pressed = false;
};

}  // namespace delta
