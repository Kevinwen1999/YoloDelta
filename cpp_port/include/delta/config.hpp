#pragma once

#include <cstdint>
#include <string>
#include <string_view>

#include "delta/recoil_types.hpp"

namespace delta {

enum class TrackingStrategy {
    Raw,
    Kalman,
    Ema,
    Dema,
    RawDelta,
    LegacyPid,
    PredictivePid,
};

inline const char* trackingStrategyName(const TrackingStrategy strategy) {
    switch (strategy) {
    case TrackingStrategy::Raw: return "raw";
    case TrackingStrategy::LegacyPid: return "legacy_pid";
    case TrackingStrategy::PredictivePid: return "predictive_pid";
    case TrackingStrategy::Kalman:
    case TrackingStrategy::Ema:
    case TrackingStrategy::Dema:
    case TrackingStrategy::RawDelta:
    default: return "raw_delta";
    }
}

inline const char* trackingStrategyLabel(const TrackingStrategy strategy) {
    switch (strategy) {
    case TrackingStrategy::Raw: return "Raw Detection";
    case TrackingStrategy::LegacyPid: return "Legacy PID";
    case TrackingStrategy::PredictivePid: return "Predictive PID";
    case TrackingStrategy::Kalman:
    case TrackingStrategy::Ema:
    case TrackingStrategy::Dema:
    case TrackingStrategy::RawDelta:
    default: return "Raw + Velocity";
    }
}

inline TrackingStrategy parseTrackingStrategy(const std::string_view value) {
    if (value == "raw") {
        return TrackingStrategy::Raw;
    }
    if (value == "legacy_pid") {
        return TrackingStrategy::LegacyPid;
    }
    if (value == "predictive_pid") {
        return TrackingStrategy::PredictivePid;
    }
    return TrackingStrategy::RawDelta;
}

enum class LeftHoldEngageButton {
    Left,
    Right,
    X1,
    Both,
};

enum class AimMode {
    Head = 0,
    Body = 1,
    Hybrid = 2,
};

inline const char* aimModeName(const AimMode mode) {
    switch (mode) {
    case AimMode::Body: return "body";
    case AimMode::Hybrid: return "hybrid";
    case AimMode::Head:
    default: return "head";
    }
}

inline const char* aimModeLabel(const AimMode mode) {
    switch (mode) {
    case AimMode::Body: return "BODY";
    case AimMode::Hybrid: return "HYBRID";
    case AimMode::Head:
    default: return "HEAD";
    }
}

inline AimMode parseAimMode(const std::string_view value) {
    if (value == "body") {
        return AimMode::Body;
    }
    if (value == "hybrid") {
        return AimMode::Hybrid;
    }
    return AimMode::Head;
}

inline AimMode nextAimMode(const AimMode mode) {
    switch (mode) {
    case AimMode::Head: return AimMode::Body;
    case AimMode::Body: return AimMode::Hybrid;
    case AimMode::Hybrid:
    default: return AimMode::Head;
    }
}

inline int aimModeTargetClass(const AimMode mode) {
    switch (mode) {
    case AimMode::Body: return 0;
    case AimMode::Hybrid: return -1;
    case AimMode::Head:
    default: return 1;
    }
}

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
    int capture_crop_size = 832;
    float conf = 0.30F;
    int max_detections = 10;
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
    bool debug_preview_enable = false;
    bool debug_overlay_enable = false;
    AimMode aim_mode = AimMode::Head;
    int capture_cached_timeout_ms = 1;
    bool capture_freeze_to_center_enable = true;
    float body_y_ratio = 0.15F;
    float head_y_ratio = 0.50F;
    TrackingStrategy tracking_strategy = TrackingStrategy::PredictivePid;
    float tracking_alpha = 0.42F;
    float tracking_velocity_alpha = 0.5F;
    float kp = 0.30F;
    float ki = 0.9F;
    float kd = 0.009F;
    float integral_limit = 2000.0F;
    float anti_windup_gain = 1.0F;
    float derivative_alpha = 0.2F;
    float output_limit = 3000.0F;
    bool pid_settle_enable = true;
    float pid_settle_error_px = 4.0F;
    float pid_settle_threshold_min_scale = 1.6F;
    float pid_settle_threshold_max_scale = 2.7F;
    int pid_settle_stable_frames = 2;
    float pid_settle_error_delta_px = 10.0F;
    float pid_settle_pre_output_scale = 0.5F;
    float legacy_pid_lock_error_px = 4.0F;
    float legacy_pid_speed_multiplier = 1.0F;
    float legacy_pid_threshold_min_scale = 1.6F;
    float legacy_pid_threshold_max_scale = 2.7F;
    float legacy_pid_transition_sharpness = 5.0F;
    float legacy_pid_transition_midpoint = 0.0F;
    int legacy_pid_stable_frames = 2;
    float legacy_pid_error_delta_px = 10.0F;
    float legacy_pid_prelock_scale = 0.2F;
    float predictive_pid_kp = 1.0F;
    float predictive_pid_ki = 0.0F;
    float predictive_pid_kd = 0.001F;
    float predictive_pid_pred_weight_x = 0.75F;
    float predictive_pid_pred_weight_y = 0.75F;
    float predictive_pid_init_scale = 0.5F;
    float predictive_pid_ramp_time_s = 0.4F;
    float predictive_pid_integral_limit = 200.0F;
    float predictive_pid_derivative_limit = 75.0F;
    float predictive_pid_output_limit = 2000000.0F;
    float predictive_pid_velocity_alpha = 0.25F;
    float predictive_pid_acceleration_alpha = 0.15F;
    float predictive_pid_max_velocity_px_s = 3000000.0F;
    float predictive_pid_max_acceleration_px_s = 5000000.0F;
    float predictive_pid_reverse_gate_px = 20.0F;
    float predictive_pid_reverse_scale = 0.01F;
    float predictive_pid_prediction_error_scale = 2.0F;
    float predictive_pid_prediction_min_px = 1.0F;
    float predictive_pid_prediction_max_px = 1000000.0F;
    float sticky_bias_px = 800.0F;
    bool target_guard_enable = true;
    int target_guard_commit_frames = 5;
    int target_guard_hold_frames = 20;
    float target_guard_window_scale = 2.25F;
    int target_guard_min_window_px = 200;
    bool target_lead_enable = false;
    int target_lead_commit_frames = 3;
    bool target_lead_auto_latency_enable = true;
    float target_lead_max_time_s = 1.0F;
    float target_lead_min_speed_px_s = 1.0F;
    float target_lead_max_offset_box_scale = 1.0F;
    float target_lead_smoothing_alpha = 0.5F;
    float prediction_time = 0.000F;
    int target_max_lost_frames = 8;
    float model_conf = 0.30F;
    float detection_min_conf = 0.30F;
    float detection_box_scale = 1.0F;
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
    float triggerbot_arm_scale_x = 0.5F;
    float triggerbot_arm_scale_y = 0.5F;
    int triggerbot_arm_min_x_px = 0;
    int triggerbot_arm_min_y_px = 0;
    float triggerbot_click_hold_s = 0.001F;
    float triggerbot_click_cooldown_s = 0.001F;
    bool mouse_move_suppress_on_fire_enable = false;
    bool mouse_move_suppress_on_fire_debug = false;
    bool side_button_key_sequence_use_key3 = true;
    double side_button_key_sequence_key3_press_time_ms = 0.0;
    bool side_button_key_sequence_use_key1 = true;
    double side_button_key_sequence_key1_press_time_ms = 0.0;
    bool side_button_key_sequence_use_right_click = true;
    double side_button_key_sequence_right_click_hold_ms = 1.0;
    bool side_button_key_sequence_use_left_click = true;
    double side_button_key_sequence_left_click_hold_ms = 0.0;
    double side_button_key_sequence_loop_delay_ms = 8.0;
    float sendinput_gain_x = 1.0F;
    float sendinput_gain_y = 1.0F;
    int sendinput_max_step = 12700;
    int raw_max_step_x = 1000;
    int raw_max_step_y = 1000;
};

struct ToggleState {
    int mode = 0;
    bool left_hold_engage = false;
    bool recoil_tune_fallback = false;
    bool left_pressed = false;
    bool right_pressed = false;
    bool x1_pressed = false;
};

}  // namespace delta
