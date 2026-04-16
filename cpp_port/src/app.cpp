#include "delta/app.hpp"
#include "delta/capture_focus.hpp"
#include "delta/mouse_suppression.hpp"
#include "delta/predictive_pid.hpp"
#include "delta/target_guard.hpp"
#include "delta/target_lead.hpp"
#include "delta/triggerbot.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <utility>

namespace delta {

struct RuntimePerfWindow {
    std::mutex mutex;
    SteadyClock::time_point window_start{SteadyClock::now()};
    std::uint64_t capture_frames = 0;
    std::uint64_t capture_none = 0;
    double capture_grab_s = 0.0;
    double capture_acquire_s = 0.0;
    double capture_d3d_copy_s = 0.0;
    double capture_d3d_sync_s = 0.0;
    double capture_cuda_copy_s = 0.0;
    double capture_cpu_copy_s = 0.0;
    double capture_cached_reuse_s = 0.0;
    std::uint64_t capture_cached_frames = 0;

    std::uint64_t infer_frames = 0;
    std::uint64_t infer_stale = 0;
    std::uint64_t infer_found = 0;
    double infer_loop_s = 0.0;
    double infer_frame_age_s = 0.0;
    double infer_frame_age_max_s = 0.0;
    double infer_select_s = 0.0;
    double infer_tracker_update_s = 0.0;
    double infer_aim_predict_s = 0.0;
    double infer_aim_pid_s = 0.0;
    double infer_aim_sync_s = 0.0;
    double infer_preview_s = 0.0;
    double infer_queue_s = 0.0;
    std::uint64_t infer_backend_samples = 0;
    double infer_backend_pre_s = 0.0;
    double infer_backend_exec_s = 0.0;
    double infer_backend_post_s = 0.0;
    std::uint64_t infer_cmd_samples = 0;
    double infer_cmd_latency_s = 0.0;

    std::uint64_t control_cmds = 0;
    std::uint64_t control_sent = 0;
    std::uint64_t control_stale_drop = 0;
    std::uint64_t control_mode_drop = 0;
    double control_send_s = 0.0;
    double control_cmd_age_s = 0.0;
    double control_total_latency_s = 0.0;
    std::uint64_t control_latency_samples = 0;
    double control_total_latency_full_s = 0.0;
    std::uint64_t control_latency_full_samples = 0;
    double control_total_apply_latency_s = 0.0;
    std::uint64_t control_apply_latency_samples = 0;
    double control_total_apply_latency_full_s = 0.0;
    std::uint64_t control_apply_latency_full_samples = 0;

    void reset(const SteadyClock::time_point now) {
        window_start = now;
        capture_frames = 0;
        capture_none = 0;
        capture_grab_s = 0.0;
        capture_acquire_s = 0.0;
        capture_d3d_copy_s = 0.0;
        capture_d3d_sync_s = 0.0;
        capture_cuda_copy_s = 0.0;
        capture_cpu_copy_s = 0.0;
        capture_cached_reuse_s = 0.0;
        capture_cached_frames = 0;

        infer_frames = 0;
        infer_stale = 0;
        infer_found = 0;
        infer_loop_s = 0.0;
        infer_frame_age_s = 0.0;
        infer_frame_age_max_s = 0.0;
        infer_select_s = 0.0;
        infer_tracker_update_s = 0.0;
        infer_aim_predict_s = 0.0;
        infer_aim_pid_s = 0.0;
        infer_aim_sync_s = 0.0;
        infer_preview_s = 0.0;
        infer_queue_s = 0.0;
        infer_backend_samples = 0;
        infer_backend_pre_s = 0.0;
        infer_backend_exec_s = 0.0;
        infer_backend_post_s = 0.0;
        infer_cmd_samples = 0;
        infer_cmd_latency_s = 0.0;

        control_cmds = 0;
        control_sent = 0;
        control_stale_drop = 0;
        control_mode_drop = 0;
        control_send_s = 0.0;
        control_cmd_age_s = 0.0;
        control_total_latency_s = 0.0;
        control_latency_samples = 0;
        control_total_latency_full_s = 0.0;
        control_latency_full_samples = 0;
        control_total_apply_latency_s = 0.0;
        control_apply_latency_samples = 0;
        control_total_apply_latency_full_s = 0.0;
        control_apply_latency_full_samples = 0;
    }
};

namespace {

constexpr auto kToggleCooldown = std::chrono::milliseconds(200);
constexpr auto kControlIdleSleep = std::chrono::milliseconds(2);
constexpr auto kControlCommandWait = std::chrono::milliseconds(1);
constexpr auto kSideButtonKeySequenceIdleSleep = std::chrono::milliseconds(1);
constexpr auto kCaptureIdleSleep = std::chrono::milliseconds(1);
constexpr auto kInferenceIdleSleep = std::chrono::milliseconds(1);
constexpr auto kPerfLoopSleep = std::chrono::milliseconds(50);
constexpr double kPerfLogIntervalSeconds = 1.0;
constexpr double kCommandTimeoutSeconds = 0.10;
constexpr double kMaxFrameAgeSeconds = 0.05;
constexpr double kTargetTimeoutSeconds = 0.08;
constexpr double kLegacyRecoilReferenceHz = 120.0;
constexpr double kMaxRecoilIntegrateDtSeconds = 0.05;
constexpr float kMinTrackDt = 1.0F / 240.0F;
constexpr float kMaxTrackDt = 0.06F;
constexpr float kMaxTrackSpeedPxS = 1800.0F;
constexpr float kEgoMotionCompAlpha = 0.30F;
constexpr float kCmdSendLatencyAlpha = 0.20F;
constexpr float kEgoMotionCompMaxPxS = 3200.0F;
constexpr float kEgoMotionCompDecay = 0.92F;
constexpr float kTargetLockMaxJump = 260.0F;
constexpr float kTrackerReinitMinIou = 0.35F;
constexpr float kAssocPredictDt = 0.02F;
constexpr float kAssocSpeedJumpGain = 0.05F;
constexpr float kAssocMaxJumpPad = 220.0F;
constexpr bool kPerfLogWhenModeOff = true;

bool risingEdge(const bool current, const bool previous) {
    return current && !previous;
}

double secondsSince(const SystemClock::time_point since, const SystemClock::time_point now) {
    return std::chrono::duration<double>(now - since).count();
}

double secondsSince(const SteadyClock::time_point since, const SteadyClock::time_point now) {
    return std::chrono::duration<double>(now - since).count();
}

int aimModeBeepFrequency(const AimMode aim_mode) {
    switch (aim_mode) {
    case AimMode::Body: return 600;
    case AimMode::Hybrid: return 900;
    case AimMode::Head:
    default: return 1200;
    }
}

bool canReusePreviousTarget(
    const AimMode aim_mode,
    const bool using_head_candidates,
    const int prev_target_cls) {
    if (aim_mode == AimMode::Hybrid) {
        return prev_target_cls == (using_head_candidates ? 1 : 0);
    }
    return prev_target_cls == aimModeTargetClass(aim_mode);
}

float emaUpdateSigned(const float prev, const float sample, const float alpha) {
    const float clamped_alpha = clamp(alpha, 0.0F, 1.0F);
    if (clamped_alpha <= 0.0F) {
        return prev;
    }
    return prev + ((sample - prev) * clamped_alpha);
}

std::pair<int, int> screenCenter(const StaticConfig& config) {
    return {config.screen_w / 2, config.screen_h / 2};
}

float bboxIou(const std::array<int, 4>& a, const std::array<int, 4>& b) {
    const float ax1 = static_cast<float>(a[0]);
    const float ay1 = static_cast<float>(a[1]);
    const float ax2 = static_cast<float>(a[2]);
    const float ay2 = static_cast<float>(a[3]);
    const float bx1 = static_cast<float>(b[0]);
    const float by1 = static_cast<float>(b[1]);
    const float bx2 = static_cast<float>(b[2]);
    const float by2 = static_cast<float>(b[3]);
    const float ix1 = std::max(ax1, bx1);
    const float iy1 = std::max(ay1, by1);
    const float ix2 = std::min(ax2, bx2);
    const float iy2 = std::min(ay2, by2);
    const float iw = std::max(0.0F, ix2 - ix1);
    const float ih = std::max(0.0F, iy2 - iy1);
    const float inter = iw * ih;
    if (inter <= 0.0F) {
        return 0.0F;
    }
    const float area_a = std::max(0.0F, ax2 - ax1) * std::max(0.0F, ay2 - ay1);
    const float area_b = std::max(0.0F, bx2 - bx1) * std::max(0.0F, by2 - by1);
    const float denom = area_a + area_b - inter;
    return denom > 1e-6F ? (inter / denom) : 0.0F;
}

float pointToBoxDistance(const std::array<int, 4>& box, const float x, const float y) {
    const float x1 = static_cast<float>(box[0]);
    const float y1 = static_cast<float>(box[1]);
    const float x2 = static_cast<float>(box[2]);
    const float y2 = static_cast<float>(box[3]);
    const float dx = x < x1 ? (x1 - x) : (x > x2 ? x - x2 : 0.0F);
    const float dy = y < y1 ? (y1 - y) : (y > y2 ? y - y2 : 0.0F);
    return std::sqrt((dx * dx) + (dy * dy));
}

CaptureRegion buildCaptureRegion(const StaticConfig& config, const int center_x, const int center_y) {
    const int size = clamp(effectiveCaptureCropSize(config), 1, std::min(config.screen_w, config.screen_h));
    const int width = size;
    const int height = size;
    return CaptureRegion{
        .left = clamp(center_x - (width / 2), 0, config.screen_w - width),
        .top = clamp(center_y - (height / 2), 0, config.screen_h - height),
        .width = width,
        .height = height,
    };
}

PIDSettleConfig buildPidSettleConfig(const RuntimeConfig& runtime) {
    return PIDSettleConfig{
        .enable = runtime.pid_settle_enable,
        .error_px = runtime.pid_settle_error_px,
        .threshold_min_scale = runtime.pid_settle_threshold_min_scale,
        .threshold_max_scale = runtime.pid_settle_threshold_max_scale,
        .stable_frames = runtime.pid_settle_stable_frames,
        .error_delta_px = runtime.pid_settle_error_delta_px,
        .pre_output_scale = runtime.pid_settle_pre_output_scale,
    };
}

LegacyPidConfig buildLegacyPidConfig(const RuntimeConfig& runtime) {
    return LegacyPidConfig{
        .kp = runtime.kp,
        .ki = runtime.ki,
        .kd = runtime.kd,
        .lock_error_px = runtime.legacy_pid_lock_error_px,
        .speed_multiplier = runtime.legacy_pid_speed_multiplier,
        .threshold_min_scale = runtime.legacy_pid_threshold_min_scale,
        .threshold_max_scale = runtime.legacy_pid_threshold_max_scale,
        .transition_sharpness = runtime.legacy_pid_transition_sharpness,
        .transition_midpoint = runtime.legacy_pid_transition_midpoint,
        .stable_frames = runtime.legacy_pid_stable_frames,
        .error_delta_px = runtime.legacy_pid_error_delta_px,
        .prelock_scale = runtime.legacy_pid_prelock_scale,
    };
}

float trackerVelocityAlpha(const RuntimeConfig& runtime) {
    return (runtime.tracking_strategy == TrackingStrategy::LegacyPid || runtime.tracking_strategy == TrackingStrategy::PredictivePid)
        ? 1.0F
        : runtime.tracking_velocity_alpha;
}

bool pidRuntimeSettingsChanged(const RuntimeConfig& lhs, const RuntimeConfig& rhs) {
    return lhs.pid_enable != rhs.pid_enable
        || lhs.kp != rhs.kp
        || lhs.ki != rhs.ki
        || lhs.kd != rhs.kd
        || lhs.integral_limit != rhs.integral_limit
        || lhs.anti_windup_gain != rhs.anti_windup_gain
        || lhs.derivative_alpha != rhs.derivative_alpha
        || lhs.output_limit != rhs.output_limit
        || lhs.pid_settle_enable != rhs.pid_settle_enable
        || lhs.pid_settle_error_px != rhs.pid_settle_error_px
        || lhs.pid_settle_threshold_min_scale != rhs.pid_settle_threshold_min_scale
        || lhs.pid_settle_threshold_max_scale != rhs.pid_settle_threshold_max_scale
        || lhs.pid_settle_stable_frames != rhs.pid_settle_stable_frames
        || lhs.pid_settle_error_delta_px != rhs.pid_settle_error_delta_px
        || lhs.pid_settle_pre_output_scale != rhs.pid_settle_pre_output_scale
        || lhs.legacy_pid_lock_error_px != rhs.legacy_pid_lock_error_px
        || lhs.legacy_pid_speed_multiplier != rhs.legacy_pid_speed_multiplier
        || lhs.legacy_pid_threshold_min_scale != rhs.legacy_pid_threshold_min_scale
        || lhs.legacy_pid_threshold_max_scale != rhs.legacy_pid_threshold_max_scale
        || lhs.legacy_pid_transition_sharpness != rhs.legacy_pid_transition_sharpness
        || lhs.legacy_pid_transition_midpoint != rhs.legacy_pid_transition_midpoint
        || lhs.legacy_pid_stable_frames != rhs.legacy_pid_stable_frames
        || lhs.legacy_pid_error_delta_px != rhs.legacy_pid_error_delta_px
        || lhs.legacy_pid_prelock_scale != rhs.legacy_pid_prelock_scale
        || lhs.predictive_pid_kp != rhs.predictive_pid_kp
        || lhs.predictive_pid_ki != rhs.predictive_pid_ki
        || lhs.predictive_pid_kd != rhs.predictive_pid_kd
        || lhs.predictive_pid_pred_weight_x != rhs.predictive_pid_pred_weight_x
        || lhs.predictive_pid_pred_weight_y != rhs.predictive_pid_pred_weight_y
        || lhs.predictive_pid_init_scale != rhs.predictive_pid_init_scale
        || lhs.predictive_pid_ramp_time_s != rhs.predictive_pid_ramp_time_s
        || lhs.predictive_pid_integral_limit != rhs.predictive_pid_integral_limit
        || lhs.predictive_pid_derivative_limit != rhs.predictive_pid_derivative_limit
        || lhs.predictive_pid_output_limit != rhs.predictive_pid_output_limit
        || lhs.predictive_pid_velocity_alpha != rhs.predictive_pid_velocity_alpha
        || lhs.predictive_pid_acceleration_alpha != rhs.predictive_pid_acceleration_alpha
        || lhs.predictive_pid_max_velocity_px_s != rhs.predictive_pid_max_velocity_px_s
        || lhs.predictive_pid_max_acceleration_px_s != rhs.predictive_pid_max_acceleration_px_s
        || lhs.predictive_pid_reverse_gate_px != rhs.predictive_pid_reverse_gate_px
        || lhs.predictive_pid_reverse_scale != rhs.predictive_pid_reverse_scale
        || lhs.predictive_pid_prediction_error_scale != rhs.predictive_pid_prediction_error_scale
        || lhs.predictive_pid_prediction_min_px != rhs.predictive_pid_prediction_min_px
        || lhs.predictive_pid_prediction_max_px != rhs.predictive_pid_prediction_max_px;
}

DebugPreviewSnapshot makeInactiveDebugPreviewSnapshot(
    const CaptureRegion& capture_region,
    const std::pair<int, int> center) {
    DebugPreviewSnapshot snapshot{};
    snapshot.capture_region = capture_region;
    snapshot.screen_center = center;
    return snapshot;
}

DebugPreviewSnapshot makeDebugPreviewSnapshot(
    const CaptureRegion& capture_region,
    const std::pair<int, int> center,
    const std::vector<Detection>& detections,
    const std::optional<Detection>& selected_detection = std::nullopt) {
    DebugPreviewSnapshot snapshot{};
    snapshot.active = true;
    snapshot.capture_region = capture_region;
    snapshot.screen_center = center;
    snapshot.detections.reserve(detections.size());
    for (const auto& detection : detections) {
        snapshot.detections.push_back(DebugPreviewDetection{
            .bbox = detection.bbox,
            .cls = detection.cls,
            .conf = detection.conf,
            .selected = selected_detection.has_value()
                && selected_detection->bbox == detection.bbox
                && selected_detection->cls == detection.cls
                && std::abs(selected_detection->conf - detection.conf) <= 1e-6F,
        });
    }
    return snapshot;
}

void clearAimStateLocked(SharedState& shared, const std::pair<int, int> center, const TrackingStrategy strategy) {
    shared.target_found = false;
    shared.target_cls = -1;
    shared.target_speed = 0.0F;
    shared.pid_settled = false;
    shared.pid_settle_error_metric_px = 0.0F;
    shared.pid_settle_threshold_px = 0.0F;
    shared.lead_active = false;
    shared.lead_time_ms = 0.0F;
    shared.aim_dx = 0;
    shared.aim_dy = 0;
    shared.last_target_full = center;
    shared.capture_focus_full = center;
    shared.target_time = {};
    shared.tracking_strategy = trackingStrategyName(strategy);
}

void resetEgoMotionStateLocked(SharedState& shared) {
    shared.ctrl_sent_vx_ema.store(0.0F, std::memory_order_relaxed);
    shared.ctrl_sent_vy_ema.store(0.0F, std::memory_order_relaxed);
    shared.ctrl_last_send_tick = {};
}

void decayEgoMotionStateLocked(SharedState& shared) {
    float ctrl_sent_vx_ema = shared.ctrl_sent_vx_ema.load(std::memory_order_relaxed) * kEgoMotionCompDecay;
    float ctrl_sent_vy_ema = shared.ctrl_sent_vy_ema.load(std::memory_order_relaxed) * kEgoMotionCompDecay;
    if (std::abs(ctrl_sent_vx_ema) < 1e-6F) {
        ctrl_sent_vx_ema = 0.0F;
    }
    if (std::abs(ctrl_sent_vy_ema) < 1e-6F) {
        ctrl_sent_vy_ema = 0.0F;
    }
    shared.ctrl_sent_vx_ema.store(ctrl_sent_vx_ema, std::memory_order_relaxed);
    shared.ctrl_sent_vy_ema.store(ctrl_sent_vy_ema, std::memory_order_relaxed);
}

struct InferenceAppTimings {
    double select_s = 0.0;
    double tracker_update_s = 0.0;
    double aim_predict_s = 0.0;
    double aim_pid_s = 0.0;
    double aim_sync_s = 0.0;
    double preview_s = 0.0;
    double queue_s = 0.0;
};

std::optional<StickyTargetPick> pickAssociatedStickyTarget(
    const std::vector<Detection>& detections,
    const int center_x,
    const int center_y,
    const std::pair<float, float>& assoc_ref,
    const float assoc_limit,
    const std::optional<std::pair<float, float>>& locked_point,
    const std::optional<std::array<int, 4>>& last_target_bbox,
    const float sticky_bias_px) {
    StickyTargetPick result{};
    bool found_any = false;
    int locked_idx = -1;
    float best_locked_distance = std::numeric_limits<float>::max();

    for (int i = 0; i < static_cast<int>(detections.size()); ++i) {
        const auto& detection = detections[static_cast<size_t>(i)];
        const bool near_ref = pointToBoxDistance(detection.bbox, assoc_ref.first, assoc_ref.second) <= assoc_limit;
        const bool overlaps_prev = last_target_bbox.has_value()
            && bboxIou(detection.bbox, *last_target_bbox) >= kTrackerReinitMinIou;
        if (!(near_ref || overlaps_prev)) {
            continue;
        }
        found_any = true;
        if (!locked_point.has_value()) {
            continue;
        }
        const float dx = detection.x - locked_point->first;
        const float dy = detection.y - locked_point->second;
        const float locked_distance = std::sqrt((dx * dx) + (dy * dy));
        if (locked_distance < best_locked_distance) {
            best_locked_distance = locked_distance;
            locked_idx = i;
        }
    }

    if (!found_any) {
        return std::nullopt;
    }

    float best_score = std::numeric_limits<float>::max();
    float best_anchor_score = std::numeric_limits<float>::max();
    float best_conf = -1.0F;
    int best_idx = -1;

    for (int i = 0; i < static_cast<int>(detections.size()); ++i) {
        const auto& detection = detections[static_cast<size_t>(i)];
        const bool near_ref = pointToBoxDistance(detection.bbox, assoc_ref.first, assoc_ref.second) <= assoc_limit;
        const bool overlaps_prev = last_target_bbox.has_value()
            && bboxIou(detection.bbox, *last_target_bbox) >= kTrackerReinitMinIou;
        if (!(near_ref || overlaps_prev)) {
            continue;
        }

        float score = pointToBoxDistance(detection.bbox, static_cast<float>(center_x), static_cast<float>(center_y));
        const float anchor_score = std::sqrt(
            ((detection.x - static_cast<float>(center_x)) * (detection.x - static_cast<float>(center_x)))
            + ((detection.y - static_cast<float>(center_y)) * (detection.y - static_cast<float>(center_y))));
        if (i == locked_idx) {
            score -= sticky_bias_px;
        }
        if (
            score < best_score
            || (
                std::abs(score - best_score) <= 1e-6F
                && (
                    anchor_score < best_anchor_score
                    || (std::abs(anchor_score - best_anchor_score) <= 1e-6F && detection.conf > best_conf)
                )
            )
        ) {
            best_idx = i;
            best_score = score;
            best_anchor_score = anchor_score;
            best_conf = detection.conf;
        }
    }

    if (best_idx < 0) {
        return std::nullopt;
    }
    result.detection = detections[static_cast<size_t>(best_idx)];
    result.switched = locked_idx >= 0 && best_idx != locked_idx;
    return result;
}

struct PerfLogSnapshot {
    double elapsed_s = 0.0;
    double cap_fps = 0.0;
    double cap_grab_ms = 0.0;
    double cap_acquire_ms = 0.0;
    double cap_d3d_copy_ms = 0.0;
    double cap_d3d_sync_ms = 0.0;
    double cap_cuda_copy_ms = 0.0;
    double cap_cpu_copy_ms = 0.0;
    double cap_cached_reuse_ms = 0.0;
    double cap_cached_rate = 0.0;
    std::uint64_t cap_none = 0;
    double infer_fps = 0.0;
    double infer_loop_ms = 0.0;
    double infer_age_ms = 0.0;
    double infer_age_max_ms = 0.0;
    std::uint64_t infer_found = 0;
    std::uint64_t infer_stale = 0;
    double infer_lock_rate = 0.0;
    double infer_select_ms = 0.0;
    double infer_tracker_update_ms = 0.0;
    double infer_aim_predict_ms = 0.0;
    double infer_aim_pid_ms = 0.0;
    double infer_aim_sync_ms = 0.0;
    double infer_preview_ms = 0.0;
    double infer_queue_ms = 0.0;
    std::uint64_t infer_backend_samples = 0;
    double infer_backend_pre_ms = 0.0;
    double infer_backend_exec_ms = 0.0;
    double infer_backend_post_ms = 0.0;
    double infer_cmd_ms = 0.0;
    double control_send_hz = 0.0;
    double control_send_ms = 0.0;
    double control_cmd_age_ms = 0.0;
    double control_total_latency_ms = 0.0;
    double control_total_latency_full_ms = 0.0;
    double control_total_apply_latency_ms = 0.0;
    double control_total_apply_latency_full_ms = 0.0;
    std::uint64_t control_stale_drop = 0;
    std::uint64_t control_mode_drop = 0;
};

void recordCapturePerf(
    RuntimePerfWindow& perf,
    const double grab_s,
    const bool is_none,
    const std::optional<CaptureTimings>& timings = std::nullopt) {
    std::lock_guard<std::mutex> lock(perf.mutex);
    if (is_none) {
        ++perf.capture_none;
        return;
    }
    ++perf.capture_frames;
    perf.capture_grab_s += std::max(0.0, grab_s);
    if (timings.has_value()) {
        perf.capture_acquire_s += std::max(0.0, timings->acquire_s);
        perf.capture_d3d_copy_s += std::max(0.0, timings->d3d_copy_s);
        perf.capture_d3d_sync_s += std::max(0.0, timings->d3d_sync_s);
        perf.capture_cuda_copy_s += std::max(0.0, timings->cuda_copy_s);
        perf.capture_cpu_copy_s += std::max(0.0, timings->cpu_copy_s);
        perf.capture_cached_reuse_s += std::max(0.0, timings->cached_reuse_s);
        if (timings->used_cached_frame) {
            ++perf.capture_cached_frames;
        }
    }
}

void recordInferencePerf(
    RuntimePerfWindow& perf,
    const double frame_age_s,
    const double loop_s,
    const bool stale_drop,
    const bool target_found,
    const InferenceAppTimings& app_timings,
    const InferenceTimings& timings,
    const std::optional<double> cmd_latency_s) {
    std::lock_guard<std::mutex> lock(perf.mutex);
    ++perf.infer_frames;
    perf.infer_loop_s += std::max(0.0, loop_s);
    perf.infer_frame_age_s += std::max(0.0, frame_age_s);
    perf.infer_frame_age_max_s = std::max(perf.infer_frame_age_max_s, std::max(0.0, frame_age_s));
    if (stale_drop) {
        ++perf.infer_stale;
    }
    if (target_found) {
        ++perf.infer_found;
    }
    perf.infer_select_s += std::max(0.0, app_timings.select_s);
    perf.infer_tracker_update_s += std::max(0.0, app_timings.tracker_update_s);
    perf.infer_aim_predict_s += std::max(0.0, app_timings.aim_predict_s);
    perf.infer_aim_pid_s += std::max(0.0, app_timings.aim_pid_s);
    perf.infer_aim_sync_s += std::max(0.0, app_timings.aim_sync_s);
    perf.infer_preview_s += std::max(0.0, app_timings.preview_s);
    perf.infer_queue_s += std::max(0.0, app_timings.queue_s);
    if (timings.preprocess_ms > 0.0 || timings.execute_ms > 0.0 || timings.postprocess_ms > 0.0) {
        ++perf.infer_backend_samples;
        perf.infer_backend_pre_s += timings.preprocess_ms / 1000.0;
        perf.infer_backend_exec_s += timings.execute_ms / 1000.0;
        perf.infer_backend_post_s += timings.postprocess_ms / 1000.0;
    }
    if (cmd_latency_s.has_value()) {
        ++perf.infer_cmd_samples;
        perf.infer_cmd_latency_s += std::max(0.0, *cmd_latency_s);
    }
}

void recordControlPerf(
    RuntimePerfWindow& perf,
    const double cmd_age_s,
    const bool sent,
    const double send_s,
    const bool stale_drop,
    const bool mode_drop,
    const std::optional<double> total_latency_s,
    const std::optional<double> total_latency_full_s,
    const std::optional<double> total_apply_latency_s,
    const std::optional<double> total_apply_latency_full_s) {
    std::lock_guard<std::mutex> lock(perf.mutex);
    ++perf.control_cmds;
    if (stale_drop) {
        ++perf.control_stale_drop;
    }
    if (mode_drop) {
        ++perf.control_mode_drop;
    }
    if (!sent) {
        return;
    }

    ++perf.control_sent;
    perf.control_send_s += std::max(0.0, send_s);
    perf.control_cmd_age_s += std::max(0.0, cmd_age_s);
    if (total_latency_s.has_value()) {
        ++perf.control_latency_samples;
        perf.control_total_latency_s += std::max(0.0, *total_latency_s);
    }
    if (total_latency_full_s.has_value()) {
        ++perf.control_latency_full_samples;
        perf.control_total_latency_full_s += std::max(0.0, *total_latency_full_s);
    }
    if (total_apply_latency_s.has_value()) {
        ++perf.control_apply_latency_samples;
        perf.control_total_apply_latency_s += std::max(0.0, *total_apply_latency_s);
    }
    if (total_apply_latency_full_s.has_value()) {
        ++perf.control_apply_latency_full_samples;
        perf.control_total_apply_latency_full_s += std::max(0.0, *total_apply_latency_full_s);
    }
}

std::optional<PerfLogSnapshot> takePerfSnapshot(RuntimePerfWindow& perf, const double min_interval_s) {
    const auto now = SteadyClock::now();
    std::lock_guard<std::mutex> lock(perf.mutex);
    const double elapsed = secondsSince(perf.window_start, now);
    if (elapsed < min_interval_s) {
        return std::nullopt;
    }

    PerfLogSnapshot snapshot{};
    snapshot.elapsed_s = elapsed;
    snapshot.cap_fps = elapsed > 0.0 ? static_cast<double>(perf.capture_frames) / elapsed : 0.0;
    snapshot.cap_grab_ms = perf.capture_frames > 0 ? (perf.capture_grab_s * 1000.0 / static_cast<double>(perf.capture_frames)) : 0.0;
    snapshot.cap_acquire_ms = perf.capture_frames > 0 ? (perf.capture_acquire_s * 1000.0 / static_cast<double>(perf.capture_frames)) : 0.0;
    snapshot.cap_d3d_copy_ms = perf.capture_frames > 0 ? (perf.capture_d3d_copy_s * 1000.0 / static_cast<double>(perf.capture_frames)) : 0.0;
    snapshot.cap_d3d_sync_ms = perf.capture_frames > 0 ? (perf.capture_d3d_sync_s * 1000.0 / static_cast<double>(perf.capture_frames)) : 0.0;
    snapshot.cap_cuda_copy_ms = perf.capture_frames > 0 ? (perf.capture_cuda_copy_s * 1000.0 / static_cast<double>(perf.capture_frames)) : 0.0;
    snapshot.cap_cpu_copy_ms = perf.capture_frames > 0 ? (perf.capture_cpu_copy_s * 1000.0 / static_cast<double>(perf.capture_frames)) : 0.0;
    snapshot.cap_cached_reuse_ms = perf.capture_cached_frames > 0
        ? (perf.capture_cached_reuse_s * 1000.0 / static_cast<double>(perf.capture_cached_frames))
        : 0.0;
    snapshot.cap_cached_rate = perf.capture_frames > 0
        ? static_cast<double>(perf.capture_cached_frames) / static_cast<double>(perf.capture_frames)
        : 0.0;
    snapshot.cap_none = perf.capture_none;
    snapshot.infer_fps = elapsed > 0.0 ? static_cast<double>(perf.infer_frames) / elapsed : 0.0;
    snapshot.infer_loop_ms = perf.infer_frames > 0 ? (perf.infer_loop_s * 1000.0 / static_cast<double>(perf.infer_frames)) : 0.0;
    snapshot.infer_age_ms = perf.infer_frames > 0 ? (perf.infer_frame_age_s * 1000.0 / static_cast<double>(perf.infer_frames)) : 0.0;
    snapshot.infer_age_max_ms = perf.infer_frame_age_max_s * 1000.0;
    snapshot.infer_found = perf.infer_found;
    snapshot.infer_stale = perf.infer_stale;
    snapshot.infer_lock_rate = perf.infer_frames > 0 ? static_cast<double>(perf.infer_found) / static_cast<double>(perf.infer_frames) : 0.0;
    snapshot.infer_select_ms = perf.infer_frames > 0 ? (perf.infer_select_s * 1000.0 / static_cast<double>(perf.infer_frames)) : 0.0;
    snapshot.infer_tracker_update_ms = perf.infer_frames > 0 ? (perf.infer_tracker_update_s * 1000.0 / static_cast<double>(perf.infer_frames)) : 0.0;
    snapshot.infer_aim_predict_ms = perf.infer_frames > 0 ? (perf.infer_aim_predict_s * 1000.0 / static_cast<double>(perf.infer_frames)) : 0.0;
    snapshot.infer_aim_pid_ms = perf.infer_frames > 0 ? (perf.infer_aim_pid_s * 1000.0 / static_cast<double>(perf.infer_frames)) : 0.0;
    snapshot.infer_aim_sync_ms = perf.infer_frames > 0 ? (perf.infer_aim_sync_s * 1000.0 / static_cast<double>(perf.infer_frames)) : 0.0;
    snapshot.infer_preview_ms = perf.infer_frames > 0 ? (perf.infer_preview_s * 1000.0 / static_cast<double>(perf.infer_frames)) : 0.0;
    snapshot.infer_queue_ms = perf.infer_frames > 0 ? (perf.infer_queue_s * 1000.0 / static_cast<double>(perf.infer_frames)) : 0.0;
    snapshot.infer_backend_samples = perf.infer_backend_samples;
    snapshot.infer_backend_pre_ms = perf.infer_backend_samples > 0 ? (perf.infer_backend_pre_s * 1000.0 / static_cast<double>(perf.infer_backend_samples)) : 0.0;
    snapshot.infer_backend_exec_ms = perf.infer_backend_samples > 0 ? (perf.infer_backend_exec_s * 1000.0 / static_cast<double>(perf.infer_backend_samples)) : 0.0;
    snapshot.infer_backend_post_ms = perf.infer_backend_samples > 0 ? (perf.infer_backend_post_s * 1000.0 / static_cast<double>(perf.infer_backend_samples)) : 0.0;
    snapshot.infer_cmd_ms = perf.infer_cmd_samples > 0 ? (perf.infer_cmd_latency_s * 1000.0 / static_cast<double>(perf.infer_cmd_samples)) : 0.0;
    snapshot.control_send_hz = elapsed > 0.0 ? static_cast<double>(perf.control_sent) / elapsed : 0.0;
    snapshot.control_send_ms = perf.control_sent > 0 ? (perf.control_send_s * 1000.0 / static_cast<double>(perf.control_sent)) : 0.0;
    snapshot.control_cmd_age_ms = perf.control_sent > 0 ? (perf.control_cmd_age_s * 1000.0 / static_cast<double>(perf.control_sent)) : 0.0;
    snapshot.control_total_latency_ms = perf.control_latency_samples > 0 ? (perf.control_total_latency_s * 1000.0 / static_cast<double>(perf.control_latency_samples)) : 0.0;
    snapshot.control_total_latency_full_ms = perf.control_latency_full_samples > 0 ? (perf.control_total_latency_full_s * 1000.0 / static_cast<double>(perf.control_latency_full_samples)) : 0.0;
    snapshot.control_total_apply_latency_ms = perf.control_apply_latency_samples > 0 ? (perf.control_total_apply_latency_s * 1000.0 / static_cast<double>(perf.control_apply_latency_samples)) : 0.0;
    snapshot.control_total_apply_latency_full_ms = perf.control_apply_latency_full_samples > 0 ? (perf.control_total_apply_latency_full_s * 1000.0 / static_cast<double>(perf.control_apply_latency_full_samples)) : 0.0;
    snapshot.control_stale_drop = perf.control_stale_drop;
    snapshot.control_mode_drop = perf.control_mode_drop;

    perf.reset(now);
    return snapshot;
}

}  // namespace

DeltaApp::DeltaApp(StaticConfig config, RuntimeConfig runtime)
    : config_(std::move(config)),
      runtime_store_(std::move(runtime)),
      capture_(makeDefaultCaptureSource(config_)),
      inference_(makeInferenceEngine(config_)),
      input_sender_(makeInputSender()),
      mouse_move_suppressor_(makeMouseMoveSuppressor()),
      recoil_scheduler_(std::make_unique<RecoilScheduler>(config_)),
      debug_preview_(makeDebugPreviewWindow(config_)),
      debug_overlay_(makeDebugOverlayWindow(config_)),
      frontend_(makeRuntimeFrontend(config_, runtime_store_, shared_)),
      perf_(std::make_unique<RuntimePerfWindow>()) {
    if (capture_ && inference_) {
        capture_->setGpuConsumerStream(inference_->gpuInputStream());
    }
    if (capture_) {
        capture_->setCachedFrameTimeoutMs(runtime_store_.snapshot().capture_cached_timeout_ms);
    }
    const auto center = screenCenter(config_);
    std::lock_guard<std::mutex> lock(shared_.mutex);
    shared_.last_target_full = center;
    shared_.capture_focus_full = center;
    shared_.tracking_strategy = trackingStrategyName(runtime_store_.snapshot().tracking_strategy);
    shared_.recoil.mode = runtime_store_.snapshot().recoil_mode;
    shared_.recoil.selected_profile_id = runtime_store_.snapshot().selected_recoil_profile_id;
}

DeltaApp::~DeltaApp() = default;

int DeltaApp::run() {
    std::cout << "Delta native scaffold initialized.\n";
    std::cout << "Capture module: " << (capture_ ? capture_->name() : "none") << "\n";
    std::cout << "Inference module: " << (inference_ ? inference_->name() : "none") << "\n";
    std::cout << "Control module: " << (input_sender_ ? input_sender_->name() : "none") << "\n";
    if (!capture_ || !inference_ || !input_sender_) {
        std::cerr << "[fatal] Native runtime is missing a required module.\n";
        return 1;
    }

    inference_->setModelConfidence(runtime_store_.snapshot().model_conf);
    inference_->warmup();
    std::cout << "Runtime ready. Open the frontend and press Insert to exit.\n";

    try {
        if (mouse_move_suppressor_) {
            mouse_move_suppressor_->setDebugLogging(runtime_store_.snapshot().mouse_move_suppress_on_fire_debug);
            mouse_move_suppressor_->start();
            const MouseMoveSuppressionStatus status = mouse_move_suppressor_->snapshot();
            std::lock_guard<std::mutex> lock(shared_.mutex);
            shared_.mouse_move_suppress_supported = status.supported;
            shared_.mouse_move_suppress_active = status.active;
            shared_.mouse_move_suppress_count = status.suppressed_count;
        }
        if (frontend_) {
            frontend_->start();
        }
        if (debug_preview_) {
            debug_preview_->start();
            debug_preview_->setEnabled(runtime_store_.snapshot().debug_preview_enable);
        }
        if (debug_overlay_) {
            debug_overlay_->start();
            debug_overlay_->setEnabled(runtime_store_.snapshot().debug_overlay_enable);
        }
        if (config_.perf_log_enable && perf_) {
            perf_thread_ = AppThread([this]() { perfLoop(); });
        }
        if (!(inference_ && inference_->supportsGpuInput())) {
            capture_thread_ = AppThread([this]() { captureLoop(); });
        }
        inference_thread_ = AppThread([this]() { inferenceLoop(); });
        recoil_thread_ = AppThread([this]() { recoilLoop(); });
        control_thread_ = AppThread([this]() { controlLoop(); });
        side_button_key_sequence_thread_ = AppThread([this]() { sideButtonKeySequenceLoop(); });

        while (true) {
            {
                std::lock_guard<std::mutex> lock(shared_.mutex);
                if (!shared_.running) {
                    break;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    } catch (...) {
        stop();
        if (mouse_move_suppressor_) {
            mouse_move_suppressor_->stop();
        }
        if (debug_preview_) {
            debug_preview_->stop();
        }
        if (debug_overlay_) {
            debug_overlay_->stop();
        }
        if (frontend_) {
            frontend_->stop();
        }
        throw;
    }

    stop();
    if (capture_thread_.joinable()) {
        capture_thread_.join();
    }
    if (inference_thread_.joinable()) {
        inference_thread_.join();
    }
    if (control_thread_.joinable()) {
        control_thread_.join();
    }
    if (recoil_thread_.joinable()) {
        recoil_thread_.join();
    }
    if (side_button_key_sequence_thread_.joinable()) {
        side_button_key_sequence_thread_.join();
    }
    if (perf_thread_.joinable()) {
        perf_thread_.join();
    }
    if (mouse_move_suppressor_) {
        mouse_move_suppressor_->stop();
    }
    if (debug_preview_) {
        debug_preview_->stop();
    }
    if (debug_overlay_) {
        debug_overlay_->stop();
    }
    if (frontend_) {
        frontend_->stop();
    }

    return 0;
}

void DeltaApp::stop() {
    {
        std::lock_guard<std::mutex> lock(shared_.mutex);
        shared_.running = false;
    }
    if (mouse_move_suppressor_) {
        mouse_move_suppressor_->setSuppressionActive(false);
        const MouseMoveSuppressionStatus status = mouse_move_suppressor_->snapshot();
        std::lock_guard<std::mutex> lock(shared_.mutex);
        shared_.mouse_move_suppress_supported = status.supported;
        shared_.mouse_move_suppress_active = status.active;
        shared_.mouse_move_suppress_count = status.suppressed_count;
    }
}

void DeltaApp::captureLoop() {
    try {
        const auto center = screenCenter(config_);
        const bool prefer_gpu = inference_ && inference_->supportsGpuInput();

        for (;;) {
            {
                std::lock_guard<std::mutex> lock(shared_.mutex);
                if (!shared_.running) {
                    break;
                }
            }

            const bool freeze_capture_to_center = runtime_store_.snapshot().capture_freeze_to_center_enable;
            std::pair<int, int> focus = center;
            {
                std::lock_guard<std::mutex> lock(shared_.mutex);
                focus = selectCaptureFocus(
                    freeze_capture_to_center,
                    shared_.target_found,
                    center,
                    shared_.capture_focus_full);
            }

            const CaptureRegion region = buildCaptureRegion(config_, focus.first, focus.second);
            bool captured = false;

            if (prefer_gpu) {
                const auto grab_start = SteadyClock::now();
                if (std::optional<GpuFramePacket> packet = capture_->grabGpu(region); packet.has_value()) {
                    const CaptureTimings timings = packet->timings;
                    frame_slot_.clear();
                    gpu_frame_slot_.put(std::move(*packet));
                    if (perf_) {
                        recordCapturePerf(*perf_, secondsSince(grab_start, SteadyClock::now()), false, timings);
                    }
                    captured = true;
                } else if (perf_) {
                    recordCapturePerf(*perf_, secondsSince(grab_start, SteadyClock::now()), true);
                }
            }

            if (!captured) {
                const auto grab_start = SteadyClock::now();
                if (std::optional<FramePacket> packet = capture_->grab(region); packet.has_value()) {
                    gpu_frame_slot_.clear();
                    const CaptureTimings timings = packet->timings;
                    frame_slot_.put(std::move(*packet));
                    if (perf_) {
                        recordCapturePerf(*perf_, secondsSince(grab_start, SteadyClock::now()), false, timings);
                    }
                    captured = true;
                } else if (perf_) {
                    recordCapturePerf(*perf_, secondsSince(grab_start, SteadyClock::now()), true);
                }
            }

            if (!captured) {
                std::this_thread::sleep_for(kCaptureIdleSleep);
            }
        }
    } catch (const std::exception& ex) {
        std::cerr << "[capture] loop failed: " << ex.what() << "\n";
        stop();
    }

    if (capture_) {
        capture_->close();
    }
}

void DeltaApp::inferenceLoop() {
    try {
        const auto center = screenCenter(config_);
        PIDController pid_x{};
        PIDController pid_y{};
        LegacyPidAxisState legacy_pid_x{};
        LegacyPidAxisState legacy_pid_y{};
        PredictivePidController predictive_pid{};

        RuntimeConfig runtime = runtime_store_.snapshot();
        const auto configurePidControllers = [&](const RuntimeConfig& current) {
            pid_x.configure(
                current.kp,
                current.ki,
                current.kd,
                current.integral_limit,
                current.anti_windup_gain,
                current.derivative_alpha,
                current.output_limit);
            pid_y.configure(
                current.kp,
                current.ki,
                current.kd,
                current.integral_limit,
                current.anti_windup_gain,
                current.derivative_alpha,
                current.output_limit);
            predictive_pid.configure(buildPredictivePidConfig(current));
        };
        configurePidControllers(runtime);
        if (capture_) {
            capture_->setCachedFrameTimeoutMs(runtime.capture_cached_timeout_ms);
        }

        RuntimeConfig last_pid_runtime = runtime;
        int last_capture_cached_timeout_ms = runtime.capture_cached_timeout_ms;
        std::uint64_t last_reset_token = runtime_store_.resetToken();
        TrackingStrategy last_tracking_strategy = runtime.tracking_strategy;
        AimMode last_aim_mode = runtime.aim_mode;
        auto tracker = makeTargetTracker(last_tracking_strategy, trackerVelocityAlpha(runtime));
        TargetGuardState target_guard_state{};
        TargetLeadState target_lead_state{};
        PIDSettleState pid_settle_state{};
        int lost_frames = 0;
        int active_target_cls = -1;
        float last_box_w = 0.0F;
        float last_box_h = 0.0F;
        std::optional<std::array<int, 4>> last_target_bbox;
        SteadyClock::time_point last_pid_tick{};
        SteadyClock::time_point last_track_tick{};
        bool last_debug_preview_enabled = runtime.debug_preview_enable;
        bool last_debug_overlay_enabled = runtime.debug_overlay_enable;
        bool preview_idle_state = true;
        const auto resetPidControllers = [&](const SteadyClock::time_point pid_tick = SteadyClock::time_point{}) {
            pid_x.reset();
            pid_y.reset();
            legacy_pid_x.reset();
            legacy_pid_y.reset();
            predictive_pid.reset();
            pid_settle_state.reset();
            last_pid_tick = pid_tick;
        };

        {
            std::lock_guard<std::mutex> lock(shared_.mutex);
            shared_.tracking_strategy = trackingStrategyName(last_tracking_strategy);
        }
        if (debug_preview_) {
            debug_preview_->setEnabled(last_debug_preview_enabled);
        }
        if (debug_overlay_) {
            debug_overlay_->setEnabled(last_debug_overlay_enabled);
        }

        for (;;) {
            {
                std::lock_guard<std::mutex> lock(shared_.mutex);
                if (!shared_.running) {
                    break;
                }
            }
            const auto loop_start = SteadyClock::now();

            runtime = runtime_store_.snapshot();
            const std::uint64_t reset_token = runtime_store_.resetToken();
            inference_->setModelConfidence(runtime.model_conf);
            const bool tracking_enabled = runtime.tracking_enabled;
            const TargetGuardConfig target_guard_config = buildTargetGuardConfig(runtime);
            const TargetLeadConfig target_lead_config = buildTargetLeadConfig(runtime);
            const TriggerbotConfig triggerbot_config = buildTriggerbotConfig(runtime);
            const bool debug_preview_enabled = runtime.debug_preview_enable;
            const bool debug_overlay_enabled = runtime.debug_overlay_enable;
            const bool debug_visuals_enabled = debug_preview_enabled || debug_overlay_enabled;
            const auto publishDebugSnapshot = [&](DebugPreviewSnapshot snapshot) {
                if (debug_preview_ && debug_preview_enabled) {
                    debug_preview_->publish(snapshot);
                }
                if (debug_overlay_ && debug_overlay_enabled) {
                    debug_overlay_->publish(std::move(snapshot));
                }
            };

            if (debug_preview_ && debug_preview_enabled != last_debug_preview_enabled) {
                debug_preview_->setEnabled(debug_preview_enabled);
                last_debug_preview_enabled = debug_preview_enabled;
                preview_idle_state = true;
            }
            if (debug_overlay_ && debug_overlay_enabled != last_debug_overlay_enabled) {
                debug_overlay_->setEnabled(debug_overlay_enabled);
                last_debug_overlay_enabled = debug_overlay_enabled;
                preview_idle_state = true;
            }

            if (capture_ && runtime.capture_cached_timeout_ms != last_capture_cached_timeout_ms) {
                capture_->setCachedFrameTimeoutMs(runtime.capture_cached_timeout_ms);
                last_capture_cached_timeout_ms = runtime.capture_cached_timeout_ms;
            }

            if (pidRuntimeSettingsChanged(runtime, last_pid_runtime)) {
                configurePidControllers(runtime);
                resetPidControllers();
                last_pid_runtime = runtime;
            }

            if (reset_token != last_reset_token) {
                resetPidControllers();
                target_guard_state.reset();
                target_lead_state.reset();
                last_reset_token = reset_token;
            }

            if (!target_guard_config.enable) {
                target_guard_state.reset();
            }
            if (!target_lead_config.enable) {
                target_lead_state.reset();
            }

            tracker->configure(trackerVelocityAlpha(runtime));
            if (runtime.tracking_strategy != last_tracking_strategy) {
                tracker = makeTargetTracker(runtime.tracking_strategy, trackerVelocityAlpha(runtime));
                resetPidControllers();
                resetAimTrackingState(
                    lost_frames,
                    active_target_cls,
                    last_box_w,
                    last_box_h,
                    last_target_bbox,
                    last_pid_tick,
                    last_track_tick);
                target_guard_state.reset();
                target_lead_state.reset();
                command_slot_.clear();
                {
                    std::lock_guard<std::mutex> lock(shared_.mutex);
                    clearAimStateLocked(shared_, center, runtime.tracking_strategy);
                }
                last_tracking_strategy = runtime.tracking_strategy;
                if (debug_visuals_enabled) {
                    publishDebugSnapshot(makeInactiveDebugPreviewSnapshot(
                        buildCaptureRegion(config_, center.first, center.second),
                        center));
                }
                preview_idle_state = true;
                std::this_thread::sleep_for(kInferenceIdleSleep);
                continue;
            }

            if (runtime.aim_mode != last_aim_mode) {
                tracker->reset();
                resetPidControllers();
                resetAimTrackingState(
                    lost_frames,
                    active_target_cls,
                    last_box_w,
                    last_box_h,
                    last_target_bbox,
                    last_pid_tick,
                    last_track_tick);
                target_guard_state.reset();
                target_lead_state.reset();
                command_slot_.clear();
                {
                    std::lock_guard<std::mutex> lock(shared_.mutex);
                    clearAimStateLocked(shared_, center, runtime.tracking_strategy);
                }
                last_aim_mode = runtime.aim_mode;
                if (debug_visuals_enabled) {
                    publishDebugSnapshot(makeInactiveDebugPreviewSnapshot(
                        buildCaptureRegion(config_, center.first, center.second),
                        center));
                }
                preview_idle_state = true;
                std::this_thread::sleep_for(kInferenceIdleSleep);
                continue;
            }

            ToggleState toggles{};
            bool prev_target_found = false;
            int prev_target_cls = -1;
            std::pair<int, int> prev_target_full = center;
            SystemClock::time_point prev_target_time{};
            {
                std::lock_guard<std::mutex> lock(shared_.mutex);
                toggles = shared_.toggles;
                prev_target_found = shared_.target_found;
                prev_target_cls = shared_.target_cls;
                prev_target_full = shared_.last_target_full;
                prev_target_time = shared_.target_time;
                shared_.tracking_strategy = trackingStrategyName(runtime.tracking_strategy);
            }

            const bool engage_active = (toggles.mode != 0)
                && isLeftHoldEngageSatisfied(
                    toggles.left_hold_engage,
                    runtime.left_hold_engage_button,
                    toggles.left_pressed,
                    toggles.right_pressed,
                    toggles.x1_pressed);
            const bool triggerbot_monitor_active = (toggles.mode != 0) && runtime.triggerbot_enable;
            const bool debug_overlay_observe_active = debug_overlay_enabled && !engage_active && !triggerbot_monitor_active;

            if (!(engage_active || triggerbot_monitor_active || debug_overlay_observe_active)) {
                tracker->reset();
                resetPidControllers();
                resetAimTrackingState(
                    lost_frames,
                    active_target_cls,
                    last_box_w,
                    last_box_h,
                    last_target_bbox,
                    last_pid_tick,
                    last_track_tick);
                target_guard_state.reset();
                target_lead_state.reset();
                command_slot_.clear();
                {
                    std::lock_guard<std::mutex> lock(shared_.mutex);
                    clearAimStateLocked(shared_, center, runtime.tracking_strategy);
                }
                if (debug_visuals_enabled && !preview_idle_state) {
                    publishDebugSnapshot(makeInactiveDebugPreviewSnapshot(
                        buildCaptureRegion(config_, center.first, center.second),
                        center));
                }
                preview_idle_state = true;
                std::this_thread::sleep_for(kInferenceIdleSleep);
                continue;
            }
            preview_idle_state = false;

            if (debug_overlay_observe_active) {
                tracker->reset();
                resetPidControllers();
                resetAimTrackingState(
                    lost_frames,
                    active_target_cls,
                    last_box_w,
                    last_box_h,
                    last_target_bbox,
                    last_pid_tick,
                    last_track_tick);
                target_guard_state.reset();
                target_lead_state.reset();
                command_slot_.clear();
                {
                    std::lock_guard<std::mutex> lock(shared_.mutex);
                    clearAimStateLocked(shared_, center, runtime.tracking_strategy);
                }
            }

            InferenceAppTimings app_timings{};
            std::optional<FramePacket> cpu_packet;
            std::optional<GpuFramePacket> gpu_packet;
            if (inference_->supportsGpuInput()) {
                std::pair<int, int> focus = center;
                {
                    std::lock_guard<std::mutex> lock(shared_.mutex);
                    focus = selectCaptureFocus(
                        runtime.capture_freeze_to_center_enable,
                        shared_.target_found,
                        center,
                        shared_.capture_focus_full);
                }
                const CaptureRegion region = buildCaptureRegion(config_, focus.first, focus.second);
                const auto grab_start = SteadyClock::now();
                gpu_packet = capture_->grabGpu(region);
                if (!gpu_packet.has_value()) {
                    cpu_packet = capture_->grab(region);
                }
                if (perf_) {
                    const std::optional<CaptureTimings> capture_timings = gpu_packet.has_value()
                        ? std::optional<CaptureTimings>(gpu_packet->timings)
                        : (cpu_packet.has_value() ? std::optional<CaptureTimings>(cpu_packet->timings) : std::nullopt);
                    recordCapturePerf(
                        *perf_,
                        secondsSince(grab_start, SteadyClock::now()),
                        !(gpu_packet.has_value() || cpu_packet.has_value()),
                        capture_timings);
                }
            } else {
                cpu_packet = frame_slot_.try_take();
            }
            if (!gpu_packet.has_value() && !cpu_packet.has_value()) {
                std::this_thread::sleep_for(kInferenceIdleSleep);
                continue;
            }

            const CaptureRegion capture_region = gpu_packet.has_value() ? gpu_packet->capture : cpu_packet->capture;
            const SteadyClock::time_point frame_ready = gpu_packet.has_value() ? gpu_packet->frame_ready : cpu_packet->frame_ready;
            const SteadyClock::time_point frame_time = frame_ready != SteadyClock::time_point{}
                ? frame_ready
                : (gpu_packet.has_value() ? gpu_packet->frame_time : cpu_packet->frame_time);
            const SteadyClock::time_point capture_done = gpu_packet.has_value() ? gpu_packet->capture_done : cpu_packet->capture_done;
            const SteadyClock::time_point acquire_started = gpu_packet.has_value() ? gpu_packet->acquire_started : cpu_packet->acquire_started;
            const double frame_age = secondsSince(frame_time, SteadyClock::now());
            if (frame_age > kMaxFrameAgeSeconds) {
                if (perf_) {
                    recordInferencePerf(*perf_, frame_age, secondsSince(loop_start, SteadyClock::now()), true, false, app_timings, {}, std::nullopt);
                }
                continue;
            }

            const int requested_target_cls = aimModeTargetClass(runtime.aim_mode);

            InferenceResult inference_result{};
            if (gpu_packet.has_value()) {
                inference_result = inference_->predictGpu(*gpu_packet, requested_target_cls);
            } else if (cpu_packet.has_value()) {
                inference_result = inference_->predict(*cpu_packet, requested_target_cls);
            }

            std::vector<Detection> detections = inference_result.detections;
            for (auto& detection : detections) {
                detection.bbox[0] += capture_region.left;
                detection.bbox[1] += capture_region.top;
                detection.bbox[2] += capture_region.left;
                detection.bbox[3] += capture_region.top;
                detection = scaleDetectionBox(detection, runtime.detection_box_scale, capture_region);
                const auto [aim_x, aim_y] = detectionAimPoint(detection, runtime.body_y_ratio, runtime.head_y_ratio);
                detection.x = aim_x;
                detection.y = aim_y;
            }
            detections.erase(
                std::remove_if(
                    detections.begin(),
                    detections.end(),
                    [&](const Detection& detection) {
                        return detection.conf < runtime.detection_min_conf;
                    }),
                detections.end());
            const AimCandidatePool aim_pool = buildAimCandidatePool(
                detections,
                runtime.aim_mode,
                runtime.body_y_ratio,
                runtime.head_y_ratio);

            const auto select_start = SteadyClock::now();
            std::optional<std::pair<float, float>> locked_point;
            bool locked_point_from_guard = false;
            if (tracking_enabled && tracker->initialized()) {
                const TrackerState tracker_state = tracker->state();
                locked_point = std::make_pair(tracker_state.x, tracker_state.y);
            } else if (tracking_enabled
                && prev_target_found
                && canReusePreviousTarget(runtime.aim_mode, aim_pool.using_head_candidates, prev_target_cls)
                && prev_target_time != SystemClock::time_point{}
                && secondsSince(prev_target_time, SystemClock::now()) <= kTargetTimeoutSeconds) {
                locked_point = std::make_pair(static_cast<float>(prev_target_full.first), static_cast<float>(prev_target_full.second));
            } else if (target_guard_config.enable
                && target_guard_state.active.active
                && target_guard_state.active.last_accepted_point.has_value()) {
                locked_point = target_guard_state.active.last_accepted_point;
                locked_point_from_guard = true;
            }

            std::pair<float, float> assoc_ref = locked_point.value_or(std::make_pair(
                static_cast<float>(center.first),
                static_cast<float>(center.second)));
            float assoc_limit = kTargetLockMaxJump;
            if (locked_point.has_value() && tracker->initialized()) {
                const TrackerState assoc_state = tracker->state();
                assoc_ref = {
                    assoc_state.x + (assoc_state.vx * kAssocPredictDt),
                    assoc_state.y + (assoc_state.vy * kAssocPredictDt),
                };
                assoc_limit += std::min(
                    kAssocMaxJumpPad,
                    std::sqrt((assoc_state.vx * assoc_state.vx) + (assoc_state.vy * assoc_state.vy)) * kAssocSpeedJumpGain);
            }

            const auto pickFromPool = [&](const AimCandidatePool& pool) -> std::optional<StickyTargetPick> {
                if (!locked_point.has_value()) {
                    return pickStickyTarget(
                        pool.candidates,
                        center.first,
                        center.second,
                        std::nullopt,
                        runtime.sticky_bias_px);
                }
                return pickAssociatedStickyTarget(
                    pool.candidates,
                    center.first,
                    center.second,
                    assoc_ref,
                    assoc_limit,
                    locked_point,
                    last_target_bbox,
                    runtime.sticky_bias_px);
            };

            std::optional<CaptureRegion> guard_region;
            std::optional<StickyTargetPick> sticky;
            TargetGuardMissResult target_guard_miss = TargetGuardMissResult::Inactive;
            const bool guard_active = target_guard_config.enable && target_guard_state.active.active;
            if (guard_active) {
                guard_region = buildTargetGuardRegion(
                    target_guard_state,
                    target_guard_config,
                    capture_region,
                    tracker->initialized() ? std::optional<std::pair<float, float>>(assoc_ref) : std::nullopt);
                const std::vector<Detection> guarded_detections = filterDetectionsInTargetGuard(detections, guard_region);
                const AimCandidatePool guarded_pool = buildAimCandidatePool(
                    guarded_detections,
                    runtime.aim_mode,
                    runtime.body_y_ratio,
                    runtime.head_y_ratio);
                sticky = pickFromPool(guarded_pool);
                if (!sticky.has_value() || !sticky->detection.has_value()) {
                    target_guard_miss = noteTargetGuardMiss(target_guard_state, target_guard_config);
                    if (target_guard_miss == TargetGuardMissResult::Expired) {
                        if (locked_point_from_guard) {
                            locked_point.reset();
                            assoc_ref = std::make_pair(static_cast<float>(center.first), static_cast<float>(center.second));
                            assoc_limit = kTargetLockMaxJump;
                        }
                        guard_region.reset();
                        sticky = pickFromPool(aim_pool);
                    }
                }
            } else {
                sticky = pickFromPool(aim_pool);
            }

            if (sticky.has_value() && sticky->detection.has_value()) {
                noteTargetGuardSelection(target_guard_state, target_guard_config, *sticky->detection);
            } else if (!guard_active || target_guard_miss == TargetGuardMissResult::Inactive) {
                (void)noteTargetGuardMiss(target_guard_state, target_guard_config);
            }
            if (target_guard_config.enable && target_guard_state.active.active) {
                const std::optional<std::pair<float, float>> guard_center = (sticky.has_value() && sticky->detection.has_value())
                    ? std::optional<std::pair<float, float>>(std::make_pair(sticky->detection->x, sticky->detection->y))
                    : (tracker->initialized() ? std::optional<std::pair<float, float>>(assoc_ref) : std::nullopt);
                guard_region = buildTargetGuardRegion(
                    target_guard_state,
                    target_guard_config,
                    capture_region,
                    guard_center);
            } else {
                guard_region.reset();
            }
            const SteadyClock::time_point lead_sample_time = frame_ready != SteadyClock::time_point{}
                ? frame_ready
                : frame_time;
            const bool selected_detection = sticky.has_value() && sticky->detection.has_value();
            const bool target_switched = selected_detection
                && (sticky->switched
                    || (last_target_bbox.has_value() && bboxIou(sticky->detection->bbox, *last_target_bbox) < 0.05F));
            if (!debug_overlay_observe_active && target_lead_config.enable) {
                if (target_switched) {
                    target_lead_state.reset();
                }
                if (selected_detection) {
                    noteTargetLeadSelection(target_lead_state, target_lead_config, *sticky->detection, lead_sample_time);
                } else {
                    noteTargetLeadMiss(target_lead_state, target_lead_config);
                }
            }
            app_timings.select_s = secondsSince(select_start, SteadyClock::now());

            if (debug_overlay_observe_active) {
                if (debug_overlay_ && debug_overlay_enabled) {
                    const auto preview_start = SteadyClock::now();
                    DebugPreviewSnapshot overlay_snapshot = makeDebugPreviewSnapshot(
                        capture_region,
                        center,
                        detections,
                        sticky.has_value() ? sticky->detection : std::nullopt);
                    overlay_snapshot.guard_region = guard_region;
                    if (sticky.has_value() && sticky->detection.has_value()) {
                        overlay_snapshot.target_found = true;
                        overlay_snapshot.target_cls = sticky->detection->cls;
                    }
                    debug_overlay_->publish(std::move(overlay_snapshot));
                    app_timings.preview_s = secondsSince(preview_start, SteadyClock::now());
                }
                if (perf_) {
                    recordInferencePerf(
                        *perf_,
                        frame_age,
                        secondsSince(loop_start, SteadyClock::now()),
                        false,
                        false,
                        app_timings,
                        inference_result.timings,
                        std::nullopt);
                }
                continue;
            }

            const auto tracker_start = SteadyClock::now();
            const SteadyClock::time_point now_tick = SteadyClock::now();
            const float dt = clamp(
                last_track_tick == SteadyClock::time_point{}
                    ? kMinTrackDt
                    : static_cast<float>(secondsSince(last_track_tick, now_tick)),
                kMinTrackDt,
                kMaxTrackDt);
            last_track_tick = now_tick;
            tracker->predict(dt);

            bool trigger_fire = false;
            if (sticky.has_value() && sticky->detection.has_value()) {
                if (target_switched) {
                    resetPidControllers(now_tick);
                }
                if (!tracking_enabled || target_switched) {
                    tracker->reset();
                }
                tracker->update(sticky->detection->x, sticky->detection->y);
                active_target_cls = sticky->detection->cls;
                lost_frames = 0;
                last_box_w = static_cast<float>(std::max(1, sticky->detection->bbox[2] - sticky->detection->bbox[0]));
                last_box_h = static_cast<float>(std::max(1, sticky->detection->bbox[3] - sticky->detection->bbox[1]));
                last_target_bbox = sticky->detection->bbox;
                if (target_switched
                    && runtime.tracking_strategy != TrackingStrategy::LegacyPid
                    && runtime.ego_motion_reset_on_switch) {
                    std::lock_guard<std::mutex> lock(shared_.mutex);
                    resetEgoMotionStateLocked(shared_);
                }
                if (triggerbot_monitor_active) {
                    trigger_fire = isTriggerbotArmed(*sticky->detection, center.first, center.second, triggerbot_config);
                }
                app_timings.tracker_update_s = secondsSince(tracker_start, SteadyClock::now());
            } else if (!tracking_enabled || !tracker->initialized()) {
                app_timings.tracker_update_s = secondsSince(tracker_start, SteadyClock::now());
                target_lead_state.reset();
                command_slot_.clear();
                {
                    std::lock_guard<std::mutex> lock(shared_.mutex);
                    clearAimStateLocked(shared_, center, runtime.tracking_strategy);
                }
                if (debug_visuals_enabled) {
                    const auto preview_start = SteadyClock::now();
                    DebugPreviewSnapshot preview_snapshot = makeDebugPreviewSnapshot(capture_region, center, detections);
                    preview_snapshot.guard_region = guard_region;
                    if (locked_point.has_value()) {
                        preview_snapshot.locked_point = *locked_point;
                    }
                    publishDebugSnapshot(std::move(preview_snapshot));
                    app_timings.preview_s = secondsSince(preview_start, SteadyClock::now());
                }
                if (perf_) {
                    recordInferencePerf(
                        *perf_,
                        frame_age,
                        secondsSince(loop_start, SteadyClock::now()),
                        false,
                        false,
                        app_timings,
                        inference_result.timings,
                        std::nullopt);
                }
                continue;
            } else {
                ++lost_frames;
                if (lost_frames > runtime.target_max_lost_frames) {
                    app_timings.tracker_update_s = secondsSince(tracker_start, SteadyClock::now());
                    tracker->reset();
                    resetPidControllers();
                    resetAimTrackingState(
                        lost_frames,
                        active_target_cls,
                        last_box_w,
                        last_box_h,
                        last_target_bbox,
                        last_pid_tick,
                        last_track_tick);
                    target_guard_state.reset();
                    target_lead_state.reset();
                    command_slot_.clear();
                    {
                        std::lock_guard<std::mutex> lock(shared_.mutex);
                        clearAimStateLocked(shared_, center, runtime.tracking_strategy);
                    }
                    if (debug_visuals_enabled) {
                        const auto preview_start = SteadyClock::now();
                        DebugPreviewSnapshot preview_snapshot = makeDebugPreviewSnapshot(capture_region, center, detections);
                        preview_snapshot.guard_region = guard_region;
                        if (locked_point.has_value()) {
                            preview_snapshot.locked_point = *locked_point;
                        }
                        publishDebugSnapshot(std::move(preview_snapshot));
                        app_timings.preview_s = secondsSince(preview_start, SteadyClock::now());
                    }
                    if (perf_) {
                        recordInferencePerf(
                            *perf_,
                            frame_age,
                            secondsSince(loop_start, SteadyClock::now()),
                            false,
                            false,
                            app_timings,
                            inference_result.timings,
                            std::nullopt);
                    }
                    continue;
                }
                app_timings.tracker_update_s = secondsSince(tracker_start, SteadyClock::now());
            }

            const auto aim_predict_start = SteadyClock::now();
            const TrackerState tracker_state = tracker->state();
            const bool use_legacy_pid = runtime.tracking_strategy == TrackingStrategy::LegacyPid;
            float predicted_x = tracker_state.x;
            float aim_y = tracker_state.y;
            float speed = 0.0F;
            float pid_error_metric_px = 0.0F;
            float pid_threshold_px = 0.0F;
            bool pid_settled = false;
            float desired_x = predicted_x - static_cast<float>(center.first);
            float desired_y = aim_y - static_cast<float>(center.second);

            float ff_scale = 0.0F;
            float vx = 0.0F;
            float vy = 0.0F;
            std::optional<TargetLeadPrediction> target_lead_prediction;
            const bool target_lead_locked = target_lead_config.enable && target_lead_state.active.active;
            const bool use_predictive_pid = runtime.tracking_strategy == TrackingStrategy::PredictivePid;
            const bool use_modern_pid = !use_legacy_pid && !use_predictive_pid;
            if (use_modern_pid) {
                ff_scale = tracker->feedforwardScale();
            }
            if (target_lead_locked) {
                float extra_velocity_x = 0.0F;
                float extra_velocity_y = 0.0F;
                if (use_modern_pid) {
                    const float ctrl_sent_vx_ema = shared_.ctrl_sent_vx_ema.load(std::memory_order_relaxed);
                    const float ctrl_sent_vy_ema = shared_.ctrl_sent_vy_ema.load(std::memory_order_relaxed);
                    const std::pair<float, float> gate_point = target_lead_state.active.last_detected_point.value_or(
                        std::make_pair(tracker_state.x, tracker_state.y));
                    float ego_gate_x = 1.0F;
                    float ego_gate_y = 1.0F;
                    if (runtime.ego_motion_error_gate_enable) {
                        if (runtime.ego_motion_error_gate_normalize_by_box && runtime.ego_motion_error_gate_norm_threshold > 1e-6F) {
                            const float norm_box_w = std::max(1.0F, last_box_w);
                            const float norm_box_h = std::max(1.0F, last_box_h);
                            const float normalized_error_x = std::abs(gate_point.first - static_cast<float>(center.first)) / norm_box_w;
                            const float normalized_error_y = std::abs(gate_point.second - static_cast<float>(center.second)) / norm_box_h;
                            ego_gate_x = clamp(
                                1.0F - (normalized_error_x / runtime.ego_motion_error_gate_norm_threshold),
                                0.0F,
                                1.0F);
                            ego_gate_y = clamp(
                                1.0F - (normalized_error_y / runtime.ego_motion_error_gate_norm_threshold),
                                0.0F,
                                1.0F);
                        } else if (runtime.ego_motion_error_gate_px > 1e-6F) {
                            ego_gate_x = clamp(
                                1.0F - (std::abs(gate_point.first - static_cast<float>(center.first)) / runtime.ego_motion_error_gate_px),
                                0.0F,
                                1.0F);
                            ego_gate_y = clamp(
                                1.0F - (std::abs(gate_point.second - static_cast<float>(center.second)) / runtime.ego_motion_error_gate_px),
                                0.0F,
                                1.0F);
                        }
                    }
                    if (runtime.ego_motion_comp_enable && ff_scale > 0.0F) {
                        extra_velocity_x = clamp(
                            ctrl_sent_vx_ema * runtime.ego_motion_comp_gain_x * ff_scale * ego_gate_x,
                            -kEgoMotionCompMaxPxS,
                            kEgoMotionCompMaxPxS);
                        extra_velocity_y = clamp(
                            ctrl_sent_vy_ema * runtime.ego_motion_comp_gain_y * ff_scale * ego_gate_y,
                            -kEgoMotionCompMaxPxS,
                            kEgoMotionCompMaxPxS);
                    }
                }
                target_lead_prediction = predictTargetLead(
                    target_lead_state,
                    target_lead_config,
                    now_tick,
                    target_lead_config.auto_latency_enable
                        ? shared_.cmd_send_latency_ema_s.load(std::memory_order_relaxed)
                        : 0.0F,
                    std::max(0.0F, runtime.prediction_time),
                    extra_velocity_x,
                    extra_velocity_y,
                    config_.screen_w,
                    config_.screen_h);
                if (target_lead_prediction.has_value()) {
                    predicted_x = target_lead_prediction->predicted_point.first;
                    aim_y = target_lead_prediction->predicted_point.second;
                    vx = target_lead_prediction->velocity_x;
                    vy = target_lead_prediction->velocity_y;
                    speed = std::sqrt((vx * vx) + (vy * vy));
                    desired_x = predicted_x - static_cast<float>(center.first);
                    desired_y = aim_y - static_cast<float>(center.second);
                }
            }
            if (use_modern_pid && !target_lead_prediction.has_value()) {
                ff_scale = tracker->feedforwardScale();
                const float ctrl_sent_vx_ema = shared_.ctrl_sent_vx_ema.load(std::memory_order_relaxed);
                const float ctrl_sent_vy_ema = shared_.ctrl_sent_vy_ema.load(std::memory_order_relaxed);
                vx = tracker_state.vx;
                vy = tracker_state.vy;
                const float prediction_time = target_lead_config.enable ? 0.0F : std::max(0.0F, runtime.prediction_time);
                const float base_predicted_x = tracker_state.x + (vx * prediction_time);
                const float base_predicted_y = tracker_state.y + (vy * prediction_time);
                const float base_aim_y = base_predicted_y;
                float ego_gate_x = 1.0F;
                float ego_gate_y = 1.0F;
                if (runtime.ego_motion_error_gate_enable) {
                    if (runtime.ego_motion_error_gate_normalize_by_box && runtime.ego_motion_error_gate_norm_threshold > 1e-6F) {
                        const float norm_box_w = std::max(1.0F, last_box_w);
                        const float norm_box_h = std::max(1.0F, last_box_h);
                        const float normalized_error_x = std::abs(base_predicted_x - static_cast<float>(center.first)) / norm_box_w;
                        const float normalized_error_y = std::abs(base_aim_y - static_cast<float>(center.second)) / norm_box_h;
                        ego_gate_x = clamp(
                            1.0F - (normalized_error_x / runtime.ego_motion_error_gate_norm_threshold),
                            0.0F,
                            1.0F);
                        ego_gate_y = clamp(
                            1.0F - (normalized_error_y / runtime.ego_motion_error_gate_norm_threshold),
                            0.0F,
                            1.0F);
                    } else if (runtime.ego_motion_error_gate_px > 1e-6F) {
                        ego_gate_x = clamp(
                            1.0F - (std::abs(base_predicted_x - static_cast<float>(center.first)) / runtime.ego_motion_error_gate_px),
                            0.0F,
                            1.0F);
                        ego_gate_y = clamp(
                            1.0F - (std::abs(base_aim_y - static_cast<float>(center.second)) / runtime.ego_motion_error_gate_px),
                            0.0F,
                            1.0F);
                    }
                }
                if (runtime.ego_motion_comp_enable && ff_scale > 0.0F) {
                    const float ego_vx = clamp(
                        ctrl_sent_vx_ema * runtime.ego_motion_comp_gain_x * ff_scale * ego_gate_x,
                        -kEgoMotionCompMaxPxS,
                        kEgoMotionCompMaxPxS);
                    const float ego_vy = clamp(
                        ctrl_sent_vy_ema * runtime.ego_motion_comp_gain_y * ff_scale * ego_gate_y,
                        -kEgoMotionCompMaxPxS,
                        kEgoMotionCompMaxPxS);
                    vx += ego_vx;
                    vy += ego_vy;
                }
                speed = std::sqrt((vx * vx) + (vy * vy));
                if (speed > kMaxTrackSpeedPxS && speed > 1e-6F) {
                    const float speed_scale = kMaxTrackSpeedPxS / speed;
                    vx *= speed_scale;
                    vy *= speed_scale;
                    speed = kMaxTrackSpeedPxS;
                }
                predicted_x = tracker_state.x + (vx * prediction_time);
                aim_y = tracker_state.y + (vy * prediction_time);
                desired_x = predicted_x - static_cast<float>(center.first);
                desired_y = aim_y - static_cast<float>(center.second);
            }
            app_timings.aim_predict_s = secondsSince(aim_predict_start, SteadyClock::now());

            const auto aim_pid_start = SteadyClock::now();
            const float pid_dt = clamp(
                last_pid_tick == SteadyClock::time_point{}
                    ? kMinTrackDt
                    : static_cast<float>(secondsSince(last_pid_tick, now_tick)),
                kMinTrackDt,
                kMaxTrackDt);
            last_pid_tick = now_tick;
            int dx = 0;
            int dy = 0;
            if (use_legacy_pid) {
                const LegacyPidConfig legacy_pid_config = buildLegacyPidConfig(runtime);
                const LegacyPidAxisResult legacy_x = updateLegacyPidAxis(
                    legacy_pid_x,
                    legacy_pid_config,
                    predicted_x - static_cast<float>(center.first),
                    pid_dt,
                    last_box_w,
                    static_cast<float>(std::max(1, capture_region.width)));
                const LegacyPidAxisResult legacy_y = updateLegacyPidAxis(
                    legacy_pid_y,
                    legacy_pid_config,
                    aim_y - static_cast<float>(center.second),
                    pid_dt,
                    last_box_w,
                    static_cast<float>(std::max(1, capture_region.width)));
                const LegacyPidStatus legacy_status = makeLegacyPidStatus(legacy_x, legacy_y);
                desired_x = runtime.pid_enable ? legacy_x.output : legacy_x.error_px;
                desired_y = runtime.pid_enable ? legacy_y.output : legacy_y.error_px;
                speed = legacy_status.speed;
                pid_settled = legacy_status.settled;
                pid_error_metric_px = legacy_status.error_metric_px;
                pid_threshold_px = legacy_status.threshold_px;
            } else if (use_predictive_pid) {
                const float raw_error_x = predicted_x - static_cast<float>(center.first);
                const float raw_error_y = aim_y - static_cast<float>(center.second);
                if (runtime.pid_enable) {
                    const PredictivePidResult predictive = predictive_pid.update(raw_error_x, raw_error_y, pid_dt);
                    desired_x = predictive.output_x;
                    desired_y = predictive.output_y;
                    speed = std::sqrt(
                        (predictive.velocity_x * predictive.velocity_x)
                        + (predictive.velocity_y * predictive.velocity_y));
                    pid_error_metric_px = std::max(
                        std::abs(predictive.fused_error_x),
                        std::abs(predictive.fused_error_y));
                    pid_threshold_px = runtime.predictive_pid_output_limit;
                } else {
                    desired_x = raw_error_x;
                    desired_y = raw_error_y;
                    speed = 0.0F;
                    pid_error_metric_px = std::max(std::abs(raw_error_x), std::abs(raw_error_y));
                    pid_threshold_px = 0.0F;
                }
                pid_settled = true;
            } else {
                const PIDSettleDecision pid_settle = updatePidSettleState(
                    pid_settle_state,
                    buildPidSettleConfig(runtime),
                    pidSettleErrorMetricPx(
                        predicted_x,
                        aim_y,
                        static_cast<float>(center.first),
                        static_cast<float>(center.second)),
                    last_box_w,
                    static_cast<float>(std::max(1, capture_region.width)));
                if (runtime.pid_enable && pid_settle.just_unsettled) {
                    pid_x.clearIntegral();
                    pid_y.clearIntegral();
                }

                const bool pid_integrate = runtime.pid_enable && std::abs(runtime.ki) > 1e-12F && pid_settle.integrate;
                const float pid_term_x = runtime.pid_enable
                    ? (pid_x.update(0.0F, static_cast<float>(center.first) - predicted_x, pid_dt, pid_integrate) * pid_settle.pid_output_scale)
                    : (predicted_x - static_cast<float>(center.first));
                const float pid_term_y = runtime.pid_enable
                    ? (pid_y.update(0.0F, static_cast<float>(center.second) - aim_y, pid_dt, pid_integrate) * pid_settle.pid_output_scale)
                    : (aim_y - static_cast<float>(center.second));
                const float ff_x = (vx * dt) * ff_scale;
                const float ff_y = (vy * dt) * ff_scale;
                desired_x = pid_term_x + ff_x;
                desired_y = pid_term_y + ff_y;
                pid_settled = pid_settle.settled;
                pid_error_metric_px = pid_settle.error_metric_px;
                pid_threshold_px = pid_settle.dynamic_threshold_px;
            }

            dx = engage_active
                ? static_cast<int>(std::lround(clamp(
                    desired_x,
                    -static_cast<float>(runtime.raw_max_step_x),
                    static_cast<float>(runtime.raw_max_step_x))))
                : 0;
            dy = engage_active
                ? static_cast<int>(std::lround(clamp(desired_y, -static_cast<float>(runtime.raw_max_step_y), static_cast<float>(runtime.raw_max_step_y))))
                : 0;
            if (use_predictive_pid && runtime.pid_enable) {
                predictive_pid.commitOutput(static_cast<float>(dx), static_cast<float>(dy));
            }
            app_timings.aim_pid_s = secondsSince(aim_pid_start, SteadyClock::now());

            const auto aim_sync_start = SteadyClock::now();
            const int focus_x = clamp(static_cast<int>(std::lround(tracker_state.x)), 0, config_.screen_w - 1);
            const int focus_y = clamp(static_cast<int>(std::lround(tracker_state.y)), 0, config_.screen_h - 1);
            const int default_target_cls = runtime.aim_mode == AimMode::Hybrid
                ? (aim_pool.using_head_candidates ? 1 : 0)
                : aimModeTargetClass(runtime.aim_mode);
            const int selected_cls = active_target_cls != -1 ? active_target_cls : default_target_cls;
            const auto now_system = SystemClock::now();

            {
                std::lock_guard<std::mutex> lock(shared_.mutex);
                shared_.target_found = true;
                shared_.target_cls = selected_cls;
                shared_.target_speed = speed;
                shared_.pid_settled = pid_settled;
                shared_.pid_settle_error_metric_px = pid_error_metric_px;
                shared_.pid_settle_threshold_px = pid_threshold_px;
                shared_.lead_active = target_lead_prediction.has_value() && target_lead_prediction->active;
                shared_.lead_time_ms = target_lead_prediction.has_value() ? (target_lead_prediction->lead_time_s * 1000.0F) : 0.0F;
                shared_.aim_dx = dx;
                shared_.aim_dy = dy;
                shared_.last_target_full = {focus_x, focus_y};
                shared_.capture_focus_full = {focus_x, focus_y};
                shared_.target_time = now_system;
                shared_.tracking_strategy = trackingStrategyName(runtime.tracking_strategy);
            }
            app_timings.aim_sync_s = secondsSince(aim_sync_start, SteadyClock::now());

            if (debug_visuals_enabled) {
                const auto preview_start = SteadyClock::now();
                DebugPreviewSnapshot preview_snapshot = makeDebugPreviewSnapshot(
                    capture_region,
                    center,
                    detections,
                    sticky.has_value() ? sticky->detection : std::nullopt);
                preview_snapshot.guard_region = guard_region;
                if (locked_point.has_value()) {
                    preview_snapshot.locked_point = *locked_point;
                }
                preview_snapshot.predicted_point = std::make_pair(predicted_x, aim_y);
                if (target_lead_prediction.has_value()) {
                    preview_snapshot.detected_point = target_lead_prediction->detected_point;
                    preview_snapshot.lead_time_s = target_lead_prediction->lead_time_s;
                    preview_snapshot.lead_active = target_lead_prediction->active;
                    preview_snapshot.detected_point_stale = target_lead_prediction->detected_point_stale;
                }
                preview_snapshot.target_found = true;
                preview_snapshot.target_cls = selected_cls;
                preview_snapshot.target_speed = speed;
                publishDebugSnapshot(std::move(preview_snapshot));
                app_timings.preview_s = secondsSince(preview_start, SteadyClock::now());
            }

            const auto queue_start = SteadyClock::now();
            if (engage_active || trigger_fire) {
                command_slot_.put(CommandPacket{
                    .dx = dx,
                    .dy = dy,
                    .acquire_started = acquire_started,
                    .frame_ready = frame_ready,
                    .capture_done = capture_done,
                    .cmd_generated = queue_start,
                    .generated_at = now_system,
                    .frame_time = gpu_packet.has_value() ? gpu_packet->capture_time : cpu_packet->capture_time,
                    .capture_time = gpu_packet.has_value() ? gpu_packet->capture_time : cpu_packet->capture_time,
                    .target_detected = true,
                    .synthetic_recoil = false,
                    .trigger_fire = trigger_fire,
                });
                app_timings.queue_s = secondsSince(queue_start, SteadyClock::now());
            }
            if (perf_) {
                const SteadyClock::time_point pipe_origin = capture_done != SteadyClock::time_point{} ? capture_done : frame_time;
                const SteadyClock::time_point pipe_end = SteadyClock::now();
                recordInferencePerf(
                    *perf_,
                    frame_age,
                    secondsSince(loop_start, SteadyClock::now()),
                    false,
                    true,
                    app_timings,
                    inference_result.timings,
                    pipe_origin == SteadyClock::time_point{} ? std::nullopt : std::optional<double>(secondsSince(pipe_origin, pipe_end)));
            }
        }
    } catch (const std::exception& ex) {
        std::cerr << "[inference] loop failed: " << ex.what() << "\n";
        stop();
    }

    if (capture_ && inference_ && inference_->supportsGpuInput()) {
        capture_->close();
    }
}

void DeltaApp::recoilLoop() {
    if (!recoil_scheduler_) {
        return;
    }

    for (;;) {
        RuntimeConfig runtime = runtime_store_.snapshot();
        bool running = true;
        bool recoil_enabled = false;
        bool left_pressed = false;
        bool x1_pressed = false;
        bool mode_active = false;
        bool hold_engage_toggle = false;
        {
            std::lock_guard<std::mutex> lock(shared_.mutex);
            running = shared_.running;
            recoil_enabled = shared_.toggles.recoil_tune_fallback;
            left_pressed = shared_.toggles.left_pressed;
            x1_pressed = shared_.toggles.x1_pressed;
            mode_active = shared_.toggles.mode != 0;
            hold_engage_toggle = shared_.toggles.left_hold_engage;
        }
        if (!running) {
            break;
        }

        const RecoilSchedulerUpdate update = recoil_scheduler_->tick(runtime, recoil_enabled, left_pressed, x1_pressed);
        {
            std::lock_guard<std::mutex> lock(shared_.mutex);
            if (!shared_.running) {
                break;
            }
            const int last_applied_dx = shared_.recoil.last_applied_dx;
            const int last_applied_dy = shared_.recoil.last_applied_dy;
            const int last_applied_shot_index = shared_.recoil.last_applied_shot_index;
            const std::uint64_t apply_count = shared_.recoil.apply_count;
            shared_.recoil = update.state;
            shared_.recoil.mode_active = mode_active;
            shared_.recoil.hold_engage_toggle = hold_engage_toggle;
            shared_.recoil.last_applied_dx = last_applied_dx;
            shared_.recoil.last_applied_dy = last_applied_dy;
            shared_.recoil.last_applied_shot_index = last_applied_shot_index;
            shared_.recoil.apply_count = apply_count;
            if (update.clear_pending) {
                shared_.pending_recoil = {};
            }
            shared_.pending_recoil.dx += update.delta.dx;
            shared_.pending_recoil.dy += update.delta.dy;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void DeltaApp::controlLoop() {
    if (!input_sender_) {
        return;
    }

    Win32HotkeySource hotkeys;
    InputSnapshot previous{};
    SteadyClock::time_point last_mode_toggle{};
    SteadyClock::time_point last_aim_mode_toggle{};
    SteadyClock::time_point last_hold_toggle{};
    SteadyClock::time_point last_recoil_toggle{};
    SteadyClock::time_point last_triggerbot_toggle{};
    SystemClock::time_point last_trigger_click{};
    SteadyClock::time_point last_recoil_integrate_tick{};
    double recoil_carry_y = 0.0;
    MouseSenderConfig last_sender_config{};
    bool sender_config_initialized = false;
    const auto center = screenCenter(config_);

    for (;;) {
        {
            std::lock_guard<std::mutex> lock(shared_.mutex);
            if (!shared_.running) {
                break;
            }
        }

        const auto steady_now = SteadyClock::now();
        const auto system_now = SystemClock::now();
        const InputSnapshot snapshot = hotkeys.poll();
        RuntimeConfig runtime = runtime_store_.snapshot();

        if (snapshot.insert_pressed) {
            stop();
            break;
        }

        bool triggerbot_toggle = false;
        bool aim_mode_changed = false;
        AimMode next_aim_mode = runtime.aim_mode;
        {
            std::lock_guard<std::mutex> lock(shared_.mutex);
            shared_.toggles.left_pressed = snapshot.left_pressed;
            shared_.toggles.right_pressed = snapshot.right_pressed;
            shared_.toggles.x1_pressed = snapshot.x1_pressed;

            if (risingEdge(snapshot.x2_pressed, previous.x2_pressed) && (steady_now - last_mode_toggle) >= kToggleCooldown) {
                shared_.toggles.mode = (shared_.toggles.mode + 1) % 2;
                last_mode_toggle = steady_now;
                const int mode = shared_.toggles.mode;
                if (mode == 0) {
                    command_slot_.clear();
                    clearAimStateLocked(shared_, center, runtime.tracking_strategy);
                }
                if (config_.debug_log) {
                    std::cout << "[control] Mode: " << mode << " (" << (mode == 1 ? "ACTIVE" : "OFF") << ")\n";
                }
                playToggleBeep(mode == 1 ? 1000 : 500);
            }
            if (risingEdge(snapshot.f4_pressed, previous.f4_pressed) && (steady_now - last_aim_mode_toggle) >= kToggleCooldown) {
                next_aim_mode = nextAimMode(runtime.aim_mode);
                last_aim_mode_toggle = steady_now;
                aim_mode_changed = true;
                command_slot_.clear();
                clearAimStateLocked(shared_, center, runtime.tracking_strategy);
            }
            if (risingEdge(snapshot.f6_pressed, previous.f6_pressed) && (steady_now - last_hold_toggle) >= kToggleCooldown) {
                shared_.toggles.left_hold_engage = !shared_.toggles.left_hold_engage;
                last_hold_toggle = steady_now;
                if (shared_.toggles.left_hold_engage && !isLeftHoldEngageSatisfied(
                        true,
                        runtime.left_hold_engage_button,
                        shared_.toggles.left_pressed,
                        shared_.toggles.right_pressed,
                        shared_.toggles.x1_pressed)) {
                    command_slot_.clear();
                    clearAimStateLocked(shared_, center, runtime.tracking_strategy);
                }
                if (config_.debug_log) {
                    std::cout << "[control] HoldEngage: " << (shared_.toggles.left_hold_engage ? "ON" : "OFF") << "\n";
                }
                playToggleBeep(shared_.toggles.left_hold_engage ? 1400 : 700);
            }
            if (risingEdge(snapshot.f7_pressed, previous.f7_pressed) && (steady_now - last_recoil_toggle) >= kToggleCooldown) {
                shared_.toggles.recoil_tune_fallback = !shared_.toggles.recoil_tune_fallback;
                last_recoil_toggle = steady_now;
                if (config_.debug_log) {
                    std::cout << "[control] RecoilTuneFallback: " << (shared_.toggles.recoil_tune_fallback ? "ON" : "OFF") << "\n";
                }
                playToggleBeep(shared_.toggles.recoil_tune_fallback ? 1500 : 800);
            }
            if (risingEdge(snapshot.f8_pressed, previous.f8_pressed) && (steady_now - last_triggerbot_toggle) >= kToggleCooldown) {
                last_triggerbot_toggle = steady_now;
                triggerbot_toggle = true;
            }
        }
        previous = snapshot;

        if (aim_mode_changed) {
            runtime.aim_mode = next_aim_mode;
            runtime_store_.update(runtime);
            if (config_.debug_log) {
                std::cout << "[control] AimMode: " << aimModeLabel(runtime.aim_mode)
                          << " (" << aimModeName(runtime.aim_mode) << ")\n";
            }
            playToggleBeep(aimModeBeepFrequency(runtime.aim_mode));
        }

        if (triggerbot_toggle) {
            runtime.triggerbot_enable = !runtime.triggerbot_enable;
            runtime_store_.update(runtime);
            if (config_.debug_log) {
                std::cout << "[control] TriggerBot: " << (runtime.triggerbot_enable ? "ON" : "OFF") << "\n";
            }
            playToggleBeep(runtime.triggerbot_enable ? 1600 : 900);
        }
        if (mouse_move_suppressor_) {
            mouse_move_suppressor_->setDebugLogging(runtime.mouse_move_suppress_on_fire_debug);
            mouse_move_suppressor_->setSuppressionActive(
                shouldSuppressMouseMoveOnFire(
                    runtime.mouse_move_suppress_on_fire_enable,
                    snapshot.left_pressed,
                    snapshot.x1_pressed));
            const MouseMoveSuppressionStatus status = mouse_move_suppressor_->snapshot();
            std::lock_guard<std::mutex> lock(shared_.mutex);
            shared_.mouse_move_suppress_supported = status.supported;
            shared_.mouse_move_suppress_active = status.active;
            shared_.mouse_move_suppress_count = status.suppressed_count;
        }
        const TriggerbotConfig triggerbot_config = buildTriggerbotConfig(runtime);

        const MouseSenderConfig sender_config{
            .gain_x = runtime.sendinput_gain_x,
            .gain_y = runtime.sendinput_gain_y,
            .max_step = runtime.sendinput_max_step,
        };
        if (!sender_config_initialized
            || sender_config.gain_x != last_sender_config.gain_x
            || sender_config.gain_y != last_sender_config.gain_y
            || sender_config.max_step != last_sender_config.max_step) {
            input_sender_->configure(sender_config);
            last_sender_config = sender_config;
            sender_config_initialized = true;
        }

        ToggleState toggles{};
        bool target_detected = false;
        PendingRecoilDelta pending_recoil{};
        RecoilRuntimeState recoil_state{};
        {
            std::lock_guard<std::mutex> lock(shared_.mutex);
            toggles = shared_.toggles;
            target_detected = shared_.target_found;
            pending_recoil = shared_.pending_recoil;
            recoil_state = shared_.recoil;
            shared_.pending_recoil = {};
        }
        const bool advanced_recoil_pending = pending_recoil.dx != 0 || pending_recoil.dy != 0;

        std::optional<CommandPacket> cmd = command_slot_.wait_take_for(kControlCommandWait);
        const double recoil_rate_y_px_s = std::abs(runtime.recoil_compensation_y_rate_px_s) > 1e-6F
            ? static_cast<double>(runtime.recoil_compensation_y_rate_px_s)
            : (static_cast<double>(runtime.recoil_compensation_y_px) * kLegacyRecoilReferenceHz);
        const bool legacy_recoil_enabled = runtime.recoil_mode == RecoilMode::Legacy && std::abs(recoil_rate_y_px_s) > 1e-6;
        const bool recoil_trigger_pressed = toggles.left_pressed || toggles.x1_pressed;
        if (!cmd.has_value()) {
            if (advanced_recoil_pending) {
                cmd = CommandPacket{
                    .dx = 0,
                    .dy = 0,
                    .cmd_generated = steady_now,
                    .generated_at = system_now,
                    .frame_time = system_now,
                    .capture_time = system_now,
                    .target_detected = false,
                    .synthetic_recoil = true,
                    .trigger_fire = false,
                };
            } else {
                const bool mode_ok = runtime.recoil_tune_fallback_ignore_mode_check || (toggles.mode != 0);
                const bool target_ok = runtime.recoil_tune_fallback_ignore_mode_check || !target_detected;
                const bool engage_ok = isLeftHoldEngageSatisfied(
                    toggles.left_hold_engage,
                    runtime.left_hold_engage_button,
                    toggles.left_pressed,
                    toggles.right_pressed,
                    toggles.x1_pressed);
                if (legacy_recoil_enabled && toggles.recoil_tune_fallback && target_ok && recoil_trigger_pressed && mode_ok && engage_ok) {
                    cmd = CommandPacket{
                        .dx = 0,
                        .dy = 0,
                        .cmd_generated = steady_now,
                        .generated_at = system_now,
                        .frame_time = system_now,
                        .capture_time = system_now,
                        .target_detected = target_ok,
                        .synthetic_recoil = true,
                        .trigger_fire = false,
                    };
                } else {
                    {
                        std::lock_guard<std::mutex> lock(shared_.mutex);
                        decayEgoMotionStateLocked(shared_);
                    }
                    std::this_thread::sleep_for(kControlIdleSleep);
                    continue;
                }
            }
        }

        if (cmd->cmd_generated != SteadyClock::time_point{} && secondsSince(cmd->cmd_generated, steady_now) > kCommandTimeoutSeconds) {
            {
                std::lock_guard<std::mutex> lock(shared_.mutex);
                decayEgoMotionStateLocked(shared_);
            }
            if (perf_) {
                recordControlPerf(
                    *perf_,
                    secondsSince(cmd->cmd_generated, steady_now),
                    false,
                    0.0,
                    true,
                    false,
                    cmd->frame_ready == SteadyClock::time_point{} ? std::nullopt : std::optional<double>(secondsSince(cmd->frame_ready, steady_now)),
                    cmd->acquire_started == SteadyClock::time_point{} ? std::nullopt : std::optional<double>(secondsSince(cmd->acquire_started, steady_now)),
                    std::nullopt,
                    std::nullopt);
            }
            continue;
        }

        const bool engage_active = cmd->synthetic_recoil
            || (toggles.mode != 0 && isLeftHoldEngageSatisfied(
                toggles.left_hold_engage,
                runtime.left_hold_engage_button,
                toggles.left_pressed,
                toggles.right_pressed,
                toggles.x1_pressed));
        const bool trigger_enabled = triggerbot_config.enable;
        const bool trigger_fire = trigger_enabled && cmd->trigger_fire;
        if (!(engage_active || trigger_fire)) {
            {
                std::lock_guard<std::mutex> lock(shared_.mutex);
                decayEgoMotionStateLocked(shared_);
            }
            if (perf_) {
                recordControlPerf(
                    *perf_,
                    cmd->cmd_generated == SteadyClock::time_point{} ? 0.0 : secondsSince(cmd->cmd_generated, steady_now),
                    false,
                    0.0,
                    false,
                    true,
                    cmd->frame_ready == SteadyClock::time_point{} ? std::nullopt : std::optional<double>(secondsSince(cmd->frame_ready, steady_now)),
                    cmd->acquire_started == SteadyClock::time_point{} ? std::nullopt : std::optional<double>(secondsSince(cmd->acquire_started, steady_now)),
                    std::nullopt,
                    std::nullopt);
            }
            continue;
        }

        int dx = cmd->dx;
        int dy = cmd->dy;
        const bool trigger_will_click = trigger_fire
            && !recoil_trigger_pressed
            && (last_trigger_click == SystemClock::time_point{}
                || secondsSince(last_trigger_click, system_now) >= triggerbot_config.click_cooldown_s);

        const bool recoil_active = legacy_recoil_enabled && cmd->target_detected && (recoil_trigger_pressed || trigger_will_click);
        if (recoil_active) {
            const auto recoil_tick = SteadyClock::now();
            if (last_recoil_integrate_tick != SteadyClock::time_point{}) {
                const double recoil_dt = clamp(
                    secondsSince(last_recoil_integrate_tick, recoil_tick),
                    0.0,
                    kMaxRecoilIntegrateDtSeconds);
                recoil_carry_y += recoil_rate_y_px_s * recoil_dt;
            }
            last_recoil_integrate_tick = recoil_tick;
            const int recoil_step = static_cast<int>(recoil_carry_y);
            recoil_carry_y -= static_cast<double>(recoil_step);
            dy = clamp(
                dy + recoil_step,
                -runtime.raw_max_step_y,
                runtime.raw_max_step_y);
        } else {
            recoil_carry_y = 0.0;
            last_recoil_integrate_tick = {};
        }
        const int base_dx_before_advanced = dx;
        const int base_dy_before_advanced = dy;
        if (advanced_recoil_pending) {
            dx += pending_recoil.dx;
            dy += pending_recoil.dy;
        }
        const int base_clamped_dx = clamp(base_dx_before_advanced, -runtime.raw_max_step_x, runtime.raw_max_step_x);
        const int base_clamped_dy = clamp(base_dy_before_advanced, -runtime.raw_max_step_y, runtime.raw_max_step_y);
        dx = clamp(dx, -runtime.raw_max_step_x, runtime.raw_max_step_x);
        dy = clamp(dy, -runtime.raw_max_step_y, runtime.raw_max_step_y);
        const int applied_advanced_dx = advanced_recoil_pending ? (dx - base_clamped_dx) : 0;
        const int applied_advanced_dy = advanced_recoil_pending ? (dy - base_clamped_dy) : 0;

        bool movement_sent = false;
        const auto send_start = SteadyClock::now();
        if (dx != 0 || dy != 0) {
            movement_sent = input_sender_->sendRelative(dx, dy);
        }
        bool trigger_sent = false;
        if (trigger_will_click) {
            trigger_sent = input_sender_->clickLeft(triggerbot_config.click_hold_s);
            if (trigger_sent) {
                last_trigger_click = SystemClock::now();
            }
        }
        const auto send_end_tick = SteadyClock::now();
        const double send_elapsed = secondsSince(send_start, send_end_tick);

        if (advanced_recoil_pending) {
            const bool recoil_applied = movement_sent && (applied_advanced_dx != 0 || applied_advanced_dy != 0);
            {
                std::lock_guard<std::mutex> lock(shared_.mutex);
                shared_.recoil.last_applied_dx = recoil_applied ? applied_advanced_dx : 0;
                shared_.recoil.last_applied_dy = recoil_applied ? applied_advanced_dy : 0;
                if (recoil_applied) {
                    shared_.recoil.last_applied_shot_index = recoil_state.shot_index;
                    ++shared_.recoil.apply_count;
                }
            }
            if (config_.debug_log) {
                std::cout << "[recoil] Advanced profile="
                          << (recoil_state.selected_profile_name.empty() ? recoil_state.selected_profile_id : recoil_state.selected_profile_name)
                          << " shot=" << recoil_state.shot_index << "/" << recoil_state.shot_count
                          << " scheduled=(" << pending_recoil.dx << ", " << pending_recoil.dy << ")"
                          << " applied=(" << (recoil_applied ? applied_advanced_dx : 0) << ", " << (recoil_applied ? applied_advanced_dy : 0) << ")"
                          << " send=" << (movement_sent ? "OK" : "SKIP")
                          << " F7=" << (toggles.recoil_tune_fallback ? "ON" : "OFF")
                          << " ignore_mode=" << (runtime.recoil_tune_fallback_ignore_mode_check ? "ON" : "OFF")
                          << " mode=" << (toggles.mode != 0 ? "ACTIVE" : "OFF")
                          << " F6=" << (toggles.left_hold_engage ? "ON" : "OFF")
                          << " trigger=" << (recoil_trigger_pressed ? "DOWN" : "UP")
                          << " left=" << (toggles.left_pressed ? "DOWN" : "UP")
                          << " x1=" << (toggles.x1_pressed ? "DOWN" : "UP")
                          << "\n";
            }
        }

        if (dx == 0 && dy == 0 && !trigger_sent) {
            {
                std::lock_guard<std::mutex> lock(shared_.mutex);
                decayEgoMotionStateLocked(shared_);
            }
            if (perf_) {
                recordControlPerf(
                    *perf_,
                    cmd->cmd_generated == SteadyClock::time_point{} ? 0.0 : secondsSince(cmd->cmd_generated, steady_now),
                    false,
                    0.0,
                    false,
                    false,
                    cmd->frame_ready == SteadyClock::time_point{} ? std::nullopt : std::optional<double>(secondsSince(cmd->frame_ready, steady_now)),
                    cmd->acquire_started == SteadyClock::time_point{} ? std::nullopt : std::optional<double>(secondsSince(cmd->acquire_started, steady_now)),
                    std::nullopt,
                    std::nullopt);
            }
            continue;
        }

        if ((dx != 0 || dy != 0) && movement_sent) {
            std::lock_guard<std::mutex> lock(shared_.mutex);
            if (cmd->cmd_generated != SteadyClock::time_point{}) {
                const float send_latency_s = static_cast<float>(secondsSince(cmd->cmd_generated, send_end_tick));
                const float next_latency_ema = emaUpdateSigned(
                    shared_.cmd_send_latency_ema_s.load(std::memory_order_relaxed),
                    std::max(0.0F, send_latency_s),
                    kCmdSendLatencyAlpha);
                shared_.cmd_send_latency_ema_s.store(next_latency_ema, std::memory_order_relaxed);
            }
            if (shared_.ctrl_last_send_tick != SteadyClock::time_point{}) {
                const float send_dt = std::max(1e-4F, static_cast<float>(secondsSince(shared_.ctrl_last_send_tick, send_end_tick)));
                const float sent_vx = clamp(static_cast<float>(dx) / send_dt, -kEgoMotionCompMaxPxS, kEgoMotionCompMaxPxS);
                const float sent_vy = clamp(static_cast<float>(dy) / send_dt, -kEgoMotionCompMaxPxS, kEgoMotionCompMaxPxS);
                const float next_vx_ema = emaUpdateSigned(
                    shared_.ctrl_sent_vx_ema.load(std::memory_order_relaxed),
                    sent_vx,
                    kEgoMotionCompAlpha);
                const float next_vy_ema = emaUpdateSigned(
                    shared_.ctrl_sent_vy_ema.load(std::memory_order_relaxed),
                    sent_vy,
                    kEgoMotionCompAlpha);
                shared_.ctrl_sent_vx_ema.store(next_vx_ema, std::memory_order_relaxed);
                shared_.ctrl_sent_vy_ema.store(next_vy_ema, std::memory_order_relaxed);
            }
            shared_.ctrl_last_send_tick = send_end_tick;
        } else if (!trigger_sent) {
            std::lock_guard<std::mutex> lock(shared_.mutex);
            decayEgoMotionStateLocked(shared_);
        }

        if (movement_sent || trigger_sent) {
            std::lock_guard<std::mutex> lock(shared_.mutex);
            shared_.aim_dx = dx;
            shared_.aim_dy = dy;
        }
        if (perf_) {
            const bool sent_ok = movement_sent || trigger_sent;
            recordControlPerf(
                *perf_,
                cmd->cmd_generated == SteadyClock::time_point{} ? 0.0 : secondsSince(cmd->cmd_generated, steady_now),
                sent_ok,
                sent_ok ? send_elapsed : 0.0,
                false,
                false,
                cmd->frame_ready == SteadyClock::time_point{} ? std::nullopt : std::optional<double>(secondsSince(cmd->frame_ready, steady_now)),
                cmd->acquire_started == SteadyClock::time_point{} ? std::nullopt : std::optional<double>(secondsSince(cmd->acquire_started, steady_now)),
                sent_ok && cmd->frame_ready != SteadyClock::time_point{} ? std::optional<double>(secondsSince(cmd->frame_ready, send_end_tick)) : std::nullopt,
                sent_ok && cmd->acquire_started != SteadyClock::time_point{} ? std::optional<double>(secondsSince(cmd->acquire_started, send_end_tick)) : std::nullopt);
        }
    }
}

void DeltaApp::sideButtonKeySequenceLoop() {
    Win32HotkeySource hotkeys;
    InputSnapshot previous{};
    SteadyClock::time_point last_toggle{};

    for (;;) {
        {
            std::lock_guard<std::mutex> lock(shared_.mutex);
            if (!shared_.running) {
                break;
            }
        }

        const auto steady_now = SteadyClock::now();
        const InputSnapshot snapshot = hotkeys.poll();
        const RuntimeConfig runtime = runtime_store_.snapshot();
        bool sequence_enabled = false;
        bool toggle_changed = false;

        {
            std::lock_guard<std::mutex> lock(shared_.mutex);
            if (risingEdge(snapshot.f5_pressed, previous.f5_pressed) && (steady_now - last_toggle) >= kToggleCooldown) {
                shared_.side_button_key_sequence_enabled = !shared_.side_button_key_sequence_enabled;
                last_toggle = steady_now;
                toggle_changed = true;
            }
            sequence_enabled = shared_.side_button_key_sequence_enabled;
        }

        if (toggle_changed) {
            if (config_.debug_log) {
                std::cout << "[control] SideButtonKeySequenceRClickLClick31: " << (sequence_enabled ? "ON" : "OFF") << "\n";
            }
            playToggleBeep(sequence_enabled ? 1700 : 950);
        }

        if (!sequence_enabled || !snapshot.x1_pressed) {
            previous = snapshot;
            std::this_thread::sleep_for(kSideButtonKeySequenceIdleSleep);
            continue;
        }

        const double key3_press_time_ms = std::max(0.0, runtime.side_button_key_sequence_key3_press_time_ms);
        const double key1_press_time_ms = std::max(0.0, runtime.side_button_key_sequence_key1_press_time_ms);
        const double right_click_hold_ms = std::max(0.0, runtime.side_button_key_sequence_right_click_hold_ms);
        const double left_click_hold_ms = std::max(0.0, runtime.side_button_key_sequence_left_click_hold_ms);
        const double loop_delay_ms = std::max(0.0, runtime.side_button_key_sequence_loop_delay_ms);

        const bool sent_right_click = !runtime.side_button_key_sequence_use_right_click
            || sendRightClickTap(static_cast<double>(right_click_hold_ms) / 1000.0);
        const bool sent_left_click = !runtime.side_button_key_sequence_use_left_click
            || sendLeftClickTap(static_cast<double>(left_click_hold_ms) / 1000.0);
        const bool sent_three = !runtime.side_button_key_sequence_use_key3
            || sendVirtualKeyTap(static_cast<std::uint16_t>('3'), key3_press_time_ms);
        const bool sent_one = !runtime.side_button_key_sequence_use_key1
            || sendVirtualKeyTap(static_cast<std::uint16_t>('1'), key1_press_time_ms);
        if (config_.debug_log && (!sent_three || !sent_one || !sent_right_click || !sent_left_click)) {
            std::cerr << "[control] SideButtonKeySequenceRClickLClick31 send failed.\n";
        }

        previous = snapshot;
        if (loop_delay_ms > 0.0) {
            std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(loop_delay_ms));
        }
    }
}

void DeltaApp::perfLoop() {
    if (!perf_) {
        return;
    }

    for (;;) {
        {
            std::lock_guard<std::mutex> lock(shared_.mutex);
            if (!shared_.running) {
                break;
            }
        }
        std::this_thread::sleep_for(kPerfLoopSleep);

        const std::optional<PerfLogSnapshot> snapshot = takePerfSnapshot(*perf_, kPerfLogIntervalSeconds);
        if (!snapshot.has_value()) {
            continue;
        }

        ToggleState toggles{};
        {
            std::lock_guard<std::mutex> lock(shared_.mutex);
            toggles = shared_.toggles;
        }
        const RuntimeConfig runtime = runtime_store_.snapshot();
        const bool perf_engaged = (toggles.mode != 0)
            && isLeftHoldEngageSatisfied(
                toggles.left_hold_engage,
                runtime.left_hold_engage_button,
                toggles.left_pressed,
                toggles.right_pressed,
                toggles.x1_pressed);
        if (!kPerfLogWhenModeOff && !perf_engaged) {
            continue;
        }
        if (snapshot->infer_found == 0) {
            continue;
        }

        std::cout << std::fixed << std::setprecision(2)
                  << "[PERF] "
                  << "cap=" << snapshot->cap_fps << "fps grab=" << snapshot->cap_grab_ms
                  << "ms acq/d3d/sync/cuda/cpu=" << snapshot->cap_acquire_ms
                  << "/" << snapshot->cap_d3d_copy_ms
                  << "/" << snapshot->cap_d3d_sync_ms
                  << "/" << snapshot->cap_cuda_copy_ms
                  << "/" << snapshot->cap_cpu_copy_ms
                  << "ms cached=" << (snapshot->cap_cached_rate * 100.0)
                  << "%@" << snapshot->cap_cached_reuse_ms << "ms none=" << snapshot->cap_none
                  << " | inf=" << snapshot->infer_fps << "fps loop=" << snapshot->infer_loop_ms << "ms age="
                  << snapshot->infer_age_ms << "/" << snapshot->infer_age_max_ms << "ms stale=" << snapshot->infer_stale
                  << " lock=" << (snapshot->infer_lock_rate * 100.0) << "% | app(sel/trk/pred/pid/sync/prev/q)="
                  << snapshot->infer_select_ms
                  << "/" << snapshot->infer_tracker_update_ms
                  << "/" << snapshot->infer_aim_predict_ms
                  << "/" << snapshot->infer_aim_pid_ms
                  << "/" << snapshot->infer_aim_sync_ms
                  << "/" << snapshot->infer_preview_ms
                  << "/" << snapshot->infer_queue_ms << "ms";
        if (snapshot->infer_backend_samples > 0) {
            std::cout << " onnx(pre/exec/post)="
                      << snapshot->infer_backend_pre_ms << "/"
                      << snapshot->infer_backend_exec_ms << "/"
                      << snapshot->infer_backend_post_ms << "ms";
        }
        std::cout << " | ctl=" << snapshot->control_send_hz << "Hz send=" << snapshot->control_send_ms
                  << "ms cmdAge=" << snapshot->control_cmd_age_ms << "ms e2e=" << snapshot->control_total_latency_ms
                  << "ms e2eIn=" << snapshot->control_total_apply_latency_ms
                  << "ms e2eFull=" << snapshot->control_total_latency_full_ms
                  << "ms e2eFullIn=" << snapshot->control_total_apply_latency_full_ms
                  << "ms drop(stale/mode)=" << snapshot->control_stale_drop
                  << "/" << snapshot->control_mode_drop
                  << " aimPipe=" << snapshot->infer_cmd_ms << "ms\n";
        std::cout.unsetf(std::ios::floatfield);
    }
}

}  // namespace delta
