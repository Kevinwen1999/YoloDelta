#include "delta/target_lead.hpp"

#include <algorithm>
#include <cmath>

namespace delta {

namespace {

constexpr float kTargetLeadStableIou = 0.20F;
constexpr float kMinDt = 1.0F / 240.0F;
constexpr float kMaxDt = 0.06F;
constexpr float kMaxTrackSpeedPxS = 1800.0F;

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

float emaUpdate(const float prev, const float sample, const float alpha) {
    const float clamped_alpha = clamp(alpha, 0.0F, 1.0F);
    return prev + ((sample - prev) * clamped_alpha);
}

}  // namespace

void TargetLeadPendingState::reset() {
    last_selected_bbox.reset();
    stable_frame_count = 0;
}

void TargetLeadActiveState::reset() {
    active = false;
    last_detected_point.reset();
    last_detected_bbox.reset();
    last_measurement_time = {};
    velocity_x = 0.0F;
    velocity_y = 0.0F;
    lead_offset_x = 0.0F;
    lead_offset_y = 0.0F;
    detected_point_stale = false;
}

void TargetLeadState::reset() {
    pending.reset();
    active.reset();
}

TargetLeadConfig buildTargetLeadConfig(const RuntimeConfig& runtime) {
    TargetLeadConfig config{};
    config.enable = runtime.target_lead_enable;
    config.commit_frames = std::max(1, runtime.target_lead_commit_frames);
    config.auto_latency_enable = runtime.target_lead_auto_latency_enable;
    config.max_time_s = std::max(0.0F, runtime.target_lead_max_time_s);
    config.min_speed_px_s = std::max(0.0F, runtime.target_lead_min_speed_px_s);
    config.max_offset_box_scale = std::max(0.0F, runtime.target_lead_max_offset_box_scale);
    config.smoothing_alpha = clamp(runtime.target_lead_smoothing_alpha, 0.0F, 1.0F);
    return config;
}

void noteTargetLeadSelection(
    TargetLeadState& state,
    const TargetLeadConfig& config,
    const Detection& detection,
    const SteadyClock::time_point sample_time) {
    if (!config.enable) {
        state.reset();
        return;
    }

    const bool had_previous = state.pending.last_selected_bbox.has_value();
    const bool stable_match = had_previous
        && bboxIou(*state.pending.last_selected_bbox, detection.bbox) >= kTargetLeadStableIou;
    if (stable_match) {
        ++state.pending.stable_frame_count;
    } else {
        state.pending.stable_frame_count = 1;
        if (had_previous) {
            state.active.reset();
        }
    }
    state.pending.last_selected_bbox = detection.bbox;

    if (state.active.last_detected_point.has_value() && state.active.last_measurement_time != SteadyClock::time_point{}) {
        const float dt = clamp(
            static_cast<float>(std::chrono::duration<double>(sample_time - state.active.last_measurement_time).count()),
            kMinDt,
            kMaxDt);
        const float raw_vx = clamp(
            (detection.x - state.active.last_detected_point->first) / dt,
            -kMaxTrackSpeedPxS,
            kMaxTrackSpeedPxS);
        const float raw_vy = clamp(
            (detection.y - state.active.last_detected_point->second) / dt,
            -kMaxTrackSpeedPxS,
            kMaxTrackSpeedPxS);
        state.active.velocity_x = emaUpdate(state.active.velocity_x, raw_vx, config.smoothing_alpha);
        state.active.velocity_y = emaUpdate(state.active.velocity_y, raw_vy, config.smoothing_alpha);
    }

    state.active.last_detected_point = std::make_pair(detection.x, detection.y);
    state.active.last_detected_bbox = detection.bbox;
    state.active.last_measurement_time = sample_time;
    state.active.detected_point_stale = false;
    if (state.pending.stable_frame_count >= config.commit_frames) {
        state.active.active = true;
    }
}

void noteTargetLeadMiss(
    TargetLeadState& state,
    const TargetLeadConfig& config) {
    if (!config.enable) {
        state.reset();
        return;
    }
    if (state.active.active) {
        state.active.detected_point_stale = true;
    }
}

std::optional<TargetLeadPrediction> predictTargetLead(
    TargetLeadState& state,
    const TargetLeadConfig& config,
    const SteadyClock::time_point now,
    const float adaptive_latency_s,
    const float extra_time_s,
    const float extra_velocity_x,
    const float extra_velocity_y,
    const int screen_w,
    const int screen_h) {
    if (!config.enable
        || !state.active.active
        || !state.active.last_detected_point.has_value()
        || !state.active.last_detected_bbox.has_value()
        || state.active.last_measurement_time == SteadyClock::time_point{}) {
        return std::nullopt;
    }

    const float measurement_age_s = std::max(
        0.0F,
        static_cast<float>(std::chrono::duration<double>(now - state.active.last_measurement_time).count()));
    const float lead_time_s = clamp(
        measurement_age_s + std::max(0.0F, adaptive_latency_s) + std::max(0.0F, extra_time_s),
        0.0F,
        config.max_time_s);

    float vx = state.active.velocity_x + extra_velocity_x;
    float vy = state.active.velocity_y + extra_velocity_y;
    const float speed = std::sqrt((vx * vx) + (vy * vy));
    if (speed > kMaxTrackSpeedPxS && speed > 1e-6F) {
        const float scale = kMaxTrackSpeedPxS / speed;
        vx *= scale;
        vy *= scale;
    }

    float raw_offset_x = 0.0F;
    float raw_offset_y = 0.0F;
    if (std::sqrt((vx * vx) + (vy * vy)) >= config.min_speed_px_s) {
        raw_offset_x = vx * lead_time_s;
        raw_offset_y = vy * lead_time_s;
    }

    const float box_w = std::max(
        1.0F,
        static_cast<float>(state.active.last_detected_bbox->at(2) - state.active.last_detected_bbox->at(0)));
    const float box_h = std::max(
        1.0F,
        static_cast<float>(state.active.last_detected_bbox->at(3) - state.active.last_detected_bbox->at(1)));
    raw_offset_x = clamp(
        raw_offset_x,
        -(box_w * config.max_offset_box_scale),
        box_w * config.max_offset_box_scale);
    raw_offset_y = clamp(
        raw_offset_y,
        -(box_h * config.max_offset_box_scale),
        box_h * config.max_offset_box_scale);

    state.active.lead_offset_x = emaUpdate(state.active.lead_offset_x, raw_offset_x, config.smoothing_alpha);
    state.active.lead_offset_y = emaUpdate(state.active.lead_offset_y, raw_offset_y, config.smoothing_alpha);

    const float predicted_x = clamp(
        state.active.last_detected_point->first + state.active.lead_offset_x,
        0.0F,
        static_cast<float>(std::max(0, screen_w - 1)));
    const float predicted_y = clamp(
        state.active.last_detected_point->second + state.active.lead_offset_y,
        0.0F,
        static_cast<float>(std::max(0, screen_h - 1)));

    return TargetLeadPrediction{
        .active = true,
        .detected_point = *state.active.last_detected_point,
        .predicted_point = {predicted_x, predicted_y},
        .velocity_x = vx,
        .velocity_y = vy,
        .lead_time_s = lead_time_s,
        .detected_point_stale = state.active.detected_point_stale,
    };
}

}  // namespace delta
