#pragma once

#include <array>
#include <optional>
#include <utility>

#include "delta/config.hpp"
#include "delta/core.hpp"

namespace delta {

struct TargetLeadConfig {
    bool enable = false;
    int commit_frames = 3;
    bool auto_latency_enable = true;
    float max_time_s = 0.08F;
    float min_speed_px_s = 120.0F;
    float max_offset_box_scale = 0.75F;
    float smoothing_alpha = 0.35F;
};

struct TargetLeadPendingState {
    std::optional<std::array<int, 4>> last_selected_bbox;
    int stable_frame_count = 0;

    void reset();
};

struct TargetLeadActiveState {
    bool active = false;
    std::optional<std::pair<float, float>> last_detected_point;
    std::optional<std::array<int, 4>> last_detected_bbox;
    SteadyClock::time_point last_measurement_time{};
    float velocity_x = 0.0F;
    float velocity_y = 0.0F;
    float lead_offset_x = 0.0F;
    float lead_offset_y = 0.0F;
    bool detected_point_stale = false;

    void reset();
};

struct TargetLeadState {
    TargetLeadPendingState pending{};
    TargetLeadActiveState active{};

    void reset();
};

struct TargetLeadPrediction {
    bool active = false;
    std::pair<float, float> detected_point{0.0F, 0.0F};
    std::pair<float, float> predicted_point{0.0F, 0.0F};
    float velocity_x = 0.0F;
    float velocity_y = 0.0F;
    float lead_time_s = 0.0F;
    bool detected_point_stale = false;
};

TargetLeadConfig buildTargetLeadConfig(const RuntimeConfig& runtime);

void noteTargetLeadSelection(
    TargetLeadState& state,
    const TargetLeadConfig& config,
    const Detection& detection,
    SteadyClock::time_point sample_time);

void noteTargetLeadMiss(
    TargetLeadState& state,
    const TargetLeadConfig& config);

std::optional<TargetLeadPrediction> predictTargetLead(
    TargetLeadState& state,
    const TargetLeadConfig& config,
    SteadyClock::time_point now,
    float adaptive_latency_s,
    float extra_time_s,
    float extra_velocity_x,
    float extra_velocity_y,
    int screen_w,
    int screen_h);

}  // namespace delta
