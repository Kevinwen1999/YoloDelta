#pragma once

#include <optional>
#include <vector>

#include "delta/config.hpp"
#include "delta/core.hpp"

namespace delta {

struct DetectionDampeningConfig {
    bool enable = false;
    int stable_frames = 3;
};

struct DetectionDampeningState {
    std::optional<std::array<int, 4>> last_selected_bbox;
    int stable_frame_count = 0;
    SteadyClock::time_point last_observation{};
    SteadyClock::time_point confirmation_started{};
    std::optional<Detection> accepted_detection;
    float vx = 0, vy = 0;
    bool confirmed = false;

    void reset();
};

struct MotionSelection {
    const char* decision = "no_candidate";
    std::optional<Detection> detection;
    bool ready = false;
    bool switched = false;
    bool ambiguous = false;
    bool duplicate = false;
    float residual_px = 0;
    std::vector<std::pair<Detection,float>> matches;
};

// Only unique accepted observations advance confirmation or refresh the deadline.
// Camera displacement is calibrated delivered motion between the previous
// accepted source frame and this source frame. Detections remain screen-local;
// the optional compensation keeps camera catch-up out of target velocity.
MotionSelection noteMotionSelection(DetectionDampeningState& state,
    const RuntimeConfig& config, const std::optional<Detection>& proposed,
    const std::vector<Detection>& candidates, SteadyClock::time_point observed,
    SteadyClock::time_point now,std::optional<CaptureRegion> observed_crop=std::nullopt,
    std::pair<float,float> camera_displacement={},
    std::pair<float,float> camera_displacement_now={});

struct DetectionDampeningResult {
    bool ready = true;
    int streak = 0;
    int required_frames = 3;
};

DetectionDampeningConfig buildDetectionDampeningConfig(const RuntimeConfig& runtime);

DetectionDampeningResult noteDetectionDampeningSelection(
    DetectionDampeningState& state,
    const DetectionDampeningConfig& config,
    const std::optional<Detection>& detection,
    bool target_switched);

}  // namespace delta
