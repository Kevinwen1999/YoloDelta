#pragma once

#include <optional>

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

    void reset();
};

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
