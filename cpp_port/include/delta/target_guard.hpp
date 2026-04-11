#pragma once

#include <optional>
#include <utility>
#include <vector>

#include "delta/config.hpp"
#include "delta/core.hpp"

namespace delta {

struct TargetGuardConfig {
    bool enable = false;
    int commit_frames = 3;
    int hold_frames = 5;
    float window_scale = 2.25F;
    int min_window_px = 120;
};

struct TargetGuardPendingState {
    std::optional<std::array<int, 4>> last_selected_bbox;
    int stable_frame_count = 0;

    void reset();
};

struct TargetGuardActiveState {
    bool active = false;
    std::optional<std::array<int, 4>> last_accepted_bbox;
    std::optional<std::pair<float, float>> last_accepted_point;
    int miss_count = 0;

    void reset();
};

struct TargetGuardState {
    TargetGuardPendingState pending{};
    TargetGuardActiveState active{};

    void reset();
};

enum class TargetGuardMissResult {
    Inactive,
    Holding,
    Expired,
};

TargetGuardConfig buildTargetGuardConfig(const RuntimeConfig& runtime);

std::optional<CaptureRegion> buildTargetGuardRegion(
    const TargetGuardState& state,
    const TargetGuardConfig& config,
    const CaptureRegion& capture_region,
    const std::optional<std::pair<float, float>>& preferred_center);

std::vector<Detection> filterDetectionsInTargetGuard(
    const std::vector<Detection>& detections,
    const std::optional<CaptureRegion>& guard_region);

void noteTargetGuardSelection(
    TargetGuardState& state,
    const TargetGuardConfig& config,
    const Detection& detection);

TargetGuardMissResult noteTargetGuardMiss(
    TargetGuardState& state,
    const TargetGuardConfig& config);

}  // namespace delta
