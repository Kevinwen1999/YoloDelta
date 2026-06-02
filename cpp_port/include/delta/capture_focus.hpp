#pragma once

#include <array>
#include <optional>
#include <utility>

#include "delta/config.hpp"
#include "delta/core.hpp"

namespace delta {

struct CaptureCropTarget {
    std::array<int, 4> bbox{0, 0, 0, 0};
    CaptureRegion capture{};
};

std::pair<int, int> selectCaptureFocus(
    bool freeze_to_center,
    bool target_found,
    std::pair<int, int> screen_center,
    std::pair<int, int> tracked_focus);

int fixedCaptureCropSize(const StaticConfig& config);
int initialCaptureCropSize(const StaticConfig& config, const RuntimeConfig& runtime);
int updateAdaptiveCaptureCropSize(
    const StaticConfig& config,
    const RuntimeConfig& runtime,
    int current_size,
    const std::optional<CaptureCropTarget>& target);

}  // namespace delta
