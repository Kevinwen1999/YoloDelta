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

struct ThirdPersonCaptureOffset {
    bool active = false;
    int x_px = 0;
    int y_px = 0;
};

ThirdPersonCaptureOffset effectiveThirdPersonCaptureOffset(
    const RuntimeConfig& runtime,
    bool right_pressed);

std::pair<int, int> applyThirdPersonCaptureOffset(
    std::pair<int, int> focus,
    const ThirdPersonCaptureOffset& offset,
    int screen_w,
    int screen_h);

bool isAdaptiveCaptureCropActive(const RuntimeConfig& runtime, bool right_pressed);

int fixedCaptureCropSize(const StaticConfig& config);
int initialCaptureCropSize(const StaticConfig& config, const RuntimeConfig& runtime);
int initialEffectiveCaptureCropSize(const StaticConfig& config, const RuntimeConfig& runtime, bool right_pressed);
int updateAdaptiveCaptureCropSize(
    const StaticConfig& config,
    const RuntimeConfig& runtime,
    int current_size,
    const std::optional<CaptureCropTarget>& target);
int updateEffectiveAdaptiveCaptureCropSize(
    const StaticConfig& config,
    const RuntimeConfig& runtime,
    bool right_pressed,
    int current_size,
    const std::optional<CaptureCropTarget>& target);

}  // namespace delta
