#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "delta/core.hpp"

namespace delta {

struct DebugPreviewDetection {
    std::array<int, 4> bbox{0, 0, 0, 0};
    int cls = -1;
    float conf = 0.0F;
    bool selected = false;
};

struct DebugPreviewSnapshot {
    bool active = false;
    CaptureRegion capture_region{};
    std::optional<CaptureRegion> guard_region;
    std::pair<int, int> screen_center{0, 0};
    std::vector<DebugPreviewDetection> detections;
    std::optional<std::pair<float, float>> locked_point;
    std::optional<std::pair<float, float>> predicted_point;
    bool target_found = false;
    int target_cls = -1;
    float target_speed = 0.0F;
    std::uint64_t sequence = 0;
};

}  // namespace delta
