#include "delta/detection_dampening.hpp"

#include <algorithm>

namespace delta {

namespace {

constexpr float kDetectionDampeningStableIou = 0.20F;

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

}  // namespace

void DetectionDampeningState::reset() {
    last_selected_bbox.reset();
    stable_frame_count = 0;
}

DetectionDampeningConfig buildDetectionDampeningConfig(const RuntimeConfig& runtime) {
    DetectionDampeningConfig config{};
    config.enable = runtime.detection_dampening_enable;
    config.stable_frames = std::max(1, runtime.detection_dampening_stable_frames);
    return config;
}

DetectionDampeningResult noteDetectionDampeningSelection(
    DetectionDampeningState& state,
    const DetectionDampeningConfig& config,
    const std::optional<Detection>& detection,
    const bool target_switched) {
    const int required_frames = std::max(1, config.stable_frames);
    if (!config.enable) {
        state.reset();
        return DetectionDampeningResult{
            .ready = true,
            .streak = 0,
            .required_frames = required_frames,
        };
    }

    if (!detection.has_value()) {
        state.reset();
        return DetectionDampeningResult{
            .ready = false,
            .streak = 0,
            .required_frames = required_frames,
        };
    }

    const bool stable_match = !target_switched
        && state.last_selected_bbox.has_value()
        && bboxIou(*state.last_selected_bbox, detection->bbox) >= kDetectionDampeningStableIou;
    state.stable_frame_count = stable_match ? (state.stable_frame_count + 1) : 1;
    state.last_selected_bbox = detection->bbox;

    return DetectionDampeningResult{
        .ready = state.stable_frame_count >= required_frames,
        .streak = state.stable_frame_count,
        .required_frames = required_frames,
    };
}

}  // namespace delta
