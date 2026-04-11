#include "delta/target_guard.hpp"

#include <algorithm>
#include <cmath>

namespace delta {

namespace {

constexpr float kTargetGuardStableIou = 0.20F;

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

bool pointInRegion(const CaptureRegion& region, const float x, const float y) {
    const float left = static_cast<float>(region.left);
    const float top = static_cast<float>(region.top);
    const float right = left + static_cast<float>(region.width);
    const float bottom = top + static_cast<float>(region.height);
    return x >= left && x <= right && y >= top && y <= bottom;
}

std::pair<float, float> fallbackCenter(const TargetGuardActiveState& active) {
    if (active.last_accepted_point.has_value()) {
        return *active.last_accepted_point;
    }
    if (active.last_accepted_bbox.has_value()) {
        const auto& bbox = *active.last_accepted_bbox;
        return {
            static_cast<float>(bbox[0] + bbox[2]) * 0.5F,
            static_cast<float>(bbox[1] + bbox[3]) * 0.5F,
        };
    }
    return {0.0F, 0.0F};
}

}  // namespace

void TargetGuardPendingState::reset() {
    last_selected_bbox.reset();
    stable_frame_count = 0;
}

void TargetGuardActiveState::reset() {
    active = false;
    last_accepted_bbox.reset();
    last_accepted_point.reset();
    miss_count = 0;
}

void TargetGuardState::reset() {
    pending.reset();
    active.reset();
}

TargetGuardConfig buildTargetGuardConfig(const RuntimeConfig& runtime) {
    TargetGuardConfig config{};
    config.enable = runtime.target_guard_enable;
    config.commit_frames = std::max(1, runtime.target_guard_commit_frames);
    config.hold_frames = std::clamp(runtime.target_guard_hold_frames, 0, runtime.target_max_lost_frames);
    config.window_scale = std::max(0.1F, runtime.target_guard_window_scale);
    config.min_window_px = std::max(1, runtime.target_guard_min_window_px);
    return config;
}

std::optional<CaptureRegion> buildTargetGuardRegion(
    const TargetGuardState& state,
    const TargetGuardConfig& config,
    const CaptureRegion& capture_region,
    const std::optional<std::pair<float, float>>& preferred_center) {
    if (!config.enable || !state.active.active || !state.active.last_accepted_bbox.has_value()) {
        return std::nullopt;
    }
    if (capture_region.width <= 0 || capture_region.height <= 0) {
        return std::nullopt;
    }

    const auto center = preferred_center.value_or(fallbackCenter(state.active));
    const auto& bbox = *state.active.last_accepted_bbox;
    const float box_width = std::max(1.0F, static_cast<float>(bbox[2] - bbox[0]));
    const float box_height = std::max(1.0F, static_cast<float>(bbox[3] - bbox[1]));
    const int width = clamp(
        static_cast<int>(std::lround(std::max(box_width * config.window_scale, static_cast<float>(config.min_window_px)))),
        1,
        capture_region.width);
    const int height = clamp(
        static_cast<int>(std::lround(std::max(box_height * config.window_scale, static_cast<float>(config.min_window_px)))),
        1,
        capture_region.height);

    return CaptureRegion{
        .left = clamp(
            static_cast<int>(std::lround(center.first)) - (width / 2),
            capture_region.left,
            capture_region.left + capture_region.width - width),
        .top = clamp(
            static_cast<int>(std::lround(center.second)) - (height / 2),
            capture_region.top,
            capture_region.top + capture_region.height - height),
        .width = width,
        .height = height,
    };
}

std::vector<Detection> filterDetectionsInTargetGuard(
    const std::vector<Detection>& detections,
    const std::optional<CaptureRegion>& guard_region) {
    if (!guard_region.has_value()) {
        return detections;
    }

    std::vector<Detection> filtered;
    filtered.reserve(detections.size());
    for (const auto& detection : detections) {
        if (pointInRegion(*guard_region, detection.x, detection.y)) {
            filtered.push_back(detection);
        }
    }
    return filtered;
}

void noteTargetGuardSelection(
    TargetGuardState& state,
    const TargetGuardConfig& config,
    const Detection& detection) {
    if (!config.enable) {
        state.reset();
        return;
    }

    const bool stable_match = state.pending.last_selected_bbox.has_value()
        && bboxIou(*state.pending.last_selected_bbox, detection.bbox) >= kTargetGuardStableIou;
    state.pending.stable_frame_count = stable_match ? (state.pending.stable_frame_count + 1) : 1;
    state.pending.last_selected_bbox = detection.bbox;

    if (state.pending.stable_frame_count >= config.commit_frames) {
        state.active.active = true;
    }
    if (state.active.active) {
        state.active.last_accepted_bbox = detection.bbox;
        state.active.last_accepted_point = std::make_pair(detection.x, detection.y);
        state.active.miss_count = 0;
    }
}

TargetGuardMissResult noteTargetGuardMiss(
    TargetGuardState& state,
    const TargetGuardConfig& config) {
    state.pending.reset();
    if (!config.enable || !state.active.active) {
        return TargetGuardMissResult::Inactive;
    }

    ++state.active.miss_count;
    if (state.active.miss_count <= config.hold_frames) {
        return TargetGuardMissResult::Holding;
    }

    state.reset();
    return TargetGuardMissResult::Expired;
}

}  // namespace delta
