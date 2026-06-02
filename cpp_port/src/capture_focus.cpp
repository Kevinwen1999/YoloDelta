#include "delta/capture_focus.hpp"

#include <algorithm>
#include <cmath>

namespace delta {

namespace {

int maxCaptureSizeForScreen(const StaticConfig& config) {
    return std::max(1, std::min(config.screen_w, config.screen_h));
}

float finiteOrDefault(const float value, const float fallback) {
    return std::isfinite(value) ? value : fallback;
}

struct AdaptiveCropBounds {
    int min_size = 1;
    int search_size = 1;
    int max_size = 1;
    int step_px = 1;
};

AdaptiveCropBounds adaptiveCropBounds(const StaticConfig& config, const RuntimeConfig& runtime) {
    const int screen_max = maxCaptureSizeForScreen(config);
    const int fixed_size = fixedCaptureCropSize(config);
    const int min_size = clamp(runtime.adaptive_capture_crop_min_size, 1, screen_max);
    const int max_size = clamp(
        std::max(runtime.adaptive_capture_crop_max_size, min_size),
        min_size,
        screen_max);
    return AdaptiveCropBounds{
        .min_size = min_size,
        .search_size = clamp(
            runtime.adaptive_capture_crop_search_size > 0
                ? runtime.adaptive_capture_crop_search_size
                : fixed_size,
            min_size,
            max_size),
        .max_size = max_size,
        .step_px = std::max(1, runtime.adaptive_capture_crop_step_px),
    };
}

int quantizeCropSize(const float value, const AdaptiveCropBounds& bounds) {
    if (bounds.step_px <= 1) {
        return clamp(static_cast<int>(std::lround(value)), bounds.min_size, bounds.max_size);
    }
    const float offset_steps = std::round(
        (value - static_cast<float>(bounds.search_size)) / static_cast<float>(bounds.step_px));
    const int quantized = bounds.search_size
        + (static_cast<int>(offset_steps) * bounds.step_px);
    return clamp(quantized, bounds.min_size, bounds.max_size);
}

bool targetNearCaptureEdge(const CaptureCropTarget& target, const float margin_ratio) {
    if (target.capture.width <= 0 || target.capture.height <= 0) {
        return false;
    }
    const float safe_ratio = clamp(finiteOrDefault(margin_ratio, 0.0F), 0.0F, 0.49F);
    if (safe_ratio <= 0.0F) {
        return false;
    }

    const float left = static_cast<float>(target.capture.left);
    const float top = static_cast<float>(target.capture.top);
    const float right = left + static_cast<float>(target.capture.width);
    const float bottom = top + static_cast<float>(target.capture.height);
    const float margin_x = static_cast<float>(target.capture.width) * safe_ratio;
    const float margin_y = static_cast<float>(target.capture.height) * safe_ratio;
    return static_cast<float>(target.bbox[0]) <= left + margin_x
        || static_cast<float>(target.bbox[2]) >= right - margin_x
        || static_cast<float>(target.bbox[1]) <= top + margin_y
        || static_cast<float>(target.bbox[3]) >= bottom - margin_y;
}

}  // namespace

std::pair<int, int> selectCaptureFocus(
    const bool freeze_to_center,
    const bool target_found,
    const std::pair<int, int> screen_center,
    const std::pair<int, int> tracked_focus) {
    if (freeze_to_center || !target_found) {
        return screen_center;
    }
    return tracked_focus;
}

int fixedCaptureCropSize(const StaticConfig& config) {
    return clamp(effectiveCaptureCropSize(config), 1, maxCaptureSizeForScreen(config));
}

int initialCaptureCropSize(const StaticConfig& config, const RuntimeConfig& runtime) {
    if (!runtime.adaptive_capture_crop_enable) {
        return fixedCaptureCropSize(config);
    }
    return adaptiveCropBounds(config, runtime).search_size;
}

int updateAdaptiveCaptureCropSize(
    const StaticConfig& config,
    const RuntimeConfig& runtime,
    const int current_size,
    const std::optional<CaptureCropTarget>& target) {
    if (!runtime.adaptive_capture_crop_enable) {
        return fixedCaptureCropSize(config);
    }

    const AdaptiveCropBounds bounds = adaptiveCropBounds(config, runtime);
    const int current = current_size > 0
        ? clamp(current_size, bounds.min_size, bounds.max_size)
        : bounds.search_size;
    float desired = static_cast<float>(bounds.search_size);

    if (target.has_value()) {
        const int box_w = std::max(1, target->bbox[2] - target->bbox[0]);
        const int box_h = std::max(1, target->bbox[3] - target->bbox[1]);
        const int box_size = std::max(box_w, box_h);
        const float target_input_px = std::max(
            1.0F,
            finiteOrDefault(runtime.adaptive_capture_crop_target_box_input_px, 96.0F));
        desired = (static_cast<float>(box_size) * static_cast<float>(std::max(1, config.imgsz))) / target_input_px;
        if (targetNearCaptureEdge(*target, runtime.adaptive_capture_crop_edge_margin_ratio)) {
            desired = std::max(desired, static_cast<float>(current + (bounds.step_px * 4)));
        }
    }

    desired = clamp(desired, static_cast<float>(bounds.min_size), static_cast<float>(bounds.max_size));
    const float alpha = clamp(
        finiteOrDefault(runtime.adaptive_capture_crop_smoothing_alpha, 0.25F),
        0.0F,
        1.0F);
    const float smoothed = static_cast<float>(current) + ((desired - static_cast<float>(current)) * alpha);
    return quantizeCropSize(smoothed, bounds);
}

}  // namespace delta
