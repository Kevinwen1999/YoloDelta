#include "delta/triggerbot.hpp"

#include <algorithm>
#include <cmath>

namespace delta {

TriggerbotConfig buildTriggerbotConfig(const RuntimeConfig& runtime) {
    TriggerbotConfig config{};
    config.enable = runtime.triggerbot_enable;
    config.arm_scale_x = std::max(0.0F, runtime.triggerbot_arm_scale_x);
    config.arm_scale_y = std::max(0.0F, runtime.triggerbot_arm_scale_y);
    config.arm_min_x_px = std::max(0, runtime.triggerbot_arm_min_x_px);
    config.arm_min_y_px = std::max(0, runtime.triggerbot_arm_min_y_px);
    config.click_hold_s = std::max(0.0F, runtime.triggerbot_click_hold_s);
    config.click_cooldown_s = std::max(0.0F, runtime.triggerbot_click_cooldown_s);
    return config;
}

std::pair<float, float> triggerbotArmThresholds(
    const Detection& detection,
    const TriggerbotConfig& config) {
    const float box_w = std::max(1.0F, static_cast<float>(detection.bbox[2] - detection.bbox[0]));
    const float box_h = std::max(1.0F, static_cast<float>(detection.bbox[3] - detection.bbox[1]));
    return {
        std::max(box_w * config.arm_scale_x, static_cast<float>(config.arm_min_x_px)),
        std::max(box_h * config.arm_scale_y, static_cast<float>(config.arm_min_y_px)),
    };
}

bool isTriggerbotArmed(
    const Detection& detection,
    const int center_x,
    const int center_y,
    const TriggerbotConfig& config) {
    const auto [threshold_x, threshold_y] = triggerbotArmThresholds(detection, config);
    return std::abs(detection.x - static_cast<float>(center_x)) <= threshold_x
        && std::abs(detection.y - static_cast<float>(center_y)) <= threshold_y;
}

}  // namespace delta
