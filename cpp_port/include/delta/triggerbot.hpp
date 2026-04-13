#pragma once

#include <utility>

#include "delta/config.hpp"
#include "delta/core.hpp"

namespace delta {

struct TriggerbotConfig {
    bool enable = false;
    float arm_scale_x = 0.5F;
    float arm_scale_y = 0.5F;
    int arm_min_x_px = 0;
    int arm_min_y_px = 0;
    float click_hold_s = 0.001F;
    float click_cooldown_s = 0.001F;
};

TriggerbotConfig buildTriggerbotConfig(const RuntimeConfig& runtime);

std::pair<float, float> triggerbotArmThresholds(
    const Detection& detection,
    const TriggerbotConfig& config);

bool isTriggerbotArmed(
    const Detection& detection,
    int center_x,
    int center_y,
    const TriggerbotConfig& config);

}  // namespace delta
