#pragma once

#include "delta/config.hpp"
#include "delta/core.hpp"

#include <algorithm>
#include <cmath>

namespace delta {

// The live output-stage legacy recoil integration, independent of device I/O.
inline double legacyRecoilRate(const RuntimeConfig& runtime) {
    return std::abs(runtime.recoil_compensation_y_rate_px_s) > 1e-6F
        ? static_cast<double>(runtime.recoil_compensation_y_rate_px_s)
        : static_cast<double>(runtime.recoil_compensation_y_px) * 120.0;
}

inline bool legacyRecoilEnabled(const RuntimeConfig& runtime) {
    return runtime.legacy_recoil_output_enable && runtime.recoil_mode == RecoilMode::Legacy &&
        std::abs(legacyRecoilRate(runtime)) > 1e-6;
}

class LegacyRecoilOutput {
public:
    int step(const RuntimeConfig& runtime, bool virtual_active, bool target_detected,
             bool trigger_pressed, bool trigger_will_click, SteadyClock::time_point now) {
        if (virtual_active || !legacyRecoilEnabled(runtime) || !target_detected ||
            !(trigger_pressed || trigger_will_click)) {
            carry_y_ = 0;
            last_tick_ = {};
            return 0;
        }
        if (last_tick_ != SteadyClock::time_point{}) {
            const auto dt = std::clamp(std::chrono::duration<double>(now - last_tick_).count(), 0.0, .05);
            carry_y_ += legacyRecoilRate(runtime) * dt;
        }
        last_tick_ = now;
        const int step = static_cast<int>(carry_y_);
        carry_y_ -= step;
        return step;
    }

private:
    double carry_y_ = 0;
    SteadyClock::time_point last_tick_{};
};

}  // namespace delta
