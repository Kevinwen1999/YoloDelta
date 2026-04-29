#include "delta/recoil_aim_offset.hpp"

#include <algorithm>
#include <cmath>

namespace delta {

RecoilAimOffsetResult RecoilAimOffsetIntegrator::update(const RecoilAimOffsetContext& context) {
    RecoilAimOffsetResult result{};
    if (!context.enabled || !context.has_target) {
        reset();
        return result;
    }

    if (context.recoil_mode == RecoilMode::AdvancedProfile) {
        reset();
        result.dx = context.advanced_delta.dx;
        result.dy = context.advanced_delta.dy;
        result.active = result.dx != 0 || result.dy != 0;
        result.consumed_advanced_delta = result.active;
        return result;
    }

    if (!context.recoil_trigger_active || std::abs(context.legacy_rate_y_px_s) <= 1e-6) {
        reset();
        return result;
    }

    if (last_legacy_tick_ != SteadyClock::time_point{}) {
        const double dt_s = clamp(
            std::chrono::duration<double>(context.now - last_legacy_tick_).count(),
            0.0,
            std::max(0.0, context.max_integrate_dt_s));
        legacy_carry_y_ += context.legacy_rate_y_px_s * dt_s;
    }
    last_legacy_tick_ = context.now;

    result.dy = static_cast<int>(legacy_carry_y_);
    legacy_carry_y_ -= static_cast<double>(result.dy);
    result.active = result.dy != 0;
    return result;
}

void RecoilAimOffsetIntegrator::reset() {
    legacy_carry_y_ = 0.0;
    last_legacy_tick_ = {};
}

}  // namespace delta
