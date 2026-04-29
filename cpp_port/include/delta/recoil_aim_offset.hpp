#pragma once

#include "delta/core.hpp"
#include "delta/recoil_types.hpp"

namespace delta {

struct RecoilAimOffsetContext {
    bool enabled = false;
    bool has_target = false;
    RecoilMode recoil_mode = RecoilMode::Legacy;
    bool recoil_trigger_active = false;
    double legacy_rate_y_px_s = 0.0;
    PendingRecoilDelta advanced_delta{};
    SteadyClock::time_point now{};
    double max_integrate_dt_s = 0.05;
};

struct RecoilAimOffsetResult {
    bool active = false;
    bool consumed_advanced_delta = false;
    int dx = 0;
    int dy = 0;
};

class RecoilAimOffsetIntegrator {
public:
    RecoilAimOffsetResult update(const RecoilAimOffsetContext& context);
    void reset();

private:
    double legacy_carry_y_ = 0.0;
    SteadyClock::time_point last_legacy_tick_{};
};

}  // namespace delta
