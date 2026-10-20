#pragma once
#include "benchmark.hpp"

namespace delta::bench {
struct ControlResponseOptions {
    int frames = 400;
    double cadence_ms = 16, observation_delay_ms = 12, transport_delay_ms = 1;
    double pixels_per_count = .4, speed_px_s = 1500;
    double initial_error_px = 0;
    int dropout_start = -1, dropout_frames = 0;
    bool reversal = false;
};
Json runControlResponse(const StaticConfig&, const RuntimeConfig&, const ControlResponseOptions&);
}  // namespace delta::bench
