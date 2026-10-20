#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <optional>

#include "delta/config.hpp"
#include "delta/core.hpp"

namespace delta {

inline bool calibratedControlActive(const RuntimeConfig& config) {
    return config.calibrated_control_enable && config.pid_enable && config.display_rate_servo_enable &&
           config.tracking_strategy == TrackingStrategy::PredictivePid;
}
// Align completed commands with the later frame in which their response is
// visible. Raw source timestamps remain unchanged for confirmation/deadlines.
inline SteadyClock::time_point calibratedHistoryTime(const RuntimeConfig& config,
                                                     SteadyClock::time_point source) {
    const double delay_ms = std::isfinite(config.calibrated_response_delay_ms)
                                ? std::clamp(double(config.calibrated_response_delay_ms), 0.0, 50.0)
                                : 0.0;
    return source - std::chrono::duration_cast<SteadyClock::duration>(
                        std::chrono::duration<double, std::milli>(delay_ms));
}
inline double calibratedPredictionTimeoutSeconds(const RuntimeConfig& config) {
    const double timeout_ms = config.motion_stability_enable && config.detection_dampening_enable
                                  ? config.motion_prediction_timeout_ms
                              : config.target_prediction_timeout_ms > 0
                                  ? config.target_prediction_timeout_ms
                                  : config.command_max_observation_age_ms;
    return std::min(100.0, timeout_ms) / 1000.0;
}

// Counts are before the selected backend transform. Calibration must include
// that transform and the application's sensitivity. Completion is an output
// adapter timestamp; it is not a measurement of physical camera application.
class DeliveredMotionHistory {
public:
    void note(SteadyClock::time_point completed, double dx, double dy);
    std::optional<std::array<double, 2>> between(SteadyClock::time_point start,
                                                 SteadyClock::time_point end) const;

private:
    struct Entry {
        SteadyClock::time_point at;
        double x, y;
    };
    std::deque<Entry> entries_;
    SteadyClock::time_point retained_since_{};
};

struct CalibratedControlResult {
    bool valid = false;
    float counts_per_second_x = 0, counts_per_second_y = 0;
    float error_x = 0, error_y = 0;
    float velocity_x = 0, velocity_y = 0;
};

// Position feedback corrects residual error; target velocity supplies the
// steady following rate. Both use source-frame time and delivered camera motion.
class CalibratedController {
public:
    void reset();
    void reanchorAimPoint();
    CalibratedControlResult update(const RuntimeConfig& config,
                                   std::optional<std::pair<float, float>> accepted_error,
                                   SteadyClock::time_point source_time, SteadyClock::time_point now,
                                   const DeliveredMotionHistory& motion);

private:
    SteadyClock::time_point previous_source_{};
    std::array<double, 2> previous_error_{}, velocity_{};
};
}  // namespace delta
