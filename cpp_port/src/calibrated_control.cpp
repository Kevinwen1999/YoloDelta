#include "delta/calibrated_control.hpp"

#include <algorithm>
#include <cmath>

#include "delta/config.hpp"

namespace delta {
void DeliveredMotionHistory::note(SteadyClock::time_point at, double dx, double dy) {
    if (at == SteadyClock::time_point{} || !std::isfinite(dx) || !std::isfinite(dy) ||
        (!entries_.empty() && at < entries_.back().at))
        return;
    entries_.push_back({at, dx, dy});
    while (!entries_.empty() &&
           (entries_.size() > 4096 || at - entries_.front().at > std::chrono::seconds(1))) {
        retained_since_ = entries_.front().at;
        entries_.pop_front();
    }
}
std::optional<std::array<double, 2>> DeliveredMotionHistory::between(SteadyClock::time_point start,
                                                                     SteadyClock::time_point end) const {
    if (start < retained_since_ || end < start) return {};
    std::array<double, 2> result{};
    for (const auto& entry : entries_) {
        if (entry.at > end) break;
        if (entry.at > start) {
            result[0] += entry.x;
            result[1] += entry.y;
        }
    }
    return result;
}
void CalibratedController::reset() {
    previous_source_ = {};
    previous_error_ = {};
    velocity_ = {};
}
void CalibratedController::reanchorAimPoint() {
    previous_source_ = {};
    previous_error_ = {};
    // Retain translational velocity when only the aim point changes.
}
CalibratedControlResult CalibratedController::update(const RuntimeConfig& config,
                                                     std::optional<std::pair<float, float>> accepted_error,
                                                     SteadyClock::time_point source_time,
                                                     SteadyClock::time_point now,
                                                     const DeliveredMotionHistory& motion) {
    CalibratedControlResult result;
    const std::array<double, 2> gain{config.calibrated_pixels_per_count_x,
                                     config.calibrated_pixels_per_count_y};
    const double response = config.calibrated_response_time_s;
    const double response_delay_s = config.calibrated_response_delay_ms / 1000.0;
    if (!std::isfinite(response) || config.calibrated_response_time_s < .02F || response > 1 ||
        !std::isfinite(gain[0]) || !std::isfinite(gain[1]) ||
        !std::isfinite(config.predictive_pid_velocity_alpha) ||
        !std::isfinite(config.predictive_pid_max_velocity_px_s) || gain[0] < .001 || gain[1] < .001 ||
        !std::isfinite(response_delay_s) || response_delay_s < 0 || response_delay_s > .05 || gain[0] > 100 ||
        gain[1] > 100) {
        reset();
        return result;
    }
    if (accepted_error) {
        if (!std::isfinite(accepted_error->first) || !std::isfinite(accepted_error->second) ||
            source_time == SteadyClock::time_point{} || source_time > now || source_time <= previous_source_)
            return result;
        const std::array<double, 2> error{accepted_error->first, accepted_error->second};
        if (previous_source_ != SteadyClock::time_point{}) {
            const double dt = std::chrono::duration<double>(source_time - previous_source_).count();
            const auto delivered = motion.between(calibratedHistoryTime(config, previous_source_),
                                                  calibratedHistoryTime(config, source_time));
            if (!delivered || dt > .1)
                velocity_ = {};
            else {
                const double alpha =
                    1 - std::pow(1 - std::clamp(double(config.predictive_pid_velocity_alpha), 0.0, 1.0),
                                 dt / .01);
                const double limit =
                    std::clamp(double(config.predictive_pid_max_velocity_px_s), 0.0, 10000.0);
                for (int axis = 0; axis < 2; ++axis) {
                    const double sample = std::clamp(
                        (error[axis] - previous_error_[axis] + (*delivered)[axis] * gain[axis]) / dt, -limit,
                        limit);
                    velocity_[axis] += alpha * (sample - velocity_[axis]);
                }
            }
        }
        previous_error_ = error;
        previous_source_ = source_time;
    }
    if (previous_source_ == SteadyClock::time_point{} || now < previous_source_) return result;
    const double age = std::chrono::duration<double>(now - previous_source_).count();
    const double timeout = calibratedPredictionTimeoutSeconds(config);
    if (age > timeout || timeout <= 0) {
        reset();
        return result;
    }
    const auto delivered = motion.between(calibratedHistoryTime(config, previous_source_), now);
    if (!delivered) {
        reset();
        return result;
    }
    std::array<double, 2> error{}, rate{},
        prediction{velocity_[0] * (age + response_delay_s) - (*delivered)[0] * gain[0],
                   velocity_[1] * (age + response_delay_s) - (*delivered)[1] * gain[1]};
    if (!accepted_error && config.motion_stability_enable && config.detection_dampening_enable) {
        const double length = std::hypot(prediction[0], prediction[1]);
        const double bound = std::max(0.0F, config.motion_prediction_max_displacement_px);
        if (length > bound)
            for (auto& axis : prediction) axis *= bound / length;
    }
    for (int axis = 0; axis < 2; ++axis) {
        error[axis] = previous_error_[axis] + prediction[axis];
        rate[axis] = (velocity_[axis] + error[axis] / response) / gain[axis];
    }
    result.valid = true;
    result.counts_per_second_x = float(rate[0]);
    result.counts_per_second_y = float(rate[1]);
    result.error_x = float(error[0]);
    result.error_y = float(error[1]);
    result.velocity_x = float(velocity_[0]);
    result.velocity_y = float(velocity_[1]);
    return result;
}
}  // namespace delta
