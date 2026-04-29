#pragma once

#include <optional>
#include <memory>
#include <vector>

#include "delta/config.hpp"
#include "delta/core.hpp"

namespace delta {

class PIDController {
public:
    PIDController() = default;
    PIDController(float kp, float ki, float kd);

    void configure(float kp, float ki, float kd, float integral_limit, float anti_windup_gain, float derivative_alpha,
                   float output_limit);
    void reset();
    void clearIntegral();
    float update(float setpoint, float measurement, float dt, bool integrate);

private:
    float kp_ = 0.0F;
    float ki_ = 0.0F;
    float kd_ = 0.0F;
    float integral_limit_ = 0.0F;
    float anti_windup_gain_ = 0.0F;
    float derivative_alpha_ = 0.0F;
    float output_limit_ = 0.0F;
    float integral_ = 0.0F;
    float prev_error_ = 0.0F;
    float prev_derivative_ = 0.0F;
    bool initialized_ = false;
};

struct PIDSettleConfig {
    bool enable = true;
    float error_px = 4.0F;
    float threshold_min_scale = 1.6F;
    float threshold_max_scale = 2.7F;
    int stable_frames = 2;
    float error_delta_px = 3.0F;
    float pre_output_scale = 0.5F;
};

struct PIDSettleState {
    bool settled = false;
    int stable_frame_count = 0;
    float previous_error_metric_px = 0.0F;
    bool initialized = false;

    void reset();
};

struct PIDSettleDecision {
    bool settled = true;
    bool integrate = true;
    bool just_unsettled = false;
    float pid_output_scale = 1.0F;
    float error_metric_px = 0.0F;
    float dynamic_threshold_px = 0.0F;
};

float pidSettleErrorMetricPx(float predicted_x, float aim_y, float center_x, float center_y);
float pidSettleDynamicThresholdPx(const PIDSettleConfig& config, float box_width_px, float capture_width_px);
PIDSettleDecision updatePidSettleState(
    PIDSettleState& state,
    const PIDSettleConfig& config,
    float error_metric_px,
    float box_width_px,
    float capture_width_px);

struct LegacyPidConfig {
    float kp = 0.0F;
    float ki = 0.0F;
    float kd = 0.0F;
    float lock_error_px = 4.0F;
    float speed_multiplier = 1.0F;
    float threshold_min_scale = 1.6F;
    float threshold_max_scale = 2.7F;
    float transition_sharpness = 5.0F;
    float transition_midpoint = 0.0F;
    int stable_frames = 2;
    float error_delta_px = 3.0F;
    float prelock_scale = 0.5F;
};

struct LegacyPidAxisState {
    float previous_error_px = 0.0F;
    float integral_accumulator = 0.0F;
    float velocity = 0.0F;
    int stable_frame_count = 0;
    bool locked = false;

    void reset();
};

struct LegacyPidAxisResult {
    float output = 0.0F;
    float raw_output = 0.0F;
    float proportional = 0.0F;
    float integral = 0.0F;
    float derivative = 0.0F;
    float velocity = 0.0F;
    bool locked = false;
    bool just_unlocked = false;
    float dynamic_threshold_px = 0.0F;
    float error_px = 0.0F;
};

struct LegacyPidStatus {
    bool settled = false;
    float error_metric_px = 0.0F;
    float threshold_px = 0.0F;
    float speed = 0.0F;
};

float legacyPidDynamicThresholdPx(const LegacyPidConfig& config, float box_width_px, float capture_width_px);
LegacyPidAxisResult updateLegacyPidAxis(
    LegacyPidAxisState& state,
    const LegacyPidConfig& config,
    float error_px,
    float dt,
    float box_width_px,
    float capture_width_px);
LegacyPidStatus makeLegacyPidStatus(const LegacyPidAxisResult& x_axis, const LegacyPidAxisResult& y_axis);

class ITargetTracker {
public:
    virtual ~ITargetTracker() = default;
    virtual void configure(float velocity_alpha) = 0;
    virtual void reset() = 0;
    virtual void predict(float dt) = 0;
    virtual void update(float x, float y) = 0;
    virtual TrackerState state() const = 0;
    virtual float feedforwardScale() const = 0;
    virtual bool initialized() const = 0;
};

class ObservedMotionTracker final : public ITargetTracker {
public:
    explicit ObservedMotionTracker(TrackingStrategy mode, float velocity_alpha);

    void configure(float velocity_alpha) override;
    void reset() override;
    void predict(float dt) override;
    void update(float x, float y) override;
    TrackerState state() const override;
    float feedforwardScale() const override;
    bool initialized() const override;

private:
    TrackingStrategy mode_ = TrackingStrategy::RawDelta;
    float velocity_alpha_ = 1.0F;
    TrackerState state_{};
    float raw_x_ = 0.0F;
    float raw_y_ = 0.0F;
    float last_dt_ = 1.0F / 240.0F;
    bool initialized_ = false;
    int measurement_updates_ = 0;
};

std::unique_ptr<ITargetTracker> makeTargetTracker(TrackingStrategy strategy, float velocity_alpha);

Detection scaleDetectionBox(const Detection& detection, float box_scale, const CaptureRegion& bounds);
std::pair<float, float> detectionAimPoint(
    const Detection& detection,
    float body_y_ratio,
    float head_x_ratio,
    float head_y_ratio);

struct AimCandidatePool {
    std::vector<Detection> candidates;
    bool using_head_candidates = false;
};

AimCandidatePool buildAimCandidatePool(
    const std::vector<Detection>& detections,
    AimMode aim_mode,
    float body_y_ratio,
    float head_x_ratio,
    float head_y_ratio);

void resetAimTrackingState(
    int& lost_frames,
    int& active_target_cls,
    float& last_box_w,
    float& last_box_h,
    std::optional<std::array<int, 4>>& last_target_bbox,
    SteadyClock::time_point& last_pid_tick,
    SteadyClock::time_point& last_track_tick);

struct StickyTargetPick {
    std::optional<Detection> detection;
    bool switched = false;
};

StickyTargetPick pickStickyTarget(
    const std::vector<Detection>& detections,
    int center_x,
    int center_y,
    std::optional<std::pair<float, float>> locked_point,
    float sticky_bias_px);

}  // namespace delta
