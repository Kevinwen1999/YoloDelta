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
