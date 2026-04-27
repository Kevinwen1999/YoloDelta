#pragma once

#include <utility>

#include "delta/config.hpp"

namespace delta {

struct PredictivePidConfig {
    float kp = 0.45F;
    float ki = 0.02F;
    float kd = 0.04F;
    float pred_weight_x = 0.5F;
    float pred_weight_y = 0.1F;
    float init_scale = 0.6F;
    float ramp_time_s = 0.5F;
    float integral_limit = 100.0F;
    float derivative_limit = 50.0F;
    float output_limit = 150.0F;
    float velocity_alpha = 0.25F;
    float acceleration_alpha = 0.15F;
    float max_velocity_px_s = 3000.0F;
    float max_acceleration_px_s = 5000.0F;
    float reverse_gate_px = 5.0F;
    float reverse_scale = 0.1F;
    float prediction_error_scale = 1.5F;
    float prediction_min_px = 30.0F;
    float prediction_max_px = 60.0F;
    bool latency_comp_enable = true;
    bool latency_auto_enable = true;
    float latency_bias_s = 0.0F;
    float latency_max_s = 0.050F;
    bool deadzone_enable = false;
    float deadzone_enter_px = 1.0F;
    float deadzone_exit_px = 1.5F;
};

struct PredictivePidResult {
    float output_x = 0.0F;
    float output_y = 0.0F;
    float raw_error_x = 0.0F;
    float raw_error_y = 0.0F;
    float prediction_x = 0.0F;
    float prediction_y = 0.0F;
    float fused_error_x = 0.0F;
    float fused_error_y = 0.0F;
    float control_error_x = 0.0F;
    float control_error_y = 0.0F;
    float velocity_x = 0.0F;
    float velocity_y = 0.0F;
    float acceleration_x = 0.0F;
    float acceleration_y = 0.0F;
    float p_x = 0.0F;
    float p_y = 0.0F;
    float i_x = 0.0F;
    float i_y = 0.0F;
    float d_x = 0.0F;
    float d_y = 0.0F;
    float ramp_scale = 1.0F;
    float dt = 0.0F;
    float latency_s = 0.0F;
    float horizon_s = 0.0F;
    bool deadzone_active_x = false;
    bool deadzone_active_y = false;
    bool first_update = false;
};

struct PredictivePidSnapshot {
    bool initialized = false;
    float previous_raw_error_x = 0.0F;
    float previous_raw_error_y = 0.0F;
    float previous_fused_error_x = 0.0F;
    float previous_fused_error_y = 0.0F;
    float previous_control_error_x = 0.0F;
    float previous_control_error_y = 0.0F;
    float previous_output_x = 0.0F;
    float previous_output_y = 0.0F;
    float velocity_x = 0.0F;
    float velocity_y = 0.0F;
    float acceleration_x = 0.0F;
    float acceleration_y = 0.0F;
    float integral_x = 0.0F;
    float integral_y = 0.0F;
    float ramp_elapsed_s = 0.0F;
    bool deadzone_active_x = false;
    bool deadzone_active_y = false;
};

PredictivePidConfig buildPredictivePidConfig(const RuntimeConfig& runtime);

class PredictivePidController {
public:
    void configure(const PredictivePidConfig& config);
    void reset();
    PredictivePidResult update(float raw_error_x, float raw_error_y, float dt, float measured_latency_s = 0.0F);
    void commitOutput(float output_x, float output_y);
    PredictivePidSnapshot snapshot() const;

private:
    PredictivePidConfig config_{};
    PredictivePidSnapshot state_{};
};

}  // namespace delta
