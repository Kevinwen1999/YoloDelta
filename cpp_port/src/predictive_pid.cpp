#include "delta/predictive_pid.hpp"

#include <algorithm>
#include <cmath>

#include "delta/core.hpp"

namespace delta {

namespace {

constexpr float kReferenceDt = 0.01F;
constexpr float kMinDt = 0.001F;
constexpr float kMaxDt = 0.05F;
constexpr float kMinAdjustedAlpha = 0.05F;
constexpr float kMaxAdjustedAlpha = 0.8F;

float sanitizeNonNegative(const float value) {
    return std::isfinite(value) ? std::max(0.0F, value) : 0.0F;
}

float sanitizeUnit(const float value) {
    return std::isfinite(value) ? clamp(value, 0.0F, 1.0F) : 0.0F;
}

float sanitizeFinite(const float value, const float fallback = 0.0F) {
    return std::isfinite(value) ? value : fallback;
}

float timeAdjustedAlpha(const float alpha, const float dt) {
    const float base = sanitizeUnit(alpha);
    if (base <= 0.0F) {
        return 0.0F;
    }
    const float adjusted = 1.0F - std::pow(1.0F - base, dt / kReferenceDt);
    return clamp(adjusted, kMinAdjustedAlpha, kMaxAdjustedAlpha);
}

float ema(const float previous, const float sample, const float alpha) {
    return previous + ((sample - previous) * alpha);
}

float signOf(const float value) {
    if (value > 0.0F) {
        return 1.0F;
    }
    if (value < 0.0F) {
        return -1.0F;
    }
    return 0.0F;
}

float dampReverseMotion(
    const float sample,
    const float current_error,
    const float gate_px,
    const float reverse_scale) {
    if (std::abs(current_error) <= gate_px || sample == 0.0F || current_error == 0.0F) {
        return sample;
    }
    if (signOf(sample) != signOf(current_error)) {
        return sample * reverse_scale;
    }
    return sample;
}

float clampMagnitude(const float value, const float magnitude) {
    const float safe_magnitude = std::max(0.0F, magnitude);
    return clamp(value, -safe_magnitude, safe_magnitude);
}

float predictionLimitForError(const PredictivePidConfig& config, const float error) {
    const float scaled = std::abs(error) * sanitizeNonNegative(config.prediction_error_scale);
    const float min_px = sanitizeNonNegative(config.prediction_min_px);
    const float max_px = std::max(min_px, sanitizeNonNegative(config.prediction_max_px));
    return std::min(std::max(scaled, min_px), max_px);
}

float clampIfLimited(const float value, const float limit) {
    const float safe_limit = sanitizeNonNegative(limit);
    return safe_limit > 0.0F ? clampMagnitude(value, safe_limit) : value;
}

}  // namespace

PredictivePidConfig buildPredictivePidConfig(const RuntimeConfig& runtime) {
    PredictivePidConfig config{};
    config.kp = runtime.predictive_pid_kp;
    config.ki = runtime.predictive_pid_ki;
    config.kd = runtime.predictive_pid_kd;
    config.pred_weight_x = runtime.predictive_pid_pred_weight_x;
    config.pred_weight_y = runtime.predictive_pid_pred_weight_y;
    config.init_scale = runtime.predictive_pid_init_scale;
    config.ramp_time_s = runtime.predictive_pid_ramp_time_s;
    config.integral_limit = runtime.predictive_pid_integral_limit;
    config.derivative_limit = runtime.predictive_pid_derivative_limit;
    config.output_limit = runtime.predictive_pid_output_limit;
    config.velocity_alpha = runtime.predictive_pid_velocity_alpha;
    config.acceleration_alpha = runtime.predictive_pid_acceleration_alpha;
    config.max_velocity_px_s = runtime.predictive_pid_max_velocity_px_s;
    config.max_acceleration_px_s = runtime.predictive_pid_max_acceleration_px_s;
    config.reverse_gate_px = runtime.predictive_pid_reverse_gate_px;
    config.reverse_scale = runtime.predictive_pid_reverse_scale;
    config.prediction_error_scale = runtime.predictive_pid_prediction_error_scale;
    config.prediction_min_px = runtime.predictive_pid_prediction_min_px;
    config.prediction_max_px = runtime.predictive_pid_prediction_max_px;
    return config;
}

void PredictivePidController::configure(const PredictivePidConfig& config) {
    config_ = config;
    config_.pred_weight_x = sanitizeFinite(config_.pred_weight_x);
    config_.pred_weight_y = sanitizeFinite(config_.pred_weight_y);
    config_.init_scale = sanitizeUnit(config_.init_scale);
    config_.ramp_time_s = sanitizeNonNegative(config_.ramp_time_s);
    config_.integral_limit = sanitizeNonNegative(config_.integral_limit);
    config_.derivative_limit = sanitizeNonNegative(config_.derivative_limit);
    config_.output_limit = sanitizeNonNegative(config_.output_limit);
    config_.velocity_alpha = sanitizeUnit(config_.velocity_alpha);
    config_.acceleration_alpha = sanitizeUnit(config_.acceleration_alpha);
    config_.max_velocity_px_s = sanitizeNonNegative(config_.max_velocity_px_s);
    config_.max_acceleration_px_s = sanitizeNonNegative(config_.max_acceleration_px_s);
    config_.reverse_gate_px = sanitizeNonNegative(config_.reverse_gate_px);
    config_.reverse_scale = sanitizeUnit(config_.reverse_scale);
    config_.prediction_error_scale = sanitizeNonNegative(config_.prediction_error_scale);
    config_.prediction_min_px = sanitizeNonNegative(config_.prediction_min_px);
    config_.prediction_max_px = std::max(config_.prediction_min_px, sanitizeNonNegative(config_.prediction_max_px));
}

void PredictivePidController::reset() {
    state_ = {};
}

PredictivePidResult PredictivePidController::update(
    const float raw_error_x,
    const float raw_error_y,
    const float dt) {
    const float clamped_dt = clamp(sanitizeNonNegative(dt), kMinDt, kMaxDt);
    const bool first_update = !state_.initialized;
    PredictivePidResult result{};
    result.raw_error_x = sanitizeFinite(raw_error_x);
    result.raw_error_y = sanitizeFinite(raw_error_y);
    result.dt = clamped_dt;
    result.first_update = first_update;

    if (!first_update) {
        float raw_vx = (result.raw_error_x - state_.previous_raw_error_x + state_.previous_output_x) / clamped_dt;
        float raw_vy = (result.raw_error_y - state_.previous_raw_error_y + state_.previous_output_y) / clamped_dt;
        raw_vx = clampIfLimited(raw_vx, config_.max_velocity_px_s);
        raw_vy = clampIfLimited(raw_vy, config_.max_velocity_px_s);
        raw_vx = dampReverseMotion(raw_vx, result.raw_error_x, config_.reverse_gate_px, config_.reverse_scale);
        raw_vy = dampReverseMotion(raw_vy, result.raw_error_y, config_.reverse_gate_px, config_.reverse_scale);

        const float previous_vx = state_.velocity_x;
        const float previous_vy = state_.velocity_y;
        const float velocity_alpha = timeAdjustedAlpha(config_.velocity_alpha, clamped_dt);
        state_.velocity_x = ema(state_.velocity_x, raw_vx, velocity_alpha);
        state_.velocity_y = ema(state_.velocity_y, raw_vy, velocity_alpha);

        float raw_ax = (state_.velocity_x - previous_vx) / clamped_dt;
        float raw_ay = (state_.velocity_y - previous_vy) / clamped_dt;
        raw_ax = clampIfLimited(raw_ax, config_.max_acceleration_px_s);
        raw_ay = clampIfLimited(raw_ay, config_.max_acceleration_px_s);
        raw_ax = dampReverseMotion(raw_ax, result.raw_error_x, config_.reverse_gate_px, config_.reverse_scale);
        raw_ay = dampReverseMotion(raw_ay, result.raw_error_y, config_.reverse_gate_px, config_.reverse_scale);

        const float acceleration_alpha = timeAdjustedAlpha(config_.acceleration_alpha, clamped_dt);
        state_.acceleration_x = ema(state_.acceleration_x, raw_ax, acceleration_alpha);
        state_.acceleration_y = ema(state_.acceleration_y, raw_ay, acceleration_alpha);

        result.prediction_x = (state_.velocity_x * clamped_dt)
            + (0.5F * state_.acceleration_x * clamped_dt * clamped_dt);
        result.prediction_y = (state_.velocity_y * clamped_dt)
            + (0.5F * state_.acceleration_y * clamped_dt * clamped_dt);
        result.prediction_x = clampMagnitude(result.prediction_x, predictionLimitForError(config_, result.raw_error_x));
        result.prediction_y = clampMagnitude(result.prediction_y, predictionLimitForError(config_, result.raw_error_y));
    }

    result.velocity_x = state_.velocity_x;
    result.velocity_y = state_.velocity_y;
    result.acceleration_x = state_.acceleration_x;
    result.acceleration_y = state_.acceleration_y;
    result.fused_error_x = result.raw_error_x + (result.prediction_x * config_.pred_weight_x);
    result.fused_error_y = result.raw_error_y + (result.prediction_y * config_.pred_weight_y);

    result.ramp_scale = 1.0F;
    if (config_.ramp_time_s > 0.0F && state_.ramp_elapsed_s < config_.ramp_time_s) {
        const float progress = clamp(state_.ramp_elapsed_s / config_.ramp_time_s, 0.0F, 1.0F);
        result.ramp_scale = config_.init_scale + ((1.0F - config_.init_scale) * progress);
    }

    const float real_kp = config_.kp * result.ramp_scale;
    result.p_x = result.fused_error_x * real_kp;
    result.p_y = result.fused_error_y * real_kp;

    state_.integral_x = clampIfLimited(
        state_.integral_x + (result.fused_error_x * clamped_dt * config_.ki),
        config_.integral_limit);
    state_.integral_y = clampIfLimited(
        state_.integral_y + (result.fused_error_y * clamped_dt * config_.ki),
        config_.integral_limit);
    result.i_x = state_.integral_x;
    result.i_y = state_.integral_y;

    if (!first_update) {
        result.d_x = ((result.fused_error_x - state_.previous_fused_error_x) / clamped_dt) * config_.kd;
        result.d_y = ((result.fused_error_y - state_.previous_fused_error_y) / clamped_dt) * config_.kd;
        result.d_x = clampIfLimited(result.d_x, config_.derivative_limit);
        result.d_y = clampIfLimited(result.d_y, config_.derivative_limit);
    }

    result.output_x = clampIfLimited(result.p_x + result.i_x + result.d_x, config_.output_limit);
    result.output_y = clampIfLimited(result.p_y + result.i_y + result.d_y, config_.output_limit);

    state_.initialized = true;
    state_.previous_raw_error_x = result.raw_error_x;
    state_.previous_raw_error_y = result.raw_error_y;
    state_.previous_fused_error_x = result.fused_error_x;
    state_.previous_fused_error_y = result.fused_error_y;
    state_.ramp_elapsed_s += clamped_dt;
    return result;
}

void PredictivePidController::commitOutput(const float output_x, const float output_y) {
    state_.previous_output_x = sanitizeFinite(output_x);
    state_.previous_output_y = sanitizeFinite(output_y);
}

PredictivePidSnapshot PredictivePidController::snapshot() const {
    return state_;
}

}  // namespace delta
