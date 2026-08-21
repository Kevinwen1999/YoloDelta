#include "delta/sigma_drift.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

#include "delta/core.hpp"

namespace delta {

namespace {

constexpr float kMinDt = 0.001F;
constexpr float kMaxDt = 0.05F;
constexpr float kMinMovementDistancePx = 1.0F;

float finiteOr(const float value, const float fallback) {
    return std::isfinite(value) ? value : fallback;
}

float nonNegative(const float value, const float fallback = 0.0F) {
    return std::max(0.0F, finiteOr(value, fallback));
}

float normalCdf(const float value) {
    return 0.5F * (1.0F + std::erf(value / std::numbers::sqrt2_v<float>));
}

float lognormalCdf(const float time_ms, const float start_ms, const float mu, const float sigma) {
    if (time_ms <= start_ms || sigma <= 0.0F) {
        return 0.0F;
    }
    return normalCdf((std::log(time_ms - start_ms) - mu) / sigma);
}

float lognormalPdf(const float time_ms, const float start_ms, const float mu, const float sigma) {
    if (time_ms <= start_ms || sigma <= 0.0F) {
        return 0.0F;
    }
    const float elapsed_ms = time_ms - start_ms;
    const float z = (std::log(elapsed_ms) - mu) / sigma;
    return std::exp(-0.5F * z * z)
        / (sigma * std::sqrt(2.0F * std::numbers::pi_v<float>) * elapsed_ms);
}

float curvatureProfile(const float progress) {
    if (progress <= 0.0F || progress >= 1.0F) {
        return 0.0F;
    }
    constexpr float kPeakNormalization = 0.4F * 0.4F * 0.6F * 0.6F * 0.6F;
    const float remaining = 1.0F - progress;
    return (progress * progress * remaining * remaining * remaining) / kPeakNormalization;
}

float directionFactor(const float direction_x, const float direction_y) {
    return 0.5F + (0.8F * std::abs(direction_y)) - (0.15F * std::abs(direction_x));
}

float effectiveTargetWidth(
    const SigmaDriftConfig& config,
    const float box_width_px,
    const float box_height_px) {
    const bool valid_width = std::isfinite(box_width_px) && box_width_px > 0.0F;
    const bool valid_height = std::isfinite(box_height_px) && box_height_px > 0.0F;
    if (valid_width && valid_height) {
        return std::max(1.0F, std::min(box_width_px, box_height_px));
    }
    if (valid_width) {
        return std::max(1.0F, box_width_px);
    }
    if (valid_height) {
        return std::max(1.0F, box_height_px);
    }
    return std::max(1.0F, config.target_width_px);
}

}  // namespace

SigmaDriftConfig buildSigmaDriftConfig(const RuntimeConfig& runtime) {
    SigmaDriftConfig config{};
    config.enable = runtime.predictive_pid_human_motion_enable;
    config.fitts_a_ms = runtime.predictive_pid_human_fitts_a_ms;
    config.fitts_b_ms = runtime.predictive_pid_human_fitts_b_ms;
    config.target_width_px = runtime.predictive_pid_human_target_width_px;
    config.overshoot_probability = runtime.predictive_pid_human_overshoot_probability;
    config.curvature_scale = runtime.predictive_pid_human_curvature_scale;
    config.ou_sigma = runtime.predictive_pid_human_ou_sigma;
    config.tremor_amplitude_max_px = runtime.predictive_pid_human_tremor_amplitude_px;
    config.signal_noise = runtime.predictive_pid_human_signal_noise;
    config.sample_interval_mean_ms = runtime.predictive_pid_human_sample_interval_ms;
    return config;
}

void SigmaDriftShaper::configure(const SigmaDriftConfig& config) {
    config_ = config;
    config_.fitts_a_ms = nonNegative(config_.fitts_a_ms, 50.0F);
    config_.fitts_b_ms = nonNegative(config_.fitts_b_ms, 150.0F);
    config_.target_width_px = std::max(1.0F, nonNegative(config_.target_width_px, 20.0F));
    config_.undershoot_min = nonNegative(config_.undershoot_min, 0.92F);
    config_.undershoot_max = std::max(config_.undershoot_min, nonNegative(config_.undershoot_max, 0.97F));
    config_.peak_time_ratio = clamp(finiteOr(config_.peak_time_ratio, 0.35F), 0.05F, 0.95F);
    config_.primary_sigma_min = std::max(0.01F, nonNegative(config_.primary_sigma_min, 0.18F));
    config_.primary_sigma_max = std::max(config_.primary_sigma_min, nonNegative(config_.primary_sigma_max, 0.28F));
    config_.overshoot_probability = clamp(finiteOr(config_.overshoot_probability, 0.15F), 0.0F, 1.0F);
    config_.overshoot_min = nonNegative(config_.overshoot_min, 1.02F);
    config_.overshoot_max = std::max(config_.overshoot_min, nonNegative(config_.overshoot_max, 1.08F));
    config_.correction_sigma_min = std::max(0.01F, nonNegative(config_.correction_sigma_min, 0.12F));
    config_.correction_sigma_max = std::max(config_.correction_sigma_min, nonNegative(config_.correction_sigma_max, 0.20F));
    config_.second_correction_probability = clamp(finiteOr(config_.second_correction_probability, 0.25F), 0.0F, 1.0F);
    config_.curvature_scale = nonNegative(config_.curvature_scale, 0.025F);
    config_.ou_theta = nonNegative(config_.ou_theta, 3.5F);
    config_.ou_sigma = nonNegative(config_.ou_sigma, 1.2F);
    config_.tremor_frequency_min_hz = nonNegative(config_.tremor_frequency_min_hz, 8.0F);
    config_.tremor_frequency_max_hz = std::max(
        config_.tremor_frequency_min_hz,
        nonNegative(config_.tremor_frequency_max_hz, 12.0F));
    config_.tremor_amplitude_min_px = nonNegative(config_.tremor_amplitude_min_px, 0.15F);
    config_.tremor_amplitude_max_px = std::max(
        config_.tremor_amplitude_min_px,
        nonNegative(config_.tremor_amplitude_max_px, 0.55F));
    config_.signal_noise = nonNegative(config_.signal_noise, 0.04F);
    config_.sample_interval_mean_ms = std::max(0.1F, nonNegative(config_.sample_interval_mean_ms, 7.8F));
    config_.gamma_shape = std::max(0.1F, nonNegative(config_.gamma_shape, 3.5F));
    reset();
}

void SigmaDriftShaper::reset() {
    state_ = {};
    if (config_.random_seed != 0) {
        random_.seed(config_.random_seed);
    } else {
        std::random_device source;
        random_.seed((static_cast<std::uint64_t>(source()) << 32U) ^ source());
    }
}

float SigmaDriftShaper::uniform(const float low, const float high) {
    return std::uniform_real_distribution<float>(low, high)(random_);
}

float SigmaDriftShaper::normal(const float mean, const float deviation) {
    return std::normal_distribution<float>(mean, deviation)(random_);
}

float SigmaDriftShaper::gamma(const float shape, const float scale) {
    return std::gamma_distribution<float>(shape, scale)(random_);
}

bool SigmaDriftShaper::beginPlan(const float target_x, const float target_y, const float target_width_px) {
    state_ = {};
    state_.target_x = finiteOr(target_x, 0.0F);
    state_.target_y = finiteOr(target_y, 0.0F);
    state_.distance = std::hypot(state_.target_x, state_.target_y);
    if (state_.distance < kMinMovementDistancePx) {
        return false;
    }

    state_.active = true;
    state_.direction_x = state_.target_x / state_.distance;
    state_.direction_y = state_.target_y / state_.distance;
    state_.normal_x = -state_.direction_y;
    state_.normal_y = state_.direction_x;

    const float safe_width = std::max(1.0F, target_width_px);
    const float index_of_difficulty = std::log2((state_.distance / safe_width) + 1.0F);
    state_.movement_time_ms = std::max(
        80.0F,
        (config_.fitts_a_ms + (config_.fitts_b_ms * index_of_difficulty))
            * std::exp(normal(0.0F, 0.08F)));
    state_.total_time_ms = state_.movement_time_ms * 1.15F;

    const bool overshoot = uniform(0.0F, 1.0F) < config_.overshoot_probability;
    const float reach = overshoot
        ? uniform(config_.overshoot_min, config_.overshoot_max)
        : uniform(config_.undershoot_min, config_.undershoot_max);
    state_.primary_distance = state_.distance * reach;
    state_.primary_sigma = uniform(config_.primary_sigma_min, config_.primary_sigma_max);
    const float peak_ratio_low = std::max(0.01F, config_.peak_time_ratio - 0.03F);
    const float peak_ratio_high = std::max(peak_ratio_low, config_.peak_time_ratio + 0.03F);
    const float peak_time_ms = state_.movement_time_ms * uniform(peak_ratio_low, peak_ratio_high);
    state_.primary_mu = std::log(std::max(0.001F, peak_time_ms))
        + (state_.primary_sigma * state_.primary_sigma);

    const float remaining = state_.distance - state_.primary_distance;
    if (std::abs(remaining) > 0.5F) {
        const float direction = remaining > 0.0F ? 1.0F : -1.0F;
        Correction& first = state_.corrections[state_.correction_count++];
        first.distance = std::abs(remaining) * uniform(0.88F, 1.02F);
        first.sigma = uniform(config_.correction_sigma_min, config_.correction_sigma_max);
        const float correction_peak_ms = state_.movement_time_ms * uniform(0.12F, 0.18F);
        first.start_ms = state_.movement_time_ms * uniform(0.55F, 0.68F);
        first.mu = std::log(std::max(0.001F, correction_peak_ms)) + (first.sigma * first.sigma);
        first.direction = direction;

        const float left = remaining - (first.distance * direction);
        if (std::abs(left) > 0.3F
            && uniform(0.0F, 1.0F) < config_.second_correction_probability) {
            Correction& second = state_.corrections[state_.correction_count++];
            second.distance = std::abs(left) * uniform(0.85F, 1.05F);
            second.sigma = uniform(0.10F, 0.16F);
            const float second_peak_ms = state_.movement_time_ms * uniform(0.08F, 0.12F);
            second.start_ms = state_.movement_time_ms * uniform(0.78F, 0.88F);
            second.mu = std::log(std::max(0.001F, second_peak_ms)) + (second.sigma * second.sigma);
            second.direction = left > 0.0F ? 1.0F : -1.0F;
        }
    }

    state_.curvature_amplitude = state_.distance * config_.curvature_scale
        * directionFactor(state_.direction_x, state_.direction_y) * normal(0.0F, 1.0F);
    state_.tremor_frequency_hz = uniform(config_.tremor_frequency_min_hz, config_.tremor_frequency_max_hz);
    state_.tremor_amplitude_px = uniform(config_.tremor_amplitude_min_px, config_.tremor_amplitude_max_px);
    state_.tremor_phase_x = uniform(0.0F, 2.0F * std::numbers::pi_v<float>);
    state_.tremor_phase_y = uniform(0.0F, 2.0F * std::numbers::pi_v<float>);
    return true;
}

bool SigmaDriftShaper::shouldRetarget(
    const float target_x,
    const float target_y,
    const float target_width_px) const {
    if (!state_.active) {
        return true;
    }
    const float expected_x = state_.target_x - state_.committed_x;
    const float expected_y = state_.target_y - state_.committed_y;
    const float drift = std::hypot(target_x - expected_x, target_y - expected_y);
    const float retarget_threshold = std::max(12.0F, target_width_px * 0.75F);
    const float direction_dot = (target_x * expected_x) + (target_y * expected_y);
    return drift > retarget_threshold || direction_dot < 0.0F;
}

SigmaDriftResult SigmaDriftShaper::update(
    const float controller_output_x,
    const float controller_output_y,
    const float target_error_x,
    const float target_error_y,
    const float dt,
    const float target_box_width_px,
    const float target_box_height_px) {
    SigmaDriftResult result{};
    const float safe_controller_x = finiteOr(controller_output_x, 0.0F);
    const float safe_controller_y = finiteOr(controller_output_y, 0.0F);
    if (!config_.enable) {
        result.output_x = safe_controller_x;
        result.output_y = safe_controller_y;
        return result;
    }

    const float controller_magnitude = std::hypot(safe_controller_x, safe_controller_y);
    if (controller_magnitude <= 1e-6F) {
        state_ = {};
        return result;
    }

    const float safe_target_x = finiteOr(target_error_x, 0.0F);
    const float safe_target_y = finiteOr(target_error_y, 0.0F);
    const float target_width = effectiveTargetWidth(
        config_,
        target_box_width_px,
        target_box_height_px);
    if (shouldRetarget(safe_target_x, safe_target_y, target_width)
        && !beginPlan(safe_target_x, safe_target_y, target_width)) {
        result.output_x = safe_controller_x;
        result.output_y = safe_controller_y;
        return result;
    }

    const float clamped_dt = clamp(nonNegative(dt), kMinDt, kMaxDt);
    state_.elapsed_ms += clamped_dt * 1000.0F;
    result.active = state_.active;
    result.movement_time_ms = state_.movement_time_ms;
    result.progress = clamp(state_.elapsed_ms / std::max(1.0F, state_.total_time_ms), 0.0F, 1.0F);

    if (state_.elapsed_ms + 1e-4F < state_.next_sample_ms) {
        return result;
    }
    result.sample_emitted = true;
    const float gamma_scale = config_.sample_interval_mean_ms / config_.gamma_shape;
    state_.next_sample_ms = state_.elapsed_ms
        + clamp(gamma(config_.gamma_shape, gamma_scale), 2.0F, 25.0F);

    const float primary_progress = lognormalCdf(
        state_.elapsed_ms,
        0.0F,
        state_.primary_mu,
        state_.primary_sigma);
    float longitudinal_position = state_.primary_distance * primary_progress;
    float speed_px_per_ms = state_.primary_distance * lognormalPdf(
        state_.elapsed_ms,
        0.0F,
        state_.primary_mu,
        state_.primary_sigma);
    for (int index = 0; index < state_.correction_count; ++index) {
        const Correction& correction = state_.corrections[index];
        const float correction_progress = lognormalCdf(
            state_.elapsed_ms,
            correction.start_ms,
            correction.mu,
            correction.sigma);
        longitudinal_position += correction.direction * correction.distance * correction_progress;
        speed_px_per_ms += correction.distance * lognormalPdf(
            state_.elapsed_ms,
            correction.start_ms,
            correction.mu,
            correction.sigma);
    }

    const float lateral_position = state_.curvature_amplitude * curvatureProfile(primary_progress);
    float desired_x = (state_.direction_x * longitudinal_position) + (state_.normal_x * lateral_position);
    float desired_y = (state_.direction_y * longitudinal_position) + (state_.normal_y * lateral_position);

    state_.ou_x += (-config_.ou_theta * state_.ou_x * clamped_dt)
        + (config_.ou_sigma * std::sqrt(clamped_dt) * normal(0.0F, 1.0F));
    state_.ou_y += (-config_.ou_theta * state_.ou_y * clamped_dt)
        + (config_.ou_sigma * std::sqrt(clamped_dt) * normal(0.0F, 1.0F));

    const float elapsed_s = state_.elapsed_ms / 1000.0F;
    const float tremor_gain = 1.0F / (1.0F + (speed_px_per_ms * 0.3F));
    const float phase = 2.0F * std::numbers::pi_v<float> * state_.tremor_frequency_hz * elapsed_s;
    const float tremor_x = state_.tremor_amplitude_px * tremor_gain
        * std::sin(phase + state_.tremor_phase_x);
    const float tremor_y = state_.tremor_amplitude_px * tremor_gain
        * std::sin(phase + state_.tremor_phase_y);
    const float signal_noise_x = config_.signal_noise * speed_px_per_ms * normal(0.0F, 1.0F);
    const float signal_noise_y = config_.signal_noise * speed_px_per_ms * normal(0.0F, 1.0F);
    desired_x += state_.ou_x + tremor_x + signal_noise_x;
    desired_y += state_.ou_y + tremor_y + signal_noise_y;

    result.output_x = desired_x - state_.committed_x;
    result.output_y = desired_y - state_.committed_y;
    const float shaped_magnitude = std::hypot(result.output_x, result.output_y);
    if (shaped_magnitude > controller_magnitude && shaped_magnitude > 0.0F) {
        const float scale = controller_magnitude / shaped_magnitude;
        result.output_x *= scale;
        result.output_y *= scale;
    }

    if (state_.elapsed_ms >= state_.total_time_ms) {
        state_.active = false;
    }
    return result;
}

void SigmaDriftShaper::commitOutput(const float output_x, const float output_y) {
    if (!config_.enable) {
        return;
    }
    state_.committed_x += finiteOr(output_x, 0.0F);
    state_.committed_y += finiteOr(output_y, 0.0F);
}

}  // namespace delta
