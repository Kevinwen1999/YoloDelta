#pragma once

#include <array>
#include <cstdint>
#include <random>

#include "delta/config.hpp"

namespace delta {

struct SigmaDriftConfig {
    bool enable = false;
    float fitts_a_ms = 50.0F;
    float fitts_b_ms = 150.0F;
    float target_width_px = 20.0F;

    float undershoot_min = 0.92F;
    float undershoot_max = 0.97F;
    float peak_time_ratio = 0.35F;
    float primary_sigma_min = 0.18F;
    float primary_sigma_max = 0.28F;

    float overshoot_probability = 0.15F;
    float overshoot_min = 1.02F;
    float overshoot_max = 1.08F;
    float correction_sigma_min = 0.12F;
    float correction_sigma_max = 0.20F;
    float second_correction_probability = 0.25F;

    float curvature_scale = 0.025F;
    float ou_theta = 3.5F;
    float ou_sigma = 1.2F;
    float tremor_frequency_min_hz = 8.0F;
    float tremor_frequency_max_hz = 12.0F;
    float tremor_amplitude_min_px = 0.15F;
    float tremor_amplitude_max_px = 0.55F;
    float signal_noise = 0.04F;
    float sample_interval_mean_ms = 7.8F;
    float gamma_shape = 3.5F;
    std::uint64_t random_seed = 0;
};

struct SigmaDriftResult {
    float output_x = 0.0F;
    float output_y = 0.0F;
    float movement_time_ms = 0.0F;
    float progress = 0.0F;
    bool active = false;
    bool sample_emitted = false;
};

SigmaDriftConfig buildSigmaDriftConfig(const RuntimeConfig& runtime);

class SigmaDriftShaper {
public:
    void configure(const SigmaDriftConfig& config);
    void reset();
    SigmaDriftResult update(
        float controller_output_x,
        float controller_output_y,
        float target_error_x,
        float target_error_y,
        float dt,
        float target_box_width_px = 0.0F,
        float target_box_height_px = 0.0F);
    void commitOutput(float output_x, float output_y);

private:
    struct Correction {
        float distance = 0.0F;
        float start_ms = 0.0F;
        float mu = 0.0F;
        float sigma = 0.0F;
        float direction = 1.0F;
    };

    struct State {
        bool active = false;
        float elapsed_ms = 0.0F;
        float movement_time_ms = 0.0F;
        float total_time_ms = 0.0F;
        float next_sample_ms = 0.0F;
        float target_x = 0.0F;
        float target_y = 0.0F;
        float distance = 0.0F;
        float direction_x = 0.0F;
        float direction_y = 0.0F;
        float normal_x = 0.0F;
        float normal_y = 0.0F;
        float primary_distance = 0.0F;
        float primary_mu = 0.0F;
        float primary_sigma = 0.0F;
        float curvature_amplitude = 0.0F;
        std::array<Correction, 2> corrections{};
        int correction_count = 0;
        float tremor_frequency_hz = 0.0F;
        float tremor_amplitude_px = 0.0F;
        float tremor_phase_x = 0.0F;
        float tremor_phase_y = 0.0F;
        float ou_x = 0.0F;
        float ou_y = 0.0F;
        float committed_x = 0.0F;
        float committed_y = 0.0F;
    };

    float uniform(float low, float high);
    float normal(float mean, float deviation);
    float gamma(float shape, float scale);
    bool beginPlan(float target_x, float target_y, float target_width_px);
    bool shouldRetarget(float target_x, float target_y, float target_width_px) const;

    SigmaDriftConfig config_{};
    State state_{};
    std::mt19937_64 random_{};
};

}  // namespace delta
