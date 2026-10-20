#include "benchmark.hpp"

#include <cmath>
#include <stdexcept>
#include <type_traits>

namespace delta::bench {
namespace {
template<class T> Json encode(const T& value) {
    if constexpr (std::is_enum_v<T>) return static_cast<int>(value);
    else return value;
}
template<class T> void decode(const Json& value, T& target) {
    if constexpr (std::is_enum_v<T> || std::is_integral_v<T> && !std::is_same_v<T, bool>) {
        if (!value.is_number_integer()) throw std::invalid_argument("Expected integer configuration value");
        const auto number = value.get<std::int64_t>();
        if (number < (std::is_enum_v<T> ? 0 : -10000000) || number > 10000000) throw std::invalid_argument("Configuration integer out of range");
        target = static_cast<T>(number);
    } else if constexpr (std::is_floating_point_v<T>) {
        if (!value.is_number()) throw std::invalid_argument("Expected numeric configuration value");
        const double number = value.get<double>();
        if (!std::isfinite(number) || std::abs(number) > 1e8) throw std::invalid_argument("Non-finite or excessive configuration value");
        target = static_cast<T>(number);
    } else {
        target = value.get<T>();
    }
}
}

Json runtimeJson(const RuntimeConfig& config) {
    Json out = Json::object();
#define DELTA_BENCH_FIELD(name) out[#name] = encode(config.name);
#include "runtime_fields.inc"
#undef DELTA_BENCH_FIELD
    return out;
}

RuntimeConfig runtimeFromJson(const Json& input) {
    if (!input.is_object()) throw std::invalid_argument("Runtime config must be a JSON object");
    RuntimeConfig config;
    const auto known = runtimeJson(config);
    for (const auto& [key, value] : input.items()) {
        if (!known.contains(key)) throw std::invalid_argument("Unknown runtime field: " + key);
    }
#define DELTA_BENCH_FIELD(name) if (input.contains(#name)) decode(input.at(#name), config.name);
#include "runtime_fields.inc"
#undef DELTA_BENCH_FIELD
    if (static_cast<int>(config.tracking_strategy) > 6 || static_cast<int>(config.aim_mode) > 2
        || static_cast<int>(config.left_hold_engage_button) > 3 || static_cast<int>(config.recoil_mode) > 1
        || static_cast<int>(config.mouse_output_method) > 1)
        throw std::invalid_argument("Unknown configuration enum value");
    if (config.target_prediction_timeout_ms < 0 || config.target_max_lost_frames < 0
        || config.motion_max_speed_px_s<=0 || config.motion_residual_scale<=0
        || config.motion_confirmation_window_ms<=0 || config.motion_prediction_timeout_ms<=0
        || config.motion_match_margin<0 || config.motion_prediction_max_displacement_px<=0
        || config.command_max_observation_age_ms<=0
        || config.raw_max_step_x < 0 || config.raw_max_step_y < 0
        || config.detection_box_scale <= 0 || config.adaptive_capture_crop_min_size <= 0
        || config.adaptive_capture_crop_max_size < config.adaptive_capture_crop_min_size)
        throw std::invalid_argument("Invalid timeout, movement limit, box scale or crop bounds");
    if (config.calibrated_pixels_per_count_x < .001F || config.calibrated_pixels_per_count_x > 100
        || config.calibrated_pixels_per_count_y < .001F || config.calibrated_pixels_per_count_y > 100
        || config.calibrated_response_time_s < .02F || config.calibrated_response_time_s > 1
        || config.calibrated_response_delay_ms<0 || config.calibrated_response_delay_ms>50)
        throw std::invalid_argument("Invalid calibrated response: pixels/count .001..100, response time .02..1s");
    return config;
}

void validateOptions(const Options& options) {
    if (options.suite != "quick" && options.suite != "sweep" && options.suite != "speed" && options.suite != "speed_sizes") throw std::invalid_argument("Suite must be quick, sweep, speed or speed_sizes");
    if (options.seed == 0 || options.seed > 0xffffffffULL) throw std::invalid_argument("Seed must be between 1 and 4294967295");
    if (options.repeats < 1 || options.repeats > 100) throw std::invalid_argument("Repeats must be between 1 and 100");
    if (!std::isfinite(options.loss_deadline_ms) || options.loss_deadline_ms <= 0 || options.loss_deadline_ms > 10000)
        throw std::invalid_argument("Loss deadline must be in (0, 10000] milliseconds");
}
} // namespace delta::bench
