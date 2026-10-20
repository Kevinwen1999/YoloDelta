#include "control_response.hpp"

#include <algorithm>
#include <cmath>
#include <deque>

#include "delta/app.hpp"
#include "delta/dispatch.hpp"

namespace delta::bench {
namespace {
auto at(double ms) {
    return SteadyClock::time_point(std::chrono::duration_cast<SteadyClock::duration>(
        std::chrono::duration<double, std::milli>(10000 + ms)));
}
class Plant final : public ITrackingReplay {
public:
    Plant(StaticConfig config, RuntimeConfig runtime, ControlResponseOptions options)
        : config_(config), runtime_(runtime), options_(options) {
        MouseSenderConfig sender;
        sender.method = runtime.mouse_output_method;
        sender.gain_x = runtime.sendinput_gain_x;
        sender.gain_y = runtime.sendinput_gain_y;
        sender.max_step = runtime.sendinput_max_step;
        transform_.configure(sender);
        period_ =
            1000.0 /
            std::clamp(runtime.display_rate_servo_hz > 0 ? double(runtime.display_rate_servo_hz) : 165.0, 1.0,
                       1000.0);
    }
    std::optional<ReplayFrame> next(const CaptureRegion&) override {
        if (index_ == options_.frames) return {};
        const double sampled = ++index_ * options_.cadence_ms;
        advance(sampled);
        const double source_error = world(sampled) - camera_;
        ReplayFrame frame;
        frame.frame.capture = {0, 0, config_.screen_w, config_.screen_h};
        frame.frame.width = config_.screen_w;
        frame.frame.height = config_.screen_h;
        frame.frame.source_id = index_;
        frame.frame.source_present = at(sampled);
        frame.frame.frame_ready = frame.frame.acquire_started = frame.frame.capture_done =
            frame.frame.frame_time = at(sampled);
        frame.frame.capture_time = SystemClock::time_point(std::chrono::milliseconds(10000 + int(sampled)));
        const double x = config_.screen_w * .5 + source_error, y = config_.screen_h * .5;
        const bool dropout = options_.dropout_start >= 0 && index_ >= options_.dropout_start &&
                             index_ < options_.dropout_start + options_.dropout_frames;
        if (!dropout && x > 24 && x < config_.screen_w - 24)
            frame.detections.push_back(
                {.bbox = {int(std::lround(x)) - 23, int(y) - 23, int(std::lround(x)) + 23, int(y) + 23},
                 .x = float(x),
                 .y = float(y),
                 .cls = 1,
                 .conf = .99F});
        now_ = sampled + options_.observation_delay_ms;
        advance(now_);
        frame.now = at(now_);
        frame.dispatch_feedback = std::move(feedback_);
        feedback_.clear();
        current_ = {{"time_ms", now_},
                    {"source_error_px", source_error},
                    {"error_px", world(now_) - camera_},
                    {"visible", !frame.detections.empty()}};
        return frame;
    }
    void observe(const ReplayResult& result) override {
        servo_ = result.servo;
        allowed_ = result.output_ready;
        if (result.command) schedule(now_, result.command);
        current_["command"] =
            result.command ? Json::array({result.command->dx, result.command->dy}) : Json(nullptr);
        current_["output_ready"] = result.output_ready;
        current_["estimated_velocity_px_s"] = result.track ? Json(result.track->vx) : Json(nullptr);
        samples.push_back(current_);
    }
    Json samples = Json::array(), moves = Json::array();

private:
    struct Move {
        double time;
        int dx;
    };
    double world(double ms) const {
        const double t = ms / 1000;
        if (!options_.reversal) return options_.initial_error_px + options_.speed_px_s * t;
        const double half = options_.frames * options_.cadence_ms / 2000;
        return options_.initial_error_px + options_.speed_px_s * (t <= half ? t : 2 * half - t);
    }
    void schedule(double ms, std::optional<CommandPacket> incoming) {
        const auto r = scheduler_.step(at(ms), SystemClock::time_point{}, std::move(incoming), servo_,
                                       runtime_, 1000 / period_, allowed_, true);
        if (!r.command) {
            if (!r.deferred) feedback_.push_back(CommandDelivery{.completed_at = at(ms)});
            return;
        }
        if (!commandFreshAt(*r.command, at(ms), runtime_.command_max_observation_age_ms)) return;
        const auto steps = transform_.apply(r.command->dx, r.command->dy);
        feedback_.push_back(CommandDelivery{.command = *r.command,
                                            .dx = r.command->dx,
                                            .dy = r.command->dy,
                                            .completed_at = at(ms),
                                            .movement_sent = !steps.empty()});
        for (const auto& [dx, dy] : steps) queue_.push_back({ms + options_.transport_delay_ms, dx});
    }
    void advance(double ms) {
        while (next_tick_ <= ms || (!queue_.empty() && queue_.front().time <= ms)) {
            if (!queue_.empty() && queue_.front().time <= ms && queue_.front().time <= next_tick_) {
                const auto move = queue_.front();
                queue_.pop_front();
                camera_ += move.dx * options_.pixels_per_count;
                moves.push_back(
                    {{"time_ms", move.time}, {"dx", move.dx}, {"error_px", world(move.time) - camera_}});
            } else if (next_tick_ <= ms) {
                schedule(next_tick_, {});
                next_tick_ += period_;
            } else
                break;
        }
    }
    StaticConfig config_;
    RuntimeConfig runtime_;
    ControlResponseOptions options_;
    int index_ = 0;
    double now_ = 0, camera_ = 0, next_tick_ = 0, period_ = 0;
    bool allowed_ = false;
    DisplayRateServoState servo_{};
    AimCommandScheduler scheduler_;
    MouseMovementTransform transform_;
    std::deque<Move> queue_;
    std::vector<CommandDelivery> feedback_;
    Json current_;
};
}  // namespace
Json runControlResponse(const StaticConfig& config, const RuntimeConfig& runtime,
                        const ControlResponseOptions& options) {
    Plant plant(config, runtime, options);
    DeltaApp::replay(config, runtime, plant);
    std::vector<double> errors;
    double squared = 0, lag = 0, max_error = 0;
    int sign = 0, crossings = 0, large_sign = 0, large_crossings = 0, ready = 0;
    for (const auto& sample : plant.samples) {
        if (sample["time_ms"].get<double>() < 500) continue;
        const double error = sample["error_px"].get<double>();
        errors.push_back(std::abs(error));
        squared += error * error;
        lag += error;
        max_error = std::max(max_error, std::abs(error));
        const int next = error > 2 ? 1 : error < -2 ? -1 : 0;
        if (next && sign && next != sign) ++crossings;
        if (next) sign = next;
        // Quarter-width of this 46px target separates large excursions from
        // the normal ripple of discrete display-rate camera steps.
        const int large_next = error > 12 ? 1 : error < -12 ? -1 : 0;
        if (large_next && large_sign && large_next != large_sign) ++large_crossings;
        if (large_next) large_sign = large_next;
        ready += sample["output_ready"].get<bool>();
    }
    std::sort(errors.begin(), errors.end());
    const double n = double(std::max<std::size_t>(1, errors.size()));
    return {{"scope", "controlled_time_perfect_detection_controller_response"},
            {"device_output", false},
            {"viewport", {config.screen_w, config.screen_h}},
            {"initial_error_px", options.initial_error_px},
            {"dropout_start", options.dropout_start},
            {"dropout_frames", options.dropout_frames},
            {"runtime", runtimeJson(runtime)},
            {"pixels_per_count", options.pixels_per_count},
            {"speed_px_s", options.speed_px_s},
            {"cadence_ms", options.cadence_ms},
            {"observation_delay_ms", options.observation_delay_ms},
            {"transport_delay_ms", options.transport_delay_ms},
            {"reversal", options.reversal},
            {"summary",
             {{"median_absolute_error_px", errors.empty() ? 0 : errors[errors.size() / 2]},
              {"evaluated_samples", errors.size()},
              {"rms_error_px", std::sqrt(squared / n)},
              {"mean_signed_error_px", lag / n},
              {"max_absolute_error_px", max_error},
              {"error_sign_changes", crossings},
              {"large_error_sign_changes", large_crossings},
              {"output_ready_fraction", ready / n}}},
            {"samples", plant.samples},
            {"movements", plant.moves}};
}
}  // namespace delta::bench
