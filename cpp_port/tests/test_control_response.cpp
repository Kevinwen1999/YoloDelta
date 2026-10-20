#include <iostream>
#include <stdexcept>

#include "control_response.hpp"
#include "delta/app.hpp"
#include "delta/calibrated_control.hpp"

namespace {
class CameraCatchupReplay final : public delta::ITrackingReplay {
public:
    bool aim_change = false;
    bool left_click = false;
    std::vector<delta::ReplayResult> results;
    std::optional<delta::ReplayFrame> next(const delta::CaptureRegion&) override {
        if (index_ == 3) return {};
        const auto at = [](int ms) { return delta::SteadyClock::time_point(std::chrono::milliseconds(ms)); };
        const int x = aim_change ? (index_ == 2 ? 680 : 650) : (index_ == 1 ? 674 : 650);
        delta::ReplayFrame frame;
        frame.now = at(1005 + index_ * 16);
        frame.frame.frame_ready = frame.frame.source_present = at(1000 + index_ * 16);
        frame.frame.source_id = index_ + 1;
        frame.toggles.left_pressed = left_click && index_ >= 1;
        frame.frame.capture = {0, 0, 1280, 720};
        frame.detections.push_back(
            {.bbox = {x - 13, 347, x + 13, 373}, .cls = aim_change && index_ == 2 ? 0 : 1, .conf = .99F});
        if (index_ == 2 && !aim_change)
            frame.dispatch_feedback.push_back({.dx = 120, .completed_at = at(1025), .movement_sent = true});
        ++index_;
        return frame;
    }
    void observe(const delta::ReplayResult& result) override { results.push_back(result); }

private:
    int index_ = 0;
};
}  // namespace

int main() try {
    delta::StaticConfig config;
    delta::RuntimeConfig runtime;
    runtime.detection_dampening_enable = false;
    runtime.target_guard_enable = false;
    runtime.adaptive_capture_crop_enable = false;
    runtime.calibrated_control_enable = true;
    {
        CameraCatchupReplay without_click;
        CameraCatchupReplay with_click;
        with_click.left_click = true;
        auto click_config = config;
        click_config.screen_w = 1280;
        click_config.screen_h = 720;
        delta::DeltaApp::replay(click_config, runtime, without_click);
        delta::DeltaApp::replay(click_config, runtime, with_click);
        for (std::size_t index = 0; index < with_click.results.size(); ++index) {
            const auto& a = without_click.results[index];
            const auto& b = with_click.results[index];
            if (!a.servo.valid || !b.servo.valid ||
                a.servo.velocity_x != b.servo.velocity_x || a.servo.velocity_y != b.servo.velocity_y)
                throw std::runtime_error("Left-click with recoil disabled must not shift calibrated tracking");
        }
    }
    {
        delta::RuntimeConfig anchor_config;
        anchor_config.predictive_pid_velocity_alpha = 1;
        delta::CalibratedController controller;
        delta::DeliveredMotionHistory motion;
        const auto at = [](int ms) { return delta::SteadyClock::time_point(std::chrono::milliseconds(ms)); };
        controller.update(anchor_config, {{0.F, 0.F}}, at(1000), at(1000), motion);
        controller.update(anchor_config, {{10.F, 0.F}}, at(1010), at(1010), motion);
        controller.reanchorAimPoint();
        const auto changed = controller.update(anchor_config, {{50.F, 0.F}}, at(1020), at(1020), motion);
        if (!changed.valid || std::abs(changed.velocity_x - 1000) > 1e-3)
            throw std::runtime_error("Changing aim-point anchor must retain established target velocity");
    }
    {
        auto delayed = runtime;
        delayed.calibrated_pixels_per_count_x = delayed.calibrated_pixels_per_count_y = .4F;
        delayed.calibrated_response_time_s = .04F;
        delayed.calibrated_response_delay_ms = 24;
        delayed.predictive_pid_velocity_alpha = .5F;
        delta::bench::ControlResponseOptions options;
        options.speed_px_s = 0;
        options.initial_error_px = 100;
        options.transport_delay_ms = 24;
        const auto summary = delta::bench::runControlResponse(config, delayed, options)["summary"];
        if (summary["median_absolute_error_px"].get<double>() > 1 ||
            summary["max_absolute_error_px"].get<double>() > 2)
            throw std::runtime_error(
                "Delayed camera response must not create self-excited stationary oscillation");
    }
    {
        auto motion_runtime = runtime;
        motion_runtime.detection_dampening_enable = true;
        motion_runtime.detection_dampening_stable_frames = 2;
        motion_runtime.motion_stability_enable = motion_runtime.motion_identity_enable = true;
        motion_runtime.motion_residual_scale = .5F;
        motion_runtime.calibrated_pixels_per_count_x = .4F;
        auto motion_config = config;
        motion_config.screen_w = 1280;
        motion_config.screen_h = 720;
        CameraCatchupReplay replay;
        delta::DeltaApp::replay(motion_config, motion_runtime, replay);
        if (!replay.results[2].output_ready || replay.results[2].motion_decision != "accepted_observation")
            throw std::runtime_error(
                "Delivered camera catch-up must not force target reversal reconfirmation");
        motion_runtime.aim_mode = delta::AimMode::Hybrid;
        CameraCatchupReplay aim_replay;
        aim_replay.aim_change = true;
        delta::DeltaApp::replay(motion_config, motion_runtime, aim_replay);
        const float expected_rate = 40.F / (motion_runtime.calibrated_response_time_s * .4F);
        if (std::abs(aim_replay.results[2].servo.velocity_x - expected_rate) > .1F)
            throw std::runtime_error("Head/body aim-point changes must not become target velocity");
    }
    for (const double gain : {.1, .4, 1.28, 2.0}) {
        runtime.calibrated_pixels_per_count_x = float(gain);
        runtime.calibrated_pixels_per_count_y = float(gain);
        delta::bench::ControlResponseOptions options;
        options.pixels_per_count = gain;
        const auto summary = delta::bench::runControlResponse(config, runtime, options)["summary"];
        std::cout << "gain=" << gain << ' ' << summary.dump() << '\n';
        if (summary["median_absolute_error_px"].get<double>() > 12 ||
            summary["large_error_sign_changes"].get<int>() > 10 ||
            summary["max_absolute_error_px"].get<double>() > 20 ||
            summary["output_ready_fraction"].get<double>() < .99)
            throw std::runtime_error(
                "Low sensitivity must follow near center without high sensitivity oscillation");
        options.speed_px_s = 0;
        options.initial_error_px = 100;
        const auto stationary = delta::bench::runControlResponse(config, runtime, options)["summary"];
        if (stationary["median_absolute_error_px"].get<double>() > 1.1 ||
            stationary["max_absolute_error_px"].get<double>() > 2.1)
            throw std::runtime_error("Stationary target must settle without sustained jitter");
        options.initial_error_px = 0;
        options.speed_px_s = 1500;
        options.reversal = true;
        const auto reversing = delta::bench::runControlResponse(config, runtime, options);
        for (const auto& sample : reversing["samples"])
            if (sample["time_ms"].get<double>() > 6000 && std::abs(sample["error_px"].get<double>()) > 12)
                throw std::runtime_error("Reversal must recover to near-center following");
        options.reversal = false;
        options.cadence_ms = 8;
        options.observation_delay_ms = 5;
        const auto fast = delta::bench::runControlResponse(config, runtime, options)["summary"];
        if (fast["median_absolute_error_px"].get<double>() > 12 ||
            fast["max_absolute_error_px"].get<double>() > 20)
            throw std::runtime_error("Changing observation cadence must preserve stable following");
        options.cadence_ms = 16;
        options.observation_delay_ms = 12;
        options.dropout_start = 120;
        options.dropout_frames = 20;
        const auto missing = delta::bench::runControlResponse(config, runtime, options);
        const double last_observation = 119 * options.cadence_ms;
        for (const auto& move : missing["movements"])
            if (move["time_ms"].get<double>() > last_observation + 100 &&
                move["time_ms"].get<double>() < 140 * options.cadence_ms)
                throw std::runtime_error(
                    "No movement may continue beyond the last accepted observation deadline");
    }
    for (const float backend_gain : {.5F, 2.F}) {
        delta::bench::ControlResponseOptions options;
        runtime.mouse_output_method = delta::MouseOutputMethod::SendInput;
        runtime.sendinput_gain_x = runtime.sendinput_gain_y = backend_gain;
        runtime.calibrated_pixels_per_count_x = runtime.calibrated_pixels_per_count_y =
            float(options.pixels_per_count) * backend_gain;
        const auto summary = delta::bench::runControlResponse(config, runtime, options)["summary"];
        if (summary["median_absolute_error_px"].get<double>() > 12 ||
            summary["max_absolute_error_px"].get<double>() > 20)
            throw std::runtime_error("Calibration must include the selected backend's numeric gain");
    }
    {
        delta::RuntimeConfig following;
        following.motion_stability_enable = following.detection_dampening_enable = true;
        following.motion_prediction_timeout_ms = 80;
        following.motion_prediction_max_displacement_px = 30;
        following.predictive_pid_velocity_alpha = 1;
        delta::CalibratedController controller;
        delta::DeliveredMotionHistory delivered;
        const auto at = [](int ms) { return delta::SteadyClock::time_point(std::chrono::milliseconds(ms)); };
        controller.update(following, {{0.F, 0.F}}, at(1000), at(1000), delivered);
        controller.update(following, {{10.F, 0.F}}, at(1010), at(1010), delivered);
        delivered.note(at(1040), 50, 0);
        const auto missing = controller.update(following, {}, at(1060), at(1060), delivered);
        if (!missing.valid || std::abs(missing.error_x - 10) > 1e-3)
            throw std::runtime_error(
                "Loss extrapolation must bound residual screen motion after subtracting the following "
                "camera");
    }
    {
        delta::DeliveredMotionHistory motion;
        delta::CalibratedController controller;
        runtime.calibrated_pixels_per_count_x = .1F;
        runtime.calibrated_pixels_per_count_y = .4F;
        const auto at = [](int ms) { return delta::SteadyClock::time_point(std::chrono::milliseconds(ms)); };
        controller.update(runtime, {{100.F, 40.F}}, at(1000), at(1000), motion);
        motion.note(at(1005), 100, 25);
        const auto stationary = controller.update(runtime, {{90.F, 30.F}}, at(1010), at(1012), motion);
        if (!stationary.valid || std::abs(stationary.velocity_x) > 1e-3 ||
            std::abs(stationary.velocity_y) > 1e-3)
            throw std::runtime_error("Own delivered camera movement must not become target velocity");
        if (controller.update(runtime, {}, at(1100), at(1111), motion).valid)
            throw std::runtime_error("Missing evidence cannot extend the prediction deadline");
        runtime.calibrated_pixels_per_count_x = 0;
        if (controller.update(runtime, {{10.F, 0.F}}, at(1120), at(1120), motion).valid)
            throw std::runtime_error("Invalid response calibration must stop output");
    }
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
