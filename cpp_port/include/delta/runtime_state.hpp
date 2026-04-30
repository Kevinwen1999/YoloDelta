#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <utility>

#include "delta/config.hpp"
#include "delta/core.hpp"

namespace delta {

template <typename T>
class LatestSlot {
public:
    void put(T value) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            value_ = std::move(value);
        }
        cv_.notify_one();
    }

    std::optional<T> try_take() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!value_.has_value()) {
            return std::nullopt;
        }
        auto out = std::move(value_);
        value_.reset();
        return out;
    }

    template <typename Rep, typename Period>
    std::optional<T> wait_take_for(const std::chrono::duration<Rep, Period>& timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!cv_.wait_for(lock, timeout, [this]() { return value_.has_value(); })) {
            return std::nullopt;
        }
        auto out = std::move(value_);
        value_.reset();
        return out;
    }

    template <typename Clock, typename Duration>
    std::optional<T> wait_take_until(const std::chrono::time_point<Clock, Duration>& deadline) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!cv_.wait_until(lock, deadline, [this]() { return value_.has_value(); })) {
            return std::nullopt;
        }
        auto out = std::move(value_);
        value_.reset();
        return out;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        value_.reset();
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    std::optional<T> value_;
};

class RuntimeConfigStore {
public:
    explicit RuntimeConfigStore(RuntimeConfig initial = {}) : value_(std::move(initial)) {}

    RuntimeConfig snapshot() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return value_;
    }

    void update(RuntimeConfig next) {
        std::lock_guard<std::mutex> lock(mutex_);
        value_ = std::move(next);
        ++version_;
    }

    void requestReset() {
        std::lock_guard<std::mutex> lock(mutex_);
        ++reset_token_;
    }

    std::uint64_t version() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return version_;
    }

    std::uint64_t resetToken() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return reset_token_;
    }

private:
    mutable std::mutex mutex_;
    RuntimeConfig value_{};
    std::uint64_t version_ = 0;
    std::uint64_t reset_token_ = 0;
};

struct DisplayRateServoState {
    bool valid = false;
    float target_x = 0.0F;
    float target_y = 0.0F;
    float velocity_x = 0.0F;
    float velocity_y = 0.0F;
    float acceleration_x = 0.0F;
    float acceleration_y = 0.0F;
    int target_cls = -1;
    SteadyClock::time_point updated_at{};
    SteadyClock::time_point acquire_started{};
    SteadyClock::time_point frame_ready{};
    SteadyClock::time_point capture_done{};
    SystemClock::time_point frame_time{};
    SystemClock::time_point capture_time{};
};

struct SharedState {
    mutable std::mutex mutex;
    bool running = true;
    ToggleState toggles{};
    bool target_found = false;
    int target_cls = -1;
    int aim_dx = 0;
    int aim_dy = 0;
    float target_speed = 0.0F;
    bool pid_settled = false;
    float pid_settle_error_metric_px = 0.0F;
    float pid_settle_threshold_px = 0.0F;
    bool lead_active = false;
    float lead_time_ms = 0.0F;
    bool kalman_prediction_enable = false;
    float kalman_residual_px = 0.0F;
    float kalman_max_residual_px = 0.0F;
    float kalman_prediction_age_ms = 0.0F;
    int kalman_predicted_only_frames = 0;
    std::uint64_t kalman_snap_count = 0;
    float predictive_pid_latency_ms = 0.0F;
    float predictive_pid_horizon_ms = 0.0F;
    bool predictive_pid_deadzone_active = false;
    bool mouse_move_suppress_active = false;
    bool mouse_move_suppress_supported = false;
    std::uint64_t mouse_move_suppress_count = 0;
    bool recoil_virtual_active = false;
    int recoil_virtual_dx = 0;
    int recoil_virtual_dy = 0;
    std::uint64_t recoil_virtual_apply_count = 0;
    std::pair<int, int> last_target_full{1280, 720};
    std::pair<int, int> capture_focus_full{1280, 720};
    SystemClock::time_point target_time{};
    std::atomic<float> ctrl_sent_vx_ema{0.0F};
    std::atomic<float> ctrl_sent_vy_ema{0.0F};
    std::atomic<float> cmd_send_latency_ema_s{0.0F};
    SteadyClock::time_point ctrl_last_send_tick{};
    double display_refresh_hz = 0.0;
    DisplayRateServoState display_rate_servo{};
    std::string tracking_strategy = "raw_delta";
    bool side_button_key_sequence_enabled = false;
    RecoilRuntimeState recoil{};
    PendingRecoilDelta pending_recoil{};
};

}  // namespace delta
