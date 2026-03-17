#include "delta/app.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <utility>

namespace delta {

struct RuntimePerfWindow {
    std::mutex mutex;
    SteadyClock::time_point window_start{SteadyClock::now()};
    std::uint64_t capture_frames = 0;
    std::uint64_t capture_none = 0;
    double capture_grab_s = 0.0;

    std::uint64_t infer_frames = 0;
    std::uint64_t infer_stale = 0;
    std::uint64_t infer_found = 0;
    double infer_loop_s = 0.0;
    double infer_frame_age_s = 0.0;
    double infer_frame_age_max_s = 0.0;
    std::uint64_t infer_tracker_calls = 0;
    double infer_tracker_s = 0.0;
    std::uint64_t infer_backend_samples = 0;
    double infer_backend_pre_s = 0.0;
    double infer_backend_exec_s = 0.0;
    double infer_backend_post_s = 0.0;
    std::uint64_t infer_cmd_samples = 0;
    double infer_cmd_latency_s = 0.0;

    std::uint64_t control_cmds = 0;
    std::uint64_t control_sent = 0;
    std::uint64_t control_stale_drop = 0;
    std::uint64_t control_mode_drop = 0;
    double control_send_s = 0.0;
    double control_cmd_age_s = 0.0;
    double control_total_latency_s = 0.0;
    std::uint64_t control_latency_samples = 0;
    double control_total_latency_full_s = 0.0;
    std::uint64_t control_latency_full_samples = 0;
    double control_total_apply_latency_s = 0.0;
    std::uint64_t control_apply_latency_samples = 0;
    double control_total_apply_latency_full_s = 0.0;
    std::uint64_t control_apply_latency_full_samples = 0;

    void reset(const SteadyClock::time_point now) {
        window_start = now;
        capture_frames = 0;
        capture_none = 0;
        capture_grab_s = 0.0;

        infer_frames = 0;
        infer_stale = 0;
        infer_found = 0;
        infer_loop_s = 0.0;
        infer_frame_age_s = 0.0;
        infer_frame_age_max_s = 0.0;
        infer_tracker_calls = 0;
        infer_tracker_s = 0.0;
        infer_backend_samples = 0;
        infer_backend_pre_s = 0.0;
        infer_backend_exec_s = 0.0;
        infer_backend_post_s = 0.0;
        infer_cmd_samples = 0;
        infer_cmd_latency_s = 0.0;

        control_cmds = 0;
        control_sent = 0;
        control_stale_drop = 0;
        control_mode_drop = 0;
        control_send_s = 0.0;
        control_cmd_age_s = 0.0;
        control_total_latency_s = 0.0;
        control_latency_samples = 0;
        control_total_latency_full_s = 0.0;
        control_latency_full_samples = 0;
        control_total_apply_latency_s = 0.0;
        control_apply_latency_samples = 0;
        control_total_apply_latency_full_s = 0.0;
        control_apply_latency_full_samples = 0;
    }
};

namespace {

constexpr auto kToggleCooldown = std::chrono::milliseconds(200);
constexpr auto kControlIdleSleep = std::chrono::milliseconds(2);
constexpr auto kControlCommandWait = std::chrono::milliseconds(1);
constexpr auto kCaptureIdleSleep = std::chrono::milliseconds(1);
constexpr auto kInferenceIdleSleep = std::chrono::milliseconds(1);
constexpr auto kPerfLoopSleep = std::chrono::milliseconds(50);
constexpr double kPerfLogIntervalSeconds = 1.0;
constexpr double kCommandTimeoutSeconds = 0.10;
constexpr double kMaxFrameAgeSeconds = 0.05;
constexpr double kTargetTimeoutSeconds = 0.08;
constexpr float kMinTrackDt = 1.0F / 240.0F;
constexpr float kMaxTrackDt = 0.06F;
constexpr float kMaxTrackSpeedPxS = 1800.0F;
constexpr float kEgoMotionCompAlpha = 0.30F;
constexpr float kEgoMotionCompMaxPxS = 3200.0F;
constexpr float kEgoMotionCompDecay = 0.92F;
constexpr float kBodyAimYRatio = 0.38F;
constexpr float kTargetLockMaxJump = 260.0F;
constexpr float kTrackerReinitMinIou = 0.35F;
constexpr float kAssocPredictDt = 0.02F;
constexpr float kAssocSpeedJumpGain = 0.05F;
constexpr float kAssocMaxJumpPad = 220.0F;
constexpr int kRawMaxStepX = 280;
constexpr bool kPerfLogWhenModeOff = true;

bool risingEdge(const bool current, const bool previous) {
    return current && !previous;
}

double secondsSince(const SystemClock::time_point since, const SystemClock::time_point now) {
    return std::chrono::duration<double>(now - since).count();
}

double secondsSince(const SteadyClock::time_point since, const SteadyClock::time_point now) {
    return std::chrono::duration<double>(now - since).count();
}

const char* trackingStrategyName(const TrackingStrategy strategy) {
    return strategy == TrackingStrategy::Raw ? "raw" : "raw_delta";
}

float emaUpdateSigned(const float prev, const float sample, const float alpha) {
    const float clamped_alpha = clamp(alpha, 0.0F, 1.0F);
    if (clamped_alpha <= 0.0F) {
        return prev;
    }
    return prev + ((sample - prev) * clamped_alpha);
}

std::pair<int, int> screenCenter(const StaticConfig& config) {
    return {config.screen_w / 2, config.screen_h / 2};
}

std::pair<float, float> detectionAimPoint(const Detection& detection) {
    const float x1 = static_cast<float>(detection.bbox[0]);
    const float y1 = static_cast<float>(detection.bbox[1]);
    const float x2 = static_cast<float>(detection.bbox[2]);
    const float y2 = static_cast<float>(detection.bbox[3]);
    const float width = std::max(1.0F, x2 - x1);
    const float height = std::max(1.0F, y2 - y1);
    const float aim_x = x1 + (width * 0.5F);
    const float aim_y = detection.cls == 0
        ? (y1 + (height * kBodyAimYRatio))
        : (y1 + (height * 0.5F));
    return {aim_x, clamp(aim_y, y1, y2)};
}

float bboxIou(const std::array<int, 4>& a, const std::array<int, 4>& b) {
    const float ax1 = static_cast<float>(a[0]);
    const float ay1 = static_cast<float>(a[1]);
    const float ax2 = static_cast<float>(a[2]);
    const float ay2 = static_cast<float>(a[3]);
    const float bx1 = static_cast<float>(b[0]);
    const float by1 = static_cast<float>(b[1]);
    const float bx2 = static_cast<float>(b[2]);
    const float by2 = static_cast<float>(b[3]);
    const float ix1 = std::max(ax1, bx1);
    const float iy1 = std::max(ay1, by1);
    const float ix2 = std::min(ax2, bx2);
    const float iy2 = std::min(ay2, by2);
    const float iw = std::max(0.0F, ix2 - ix1);
    const float ih = std::max(0.0F, iy2 - iy1);
    const float inter = iw * ih;
    if (inter <= 0.0F) {
        return 0.0F;
    }
    const float area_a = std::max(0.0F, ax2 - ax1) * std::max(0.0F, ay2 - ay1);
    const float area_b = std::max(0.0F, bx2 - bx1) * std::max(0.0F, by2 - by1);
    const float denom = area_a + area_b - inter;
    return denom > 1e-6F ? (inter / denom) : 0.0F;
}

float pointToBoxDistance(const std::array<int, 4>& box, const float x, const float y) {
    const float x1 = static_cast<float>(box[0]);
    const float y1 = static_cast<float>(box[1]);
    const float x2 = static_cast<float>(box[2]);
    const float y2 = static_cast<float>(box[3]);
    const float dx = x < x1 ? (x1 - x) : (x > x2 ? x - x2 : 0.0F);
    const float dy = y < y1 ? (y1 - y) : (y > y2 ? y - y2 : 0.0F);
    return std::sqrt((dx * dx) + (dy * dy));
}

CaptureRegion buildCaptureRegion(const StaticConfig& config, const int center_x, const int center_y) {
    const int width = clamp(config.imgsz, 1, config.screen_w);
    const int height = clamp(config.imgsz, 1, config.screen_h);
    return CaptureRegion{
        .left = clamp(center_x - (width / 2), 0, config.screen_w - width),
        .top = clamp(center_y - (height / 2), 0, config.screen_h - height),
        .width = width,
        .height = height,
    };
}

void clearAimStateLocked(SharedState& shared, const std::pair<int, int> center, const TrackingStrategy strategy) {
    shared.target_found = false;
    shared.target_cls = -1;
    shared.target_speed = 0.0F;
    shared.aim_dx = 0;
    shared.aim_dy = 0;
    shared.last_target_full = center;
    shared.capture_focus_full = center;
    shared.target_time = {};
    shared.tracking_strategy = trackingStrategyName(strategy);
}

void resetEgoMotionStateLocked(SharedState& shared) {
    shared.ctrl_sent_vx_ema = 0.0F;
    shared.ctrl_sent_vy_ema = 0.0F;
    shared.ctrl_last_send_tick = {};
}

void decayEgoMotionStateLocked(SharedState& shared) {
    shared.ctrl_sent_vx_ema *= kEgoMotionCompDecay;
    shared.ctrl_sent_vy_ema *= kEgoMotionCompDecay;
    if (std::abs(shared.ctrl_sent_vx_ema) < 1e-6F) {
        shared.ctrl_sent_vx_ema = 0.0F;
    }
    if (std::abs(shared.ctrl_sent_vy_ema) < 1e-6F) {
        shared.ctrl_sent_vy_ema = 0.0F;
    }
}

struct PerfLogSnapshot {
    double elapsed_s = 0.0;
    double cap_fps = 0.0;
    double cap_grab_ms = 0.0;
    std::uint64_t cap_none = 0;
    double infer_fps = 0.0;
    double infer_loop_ms = 0.0;
    double infer_age_ms = 0.0;
    double infer_age_max_ms = 0.0;
    std::uint64_t infer_stale = 0;
    double infer_lock_rate = 0.0;
    double tracker_hz = 0.0;
    double tracker_ms = 0.0;
    std::uint64_t infer_backend_samples = 0;
    double infer_backend_pre_ms = 0.0;
    double infer_backend_exec_ms = 0.0;
    double infer_backend_post_ms = 0.0;
    double infer_cmd_ms = 0.0;
    double control_send_hz = 0.0;
    double control_send_ms = 0.0;
    double control_cmd_age_ms = 0.0;
    double control_total_latency_ms = 0.0;
    double control_total_latency_full_ms = 0.0;
    double control_total_apply_latency_ms = 0.0;
    double control_total_apply_latency_full_ms = 0.0;
    std::uint64_t control_stale_drop = 0;
    std::uint64_t control_mode_drop = 0;
};

void recordCapturePerf(RuntimePerfWindow& perf, const double grab_s, const bool is_none) {
    std::lock_guard<std::mutex> lock(perf.mutex);
    if (is_none) {
        ++perf.capture_none;
        return;
    }
    ++perf.capture_frames;
    perf.capture_grab_s += std::max(0.0, grab_s);
}

void recordInferencePerf(
    RuntimePerfWindow& perf,
    const double frame_age_s,
    const double loop_s,
    const bool stale_drop,
    const bool target_found,
    const double tracker_s,
    const InferenceTimings& timings,
    const std::optional<double> cmd_latency_s) {
    std::lock_guard<std::mutex> lock(perf.mutex);
    ++perf.infer_frames;
    perf.infer_loop_s += std::max(0.0, loop_s);
    perf.infer_frame_age_s += std::max(0.0, frame_age_s);
    perf.infer_frame_age_max_s = std::max(perf.infer_frame_age_max_s, std::max(0.0, frame_age_s));
    if (stale_drop) {
        ++perf.infer_stale;
    }
    if (target_found) {
        ++perf.infer_found;
    }
    if (tracker_s > 0.0) {
        ++perf.infer_tracker_calls;
        perf.infer_tracker_s += tracker_s;
    }
    if (timings.preprocess_ms > 0.0 || timings.execute_ms > 0.0 || timings.postprocess_ms > 0.0) {
        ++perf.infer_backend_samples;
        perf.infer_backend_pre_s += timings.preprocess_ms / 1000.0;
        perf.infer_backend_exec_s += timings.execute_ms / 1000.0;
        perf.infer_backend_post_s += timings.postprocess_ms / 1000.0;
    }
    if (cmd_latency_s.has_value()) {
        ++perf.infer_cmd_samples;
        perf.infer_cmd_latency_s += std::max(0.0, *cmd_latency_s);
    }
}

void recordControlPerf(
    RuntimePerfWindow& perf,
    const double cmd_age_s,
    const bool sent,
    const double send_s,
    const bool stale_drop,
    const bool mode_drop,
    const std::optional<double> total_latency_s,
    const std::optional<double> total_latency_full_s,
    const std::optional<double> total_apply_latency_s,
    const std::optional<double> total_apply_latency_full_s) {
    std::lock_guard<std::mutex> lock(perf.mutex);
    ++perf.control_cmds;
    if (stale_drop) {
        ++perf.control_stale_drop;
    }
    if (mode_drop) {
        ++perf.control_mode_drop;
    }
    if (!sent) {
        return;
    }

    ++perf.control_sent;
    perf.control_send_s += std::max(0.0, send_s);
    perf.control_cmd_age_s += std::max(0.0, cmd_age_s);
    if (total_latency_s.has_value()) {
        ++perf.control_latency_samples;
        perf.control_total_latency_s += std::max(0.0, *total_latency_s);
    }
    if (total_latency_full_s.has_value()) {
        ++perf.control_latency_full_samples;
        perf.control_total_latency_full_s += std::max(0.0, *total_latency_full_s);
    }
    if (total_apply_latency_s.has_value()) {
        ++perf.control_apply_latency_samples;
        perf.control_total_apply_latency_s += std::max(0.0, *total_apply_latency_s);
    }
    if (total_apply_latency_full_s.has_value()) {
        ++perf.control_apply_latency_full_samples;
        perf.control_total_apply_latency_full_s += std::max(0.0, *total_apply_latency_full_s);
    }
}

std::optional<PerfLogSnapshot> takePerfSnapshot(RuntimePerfWindow& perf, const double min_interval_s) {
    const auto now = SteadyClock::now();
    std::lock_guard<std::mutex> lock(perf.mutex);
    const double elapsed = secondsSince(perf.window_start, now);
    if (elapsed < min_interval_s) {
        return std::nullopt;
    }

    PerfLogSnapshot snapshot{};
    snapshot.elapsed_s = elapsed;
    snapshot.cap_fps = elapsed > 0.0 ? static_cast<double>(perf.capture_frames) / elapsed : 0.0;
    snapshot.cap_grab_ms = perf.capture_frames > 0 ? (perf.capture_grab_s * 1000.0 / static_cast<double>(perf.capture_frames)) : 0.0;
    snapshot.cap_none = perf.capture_none;
    snapshot.infer_fps = elapsed > 0.0 ? static_cast<double>(perf.infer_frames) / elapsed : 0.0;
    snapshot.infer_loop_ms = perf.infer_frames > 0 ? (perf.infer_loop_s * 1000.0 / static_cast<double>(perf.infer_frames)) : 0.0;
    snapshot.infer_age_ms = perf.infer_frames > 0 ? (perf.infer_frame_age_s * 1000.0 / static_cast<double>(perf.infer_frames)) : 0.0;
    snapshot.infer_age_max_ms = perf.infer_frame_age_max_s * 1000.0;
    snapshot.infer_stale = perf.infer_stale;
    snapshot.infer_lock_rate = perf.infer_frames > 0 ? static_cast<double>(perf.infer_found) / static_cast<double>(perf.infer_frames) : 0.0;
    snapshot.tracker_hz = elapsed > 0.0 ? static_cast<double>(perf.infer_tracker_calls) / elapsed : 0.0;
    snapshot.tracker_ms = perf.infer_tracker_calls > 0 ? (perf.infer_tracker_s * 1000.0 / static_cast<double>(perf.infer_tracker_calls)) : 0.0;
    snapshot.infer_backend_samples = perf.infer_backend_samples;
    snapshot.infer_backend_pre_ms = perf.infer_backend_samples > 0 ? (perf.infer_backend_pre_s * 1000.0 / static_cast<double>(perf.infer_backend_samples)) : 0.0;
    snapshot.infer_backend_exec_ms = perf.infer_backend_samples > 0 ? (perf.infer_backend_exec_s * 1000.0 / static_cast<double>(perf.infer_backend_samples)) : 0.0;
    snapshot.infer_backend_post_ms = perf.infer_backend_samples > 0 ? (perf.infer_backend_post_s * 1000.0 / static_cast<double>(perf.infer_backend_samples)) : 0.0;
    snapshot.infer_cmd_ms = perf.infer_cmd_samples > 0 ? (perf.infer_cmd_latency_s * 1000.0 / static_cast<double>(perf.infer_cmd_samples)) : 0.0;
    snapshot.control_send_hz = elapsed > 0.0 ? static_cast<double>(perf.control_sent) / elapsed : 0.0;
    snapshot.control_send_ms = perf.control_sent > 0 ? (perf.control_send_s * 1000.0 / static_cast<double>(perf.control_sent)) : 0.0;
    snapshot.control_cmd_age_ms = perf.control_sent > 0 ? (perf.control_cmd_age_s * 1000.0 / static_cast<double>(perf.control_sent)) : 0.0;
    snapshot.control_total_latency_ms = perf.control_latency_samples > 0 ? (perf.control_total_latency_s * 1000.0 / static_cast<double>(perf.control_latency_samples)) : 0.0;
    snapshot.control_total_latency_full_ms = perf.control_latency_full_samples > 0 ? (perf.control_total_latency_full_s * 1000.0 / static_cast<double>(perf.control_latency_full_samples)) : 0.0;
    snapshot.control_total_apply_latency_ms = perf.control_apply_latency_samples > 0 ? (perf.control_total_apply_latency_s * 1000.0 / static_cast<double>(perf.control_apply_latency_samples)) : 0.0;
    snapshot.control_total_apply_latency_full_ms = perf.control_apply_latency_full_samples > 0 ? (perf.control_total_apply_latency_full_s * 1000.0 / static_cast<double>(perf.control_apply_latency_full_samples)) : 0.0;
    snapshot.control_stale_drop = perf.control_stale_drop;
    snapshot.control_mode_drop = perf.control_mode_drop;

    perf.reset(now);
    return snapshot;
}

}  // namespace

DeltaApp::DeltaApp(StaticConfig config, RuntimeConfig runtime)
    : config_(std::move(config)),
      runtime_store_(std::move(runtime)),
      capture_(makeDefaultCaptureSource(config_)),
      inference_(makeInferenceEngine(config_)),
      input_sender_(makeInputSender()),
      frontend_(makeRuntimeFrontend(config_, runtime_store_, shared_)),
      perf_(std::make_unique<RuntimePerfWindow>()) {
    if (capture_ && inference_) {
        capture_->setGpuConsumerStream(inference_->gpuInputStream());
    }
    const auto center = screenCenter(config_);
    std::lock_guard<std::mutex> lock(shared_.mutex);
    shared_.last_target_full = center;
    shared_.capture_focus_full = center;
    shared_.tracking_strategy = trackingStrategyName(runtime_store_.snapshot().tracking_strategy);
}

DeltaApp::~DeltaApp() = default;

int DeltaApp::run() {
    std::cout << "Delta native scaffold initialized.\n";
    std::cout << "Capture module: " << (capture_ ? capture_->name() : "none") << "\n";
    std::cout << "Inference module: " << (inference_ ? inference_->name() : "none") << "\n";
    std::cout << "Control module: " << (input_sender_ ? input_sender_->name() : "none") << "\n";
    if (!capture_ || !inference_ || !input_sender_) {
        std::cerr << "[fatal] Native runtime is missing a required module.\n";
        return 1;
    }

    inference_->setModelConfidence(runtime_store_.snapshot().model_conf);
    inference_->warmup();
    std::cout << "Runtime ready. Open the frontend and press Insert to exit.\n";

    try {
        if (frontend_) {
            frontend_->start();
        }
        if (config_.perf_log_enable && perf_) {
            perf_thread_ = AppThread([this]() { perfLoop(); });
        }
        if (!(inference_ && inference_->supportsGpuInput())) {
            capture_thread_ = AppThread([this]() { captureLoop(); });
        }
        inference_thread_ = AppThread([this]() { inferenceLoop(); });
        control_thread_ = AppThread([this]() { controlLoop(); });

        while (true) {
            {
                std::lock_guard<std::mutex> lock(shared_.mutex);
                if (!shared_.running) {
                    break;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    } catch (...) {
        stop();
        if (frontend_) {
            frontend_->stop();
        }
        throw;
    }

    stop();
    if (capture_thread_.joinable()) {
        capture_thread_.join();
    }
    if (inference_thread_.joinable()) {
        inference_thread_.join();
    }
    if (control_thread_.joinable()) {
        control_thread_.join();
    }
    if (perf_thread_.joinable()) {
        perf_thread_.join();
    }
    if (frontend_) {
        frontend_->stop();
    }

    return 0;
}

void DeltaApp::stop() {
    std::lock_guard<std::mutex> lock(shared_.mutex);
    shared_.running = false;
}

void DeltaApp::captureLoop() {
    try {
        const auto center = screenCenter(config_);
        const bool prefer_gpu = inference_ && inference_->supportsGpuInput();

        for (;;) {
            {
                std::lock_guard<std::mutex> lock(shared_.mutex);
                if (!shared_.running) {
                    break;
                }
            }

            std::pair<int, int> focus = center;
            {
                std::lock_guard<std::mutex> lock(shared_.mutex);
                if (shared_.target_found) {
                    focus = shared_.capture_focus_full;
                }
            }

            const CaptureRegion region = buildCaptureRegion(config_, focus.first, focus.second);
            bool captured = false;

            if (prefer_gpu) {
                const auto grab_start = SteadyClock::now();
                if (std::optional<GpuFramePacket> packet = capture_->grabGpu(region); packet.has_value()) {
                    frame_slot_.clear();
                    gpu_frame_slot_.put(std::move(*packet));
                    if (perf_) {
                        recordCapturePerf(*perf_, secondsSince(grab_start, SteadyClock::now()), false);
                    }
                    captured = true;
                } else if (perf_) {
                    recordCapturePerf(*perf_, secondsSince(grab_start, SteadyClock::now()), true);
                }
            }

            if (!captured) {
                const auto grab_start = SteadyClock::now();
                if (std::optional<FramePacket> packet = capture_->grab(region); packet.has_value()) {
                    gpu_frame_slot_.clear();
                    frame_slot_.put(std::move(*packet));
                    if (perf_) {
                        recordCapturePerf(*perf_, secondsSince(grab_start, SteadyClock::now()), false);
                    }
                    captured = true;
                } else if (perf_) {
                    recordCapturePerf(*perf_, secondsSince(grab_start, SteadyClock::now()), true);
                }
            }

            if (!captured) {
                std::this_thread::sleep_for(kCaptureIdleSleep);
            }
        }
    } catch (const std::exception& ex) {
        std::cerr << "[capture] loop failed: " << ex.what() << "\n";
        stop();
    }

    if (capture_) {
        capture_->close();
    }
}

void DeltaApp::inferenceLoop() {
    try {
        const auto center = screenCenter(config_);
        PIDController pid_x{};
        PIDController pid_y{};

        RuntimeConfig runtime = runtime_store_.snapshot();
        pid_x.configure(
            runtime.kp,
            runtime.ki,
            runtime.kd,
            runtime.integral_limit,
            runtime.anti_windup_gain,
            runtime.derivative_alpha,
            runtime.output_limit);
        pid_y.configure(
            runtime.kp,
            runtime.ki,
            runtime.kd,
            runtime.integral_limit,
            runtime.anti_windup_gain,
            runtime.derivative_alpha,
            runtime.output_limit);

        std::uint64_t last_pid_version = runtime_store_.version();
        std::uint64_t last_reset_token = runtime_store_.resetToken();
        TrackingStrategy last_tracking_strategy = runtime.tracking_strategy;
        auto tracker = makeTargetTracker(last_tracking_strategy, runtime.tracking_velocity_alpha);
        int lost_frames = 0;
        int active_target_cls = -1;
        float last_box_w = 0.0F;
        float last_box_h = 0.0F;
        std::optional<std::array<int, 4>> last_target_bbox;
        SteadyClock::time_point last_pid_tick{};
        SteadyClock::time_point last_track_tick{};

        {
            std::lock_guard<std::mutex> lock(shared_.mutex);
            shared_.tracking_strategy = trackingStrategyName(last_tracking_strategy);
        }

        for (;;) {
            {
                std::lock_guard<std::mutex> lock(shared_.mutex);
                if (!shared_.running) {
                    break;
                }
            }
            const auto loop_start = SteadyClock::now();

            runtime = runtime_store_.snapshot();
            const std::uint64_t pid_version = runtime_store_.version();
            const std::uint64_t reset_token = runtime_store_.resetToken();
            inference_->setModelConfidence(runtime.model_conf);
            const bool tracking_enabled = runtime.tracking_enabled;

            if (pid_version != last_pid_version) {
                pid_x.configure(
                    runtime.kp,
                    runtime.ki,
                    runtime.kd,
                    runtime.integral_limit,
                    runtime.anti_windup_gain,
                    runtime.derivative_alpha,
                    runtime.output_limit);
                pid_y.configure(
                    runtime.kp,
                    runtime.ki,
                    runtime.kd,
                    runtime.integral_limit,
                    runtime.anti_windup_gain,
                    runtime.derivative_alpha,
                    runtime.output_limit);
                last_pid_version = pid_version;
            }

            if (reset_token != last_reset_token) {
                pid_x.reset();
                pid_y.reset();
                last_reset_token = reset_token;
            }

            tracker->configure(runtime.tracking_velocity_alpha);
            if (runtime.tracking_strategy != last_tracking_strategy) {
                tracker = makeTargetTracker(runtime.tracking_strategy, runtime.tracking_velocity_alpha);
                pid_x.reset();
                pid_y.reset();
                lost_frames = 0;
                active_target_cls = -1;
                last_box_w = 0.0F;
                last_box_h = 0.0F;
                last_target_bbox.reset();
                last_pid_tick = {};
                last_track_tick = {};
                command_slot_.clear();
                {
                    std::lock_guard<std::mutex> lock(shared_.mutex);
                    clearAimStateLocked(shared_, center, runtime.tracking_strategy);
                }
                last_tracking_strategy = runtime.tracking_strategy;
                std::this_thread::sleep_for(kInferenceIdleSleep);
                continue;
            }

            ToggleState toggles{};
            bool prev_target_found = false;
            int prev_target_cls = -1;
            std::pair<int, int> prev_target_full = center;
            SystemClock::time_point prev_target_time{};
            {
                std::lock_guard<std::mutex> lock(shared_.mutex);
                toggles = shared_.toggles;
                prev_target_found = shared_.target_found;
                prev_target_cls = shared_.target_cls;
                prev_target_full = shared_.last_target_full;
                prev_target_time = shared_.target_time;
                shared_.tracking_strategy = trackingStrategyName(runtime.tracking_strategy);
            }

            const bool engage_active = (toggles.mode != 0)
                && isLeftHoldEngageSatisfied(
                    toggles.left_hold_engage,
                    runtime.left_hold_engage_button,
                    toggles.left_pressed,
                    toggles.right_pressed);
            const bool triggerbot_monitor_active = (toggles.mode != 0) && runtime.triggerbot_enable;

            if (!(engage_active || triggerbot_monitor_active)) {
                tracker->reset();
                pid_x.reset();
                pid_y.reset();
                lost_frames = 0;
                active_target_cls = -1;
                last_box_w = 0.0F;
                last_box_h = 0.0F;
                last_target_bbox.reset();
                last_pid_tick = {};
                last_track_tick = {};
                command_slot_.clear();
                {
                    std::lock_guard<std::mutex> lock(shared_.mutex);
                    clearAimStateLocked(shared_, center, runtime.tracking_strategy);
                }
                std::this_thread::sleep_for(kInferenceIdleSleep);
                continue;
            }

            std::optional<FramePacket> cpu_packet;
            std::optional<GpuFramePacket> gpu_packet;
            if (inference_->supportsGpuInput()) {
                std::pair<int, int> focus = center;
                {
                    std::lock_guard<std::mutex> lock(shared_.mutex);
                    if (shared_.target_found) {
                        focus = shared_.capture_focus_full;
                    }
                }
                const CaptureRegion region = buildCaptureRegion(config_, focus.first, focus.second);
                const auto grab_start = SteadyClock::now();
                gpu_packet = capture_->grabGpu(region);
                if (!gpu_packet.has_value()) {
                    cpu_packet = capture_->grab(region);
                }
                if (perf_) {
                    recordCapturePerf(*perf_, secondsSince(grab_start, SteadyClock::now()), !(gpu_packet.has_value() || cpu_packet.has_value()));
                }
            } else {
                cpu_packet = frame_slot_.try_take();
            }
            if (!gpu_packet.has_value() && !cpu_packet.has_value()) {
                std::this_thread::sleep_for(kInferenceIdleSleep);
                continue;
            }

            const SteadyClock::time_point frame_time = gpu_packet.has_value() ? gpu_packet->frame_time : cpu_packet->frame_time;
            const double frame_age = secondsSince(frame_time, SteadyClock::now());
            if (frame_age > kMaxFrameAgeSeconds) {
                if (perf_) {
                    recordInferencePerf(*perf_, frame_age, secondsSince(loop_start, SteadyClock::now()), true, false, 0.0, {}, std::nullopt);
                }
                continue;
            }

            const int target_cls = toggles.aimmode == 0 ? 1 : 0;
            if (active_target_cls != -1 && active_target_cls != target_cls) {
                tracker->reset();
                pid_x.reset();
                pid_y.reset();
                lost_frames = 0;
                active_target_cls = -1;
                last_box_w = 0.0F;
                last_box_h = 0.0F;
                last_target_bbox.reset();
                last_pid_tick = {};
                last_track_tick = {};
            }

            InferenceResult inference_result{};
            if (gpu_packet.has_value()) {
                inference_result = inference_->predictGpu(*gpu_packet, target_cls);
            } else if (cpu_packet.has_value()) {
                inference_result = inference_->predict(*cpu_packet, target_cls);
            }

            const CaptureRegion capture_region = gpu_packet.has_value() ? gpu_packet->capture : cpu_packet->capture;
            std::vector<Detection> detections = inference_result.detections;
            for (auto& detection : detections) {
                detection.bbox[0] += capture_region.left;
                detection.bbox[1] += capture_region.top;
                detection.bbox[2] += capture_region.left;
                detection.bbox[3] += capture_region.top;
                const auto [aim_x, aim_y] = detectionAimPoint(detection);
                detection.x = aim_x;
                detection.y = aim_y;
            }
            detections.erase(
                std::remove_if(
                    detections.begin(),
                    detections.end(),
                    [&](const Detection& detection) {
                        return detection.conf < runtime.detection_min_conf;
                    }),
                detections.end());

            std::optional<std::pair<float, float>> locked_point;
            if (tracking_enabled && tracker->initialized()) {
                const TrackerState tracker_state = tracker->state();
                locked_point = std::make_pair(tracker_state.x, tracker_state.y);
            } else if (tracking_enabled
                && prev_target_found
                && prev_target_cls == target_cls
                && prev_target_time != SystemClock::time_point{}
                && secondsSince(prev_target_time, SystemClock::now()) <= kTargetTimeoutSeconds) {
                locked_point = std::make_pair(static_cast<float>(prev_target_full.first), static_cast<float>(prev_target_full.second));
            }

            std::optional<StickyTargetPick> sticky;
            if (!locked_point.has_value()) {
                sticky = pickStickyTarget(
                    detections,
                    center.first,
                    center.second,
                    std::nullopt,
                    runtime.sticky_bias_px);
            } else {
                std::pair<float, float> assoc_ref = *locked_point;
                float assoc_limit = kTargetLockMaxJump;
                if (tracker->initialized()) {
                    const TrackerState assoc_state = tracker->state();
                    assoc_ref = {
                        assoc_state.x + (assoc_state.vx * kAssocPredictDt),
                        assoc_state.y + (assoc_state.vy * kAssocPredictDt),
                    };
                    assoc_limit += std::min(
                        kAssocMaxJumpPad,
                        std::sqrt((assoc_state.vx * assoc_state.vx) + (assoc_state.vy * assoc_state.vy)) * kAssocSpeedJumpGain);
                }
                std::vector<Detection> associated;
                associated.reserve(detections.size());
                for (const auto& detection : detections) {
                    const bool near_ref = pointToBoxDistance(detection.bbox, assoc_ref.first, assoc_ref.second) <= assoc_limit;
                    const bool overlaps_prev = last_target_bbox.has_value()
                        && bboxIou(detection.bbox, *last_target_bbox) >= kTrackerReinitMinIou;
                    if (near_ref || overlaps_prev) {
                        associated.push_back(detection);
                    }
                }
                if (!associated.empty()) {
                    sticky = pickStickyTarget(
                        associated,
                        center.first,
                        center.second,
                        locked_point,
                        runtime.sticky_bias_px);
                }
            }

            const auto tracker_start = SteadyClock::now();
            const SteadyClock::time_point now_tick = SteadyClock::now();
            const float dt = clamp(
                last_track_tick == SteadyClock::time_point{}
                    ? kMinTrackDt
                    : static_cast<float>(secondsSince(last_track_tick, now_tick)),
                kMinTrackDt,
                kMaxTrackDt);
            last_track_tick = now_tick;
            tracker->predict(dt);

            bool trigger_fire = false;
            if (sticky.has_value() && sticky->detection.has_value()) {
                const bool target_switched = sticky->switched
                    || (last_target_bbox.has_value() && bboxIou(sticky->detection->bbox, *last_target_bbox) < 0.05F);
                if (target_switched) {
                    pid_x.reset();
                    pid_y.reset();
                    last_pid_tick = now_tick;
                }
                if (!tracking_enabled || target_switched) {
                    tracker->reset();
                }
                tracker->update(sticky->detection->x, sticky->detection->y);
                active_target_cls = sticky->detection->cls;
                lost_frames = 0;
                last_box_w = static_cast<float>(std::max(1, sticky->detection->bbox[2] - sticky->detection->bbox[0]));
                last_box_h = static_cast<float>(std::max(1, sticky->detection->bbox[3] - sticky->detection->bbox[1]));
                last_target_bbox = sticky->detection->bbox;
                if (target_switched && runtime.ego_motion_reset_on_switch) {
                    std::lock_guard<std::mutex> lock(shared_.mutex);
                    resetEgoMotionStateLocked(shared_);
                }
                if (triggerbot_monitor_active) {
                    trigger_fire =
                        std::abs(sticky->detection->x - static_cast<float>(center.first)) <= (last_box_w * 0.5F)
                        && std::abs(sticky->detection->y - static_cast<float>(center.second)) <= (last_box_h * 0.5F);
                }
            } else if (!tracking_enabled || !tracker->initialized()) {
                command_slot_.clear();
                {
                    std::lock_guard<std::mutex> lock(shared_.mutex);
                    clearAimStateLocked(shared_, center, runtime.tracking_strategy);
                }
                if (perf_) {
                    recordInferencePerf(
                        *perf_,
                        frame_age,
                        secondsSince(loop_start, SteadyClock::now()),
                        false,
                        false,
                        secondsSince(tracker_start, SteadyClock::now()),
                        inference_result.timings,
                        std::nullopt);
                }
                continue;
            } else {
                ++lost_frames;
                if (lost_frames > runtime.target_max_lost_frames) {
                    tracker->reset();
                    pid_x.reset();
                    pid_y.reset();
                    lost_frames = 0;
                    active_target_cls = -1;
                    last_box_w = 0.0F;
                    last_box_h = 0.0F;
                    last_target_bbox.reset();
                    last_pid_tick = {};
                    command_slot_.clear();
                    {
                        std::lock_guard<std::mutex> lock(shared_.mutex);
                        clearAimStateLocked(shared_, center, runtime.tracking_strategy);
                    }
                    if (perf_) {
                        recordInferencePerf(
                            *perf_,
                            frame_age,
                            secondsSince(loop_start, SteadyClock::now()),
                            false,
                            false,
                            secondsSince(tracker_start, SteadyClock::now()),
                            inference_result.timings,
                            std::nullopt);
                    }
                    continue;
                }
            }

            const TrackerState tracker_state = tracker->state();
            const float ff_scale = tracker->feedforwardScale();
            float ctrl_sent_vx_ema = 0.0F;
            float ctrl_sent_vy_ema = 0.0F;
            {
                std::lock_guard<std::mutex> lock(shared_.mutex);
                ctrl_sent_vx_ema = shared_.ctrl_sent_vx_ema;
                ctrl_sent_vy_ema = shared_.ctrl_sent_vy_ema;
            }
            float vx = tracker_state.vx;
            float vy = tracker_state.vy;
            const float prediction_time = std::max(0.0F, runtime.prediction_time);
            const float base_predicted_x = tracker_state.x + (vx * prediction_time);
            const float base_predicted_y = tracker_state.y + (vy * prediction_time);
            const float base_aim_y = base_predicted_y;
            float ego_gate_x = 1.0F;
            float ego_gate_y = 1.0F;
            if (runtime.ego_motion_error_gate_enable) {
                if (runtime.ego_motion_error_gate_normalize_by_box && runtime.ego_motion_error_gate_norm_threshold > 1e-6F) {
                    const float norm_box_w = std::max(1.0F, last_box_w);
                    const float norm_box_h = std::max(1.0F, last_box_h);
                    const float normalized_error_x = std::abs(base_predicted_x - static_cast<float>(center.first)) / norm_box_w;
                    const float normalized_error_y = std::abs(base_aim_y - static_cast<float>(center.second)) / norm_box_h;
                    ego_gate_x = clamp(
                        1.0F - (normalized_error_x / runtime.ego_motion_error_gate_norm_threshold),
                        0.0F,
                        1.0F);
                    ego_gate_y = clamp(
                        1.0F - (normalized_error_y / runtime.ego_motion_error_gate_norm_threshold),
                        0.0F,
                        1.0F);
                } else if (runtime.ego_motion_error_gate_px > 1e-6F) {
                    ego_gate_x = clamp(
                        1.0F - (std::abs(base_predicted_x - static_cast<float>(center.first)) / runtime.ego_motion_error_gate_px),
                        0.0F,
                        1.0F);
                    ego_gate_y = clamp(
                        1.0F - (std::abs(base_aim_y - static_cast<float>(center.second)) / runtime.ego_motion_error_gate_px),
                        0.0F,
                        1.0F);
                }
            }
            if (runtime.ego_motion_comp_enable && ff_scale > 0.0F) {
                const float ego_vx = clamp(
                    ctrl_sent_vx_ema * runtime.ego_motion_comp_gain_x * ff_scale * ego_gate_x,
                    -kEgoMotionCompMaxPxS,
                    kEgoMotionCompMaxPxS);
                const float ego_vy = clamp(
                    ctrl_sent_vy_ema * runtime.ego_motion_comp_gain_y * ff_scale * ego_gate_y,
                    -kEgoMotionCompMaxPxS,
                    kEgoMotionCompMaxPxS);
                vx += ego_vx;
                vy += ego_vy;
            }
            float speed = std::sqrt((vx * vx) + (vy * vy));
            if (speed > kMaxTrackSpeedPxS && speed > 1e-6F) {
                const float speed_scale = kMaxTrackSpeedPxS / speed;
                vx *= speed_scale;
                vy *= speed_scale;
                speed = kMaxTrackSpeedPxS;
            }
            const float predicted_x = tracker_state.x + (vx * prediction_time);
            const float predicted_y = tracker_state.y + (vy * prediction_time);
            const float aim_y = predicted_y;
            const float pid_dt = clamp(
                last_pid_tick == SteadyClock::time_point{}
                    ? kMinTrackDt
                    : static_cast<float>(secondsSince(last_pid_tick, now_tick)),
                kMinTrackDt,
                kMaxTrackDt);
            last_pid_tick = now_tick;

            const bool pid_integrate = std::abs(runtime.ki) > 1e-12F;
            const float pid_term_x = runtime.pid_enable
                ? pid_x.update(0.0F, static_cast<float>(center.first) - predicted_x, pid_dt, pid_integrate)
                : (predicted_x - static_cast<float>(center.first));
            const float pid_term_y = runtime.pid_enable
                ? pid_y.update(0.0F, static_cast<float>(center.second) - aim_y, pid_dt, pid_integrate)
                : (aim_y - static_cast<float>(center.second));

            const float ff_x = (vx * dt) * ff_scale;
            const float ff_y = (vy * dt) * ff_scale;
            const float desired_x = pid_term_x + ff_x;
            const float desired_y = pid_term_y + ff_y;
            const int dx = engage_active ? static_cast<int>(std::lround(clamp(desired_x, -static_cast<float>(kRawMaxStepX), static_cast<float>(kRawMaxStepX)))) : 0;
            const int dy = engage_active
                ? static_cast<int>(std::lround(clamp(desired_y, -static_cast<float>(runtime.raw_max_step_y), static_cast<float>(runtime.raw_max_step_y))))
                : 0;

            const int focus_x = clamp(static_cast<int>(std::lround(tracker_state.x)), 0, config_.screen_w - 1);
            const int focus_y = clamp(static_cast<int>(std::lround(tracker_state.y)), 0, config_.screen_h - 1);
            const int selected_cls = active_target_cls != -1 ? active_target_cls : target_cls;
            const auto now_system = SystemClock::now();
            const double tracker_elapsed = secondsSince(tracker_start, SteadyClock::now());

            {
                std::lock_guard<std::mutex> lock(shared_.mutex);
                shared_.target_found = true;
                shared_.target_cls = selected_cls;
                shared_.target_speed = speed;
                shared_.aim_dx = dx;
                shared_.aim_dy = dy;
                shared_.last_target_full = {focus_x, focus_y};
                shared_.capture_focus_full = {focus_x, focus_y};
                shared_.target_time = now_system;
                shared_.tracking_strategy = trackingStrategyName(runtime.tracking_strategy);
            }

            if (engage_active || trigger_fire) {
                command_slot_.put(CommandPacket{
                    .dx = dx,
                    .dy = dy,
                    .generated_at = now_system,
                    .frame_time = gpu_packet.has_value() ? gpu_packet->capture_time : cpu_packet->capture_time,
                    .capture_time = gpu_packet.has_value() ? gpu_packet->capture_time : cpu_packet->capture_time,
                    .synthetic_recoil = false,
                    .trigger_fire = trigger_fire,
                });
            }
            if (perf_) {
                recordInferencePerf(
                    *perf_,
                    frame_age,
                    secondsSince(loop_start, SteadyClock::now()),
                    false,
                    true,
                    tracker_elapsed,
                    inference_result.timings,
                    secondsSince(gpu_packet.has_value() ? gpu_packet->capture_time : cpu_packet->capture_time, now_system));
            }
        }
    } catch (const std::exception& ex) {
        std::cerr << "[inference] loop failed: " << ex.what() << "\n";
        stop();
    }

    if (capture_ && inference_ && inference_->supportsGpuInput()) {
        capture_->close();
    }
}

void DeltaApp::controlLoop() {
    if (!input_sender_) {
        return;
    }

    Win32HotkeySource hotkeys;
    InputSnapshot previous{};
    SteadyClock::time_point last_mode_toggle{};
    SteadyClock::time_point last_aimmode_toggle{};
    SteadyClock::time_point last_hold_toggle{};
    SteadyClock::time_point last_recoil_toggle{};
    SteadyClock::time_point last_triggerbot_toggle{};
    SystemClock::time_point last_trigger_click{};
    const auto center = screenCenter(config_);

    for (;;) {
        {
            std::lock_guard<std::mutex> lock(shared_.mutex);
            if (!shared_.running) {
                break;
            }
        }

        const auto steady_now = SteadyClock::now();
        const auto system_now = SystemClock::now();
        const InputSnapshot snapshot = hotkeys.poll();
        RuntimeConfig runtime = runtime_store_.snapshot();

        if (snapshot.insert_pressed) {
            stop();
            break;
        }

        bool triggerbot_toggle = false;
        {
            std::lock_guard<std::mutex> lock(shared_.mutex);
            shared_.toggles.left_pressed = snapshot.left_pressed;
            shared_.toggles.right_pressed = snapshot.right_pressed;

            if (risingEdge(snapshot.x2_pressed, previous.x2_pressed) && (steady_now - last_mode_toggle) >= kToggleCooldown) {
                shared_.toggles.mode = (shared_.toggles.mode + 1) % 2;
                last_mode_toggle = steady_now;
                const int mode = shared_.toggles.mode;
                if (mode == 0) {
                    command_slot_.clear();
                    clearAimStateLocked(shared_, center, runtime.tracking_strategy);
                }
                if (config_.debug_log) {
                    std::cout << "[control] Mode: " << mode << " (" << (mode == 1 ? "ACTIVE" : "OFF") << ")\n";
                }
                playToggleBeep(mode == 1 ? 1000 : 500);
            }
            if (risingEdge(snapshot.x1_pressed, previous.x1_pressed) && (steady_now - last_aimmode_toggle) >= kToggleCooldown) {
                shared_.toggles.aimmode = (shared_.toggles.aimmode + 1) % 2;
                last_aimmode_toggle = steady_now;
                if (config_.debug_log) {
                    std::cout << "[control] AimMode: " << shared_.toggles.aimmode << "\n";
                }
                playToggleBeep(shared_.toggles.aimmode == 0 ? 1200 : 600);
            }
            if (risingEdge(snapshot.f6_pressed, previous.f6_pressed) && (steady_now - last_hold_toggle) >= kToggleCooldown) {
                shared_.toggles.left_hold_engage = !shared_.toggles.left_hold_engage;
                last_hold_toggle = steady_now;
                if (shared_.toggles.left_hold_engage && !isLeftHoldEngageSatisfied(
                        true,
                        runtime.left_hold_engage_button,
                        shared_.toggles.left_pressed,
                        shared_.toggles.right_pressed)) {
                    command_slot_.clear();
                    clearAimStateLocked(shared_, center, runtime.tracking_strategy);
                }
                if (config_.debug_log) {
                    std::cout << "[control] HoldEngage: " << (shared_.toggles.left_hold_engage ? "ON" : "OFF") << "\n";
                }
                playToggleBeep(shared_.toggles.left_hold_engage ? 1400 : 700);
            }
            if (risingEdge(snapshot.f7_pressed, previous.f7_pressed) && (steady_now - last_recoil_toggle) >= kToggleCooldown) {
                shared_.toggles.recoil_tune_fallback = !shared_.toggles.recoil_tune_fallback;
                last_recoil_toggle = steady_now;
                if (config_.debug_log) {
                    std::cout << "[control] RecoilTuneFallback: " << (shared_.toggles.recoil_tune_fallback ? "ON" : "OFF") << "\n";
                }
                playToggleBeep(shared_.toggles.recoil_tune_fallback ? 1500 : 800);
            }
            if (risingEdge(snapshot.f8_pressed, previous.f8_pressed) && (steady_now - last_triggerbot_toggle) >= kToggleCooldown) {
                last_triggerbot_toggle = steady_now;
                triggerbot_toggle = true;
            }
        }
        previous = snapshot;

        if (triggerbot_toggle) {
            runtime.triggerbot_enable = !runtime.triggerbot_enable;
            runtime_store_.update(runtime);
            if (config_.debug_log) {
                std::cout << "[control] TriggerBot: " << (runtime.triggerbot_enable ? "ON" : "OFF") << "\n";
            }
            playToggleBeep(runtime.triggerbot_enable ? 1600 : 900);
        }

        input_sender_->configure(MouseSenderConfig{
            .gain_x = runtime.sendinput_gain_x,
            .gain_y = runtime.sendinput_gain_y,
            .max_step = runtime.sendinput_max_step,
        });

        ToggleState toggles{};
        {
            std::lock_guard<std::mutex> lock(shared_.mutex);
            toggles = shared_.toggles;
        }

        std::optional<CommandPacket> cmd = command_slot_.wait_take_for(kControlCommandWait);
        if (!cmd.has_value()) {
            const bool mode_ok = runtime.recoil_tune_fallback_ignore_mode_check || (toggles.mode != 0);
            const bool engage_ok = isLeftHoldEngageSatisfied(
                toggles.left_hold_engage,
                runtime.left_hold_engage_button,
                toggles.left_pressed,
                toggles.right_pressed);
            if (toggles.recoil_tune_fallback && toggles.left_pressed && mode_ok && engage_ok) {
                cmd = CommandPacket{
                    .dx = 0,
                    .dy = 0,
                    .generated_at = system_now,
                    .frame_time = system_now,
                    .capture_time = system_now,
                    .synthetic_recoil = true,
                    .trigger_fire = false,
                };
            } else {
                std::lock_guard<std::mutex> lock(shared_.mutex);
                decayEgoMotionStateLocked(shared_);
                std::this_thread::sleep_for(kControlIdleSleep);
                continue;
            }
        }

        if (cmd->generated_at != SystemClock::time_point{} && secondsSince(cmd->generated_at, system_now) > kCommandTimeoutSeconds) {
            {
                std::lock_guard<std::mutex> lock(shared_.mutex);
                decayEgoMotionStateLocked(shared_);
            }
            if (perf_) {
                recordControlPerf(
                    *perf_,
                    secondsSince(cmd->generated_at, system_now),
                    false,
                    0.0,
                    true,
                    false,
                    cmd->frame_time == SystemClock::time_point{} ? std::nullopt : std::optional<double>(secondsSince(cmd->frame_time, system_now)),
                    cmd->capture_time == SystemClock::time_point{} ? std::nullopt : std::optional<double>(secondsSince(cmd->capture_time, system_now)),
                    std::nullopt,
                    std::nullopt);
            }
            continue;
        }

        const bool engage_active = cmd->synthetic_recoil
            || (toggles.mode != 0 && isLeftHoldEngageSatisfied(
                toggles.left_hold_engage,
                runtime.left_hold_engage_button,
                toggles.left_pressed,
                toggles.right_pressed));
        const bool trigger_enabled = runtime.triggerbot_enable;
        const bool trigger_fire = trigger_enabled && cmd->trigger_fire;
        if (!(engage_active || trigger_fire)) {
            {
                std::lock_guard<std::mutex> lock(shared_.mutex);
                decayEgoMotionStateLocked(shared_);
            }
            if (perf_) {
                recordControlPerf(
                    *perf_,
                    cmd->generated_at == SystemClock::time_point{} ? 0.0 : secondsSince(cmd->generated_at, system_now),
                    false,
                    0.0,
                    false,
                    true,
                    cmd->frame_time == SystemClock::time_point{} ? std::nullopt : std::optional<double>(secondsSince(cmd->frame_time, system_now)),
                    cmd->capture_time == SystemClock::time_point{} ? std::nullopt : std::optional<double>(secondsSince(cmd->capture_time, system_now)),
                    std::nullopt,
                    std::nullopt);
            }
            continue;
        }

        int dx = cmd->dx;
        int dy = cmd->dy;
        const bool trigger_will_click = trigger_fire
            && !toggles.left_pressed
            && (last_trigger_click == SystemClock::time_point{}
                || secondsSince(last_trigger_click, system_now) >= runtime.triggerbot_click_cooldown_s);

        if (toggles.left_pressed || trigger_will_click) {
            dy = clamp(
                static_cast<int>(std::lround(static_cast<double>(dy) + static_cast<double>(runtime.recoil_compensation_y_px))),
                -runtime.raw_max_step_y,
                runtime.raw_max_step_y);
        }

        bool movement_sent = false;
        const auto send_start = SteadyClock::now();
        if (dx != 0 || dy != 0) {
            movement_sent = input_sender_->sendRelative(dx, dy);
        }
        bool trigger_sent = false;
        if (trigger_will_click) {
            trigger_sent = input_sender_->clickLeft(runtime.triggerbot_click_hold_s);
            if (trigger_sent) {
                last_trigger_click = SystemClock::now();
            }
        }
        const auto send_end_tick = SteadyClock::now();
        const double send_elapsed = secondsSince(send_start, send_end_tick);
        const auto send_end_system = SystemClock::now();

        if (dx == 0 && dy == 0 && !trigger_sent) {
            {
                std::lock_guard<std::mutex> lock(shared_.mutex);
                decayEgoMotionStateLocked(shared_);
            }
            if (perf_) {
                recordControlPerf(
                    *perf_,
                    cmd->generated_at == SystemClock::time_point{} ? 0.0 : secondsSince(cmd->generated_at, system_now),
                    false,
                    0.0,
                    false,
                    false,
                    cmd->frame_time == SystemClock::time_point{} ? std::nullopt : std::optional<double>(secondsSince(cmd->frame_time, system_now)),
                    cmd->capture_time == SystemClock::time_point{} ? std::nullopt : std::optional<double>(secondsSince(cmd->capture_time, system_now)),
                    std::nullopt,
                    std::nullopt);
            }
            continue;
        }

        if ((dx != 0 || dy != 0) && movement_sent) {
            std::lock_guard<std::mutex> lock(shared_.mutex);
            if (shared_.ctrl_last_send_tick != SteadyClock::time_point{}) {
                const float send_dt = std::max(1e-4F, static_cast<float>(secondsSince(shared_.ctrl_last_send_tick, send_end_tick)));
                const float sent_vx = clamp(static_cast<float>(dx) / send_dt, -kEgoMotionCompMaxPxS, kEgoMotionCompMaxPxS);
                const float sent_vy = clamp(static_cast<float>(dy) / send_dt, -kEgoMotionCompMaxPxS, kEgoMotionCompMaxPxS);
                shared_.ctrl_sent_vx_ema = emaUpdateSigned(shared_.ctrl_sent_vx_ema, sent_vx, kEgoMotionCompAlpha);
                shared_.ctrl_sent_vy_ema = emaUpdateSigned(shared_.ctrl_sent_vy_ema, sent_vy, kEgoMotionCompAlpha);
            }
            shared_.ctrl_last_send_tick = send_end_tick;
        } else if (!trigger_sent) {
            std::lock_guard<std::mutex> lock(shared_.mutex);
            decayEgoMotionStateLocked(shared_);
        }

        if (movement_sent || trigger_sent) {
            std::lock_guard<std::mutex> lock(shared_.mutex);
            shared_.aim_dx = dx;
            shared_.aim_dy = dy;
        }
        if (perf_) {
            const bool sent_ok = movement_sent || trigger_sent;
            recordControlPerf(
                *perf_,
                cmd->generated_at == SystemClock::time_point{} ? 0.0 : secondsSince(cmd->generated_at, system_now),
                sent_ok,
                sent_ok ? send_elapsed : 0.0,
                false,
                false,
                cmd->frame_time == SystemClock::time_point{} ? std::nullopt : std::optional<double>(secondsSince(cmd->frame_time, system_now)),
                cmd->capture_time == SystemClock::time_point{} ? std::nullopt : std::optional<double>(secondsSince(cmd->capture_time, system_now)),
                sent_ok && cmd->frame_time != SystemClock::time_point{} ? std::optional<double>(secondsSince(cmd->frame_time, send_end_system)) : std::nullopt,
                sent_ok && cmd->capture_time != SystemClock::time_point{} ? std::optional<double>(secondsSince(cmd->capture_time, send_end_system)) : std::nullopt);
        }
    }
}

void DeltaApp::perfLoop() {
    if (!perf_) {
        return;
    }

    for (;;) {
        {
            std::lock_guard<std::mutex> lock(shared_.mutex);
            if (!shared_.running) {
                break;
            }
        }
        std::this_thread::sleep_for(kPerfLoopSleep);

        const std::optional<PerfLogSnapshot> snapshot = takePerfSnapshot(*perf_, kPerfLogIntervalSeconds);
        if (!snapshot.has_value()) {
            continue;
        }

        ToggleState toggles{};
        {
            std::lock_guard<std::mutex> lock(shared_.mutex);
            toggles = shared_.toggles;
        }
        const RuntimeConfig runtime = runtime_store_.snapshot();
        const bool perf_engaged = (toggles.mode != 0)
            && isLeftHoldEngageSatisfied(
                toggles.left_hold_engage,
                runtime.left_hold_engage_button,
                toggles.left_pressed,
                toggles.right_pressed);
        if (!kPerfLogWhenModeOff && !perf_engaged) {
            continue;
        }

        std::cout << std::fixed << std::setprecision(2)
                  << "[PERF] "
                  << "cap=" << snapshot->cap_fps << "fps grab=" << snapshot->cap_grab_ms << "ms none=" << snapshot->cap_none
                  << " | inf=" << snapshot->infer_fps << "fps loop=" << snapshot->infer_loop_ms << "ms age="
                  << snapshot->infer_age_ms << "/" << snapshot->infer_age_max_ms << "ms stale=" << snapshot->infer_stale
                  << " lock=" << (snapshot->infer_lock_rate * 100.0) << "% | trk=" << snapshot->tracker_hz
                  << "Hz@" << snapshot->tracker_ms << "ms";
        if (snapshot->infer_backend_samples > 0) {
            std::cout << " onnx(pre/exec/post)="
                      << snapshot->infer_backend_pre_ms << "/"
                      << snapshot->infer_backend_exec_ms << "/"
                      << snapshot->infer_backend_post_ms << "ms";
        }
        std::cout << " | ctl=" << snapshot->control_send_hz << "Hz send=" << snapshot->control_send_ms
                  << "ms cmdAge=" << snapshot->control_cmd_age_ms << "ms e2e=" << snapshot->control_total_latency_ms
                  << "ms e2eIn=" << snapshot->control_total_apply_latency_ms
                  << "ms e2eFull=" << snapshot->control_total_latency_full_ms
                  << "ms e2eFullIn=" << snapshot->control_total_apply_latency_full_ms
                  << "ms drop(stale/mode)=" << snapshot->control_stale_drop
                  << "/" << snapshot->control_mode_drop
                  << " aimPipe=" << snapshot->infer_cmd_ms << "ms\n";
        std::cout.unsetf(std::ios::floatfield);
    }
}

}  // namespace delta
