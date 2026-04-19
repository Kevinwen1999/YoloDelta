#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace delta {

using SteadyClock = std::chrono::steady_clock;
using SystemClock = std::chrono::system_clock;

template <typename T>
constexpr T clamp(T value, T low, T high) {
    return value < low ? low : (value > high ? high : value);
}

struct CaptureRegion {
    int left = 0;
    int top = 0;
    int width = 0;
    int height = 0;
};

enum class PixelFormat {
    Bgr8,
    Bgra8,
};

struct Detection {
    std::array<int, 4> bbox{0, 0, 0, 0};
    float x = 0.0F;
    float y = 0.0F;
    int cls = -1;
    float conf = 0.0F;
};

struct TrackerState {
    float x = 0.0F;
    float y = 0.0F;
    float vx = 0.0F;
    float vy = 0.0F;
    float ax = 0.0F;
    float ay = 0.0F;
};

struct CaptureTimings {
    double acquire_s = 0.0;
    double d3d_copy_s = 0.0;
    double d3d_sync_s = 0.0;
    double cuda_copy_s = 0.0;
    double cpu_copy_s = 0.0;
    double cached_reuse_s = 0.0;
    bool used_cached_frame = false;
};

struct FramePacket {
    std::vector<std::uint8_t> bgr;
    int width = 0;
    int height = 0;
    CaptureRegion capture{};
    SteadyClock::time_point acquire_started{};
    SteadyClock::time_point frame_ready{};
    SteadyClock::time_point capture_done{};
    SteadyClock::time_point frame_time{};
    SystemClock::time_point capture_time{};
    CaptureTimings timings{};
};

struct GpuFramePacket {
    void* device_ptr = nullptr;
    std::size_t pitch_bytes = 0;
    int width = 0;
    int height = 0;
    PixelFormat pixel_format = PixelFormat::Bgra8;
    CaptureRegion capture{};
    SteadyClock::time_point acquire_started{};
    SteadyClock::time_point frame_ready{};
    SteadyClock::time_point capture_done{};
    SteadyClock::time_point frame_time{};
    SystemClock::time_point capture_time{};
    CaptureTimings timings{};
    void* cuda_stream = nullptr;
    void* ready_event = nullptr;
    std::shared_ptr<void> lifetime;
};

struct CommandPacket {
    int dx = 0;
    int dy = 0;
    SteadyClock::time_point acquire_started{};
    SteadyClock::time_point frame_ready{};
    SteadyClock::time_point capture_done{};
    SteadyClock::time_point cmd_generated{};
    SystemClock::time_point generated_at{};
    SystemClock::time_point frame_time{};
    SystemClock::time_point capture_time{};
    bool target_detected = false;
    bool synthetic_recoil = false;
    bool trigger_fire = false;
};

struct PerfSnapshot {
    double capture_fps = 0.0;
    double inference_fps = 0.0;
    double control_hz = 0.0;
    double capture_grab_ms = 0.0;
    double infer_pre_ms = 0.0;
    double infer_exec_ms = 0.0;
    double infer_post_ms = 0.0;
    double end_to_end_ms = 0.0;
};

}  // namespace delta
