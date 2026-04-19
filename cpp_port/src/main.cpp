#include "delta/app.hpp"
#include "delta/config.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <optional>
#include <string>

namespace {

std::optional<int> envInt(const char* key) {
    const char* raw = std::getenv(key);
    if (raw == nullptr || *raw == '\0') {
        return std::nullopt;
    }
    try {
        return std::stoi(raw);
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<bool> envBool(const char* key) {
    const char* raw = std::getenv(key);
    if (raw == nullptr || *raw == '\0') {
        return std::nullopt;
    }
    std::string value(raw);
    std::transform(value.begin(), value.end(), value.begin(), [](const unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    if (value == "1" || value == "true" || value == "yes" || value == "on") {
        return true;
    }
    if (value == "0" || value == "false" || value == "no" || value == "off") {
        return false;
    }
    return std::nullopt;
}

}  // namespace

int main() {
    std::cout.setf(std::ios::unitbuf);
    try {
        delta::StaticConfig config{};
        if (const auto capture_crop_size = envInt("DELTA_CAPTURE_CROP_SIZE"); capture_crop_size.has_value()) {
            config.capture_crop_size = std::max(0, *capture_crop_size);
        }
        if (const auto async_gpu_capture = envBool("DELTA_ASYNC_GPU_CAPTURE"); async_gpu_capture.has_value()) {
            config.async_gpu_capture_enable = *async_gpu_capture;
        }
        delta::RuntimeConfig runtime{};
        if (const auto async_gpu_capture_fresh_only = envBool("DELTA_ASYNC_GPU_CAPTURE_FRESH_ONLY");
            async_gpu_capture_fresh_only.has_value()) {
            runtime.async_gpu_capture_fresh_only_enable = *async_gpu_capture_fresh_only;
        }
        if (const auto tensorrt_inline_fresh_only = envBool("DELTA_TENSORRT_INLINE_FRESH_ONLY");
            tensorrt_inline_fresh_only.has_value()) {
            runtime.tensorrt_inline_fresh_only_enable = *tensorrt_inline_fresh_only;
        }
        delta::DeltaApp app(config, runtime);
        return app.run();
    } catch (const std::exception& error) {
        std::cerr << "[fatal] " << error.what() << "\n";
        return 1;
    }
}
