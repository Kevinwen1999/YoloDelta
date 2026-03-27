#include "delta/app.hpp"
#include "delta/config.hpp"

#include <algorithm>
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

}  // namespace

int main() {
    std::cout.setf(std::ios::unitbuf);
    try {
        delta::StaticConfig config{};
        if (const auto capture_crop_size = envInt("DELTA_CAPTURE_CROP_SIZE"); capture_crop_size.has_value()) {
            config.capture_crop_size = std::max(0, *capture_crop_size);
        }
        delta::RuntimeConfig runtime{};
        delta::DeltaApp app(config, runtime);
        return app.run();
    } catch (const std::exception& error) {
        std::cerr << "[fatal] " << error.what() << "\n";
        return 1;
    }
}
