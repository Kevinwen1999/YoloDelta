#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "control_response.hpp"

int main(int argc, char** argv) {
    try {
        delta::StaticConfig config;
        delta::RuntimeConfig runtime;
        delta::bench::ControlResponseOptions options;
        // Isolate continuous control from acquisition and detector uncertainty.
        runtime.detection_dampening_enable = false;
        runtime.target_guard_enable = false;
        runtime.adaptive_capture_crop_enable = false;
        std::string output;
        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            if (++i >= argc) throw std::invalid_argument("Missing argument");
            const std::string value = argv[i];
            if (key == "--output")
                output = value;
            else if (key == "--gain")
                options.pixels_per_count = std::stod(value);
            else if (key == "--delay-ms")
                options.observation_delay_ms = std::stod(value);
            else if (key == "--transport-delay-ms")
                options.transport_delay_ms = std::stod(value);
            else if (key == "--speed")
                options.speed_px_s = std::stod(value);
            else if (key == "--initial-error")
                options.initial_error_px = std::stod(value);
            else if (key == "--cadence-ms")
                options.cadence_ms = std::stod(value);
            else if (key == "--frames")
                options.frames = std::stoi(value);
            else if (key == "--reversal")
                options.reversal = std::stoi(value) != 0;
            else if (key == "--config") {
                std::ifstream file(value);
                delta::bench::Json overrides;
                file >> overrides;
                auto json = delta::bench::runtimeJson(runtime);
                json.update(overrides);
                runtime = delta::bench::runtimeFromJson(json);
            } else
                throw std::invalid_argument("Unknown argument " + key);
        }
        if (output.empty() || std::filesystem::exists(output) || options.frames < 20 ||
            options.frames > 10000 || !std::isfinite(options.cadence_ms) ||
            !std::isfinite(options.observation_delay_ms) || !std::isfinite(options.pixels_per_count) ||
            !std::isfinite(options.transport_delay_ms) || options.transport_delay_ms<0 || options.transport_delay_ms>50 ||
            !std::isfinite(options.speed_px_s) || !std::isfinite(options.initial_error_px) ||
            std::abs(options.initial_error_px) > 500 || std::abs(options.speed_px_s) > 6000 ||
            options.cadence_ms < 1 || options.cadence_ms > 50 || options.pixels_per_count <= 0 ||
            options.observation_delay_ms < 0 || options.observation_delay_ms > options.cadence_ms ||
            options.frames*options.cadence_ms+options.observation_delay_ms<=500)
            throw std::invalid_argument("Provide output, positive gain and delay no greater than cadence");
        const auto result = delta::bench::runControlResponse(config, runtime, options);
        std::ofstream file(output);
        file << result.dump(2);
        if (!file) throw std::runtime_error("Cannot write response result");
        std::cout << result["summary"].dump() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
