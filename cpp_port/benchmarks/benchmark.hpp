#pragma once

#include <filesystem>
#include <string>
#include <nlohmann/json.hpp>
#include "delta/config.hpp"

namespace delta::bench {
using Json = nlohmann::json;

struct Options {
    std::string suite = "quick";
    std::uint64_t seed = 42;
    bool timing = false;
    int repeats = 3;
    std::string scenario;
    // An evaluation policy, not a tuned production requirement.
    double loss_deadline_ms = 150.0;
};

Json runtimeJson(const RuntimeConfig& config);
RuntimeConfig runtimeFromJson(const Json& config);
Json runSuite(const StaticConfig& config, const RuntimeConfig& runtime, const Options& options);
Json compareRuns(const Json& baseline, const Json& candidate);
Json scoreFrames(const Json& frames, double loss_deadline_ms);
void writeReport(const std::filesystem::path& path, const Json& run, const Json& baseline = nullptr);
void validateOptions(const Options& options);
} // namespace delta::bench
