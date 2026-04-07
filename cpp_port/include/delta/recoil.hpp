#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "delta/config.hpp"
#include "delta/core.hpp"
#include "delta/recoil_types.hpp"

namespace delta {

struct RecoilMarker {
    std::string id;
    double x = 0.0;
    double y = 0.0;
};

struct RecoilImage {
    std::string id;
    std::string name;
    std::string asset_path;
    int width = 0;
    int height = 0;
    std::vector<RecoilMarker> markers;
};

struct RecoilStep {
    int index = 0;
    double pattern_x = 0.0;
    double pattern_y = 0.0;
    int duration_ms = 0;
    std::string source_image_id;
    std::string source_marker_id;
};

struct RecoilProfileSummary {
    std::string id;
    std::string name;
    std::string updated_at;
    int shot_count = 0;
    double scale_factor = 0.0;
    double horizontal_scale_factor = 0.0;
    int fire_interval_ms = 0;
    bool valid = true;
    std::string error;
};

struct RecoilProfile {
    int schema_version = 1;
    std::string id;
    std::string name;
    std::string created_at;
    std::string updated_at;
    double scale_factor = 1.0;
    double horizontal_scale_factor = 1.0;
    int fire_interval_ms = 100;
    std::vector<RecoilImage> images;
    std::vector<RecoilStep> steps;
    std::filesystem::path source_path;
    std::filesystem::file_time_type file_write_time{};
};

struct RecoilSchedulerUpdate {
    RecoilRuntimeState state{};
    PendingRecoilDelta delta{};
    bool clear_pending = false;
};

const char* recoilModeName(RecoilMode mode);
RecoilMode parseRecoilMode(std::string_view value);

bool ensureRecoilProfilesDirectory(const StaticConfig& config, std::string* error = nullptr);
std::filesystem::path recoilProfilesRoot(const StaticConfig& config);
std::filesystem::path recoilProfilePath(const StaticConfig& config, std::string_view profile_id);
std::filesystem::path recoilAssetsDir(const StaticConfig& config, std::string_view profile_id);

std::vector<RecoilProfileSummary> listRecoilProfiles(const StaticConfig& config);
std::optional<RecoilProfile> loadRecoilProfile(const StaticConfig& config, std::string_view profile_id, std::string& error);

class RecoilScheduler {
public:
    explicit RecoilScheduler(StaticConfig config);

    RecoilSchedulerUpdate tick(
        const RuntimeConfig& runtime,
        bool recoil_enabled,
        bool left_pressed,
        bool x1_pressed,
        const SteadyClock::time_point& now = SteadyClock::now());

private:
    void resetSequence();
    void resetStateForProfile(const RuntimeConfig& runtime);
    bool refreshProfile(const RuntimeConfig& runtime, const SteadyClock::time_point& now);
    int durationForStep(std::size_t index) const;

    StaticConfig config_{};
    RecoilRuntimeState state_{};
    std::optional<RecoilProfile> loaded_profile_;
    std::string watched_profile_id_;
    std::filesystem::file_time_type watched_write_time_{};
    bool spray_active_ = false;
    std::size_t current_step_index_ = 0;
    SteadyClock::time_point segment_started_at_{};
    SteadyClock::time_point next_step_at_{};
    SteadyClock::time_point last_file_check_{};
    double last_target_x_ = 0.0;
    double last_target_y_ = 0.0;
    double carry_x_ = 0.0;
    double carry_y_ = 0.0;
    bool pending_clear_requested_ = false;
    RecoilMode last_mode_ = RecoilMode::Legacy;
    std::string last_selected_profile_id_;
};

}  // namespace delta
