#include "delta/recoil.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <sstream>
#include <system_error>

#include <nlohmann/json.hpp>

namespace delta {

namespace {

namespace fs = std::filesystem;
using json = nlohmann::json;

constexpr auto kProfileWatchInterval = std::chrono::milliseconds(250);

std::string stringOr(const json& node, const char* key, const std::string& fallback = {}) {
    if (const auto it = node.find(key); it != node.end() && it->is_string()) {
        return it->get<std::string>();
    }
    return fallback;
}

double numberOr(const json& node, const char* key, const double fallback = 0.0) {
    if (const auto it = node.find(key); it != node.end() && it->is_number()) {
        return it->get<double>();
    }
    return fallback;
}

int intOr(const json& node, const char* key, const int fallback = 0) {
    if (const auto it = node.find(key); it != node.end() && it->is_number_integer()) {
        return it->get<int>();
    }
    if (const auto it = node.find(key); it != node.end() && it->is_number_float()) {
        return static_cast<int>(std::lround(it->get<double>()));
    }
    return fallback;
}

std::string fallbackId(const char* prefix, const std::size_t index) {
    std::ostringstream oss;
    oss << prefix << (index + 1);
    return oss.str();
}

bool parseProfileJson(const json& root, RecoilProfile& profile, std::string& error) {
    if (!root.is_object()) {
        error = "Recoil profile root must be a JSON object.";
        return false;
    }

    profile.schema_version = intOr(root, "schema_version", 1);
    profile.id = stringOr(root, "id");
    profile.name = stringOr(root, "name", profile.id);
    profile.created_at = stringOr(root, "created_at");
    profile.updated_at = stringOr(root, "updated_at");
    profile.scale_factor = numberOr(root, "scale_factor", 0.0);
    profile.horizontal_scale_factor = numberOr(root, "horizontal_scale_factor", profile.scale_factor);
    profile.fire_interval_ms = intOr(root, "fire_interval_ms", 0);

    if (profile.id.empty()) {
        error = "Recoil profile id is required.";
        return false;
    }
    if (profile.name.empty()) {
        profile.name = profile.id;
    }
    if (!(profile.scale_factor > 0.0)) {
        error = "Recoil profile scale_factor must be greater than 0.";
        return false;
    }
    if (!(profile.horizontal_scale_factor > 0.0)) {
        error = "Recoil profile horizontal_scale_factor must be greater than 0.";
        return false;
    }
    if (profile.fire_interval_ms <= 0) {
        error = "Recoil profile fire_interval_ms must be greater than 0.";
        return false;
    }

    if (const auto it = root.find("images"); it == root.end() || !it->is_array()) {
        error = "Recoil profile images must be a JSON array.";
        return false;
    } else {
        const auto& images = *it;
        profile.images.clear();
        profile.images.reserve(images.size());
        for (std::size_t image_index = 0; image_index < images.size(); ++image_index) {
            const auto& image_node = images[image_index];
            if (!image_node.is_object()) {
                error = "Recoil profile images entries must be objects.";
                return false;
            }
            RecoilImage image{};
            image.id = stringOr(image_node, "id", fallbackId("image_", image_index));
            image.name = stringOr(image_node, "name", image.id);
            image.asset_path = stringOr(image_node, "asset_path");
            image.width = std::max(0, intOr(image_node, "width", 0));
            image.height = std::max(0, intOr(image_node, "height", 0));

            if (const auto markers_it = image_node.find("markers"); markers_it != image_node.end()) {
                if (!markers_it->is_array()) {
                    error = "Recoil profile image markers must be a JSON array.";
                    return false;
                }
                image.markers.reserve(markers_it->size());
                for (std::size_t marker_index = 0; marker_index < markers_it->size(); ++marker_index) {
                    const auto& marker_node = (*markers_it)[marker_index];
                    if (!marker_node.is_object()) {
                        error = "Recoil profile markers entries must be objects.";
                        return false;
                    }
                    RecoilMarker marker{};
                    marker.id = stringOr(marker_node, "id", fallbackId("marker_", marker_index));
                    marker.x = numberOr(marker_node, "x", 0.0);
                    marker.y = numberOr(marker_node, "y", 0.0);
                    image.markers.push_back(std::move(marker));
                }
            }

            profile.images.push_back(std::move(image));
        }
    }

    if (profile.images.empty()) {
        error = "Recoil profile must contain at least one image.";
        return false;
    }

    if (const auto it = root.find("steps"); it == root.end() || !it->is_array()) {
        error = "Recoil profile steps must be a JSON array.";
        return false;
    } else {
        const auto& steps = *it;
        profile.steps.clear();
        profile.steps.reserve(steps.size());
        for (std::size_t step_index = 0; step_index < steps.size(); ++step_index) {
            const auto& step_node = steps[step_index];
            if (!step_node.is_object()) {
                error = "Recoil profile steps entries must be objects.";
                return false;
            }
            RecoilStep step{};
            step.index = static_cast<int>(step_index);
            step.pattern_x = numberOr(step_node, "pattern_x", 0.0);
            step.pattern_y = numberOr(step_node, "pattern_y", 0.0);
            step.duration_ms = intOr(step_node, "duration_ms", profile.fire_interval_ms);
            step.source_image_id = stringOr(step_node, "source_image_id");
            step.source_marker_id = stringOr(step_node, "source_marker_id");
            profile.steps.push_back(std::move(step));
        }
    }

    if (profile.steps.empty()) {
        error = "Recoil profile must contain at least one step.";
        return false;
    }

    const double first_x = profile.steps.front().pattern_x;
    const double first_y = profile.steps.front().pattern_y;
    for (std::size_t index = 0; index < profile.steps.size(); ++index) {
        auto& step = profile.steps[index];
        step.index = static_cast<int>(index);
        step.pattern_x -= first_x;
        step.pattern_y -= first_y;
        step.duration_ms = std::max(1, step.duration_ms > 0 ? step.duration_ms : profile.fire_interval_ms);
    }
    profile.steps.front().pattern_x = 0.0;
    profile.steps.front().pattern_y = 0.0;

    return true;
}

RecoilProfileSummary makeInvalidSummary(const fs::path& path, const std::string& error) {
    RecoilProfileSummary summary{};
    summary.id = path.stem().string();
    summary.name = summary.id;
    summary.valid = false;
    summary.error = error;
    return summary;
}

}  // namespace

const char* recoilModeName(const RecoilMode mode) {
    switch (mode) {
    case RecoilMode::AdvancedProfile:
        return "advanced_profile";
    case RecoilMode::Legacy:
    default:
        return "legacy";
    }
}

RecoilMode parseRecoilMode(const std::string_view value) {
    if (value == "advanced_profile") {
        return RecoilMode::AdvancedProfile;
    }
    return RecoilMode::Legacy;
}

bool ensureRecoilProfilesDirectory(const StaticConfig& config, std::string* error) {
    std::error_code ec;
    const fs::path root = recoilProfilesRoot(config);
    fs::create_directories(root, ec);
    if (ec) {
        if (error != nullptr) {
            *error = "Failed to create recoil profile directory: " + root.string();
        }
        return false;
    }
    fs::create_directories(root / "assets", ec);
    if (ec) {
        if (error != nullptr) {
            *error = "Failed to create recoil asset directory: " + (root / "assets").string();
        }
        return false;
    }
    return true;
}

std::filesystem::path recoilProfilesRoot(const StaticConfig& config) {
    return fs::path(config.recoil_profiles_dir);
}

std::filesystem::path recoilProfilePath(const StaticConfig& config, const std::string_view profile_id) {
    return recoilProfilesRoot(config) / (std::string(profile_id) + ".json");
}

std::filesystem::path recoilAssetsDir(const StaticConfig& config, const std::string_view profile_id) {
    return recoilProfilesRoot(config) / "assets" / std::string(profile_id);
}

std::vector<RecoilProfileSummary> listRecoilProfiles(const StaticConfig& config) {
    std::vector<RecoilProfileSummary> profiles;
    std::string ignored_error;
    if (!ensureRecoilProfilesDirectory(config, &ignored_error)) {
        return profiles;
    }

    std::error_code ec;
    for (const auto& entry : fs::directory_iterator(recoilProfilesRoot(config), ec)) {
        if (ec) {
            break;
        }
        if (!entry.is_regular_file() || entry.path().extension() != ".json") {
            continue;
        }

        std::string error;
        const auto profile = loadRecoilProfile(config, entry.path().stem().string(), error);
        if (!profile.has_value()) {
            profiles.push_back(makeInvalidSummary(entry.path(), error));
            continue;
        }

        RecoilProfileSummary summary{};
        summary.id = profile->id;
        summary.name = profile->name;
        summary.updated_at = profile->updated_at;
        summary.shot_count = static_cast<int>(profile->steps.size());
        summary.scale_factor = profile->scale_factor;
        summary.horizontal_scale_factor = profile->horizontal_scale_factor;
        summary.fire_interval_ms = profile->fire_interval_ms;
        profiles.push_back(std::move(summary));
    }

    std::sort(
        profiles.begin(),
        profiles.end(),
        [](const RecoilProfileSummary& left, const RecoilProfileSummary& right) {
            if (left.valid != right.valid) {
                return left.valid > right.valid;
            }
            if (left.name != right.name) {
                return left.name < right.name;
            }
            return left.id < right.id;
        });
    return profiles;
}

std::optional<RecoilProfile> loadRecoilProfile(const StaticConfig& config, const std::string_view profile_id, std::string& error) {
    error.clear();
    if (profile_id.empty()) {
        error = "No recoil profile selected.";
        return std::nullopt;
    }

    std::string dir_error;
    if (!ensureRecoilProfilesDirectory(config, &dir_error)) {
        error = dir_error;
        return std::nullopt;
    }

    const fs::path path = recoilProfilePath(config, profile_id);
    std::error_code ec;
    if (!fs::exists(path, ec) || ec) {
        error = "Recoil profile not found: " + std::string(profile_id);
        return std::nullopt;
    }

    std::ifstream input(path, std::ios::binary);
    if (!input) {
        error = "Failed to open recoil profile: " + path.string();
        return std::nullopt;
    }

    json root;
    try {
        input >> root;
    } catch (const std::exception& ex) {
        error = "Failed to parse recoil profile JSON: " + std::string(ex.what());
        return std::nullopt;
    }

    RecoilProfile profile{};
    if (!parseProfileJson(root, profile, error)) {
        return std::nullopt;
    }

    profile.source_path = path;
    profile.file_write_time = fs::last_write_time(path, ec);
    if (ec) {
        profile.file_write_time = {};
    }
    return profile;
}

RecoilScheduler::RecoilScheduler(StaticConfig config) : config_(std::move(config)) {
    std::string ignored_error;
    ensureRecoilProfilesDirectory(config_, &ignored_error);
}

RecoilSchedulerUpdate RecoilScheduler::tick(
    const RuntimeConfig& runtime,
    const bool recoil_enabled,
    const bool left_pressed,
    const bool x1_pressed,
    const SteadyClock::time_point& now) {
    const bool trigger_pressed = left_pressed || x1_pressed;
    state_.mode = runtime.recoil_mode;
    state_.enabled = recoil_enabled;
    state_.ignore_mode_check = runtime.recoil_tune_fallback_ignore_mode_check;
    state_.left_pressed = left_pressed;
    state_.x1_pressed = x1_pressed;
    state_.trigger_pressed = trigger_pressed;
    state_.spray_active = spray_active_;
    state_.selected_profile_id = runtime.selected_recoil_profile_id;
    state_.scheduled_dx = 0;
    state_.scheduled_dy = 0;
    state_.debug_state = runtime.recoil_mode == RecoilMode::AdvancedProfile ? "initializing" : "legacy_mode";

    const bool mode_changed = runtime.recoil_mode != last_mode_;
    const bool selected_profile_changed = runtime.selected_recoil_profile_id != last_selected_profile_id_;
    if (mode_changed || selected_profile_changed) {
        resetSequence();
    }

    const bool profile_ready = refreshProfile(runtime, now);
    const bool active = runtime.recoil_mode == RecoilMode::AdvancedProfile
        && recoil_enabled
        && trigger_pressed
        && profile_ready
        && loaded_profile_.has_value()
        && state_.error.empty();

    RecoilSchedulerUpdate update{};
    if (!active) {
        resetSequence();
        state_.shot_index = 0;
        state_.spray_active = false;
        if (runtime.recoil_mode != RecoilMode::AdvancedProfile) {
            state_.debug_state = "legacy_mode";
        } else if (!recoil_enabled) {
            state_.debug_state = "f7_off";
        } else if (!state_.error.empty()) {
            state_.debug_state = "profile_error";
        } else if (!profile_ready || !loaded_profile_.has_value()) {
            state_.debug_state = "profile_unavailable";
        } else if (!trigger_pressed) {
            state_.debug_state = "waiting_left_or_x1";
        } else {
            state_.debug_state = "idle";
        }
    } else if (loaded_profile_.has_value() && !loaded_profile_->steps.empty()) {
        if (!spray_active_) {
            spray_active_ = true;
            current_step_index_ = 0;
            segment_started_at_ = now;
            state_.shot_index = 1;
            next_step_at_ = now + std::chrono::milliseconds(durationForStep(0));
            state_.debug_state = "spray_started";
        }

        while (current_step_index_ + 1 < loaded_profile_->steps.size() && now >= next_step_at_) {
            ++current_step_index_;
            state_.shot_index = static_cast<int>(current_step_index_ + 1);
            segment_started_at_ = next_step_at_;
            next_step_at_ = segment_started_at_ + std::chrono::milliseconds(durationForStep(current_step_index_));
        }

        if (state_.shot_index == 0) {
            state_.shot_index = 1;
        }

        const auto& current = loaded_profile_->steps[current_step_index_];
        double target_pattern_x = current.pattern_x;
        double target_pattern_y = current.pattern_y;
        if (current_step_index_ + 1 < loaded_profile_->steps.size()) {
            const auto& next = loaded_profile_->steps[current_step_index_ + 1];
            const double duration_ms = static_cast<double>(durationForStep(current_step_index_));
            const double elapsed_ms = std::chrono::duration<double, std::milli>(now - segment_started_at_).count();
            const double alpha = duration_ms > 0.0
                ? std::clamp(elapsed_ms / duration_ms, 0.0, 1.0)
                : 1.0;
            target_pattern_x += (next.pattern_x - current.pattern_x) * alpha;
            target_pattern_y += (next.pattern_y - current.pattern_y) * alpha;
        }

        const double target_x = -(target_pattern_x * loaded_profile_->horizontal_scale_factor);
        const double target_y = -(target_pattern_y * loaded_profile_->scale_factor);
        carry_x_ += target_x - last_target_x_;
        carry_y_ += target_y - last_target_y_;
        last_target_x_ = target_x;
        last_target_y_ = target_y;

        const int step_dx = static_cast<int>(std::lround(carry_x_));
        const int step_dy = static_cast<int>(std::lround(carry_y_));
        carry_x_ -= static_cast<double>(step_dx);
        carry_y_ -= static_cast<double>(step_dy);
        update.delta.dx += step_dx;
        update.delta.dy += step_dy;

        state_.spray_active = true;
        state_.scheduled_dx = update.delta.dx;
        state_.scheduled_dy = update.delta.dy;
        if (update.delta.dx != 0 || update.delta.dy != 0) {
            state_.debug_state = current_step_index_ + 1 < loaded_profile_->steps.size()
                ? "segment_scheduled"
                : "step_scheduled";
        } else if (current_step_index_ + 1 < loaded_profile_->steps.size()) {
            state_.debug_state = "tracking_segment";
        } else if (current_step_index_ + 1 >= loaded_profile_->steps.size()) {
            state_.debug_state = "holding_final_shot";
        } else {
            state_.debug_state = "waiting_next_step";
        }
    }

    last_mode_ = runtime.recoil_mode;
    last_selected_profile_id_ = runtime.selected_recoil_profile_id;
    state_.mode = runtime.recoil_mode;
    state_.enabled = recoil_enabled;
    state_.ignore_mode_check = runtime.recoil_tune_fallback_ignore_mode_check;
    state_.left_pressed = left_pressed;
    state_.x1_pressed = x1_pressed;
    state_.trigger_pressed = trigger_pressed;
    state_.spray_active = active && spray_active_;
    state_.selected_profile_id = runtime.selected_recoil_profile_id;
    update.clear_pending = pending_clear_requested_;
    pending_clear_requested_ = false;
    update.state = state_;
    return update;
}

void RecoilScheduler::resetSequence() {
    spray_active_ = false;
    current_step_index_ = 0;
    segment_started_at_ = {};
    next_step_at_ = {};
    last_target_x_ = 0.0;
    last_target_y_ = 0.0;
    carry_x_ = 0.0;
    carry_y_ = 0.0;
    pending_clear_requested_ = true;
}

void RecoilScheduler::resetStateForProfile(const RuntimeConfig& runtime) {
    state_.mode = runtime.recoil_mode;
    state_.enabled = false;
    state_.ignore_mode_check = runtime.recoil_tune_fallback_ignore_mode_check;
    state_.left_pressed = false;
    state_.x1_pressed = false;
    state_.trigger_pressed = false;
    state_.spray_active = false;
    state_.selected_profile_id = runtime.selected_recoil_profile_id;
    state_.selected_profile_name.clear();
    state_.profile_loaded = false;
    state_.shot_index = 0;
    state_.shot_count = 0;
    state_.scale_factor = 0.0;
    state_.horizontal_scale_factor = 0.0;
    state_.fire_interval_ms = 0;
    state_.scheduled_dx = 0;
    state_.scheduled_dy = 0;
    state_.debug_state.clear();
    state_.error.clear();
}

bool RecoilScheduler::refreshProfile(const RuntimeConfig& runtime, const SteadyClock::time_point& now) {
    if (runtime.selected_recoil_profile_id.empty()) {
        loaded_profile_.reset();
        watched_profile_id_.clear();
        watched_write_time_ = {};
        resetStateForProfile(runtime);
        if (runtime.recoil_mode == RecoilMode::AdvancedProfile) {
            state_.error = "No recoil profile selected.";
        }
        return false;
    }

    const bool selected_profile_changed = runtime.selected_recoil_profile_id != watched_profile_id_;
    const bool should_check_disk = selected_profile_changed
        || !loaded_profile_.has_value()
        || last_file_check_ == SteadyClock::time_point{}
        || (now - last_file_check_) >= kProfileWatchInterval;

    if (!should_check_disk) {
        return loaded_profile_.has_value() && state_.error.empty();
    }
    last_file_check_ = now;

    if (!selected_profile_changed && loaded_profile_.has_value()) {
        std::error_code ec;
        const auto current_write_time = fs::last_write_time(recoilProfilePath(config_, runtime.selected_recoil_profile_id), ec);
        if (!ec && current_write_time == watched_write_time_) {
            return true;
        }
    }

    std::string error;
    const auto profile = loadRecoilProfile(config_, runtime.selected_recoil_profile_id, error);
    if (!profile.has_value()) {
        loaded_profile_.reset();
        watched_profile_id_ = runtime.selected_recoil_profile_id;
        watched_write_time_ = {};
        resetStateForProfile(runtime);
        state_.error = error;
        return false;
    }

    loaded_profile_ = profile;
    watched_profile_id_ = loaded_profile_->id;
    watched_write_time_ = loaded_profile_->file_write_time;
    resetSequence();
    resetStateForProfile(runtime);
    state_.profile_loaded = true;
    state_.selected_profile_name = loaded_profile_->name;
    state_.shot_count = static_cast<int>(loaded_profile_->steps.size());
    state_.scale_factor = loaded_profile_->scale_factor;
    state_.horizontal_scale_factor = loaded_profile_->horizontal_scale_factor;
    state_.fire_interval_ms = loaded_profile_->fire_interval_ms;
    return true;
}

int RecoilScheduler::durationForStep(const std::size_t index) const {
    if (!loaded_profile_.has_value() || index >= loaded_profile_->steps.size()) {
        return 1;
    }
    return std::max(1, loaded_profile_->steps[index].duration_ms);
}

}  // namespace delta
