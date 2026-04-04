#pragma once

#include <cstdint>
#include <string>

namespace delta {

enum class RecoilMode {
    Legacy,
    AdvancedProfile,
};

struct PendingRecoilDelta {
    int dx = 0;
    int dy = 0;
};

struct RecoilRuntimeState {
    RecoilMode mode = RecoilMode::Legacy;
    bool enabled = false;
    bool profile_loaded = false;
    bool ignore_mode_check = false;
    bool mode_active = false;
    bool hold_engage_toggle = false;
    bool left_pressed = false;
    bool x1_pressed = false;
    bool trigger_pressed = false;
    bool spray_active = false;
    std::string selected_profile_id;
    std::string selected_profile_name;
    int shot_index = 0;
    int shot_count = 0;
    double scale_factor = 0.0;
    double horizontal_scale_factor = 0.0;
    int fire_interval_ms = 0;
    int scheduled_dx = 0;
    int scheduled_dy = 0;
    int last_applied_dx = 0;
    int last_applied_dy = 0;
    int last_applied_shot_index = 0;
    std::uint64_t apply_count = 0;
    std::string debug_state;
    std::string error;
};

}  // namespace delta
