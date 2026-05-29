#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "delta/config.hpp"
#include "delta/core.hpp"

namespace delta {

struct TargetAssociationConfig {
    bool enable = true;
    float min_iou = 0.08F;
    float max_distance_px = 220.0F;
    float speed_distance_gain = 0.05F;
    float iou_weight = 4.0F;
    float distance_weight = 2.0F;
    float confidence_weight = 1.0F;
    float same_class_bonus = 0.75F;
    float locked_bonus = 6.0F;
    float min_match_score = 1.25F;
    int lock_hold_frames = 8;
    bool hybrid_class_switch_enable = true;
    float hybrid_class_switch_distance_px = 90.0F;
};

struct TargetAssociationDiagnostics {
    int active_id = -1;
    int track_count = 0;
    bool locked = false;
    bool locked_missing = false;
    std::uint64_t switch_count = 0;
};

struct TargetAssociationPick {
    std::optional<Detection> detection;
    bool switched = false;
    bool locked_missing = false;
    int active_id = -1;
};

struct TargetAssociationTrack {
    int id = -1;
    Detection detection{};
    float vx = 0.0F;
    float vy = 0.0F;
    int hit_streak = 0;
    int missed_frames = 0;
    std::uint64_t last_seen_frame = 0;
    bool matched = false;
};

TargetAssociationConfig buildTargetAssociationConfig(const RuntimeConfig& runtime);

class TargetAssociationTracker {
public:
    void configure(const TargetAssociationConfig& config);
    void reset();
    void update(const std::vector<Detection>& detections, AimMode aim_mode, float dt);
    TargetAssociationPick pick(const std::vector<Detection>& candidates, int center_x, int center_y);

    [[nodiscard]] TargetAssociationDiagnostics diagnostics() const;
    [[nodiscard]] bool hasLockedTrack() const;
    [[nodiscard]] const std::vector<TargetAssociationTrack>& tracks() const;

private:
    TargetAssociationConfig config_{};
    std::vector<TargetAssociationTrack> tracks_;
    std::optional<int> locked_id_;
    std::uint64_t frame_index_ = 0;
    std::uint64_t switch_count_ = 0;
    int next_id_ = 1;
    bool locked_missing_ = false;
};

}  // namespace delta
