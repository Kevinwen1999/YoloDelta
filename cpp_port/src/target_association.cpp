#include "delta/target_association.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace delta {

namespace {

constexpr float kMinDt = 1.0F / 240.0F;
constexpr float kMaxDt = 0.10F;
constexpr float kVelocityAlpha = 0.45F;

float hypot2(const float x, const float y) {
    return std::sqrt((x * x) + (y * y));
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

float pointToBoxDistance(const Detection& detection, const float x, const float y) {
    const float x1 = static_cast<float>(detection.bbox[0]);
    const float y1 = static_cast<float>(detection.bbox[1]);
    const float x2 = static_cast<float>(detection.bbox[2]);
    const float y2 = static_cast<float>(detection.bbox[3]);
    const float dx = x < x1 ? (x1 - x) : (x > x2 ? x - x2 : 0.0F);
    const float dy = y < y1 ? (y1 - y) : (y > y2 ? y - y2 : 0.0F);
    return hypot2(dx, dy);
}

bool sameDetection(const Detection& lhs, const Detection& rhs) {
    return lhs.cls == rhs.cls
        && lhs.bbox == rhs.bbox
        && std::abs(lhs.conf - rhs.conf) <= 1e-4F
        && std::abs(lhs.x - rhs.x) <= 1e-3F
        && std::abs(lhs.y - rhs.y) <= 1e-3F;
}

bool candidateContainsDetection(const std::vector<Detection>& candidates, const Detection& detection) {
    return std::any_of(
        candidates.begin(),
        candidates.end(),
        [&](const Detection& candidate) {
            return sameDetection(candidate, detection);
        });
}

std::pair<float, float> predictedPoint(const TargetAssociationTrack& track, const float dt) {
    const float horizon = dt * static_cast<float>(std::max(1, track.missed_frames + 1));
    return {
        track.detection.x + (track.vx * horizon),
        track.detection.y + (track.vy * horizon),
    };
}

float trackSpeed(const TargetAssociationTrack& track) {
    return hypot2(track.vx, track.vy);
}

bool classCompatible(
    const TargetAssociationConfig& config,
    const AimMode aim_mode,
    const TargetAssociationTrack& track,
    const Detection& detection,
    const float distance) {
    if (track.detection.cls == detection.cls) {
        return true;
    }
    return aim_mode == AimMode::Hybrid
        && config.hybrid_class_switch_enable
        && distance <= config.hybrid_class_switch_distance_px;
}

struct MatchPair {
    int track_index = -1;
    int detection_index = -1;
    float score = 0.0F;
};

}  // namespace

TargetAssociationConfig buildTargetAssociationConfig(const RuntimeConfig& runtime) {
    TargetAssociationConfig config{};
    config.enable = runtime.target_association_enable;
    config.min_iou = clamp(runtime.target_association_min_iou, 0.0F, 1.0F);
    config.max_distance_px = std::max(1.0F, runtime.target_association_max_distance_px);
    config.speed_distance_gain = std::max(0.0F, runtime.target_association_speed_distance_gain);
    config.iou_weight = std::max(0.0F, runtime.target_association_iou_weight);
    config.distance_weight = std::max(0.0F, runtime.target_association_distance_weight);
    config.confidence_weight = std::max(0.0F, runtime.target_association_confidence_weight);
    config.same_class_bonus = std::max(0.0F, runtime.target_association_same_class_bonus);
    config.locked_bonus = std::max(0.0F, runtime.target_association_locked_bonus);
    config.min_match_score = std::max(0.0F, runtime.target_association_min_match_score);
    config.lock_hold_frames = std::max(0, runtime.target_association_lock_hold_frames);
    config.hybrid_class_switch_enable = runtime.target_association_hybrid_class_switch_enable;
    config.hybrid_class_switch_distance_px = std::max(1.0F, runtime.target_association_hybrid_class_switch_distance_px);
    return config;
}

void TargetAssociationTracker::configure(const TargetAssociationConfig& config) {
    config_ = config;
    if (!config_.enable) {
        reset();
    }
}

void TargetAssociationTracker::reset() {
    tracks_.clear();
    locked_id_.reset();
    frame_index_ = 0;
    switch_count_ = 0;
    next_id_ = 1;
    locked_missing_ = false;
}

void TargetAssociationTracker::update(
    const std::vector<Detection>& detections,
    const AimMode aim_mode,
    const float dt) {
    locked_missing_ = false;
    if (!config_.enable) {
        return;
    }

    ++frame_index_;
    const float clamped_dt = clamp(std::isfinite(dt) ? dt : kMinDt, kMinDt, kMaxDt);
    for (auto& track : tracks_) {
        track.matched = false;
    }

    std::vector<MatchPair> pairs;
    pairs.reserve(tracks_.size() * detections.size());
    for (int track_index = 0; track_index < static_cast<int>(tracks_.size()); ++track_index) {
        const auto& track = tracks_[static_cast<size_t>(track_index)];
        const auto [predicted_x, predicted_y] = predictedPoint(track, clamped_dt);
        const float max_distance = config_.max_distance_px
            + std::min(config_.max_distance_px, trackSpeed(track) * config_.speed_distance_gain);
        for (int detection_index = 0; detection_index < static_cast<int>(detections.size()); ++detection_index) {
            const auto& detection = detections[static_cast<size_t>(detection_index)];
            const float distance = hypot2(detection.x - predicted_x, detection.y - predicted_y);
            if (distance > max_distance) {
                continue;
            }
            if (!classCompatible(config_, aim_mode, track, detection, distance)) {
                continue;
            }

            const float iou = bboxIou(track.detection.bbox, detection.bbox);
            const bool allowed_hybrid_class_switch = track.detection.cls != detection.cls
                && aim_mode == AimMode::Hybrid
                && config_.hybrid_class_switch_enable
                && distance <= config_.hybrid_class_switch_distance_px;
            if (iou < config_.min_iou && !allowed_hybrid_class_switch) {
                continue;
            }

            const float distance_score = 1.0F - clamp(distance / std::max(1.0F, max_distance), 0.0F, 1.0F);
            float score = (config_.iou_weight * iou)
                + (config_.distance_weight * distance_score)
                + (config_.confidence_weight * clamp(detection.conf, 0.0F, 1.0F));
            if (track.detection.cls == detection.cls) {
                score += config_.same_class_bonus;
            }
            if (locked_id_.has_value() && *locked_id_ == track.id) {
                score += config_.locked_bonus;
            }
            if (score >= config_.min_match_score) {
                pairs.push_back(MatchPair{track_index, detection_index, score});
            }
        }
    }

    std::sort(
        pairs.begin(),
        pairs.end(),
        [](const MatchPair& lhs, const MatchPair& rhs) {
            return lhs.score > rhs.score;
        });

    std::vector<bool> track_used(tracks_.size(), false);
    std::vector<bool> detection_used(detections.size(), false);
    for (const auto& pair : pairs) {
        if (track_used[static_cast<size_t>(pair.track_index)]
            || detection_used[static_cast<size_t>(pair.detection_index)]) {
            continue;
        }
        auto& track = tracks_[static_cast<size_t>(pair.track_index)];
        const Detection& detection = detections[static_cast<size_t>(pair.detection_index)];
        const float raw_vx = (detection.x - track.detection.x) / clamped_dt;
        const float raw_vy = (detection.y - track.detection.y) / clamped_dt;
        if (track.hit_streak <= 1 || track.missed_frames > 0) {
            track.vx = raw_vx;
            track.vy = raw_vy;
        } else {
            track.vx = ((1.0F - kVelocityAlpha) * track.vx) + (kVelocityAlpha * raw_vx);
            track.vy = ((1.0F - kVelocityAlpha) * track.vy) + (kVelocityAlpha * raw_vy);
        }
        track.detection = detection;
        track.hit_streak += 1;
        track.missed_frames = 0;
        track.last_seen_frame = frame_index_;
        track.matched = true;
        track_used[static_cast<size_t>(pair.track_index)] = true;
        detection_used[static_cast<size_t>(pair.detection_index)] = true;
    }

    for (int track_index = 0; track_index < static_cast<int>(tracks_.size()); ++track_index) {
        if (!track_used[static_cast<size_t>(track_index)]) {
            ++tracks_[static_cast<size_t>(track_index)].missed_frames;
        }
    }

    for (int detection_index = 0; detection_index < static_cast<int>(detections.size()); ++detection_index) {
        if (detection_used[static_cast<size_t>(detection_index)]) {
            continue;
        }
        TargetAssociationTrack track{};
        track.id = next_id_++;
        track.detection = detections[static_cast<size_t>(detection_index)];
        track.hit_streak = 1;
        track.missed_frames = 0;
        track.last_seen_frame = frame_index_;
        track.matched = true;
        tracks_.push_back(track);
    }

    const int max_missed = std::max(0, config_.lock_hold_frames);
    tracks_.erase(
        std::remove_if(
            tracks_.begin(),
            tracks_.end(),
            [&](const TargetAssociationTrack& track) {
                return track.missed_frames > max_missed;
            }),
        tracks_.end());

    if (locked_id_.has_value()) {
        const bool lock_exists = std::any_of(
            tracks_.begin(),
            tracks_.end(),
            [&](const TargetAssociationTrack& track) {
                return track.id == *locked_id_;
            });
        if (!lock_exists) {
            locked_id_.reset();
        }
    }
}

TargetAssociationPick TargetAssociationTracker::pick(
    const std::vector<Detection>& candidates,
    const int center_x,
    const int center_y) {
    TargetAssociationPick pick{};
    locked_missing_ = false;
    if (!config_.enable) {
        return pick;
    }

    if (locked_id_.has_value()) {
        auto locked = std::find_if(
            tracks_.begin(),
            tracks_.end(),
            [&](const TargetAssociationTrack& track) {
                return track.id == *locked_id_;
            });
        if (locked != tracks_.end()) {
            pick.active_id = locked->id;
            const bool current_candidate = locked->matched
                && locked->missed_frames == 0
                && candidateContainsDetection(candidates, locked->detection);
            if (current_candidate) {
                pick.detection = locked->detection;
                return pick;
            }
            if (locked->matched && locked->missed_frames == 0) {
                float best_score = std::numeric_limits<float>::lowest();
                std::optional<Detection> best_candidate;
                for (const auto& candidate : candidates) {
                    const float distance = hypot2(candidate.x - locked->detection.x, candidate.y - locked->detection.y);
                    const bool same_class = candidate.cls == locked->detection.cls;
                    const bool hybrid_class_switch = !same_class
                        && config_.hybrid_class_switch_enable
                        && distance <= config_.hybrid_class_switch_distance_px;
                    if (!same_class && !hybrid_class_switch) {
                        continue;
                    }
                    const float iou = bboxIou(locked->detection.bbox, candidate.bbox);
                    if (iou < config_.min_iou && !hybrid_class_switch) {
                        continue;
                    }
                    const float distance_limit = same_class
                        ? config_.max_distance_px
                        : config_.hybrid_class_switch_distance_px;
                    const float distance_score = 1.0F - clamp(distance / std::max(1.0F, distance_limit), 0.0F, 1.0F);
                    const float score = (config_.iou_weight * iou)
                        + (config_.distance_weight * distance_score)
                        + (config_.confidence_weight * clamp(candidate.conf, 0.0F, 1.0F))
                        + (same_class ? config_.same_class_bonus : 0.0F);
                    if (score > best_score) {
                        best_score = score;
                        best_candidate = candidate;
                    }
                }
                if (best_candidate.has_value()) {
                    locked->detection = *best_candidate;
                    pick.detection = *best_candidate;
                    return pick;
                }
            }
            if (locked->missed_frames <= config_.lock_hold_frames) {
                pick.locked_missing = true;
                locked_missing_ = true;
                return pick;
            }
            locked_id_.reset();
        } else {
            locked_id_.reset();
        }
    }

    float best_score = std::numeric_limits<float>::lowest();
    float best_distance = std::numeric_limits<float>::max();
    int best_track_id = -1;
    Detection best_detection{};
    for (const auto& track : tracks_) {
        if (!track.matched || track.missed_frames != 0) {
            continue;
        }
        if (!candidateContainsDetection(candidates, track.detection)) {
            continue;
        }
        const float box_distance = pointToBoxDistance(
            track.detection,
            static_cast<float>(center_x),
            static_cast<float>(center_y));
        const float aim_distance = hypot2(
            track.detection.x - static_cast<float>(center_x),
            track.detection.y - static_cast<float>(center_y));
        const float score = -(box_distance * 2.0F) - aim_distance + (track.detection.conf * 25.0F);
        if (
            score > best_score
            || (std::abs(score - best_score) <= 1e-5F && box_distance < best_distance)
        ) {
            best_score = score;
            best_distance = box_distance;
            best_track_id = track.id;
            best_detection = track.detection;
        }
    }

    if (best_track_id >= 0) {
        const std::optional<int> previous_lock = locked_id_;
        locked_id_ = best_track_id;
        if (previous_lock.has_value() && *previous_lock != best_track_id) {
            pick.switched = true;
            ++switch_count_;
        }
        pick.active_id = best_track_id;
        pick.detection = best_detection;
    }
    return pick;
}

TargetAssociationDiagnostics TargetAssociationTracker::diagnostics() const {
    return TargetAssociationDiagnostics{
        .active_id = locked_id_.value_or(-1),
        .track_count = static_cast<int>(tracks_.size()),
        .locked = locked_id_.has_value(),
        .locked_missing = locked_missing_,
        .switch_count = switch_count_,
    };
}

bool TargetAssociationTracker::hasLockedTrack() const {
    return locked_id_.has_value();
}

const std::vector<TargetAssociationTrack>& TargetAssociationTracker::tracks() const {
    return tracks_;
}

}  // namespace delta
