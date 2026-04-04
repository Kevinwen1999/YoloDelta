#include "delta/tracking.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace delta {

namespace {

constexpr float kMinDt = 1.0F / 240.0F;
constexpr float kMaxDt = 0.06F;
constexpr float kMaxSpeed = 1800.0F;
constexpr int kBodyClass = 0;
constexpr int kHeadClass = 1;
constexpr float kHybridHeadBottomMaxBodyRatio = 0.45F;
constexpr float kHybridHeadCenterXMaxBodyRatio = 0.30F;
constexpr float kHybridHeadClampHalfWidthRatio = 0.18F;
constexpr float kHybridHeadClampTopRatio = 0.03F;
constexpr float kHybridHeadClampBottomRatio = 0.32F;

float hypot2(const float x, const float y) {
    return std::sqrt((x * x) + (y * y));
}

float pointToBoxDistance(const Detection& det, const float x, const float y) {
    const float dx = x < static_cast<float>(det.bbox[0])
        ? static_cast<float>(det.bbox[0]) - x
        : (x > static_cast<float>(det.bbox[2]) ? x - static_cast<float>(det.bbox[2]) : 0.0F);
    const float dy = y < static_cast<float>(det.bbox[1])
        ? static_cast<float>(det.bbox[1]) - y
        : (y > static_cast<float>(det.bbox[3]) ? y - static_cast<float>(det.bbox[3]) : 0.0F);
    return hypot2(dx, dy);
}

bool pointInsideBox(const std::array<int, 4>& box, const float x, const float y) {
    return x >= static_cast<float>(box[0])
        && x <= static_cast<float>(box[2])
        && y >= static_cast<float>(box[1])
        && y <= static_cast<float>(box[3]);
}

Detection aimedDetection(const Detection& detection, const float body_y_ratio, const float head_y_ratio) {
    Detection aimed = detection;
    const auto [aim_x, aim_y] = detectionAimPoint(detection, body_y_ratio, head_y_ratio);
    aimed.x = aim_x;
    aimed.y = aim_y;
    return aimed;
}

std::optional<Detection> matchedBodyForHead(const Detection& head, const std::vector<Detection>& bodies) {
    const float head_center_x = static_cast<float>(head.bbox[0] + head.bbox[2]) * 0.5F;
    const float head_center_y = static_cast<float>(head.bbox[1] + head.bbox[3]) * 0.5F;
    const float head_bottom = static_cast<float>(head.bbox[3]);

    std::optional<Detection> best_match;
    float best_score = std::numeric_limits<float>::max();
    for (const auto& body : bodies) {
        const float body_x1 = static_cast<float>(body.bbox[0]);
        const float body_y1 = static_cast<float>(body.bbox[1]);
        const float body_x2 = static_cast<float>(body.bbox[2]);
        const float body_y2 = static_cast<float>(body.bbox[3]);
        const float body_w = std::max(1.0F, body_x2 - body_x1);
        const float body_h = std::max(1.0F, body_y2 - body_y1);
        const float body_center_x = body_x1 + (body_w * 0.5F);
        if (!pointInsideBox(body.bbox, head_center_x, head_center_y)) {
            continue;
        }
        if (head_bottom > body_y1 + (body_h * kHybridHeadBottomMaxBodyRatio)) {
            continue;
        }
        if (std::abs(head_center_x - body_center_x) > (body_w * kHybridHeadCenterXMaxBodyRatio)) {
            continue;
        }

        const float score = std::abs(head_center_x - body_center_x) + std::abs(head_center_y - body_y1);
        if (!best_match.has_value() || score < best_score || (std::abs(score - best_score) <= 1e-6F && body.conf > best_match->conf)) {
            best_match = body;
            best_score = score;
        }
    }
    return best_match;
}

void clampHeadAimByBody(Detection& head, const Detection& body) {
    const float body_x1 = static_cast<float>(body.bbox[0]);
    const float body_y1 = static_cast<float>(body.bbox[1]);
    const float body_x2 = static_cast<float>(body.bbox[2]);
    const float body_y2 = static_cast<float>(body.bbox[3]);
    const float body_w = std::max(1.0F, body_x2 - body_x1);
    const float body_h = std::max(1.0F, body_y2 - body_y1);
    const float body_center_x = body_x1 + (body_w * 0.5F);
    const float min_x = body_center_x - (body_w * kHybridHeadClampHalfWidthRatio);
    const float max_x = body_center_x + (body_w * kHybridHeadClampHalfWidthRatio);
    const float min_y = body_y1 + (body_h * kHybridHeadClampTopRatio);
    const float max_y = body_y1 + (body_h * kHybridHeadClampBottomRatio);
    head.x = clamp(head.x, min_x, max_x);
    head.y = clamp(head.y, min_y, max_y);
}

}  // namespace

PIDController::PIDController(const float kp, const float ki, const float kd) {
    configure(kp, ki, kd, 20.0F, 1.0F, 0.2F, 350.0F);
}

void PIDController::configure(
    const float kp,
    const float ki,
    const float kd,
    const float integral_limit,
    const float anti_windup_gain,
    const float derivative_alpha,
    const float output_limit) {
    kp_ = kp;
    ki_ = ki;
    kd_ = kd;
    integral_limit_ = std::max(0.0F, integral_limit);
    anti_windup_gain_ = clamp(anti_windup_gain, 0.0F, 1.0F);
    derivative_alpha_ = clamp(derivative_alpha, 0.0F, 1.0F);
    output_limit_ = std::max(0.0F, output_limit);
}

void PIDController::reset() {
    integral_ = 0.0F;
    prev_error_ = 0.0F;
    prev_derivative_ = 0.0F;
    initialized_ = false;
}

float PIDController::update(const float setpoint, const float measurement, const float dt, const bool integrate) {
    const float clamped_dt = std::max(1e-4F, dt);
    const float error = setpoint - measurement;
    const float p_term = kp_ * error;
    float integral_term = integral_;
    if (integrate) {
        integral_term = clamp(integral_term + (ki_ * error * clamped_dt), -integral_limit_, integral_limit_);
    }

    float derivative = 0.0F;
    if (initialized_) {
        const float derivative_raw = (measurement - prev_error_) / clamped_dt;
        prev_derivative_ = (derivative_alpha_ * prev_derivative_) + ((1.0F - derivative_alpha_) * derivative_raw);
        derivative = prev_derivative_;
    } else {
        initialized_ = true;
        prev_derivative_ = 0.0F;
    }

    float output = p_term + integral_term - (kd_ * derivative);
    if (output_limit_ > 0.0F) {
        const float clamped_output = clamp(output, -output_limit_, output_limit_);
        if (anti_windup_gain_ > 0.0F) {
            integral_term += (clamped_output - output) * anti_windup_gain_;
            integral_term = clamp(integral_term, -integral_limit_, integral_limit_);
            output = clamp(p_term + integral_term - (kd_ * derivative), -output_limit_, output_limit_);
        } else {
            output = clamped_output;
        }
    }

    integral_ = integral_term;
    prev_error_ = measurement;
    return output;
}

ObservedMotionTracker::ObservedMotionTracker(const TrackingStrategy mode, const float velocity_alpha)
    : mode_(mode) {
    configure(velocity_alpha);
    reset();
}

void ObservedMotionTracker::configure(const float velocity_alpha) {
    velocity_alpha_ = clamp(velocity_alpha, 0.0F, 1.0F);
}

void ObservedMotionTracker::reset() {
    state_ = {};
    raw_x_ = 0.0F;
    raw_y_ = 0.0F;
    last_dt_ = kMinDt;
    initialized_ = false;
    measurement_updates_ = 0;
}

void ObservedMotionTracker::predict(const float dt) {
    last_dt_ = clamp(dt, kMinDt, kMaxDt);
}

void ObservedMotionTracker::update(const float x, const float y) {
    if (!initialized_) {
        state_.x = x;
        state_.y = y;
        raw_x_ = x;
        raw_y_ = y;
        initialized_ = true;
        measurement_updates_ = 1;
        return;
    }

    const float dt = std::max(kMinDt, last_dt_);
    float raw_vx = 0.0F;
    float raw_vy = 0.0F;
    if (mode_ == TrackingStrategy::RawDelta) {
        raw_vx = clamp((x - raw_x_) / dt, -kMaxSpeed, kMaxSpeed);
        raw_vy = clamp((y - raw_y_) / dt, -kMaxSpeed, kMaxSpeed);
    }

    if (mode_ == TrackingStrategy::Raw) {
        state_.vx = 0.0F;
        state_.vy = 0.0F;
    } else if (measurement_updates_ <= 1) {
        state_.vx = raw_vx;
        state_.vy = raw_vy;
    } else {
        state_.vx = ((1.0F - velocity_alpha_) * state_.vx) + (velocity_alpha_ * raw_vx);
        state_.vy = ((1.0F - velocity_alpha_) * state_.vy) + (velocity_alpha_ * raw_vy);
    }

    state_.x = x;
    state_.y = y;
    raw_x_ = x;
    raw_y_ = y;
    ++measurement_updates_;
}

TrackerState ObservedMotionTracker::state() const {
    return state_;
}

float ObservedMotionTracker::feedforwardScale() const {
    if (mode_ == TrackingStrategy::Raw) {
        return 0.0F;
    }
    const int warm_min = 1;
    const int warm_ramp = 3;
    const int updates = std::max(0, measurement_updates_ - warm_min);
    if (updates <= 0) {
        return 0.0F;
    }
    return clamp(static_cast<float>(updates) / static_cast<float>(warm_ramp), 0.0F, 1.0F);
}

bool ObservedMotionTracker::initialized() const {
    return initialized_;
}

std::unique_ptr<ITargetTracker> makeTargetTracker(const TrackingStrategy strategy, const float velocity_alpha) {
    return std::make_unique<ObservedMotionTracker>(strategy, velocity_alpha);
}

std::pair<float, float> detectionAimPoint(
    const Detection& detection,
    const float body_y_ratio,
    const float head_y_ratio) {
    const float x1 = static_cast<float>(detection.bbox[0]);
    const float y1 = static_cast<float>(detection.bbox[1]);
    const float x2 = static_cast<float>(detection.bbox[2]);
    const float y2 = static_cast<float>(detection.bbox[3]);
    const float width = std::max(1.0F, x2 - x1);
    const float height = std::max(1.0F, y2 - y1);
    const float aim_x = x1 + (width * 0.5F);
    const float aim_y = detection.cls == kBodyClass
        ? (y1 + (height * clamp(body_y_ratio, 0.0F, 1.0F)))
        : (detection.cls == kHeadClass
            ? (y1 + (height * clamp(head_y_ratio, 0.0F, 1.0F)))
            : (y1 + (height * 0.5F)));
    return {aim_x, clamp(aim_y, y1, y2)};
}

AimCandidatePool buildAimCandidatePool(
    const std::vector<Detection>& detections,
    const AimMode aim_mode,
    const float body_y_ratio,
    const float head_y_ratio) {
    AimCandidatePool pool{};
    std::vector<Detection> heads;
    std::vector<Detection> bodies;
    heads.reserve(detections.size());
    bodies.reserve(detections.size());

    for (const auto& detection : detections) {
        Detection aimed = aimedDetection(detection, body_y_ratio, head_y_ratio);
        if (aimed.cls == kHeadClass) {
            heads.push_back(aimed);
        } else if (aimed.cls == kBodyClass) {
            bodies.push_back(aimed);
        }
    }

    switch (aim_mode) {
    case AimMode::Head:
        pool.candidates = std::move(heads);
        pool.using_head_candidates = true;
        return pool;
    case AimMode::Body:
        pool.candidates = std::move(bodies);
        return pool;
    case AimMode::Hybrid:
    default:
        if (!heads.empty()) {
            for (auto& head : heads) {
                if (const auto body = matchedBodyForHead(head, bodies); body.has_value()) {
                    clampHeadAimByBody(head, *body);
                }
            }
            pool.candidates = std::move(heads);
            pool.using_head_candidates = true;
            return pool;
        }
        pool.candidates = std::move(bodies);
        return pool;
    }
}

void resetAimTrackingState(
    int& lost_frames,
    int& active_target_cls,
    float& last_box_w,
    float& last_box_h,
    std::optional<std::array<int, 4>>& last_target_bbox,
    SteadyClock::time_point& last_pid_tick,
    SteadyClock::time_point& last_track_tick) {
    lost_frames = 0;
    active_target_cls = -1;
    last_box_w = 0.0F;
    last_box_h = 0.0F;
    last_target_bbox.reset();
    last_pid_tick = {};
    last_track_tick = {};
}

StickyTargetPick pickStickyTarget(
    const std::vector<Detection>& detections,
    const int center_x,
    const int center_y,
    const std::optional<std::pair<float, float>> locked_point,
    const float sticky_bias_px) {
    StickyTargetPick result{};
    if (detections.empty()) {
        return result;
    }

    int locked_idx = -1;
    if (locked_point.has_value()) {
        float best_locked_distance = std::numeric_limits<float>::max();
        for (int i = 0; i < static_cast<int>(detections.size()); ++i) {
            const float dx = detections[static_cast<size_t>(i)].x - locked_point->first;
            const float dy = detections[static_cast<size_t>(i)].y - locked_point->second;
            const float distance = hypot2(dx, dy);
            if (distance < best_locked_distance) {
                best_locked_distance = distance;
                locked_idx = i;
            }
        }
    }

    float best_score = std::numeric_limits<float>::max();
    float best_anchor_score = std::numeric_limits<float>::max();
    float best_conf = -1.0F;
    int best_idx = -1;
    for (int i = 0; i < static_cast<int>(detections.size()); ++i) {
        const auto& det = detections[static_cast<size_t>(i)];
        float score = pointToBoxDistance(det, static_cast<float>(center_x), static_cast<float>(center_y));
        const float anchor_score = hypot2(det.x - static_cast<float>(center_x), det.y - static_cast<float>(center_y));
        if (i == locked_idx) {
            score -= sticky_bias_px;
        }
        if (
            score < best_score
            || (
                std::abs(score - best_score) <= 1e-6F
                && (
                    anchor_score < best_anchor_score
                    || (std::abs(anchor_score - best_anchor_score) <= 1e-6F && det.conf > best_conf)
                )
            )
        ) {
            best_idx = i;
            best_score = score;
            best_anchor_score = anchor_score;
            best_conf = det.conf;
        }
    }

    if (best_idx >= 0) {
        result.detection = detections[static_cast<size_t>(best_idx)];
        result.switched = locked_idx >= 0 && best_idx != locked_idx;
    }
    return result;
}

}  // namespace delta
