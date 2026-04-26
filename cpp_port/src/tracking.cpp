#include "delta/tracking.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace delta {

namespace {

constexpr float kMinDt = 1.0F / 240.0F;
constexpr float kMaxDt = 0.06F;
constexpr float kMaxSpeed = 1800.0F;
constexpr float kPidSettleThresholdSharpness = 5.0F;
constexpr int kBodyClass = 0;
constexpr int kHeadClass = 1;

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

Detection aimedDetection(const Detection& detection, const float body_y_ratio, const float head_y_ratio) {
    Detection aimed = detection;
    const auto [aim_x, aim_y] = detectionAimPoint(detection, body_y_ratio, head_y_ratio);
    aimed.x = aim_x;
    aimed.y = aim_y;
    return aimed;
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

void PIDController::clearIntegral() {
    integral_ = 0.0F;
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

void PIDSettleState::reset() {
    settled = false;
    stable_frame_count = 0;
    previous_error_metric_px = 0.0F;
    initialized = false;
}

void LegacyPidAxisState::reset() {
    previous_error_px = 0.0F;
    integral_accumulator = 0.0F;
    velocity = 0.0F;
    stable_frame_count = 0;
    locked = false;
}

float pidSettleErrorMetricPx(const float predicted_x, const float aim_y, const float center_x, const float center_y) {
    return std::max(std::abs(predicted_x - center_x), std::abs(aim_y - center_y));
}

float pidSettleDynamicThresholdPx(const PIDSettleConfig& config, const float box_width_px, const float capture_width_px) {
    const float safe_box_width = std::max(0.0F, box_width_px);
    const float safe_capture_width = std::max(1e-3F, capture_width_px);
    const float ratio = clamp(safe_box_width / safe_capture_width, 0.0F, 1.0F);
    const float min_scale = std::max(0.0F, config.threshold_min_scale);
    const float max_scale = std::max(min_scale, config.threshold_max_scale);
    const float dynamic_scale = min_scale
        + ((max_scale - min_scale) / (1.0F + std::exp(-kPidSettleThresholdSharpness * ratio)));
    return safe_box_width * dynamic_scale;
}

PIDSettleDecision updatePidSettleState(
    PIDSettleState& state,
    const PIDSettleConfig& config,
    const float error_metric_px,
    const float box_width_px,
    const float capture_width_px) {
    PIDSettleDecision decision{};
    decision.error_metric_px = std::max(0.0F, error_metric_px);
    decision.dynamic_threshold_px = pidSettleDynamicThresholdPx(config, box_width_px, capture_width_px);

    if (!config.enable) {
        state.settled = true;
        state.stable_frame_count = 0;
        state.previous_error_metric_px = decision.error_metric_px;
        state.initialized = true;
        decision.settled = true;
        decision.integrate = true;
        decision.pid_output_scale = 1.0F;
        return decision;
    }

    if (!state.initialized) {
        state.initialized = true;
        state.previous_error_metric_px = decision.error_metric_px;
    }

    const float settle_error_px = std::max(0.0F, config.error_px);
    const float error_delta_px = std::max(0.0F, config.error_delta_px);
    const int stable_frames = std::max(1, config.stable_frames);
    const float pre_output_scale = clamp(config.pre_output_scale, 0.0F, 1.0F);
    const bool was_settled = state.settled;

    if (decision.error_metric_px < settle_error_px) {
        state.settled = true;
        state.stable_frame_count = 0;
    } else if (decision.error_metric_px >= decision.dynamic_threshold_px) {
        state.settled = false;
        state.stable_frame_count = 0;
        decision.just_unsettled = was_settled;
    } else if (!state.settled) {
        if (std::abs(decision.error_metric_px - state.previous_error_metric_px) < error_delta_px) {
            ++state.stable_frame_count;
        } else {
            state.stable_frame_count = 0;
        }
        if (state.stable_frame_count >= stable_frames) {
            state.settled = true;
            state.stable_frame_count = 0;
        }
    }

    state.previous_error_metric_px = decision.error_metric_px;
    decision.settled = state.settled;
    decision.integrate = state.settled;
    decision.pid_output_scale = state.settled ? 1.0F : pre_output_scale;
    return decision;
}

float legacyPidDynamicThresholdPx(const LegacyPidConfig& config, const float box_width_px, const float capture_width_px) {
    const float safe_box_width = std::max(0.0F, box_width_px);
    const float safe_capture_width = std::max(1e-3F, capture_width_px);
    const float min_scale = std::max(0.0F, config.threshold_min_scale);
    const float max_scale = std::max(min_scale, config.threshold_max_scale);
    const float ratio = safe_box_width / safe_capture_width;
    const float transition = 1.0F + std::exp(
        -std::max(0.0F, config.transition_sharpness) * (ratio - config.transition_midpoint));
    const float dynamic_scale = min_scale + ((max_scale - min_scale) / transition);
    return safe_box_width * dynamic_scale;
}

LegacyPidAxisResult updateLegacyPidAxis(
    LegacyPidAxisState& state,
    const LegacyPidConfig& config,
    const float error_px,
    const float dt,
    const float box_width_px,
    const float capture_width_px) {
    LegacyPidAxisResult result{};
    result.error_px = error_px;
    result.dynamic_threshold_px = legacyPidDynamicThresholdPx(config, box_width_px, capture_width_px);

    const float clamped_dt = std::max(1e-4F, dt);
    const float absolute_error = std::abs(error_px);
    const float lock_error_px = std::max(0.0F, config.lock_error_px);
    const float error_delta_px = std::max(0.0F, config.error_delta_px);
    const int stable_frames = std::max(1, config.stable_frames);
    const float prelock_scale = clamp(config.prelock_scale, 0.0F, 1.0F);

    if (!state.locked && absolute_error < lock_error_px) {
        state.locked = true;
    } else if (absolute_error >= result.dynamic_threshold_px) {
        result.just_unlocked = state.locked;
        state.locked = false;
        state.integral_accumulator = 0.0F;
        state.stable_frame_count = 0;
    } else if (!state.locked && absolute_error >= lock_error_px && absolute_error <= result.dynamic_threshold_px) {
        const float error_change = std::abs(error_px - state.previous_error_px);
        if (error_change < error_delta_px) {
            ++state.stable_frame_count;
        } else {
            state.stable_frame_count = 0;
        }
        if (state.stable_frame_count >= stable_frames) {
            state.locked = true;
            state.stable_frame_count = 0;
            state.integral_accumulator = 0.0F;
        }
    }

    const float error_rate = (error_px - state.previous_error_px) / clamped_dt;
    if (state.locked) {
        state.integral_accumulator += error_px * clamped_dt;
        result.proportional = config.kp * error_px;
    } else {
        state.integral_accumulator += (error_px * prelock_scale) * clamped_dt;
        result.proportional = (config.kp * prelock_scale) * error_px;
    }
    result.integral = config.ki * state.integral_accumulator;
    result.derivative = config.kd * error_rate;
    result.raw_output = result.proportional + result.integral + result.derivative;
    result.output = result.raw_output;
    result.velocity = state.locked
        ? (error_rate + ((result.raw_output / clamped_dt) * config.speed_multiplier))
        : 0.0F;
    result.locked = state.locked;

    state.velocity = result.velocity;
    state.previous_error_px = error_px;
    return result;
}

LegacyPidStatus makeLegacyPidStatus(const LegacyPidAxisResult& x_axis, const LegacyPidAxisResult& y_axis) {
    return LegacyPidStatus{
        .settled = x_axis.locked && y_axis.locked,
        .error_metric_px = std::max(std::abs(x_axis.error_px), std::abs(y_axis.error_px)),
        .threshold_px = std::max(x_axis.dynamic_threshold_px, y_axis.dynamic_threshold_px),
        .speed = std::sqrt((x_axis.velocity * x_axis.velocity) + (y_axis.velocity * y_axis.velocity)),
    };
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
    const TrackingStrategy tracker_mode = (strategy == TrackingStrategy::LegacyPid || strategy == TrackingStrategy::PredictivePid)
        ? TrackingStrategy::RawDelta
        : strategy;
    return std::make_unique<ObservedMotionTracker>(tracker_mode, velocity_alpha);
}

Detection scaleDetectionBox(const Detection& detection, const float box_scale, const CaptureRegion& bounds) {
    Detection scaled = detection;
    const float safe_scale = std::max(0.01F, std::isfinite(box_scale) ? box_scale : 1.0F);
    const float x1 = static_cast<float>(detection.bbox[0]);
    const float y1 = static_cast<float>(detection.bbox[1]);
    const float x2 = static_cast<float>(detection.bbox[2]);
    const float y2 = static_cast<float>(detection.bbox[3]);
    const float center_x = (x1 + x2) * 0.5F;
    const float center_y = (y1 + y2) * 0.5F;
    const float half_w = std::max(0.5F, (x2 - x1) * 0.5F * safe_scale);
    const float half_h = std::max(0.5F, (y2 - y1) * 0.5F * safe_scale);

    const int min_x = bounds.left;
    const int min_y = bounds.top;
    const int max_x = bounds.left + std::max(1, bounds.width) - 1;
    const int max_y = bounds.top + std::max(1, bounds.height) - 1;
    int new_x1 = clamp(static_cast<int>(std::lround(center_x - half_w)), min_x, max_x);
    int new_y1 = clamp(static_cast<int>(std::lround(center_y - half_h)), min_y, max_y);
    int new_x2 = clamp(static_cast<int>(std::lround(center_x + half_w)), min_x, max_x);
    int new_y2 = clamp(static_cast<int>(std::lround(center_y + half_h)), min_y, max_y);

    if (new_x2 <= new_x1) {
        if (new_x1 < max_x) {
            new_x2 = new_x1 + 1;
        } else {
            new_x1 = std::max(min_x, new_x2 - 1);
        }
    }
    if (new_y2 <= new_y1) {
        if (new_y1 < max_y) {
            new_y2 = new_y1 + 1;
        } else {
            new_y1 = std::max(min_y, new_y2 - 1);
        }
    }

    scaled.bbox = {new_x1, new_y1, new_x2, new_y2};
    scaled.x = center_x;
    scaled.y = center_y;
    return scaled;
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
