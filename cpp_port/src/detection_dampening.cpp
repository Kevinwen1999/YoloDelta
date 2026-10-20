#include "delta/detection_dampening.hpp"
#include "delta/calibrated_control.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace delta {

namespace {

constexpr float kDetectionDampeningStableIou = 0.20F;

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

}  // namespace

void DetectionDampeningState::reset() {
    last_selected_bbox.reset();
    stable_frame_count = 0;
    last_observation = {};
    confirmation_started = {};
    accepted_detection.reset();
    vx = vy = 0;
    confirmed = false;
}

MotionSelection noteMotionSelection(DetectionDampeningState& state,
    const RuntimeConfig& config, const std::optional<Detection>& proposed,
    const std::vector<Detection>& candidates, SteadyClock::time_point observed,
    SteadyClock::time_point now,std::optional<CaptureRegion> observed_crop,
    std::pair<float,float> camera_displacement,std::pair<float,float> camera_displacement_now) {
    MotionSelection result;
    const auto ms = [](auto a, auto b) { return std::chrono::duration<double,std::milli>(b-a).count(); };
    const double timeout = std::max(1.0F, config.motion_prediction_timeout_ms);
    if (state.accepted_detection && ms(state.last_observation, now) > timeout) {
        state.reset();result.decision="loss_deadline";
    }
    result.ready = state.confirmed;
    if (state.confirmed)result.decision="bounded_prediction";
    const float prediction_dt=static_cast<float>(ms(state.last_observation,now)/1000.0)
        +(calibratedControlActive(config)?config.calibrated_response_delay_ms/1000.0F:0.0F);
    if (state.confirmed && std::hypot(state.vx*prediction_dt-camera_displacement_now.first,
        state.vy*prediction_dt-camera_displacement_now.second)
        > config.motion_prediction_max_displacement_px) {result.ready=false;result.decision="prediction_displacement_limit";}
    if (observed == SteadyClock::time_point{} || observed > now || ms(observed,now) > timeout) {
        result.decision="invalid_observation_time";return result;
    }
    if (state.accepted_detection && observed <= state.last_observation) {
        result.duplicate = true;
        result.decision="duplicate_observation";
        return result;
    }
    const auto complete=[&](const Detection& d) {
        if(!observed_crop)return true;
        const auto& crop=*observed_crop;
        return d.bbox[0]>crop.left && d.bbox[1]>crop.top
            && d.bbox[2]<crop.left+crop.width-1 && d.bbox[3]<crop.top+crop.height-1;
    };
    std::vector<Detection> eligible;
    for(const auto& candidate:candidates)if(complete(candidate))eligible.push_back(candidate);
    const auto proposal=proposed && complete(*proposed)?proposed:std::optional<Detection>{};
    if((eligible.empty() && !candidates.empty()) || (!config.motion_identity_enable && proposed && !proposal)) {
        result.decision="clipped_observation";return result;
    }
    std::vector<Detection> distinct_candidates;
    if(config.motion_identity_enable) {
        for(const auto& candidate:eligible) {
            const auto existing=std::find_if(distinct_candidates.begin(),distinct_candidates.end(),
                [&](const Detection& other){return other.cls==candidate.cls && other.bbox==candidate.bbox;});
            if(existing==distinct_candidates.end())distinct_candidates.push_back(candidate);
            else if(candidate.conf>existing->conf)*existing=candidate;
        }
    }
    std::optional<Detection> accepted = proposal;
    bool new_candidate = !state.accepted_detection;
    if (state.accepted_detection) {
        const auto& previous = *state.accepted_detection;
        const float dt = static_cast<float>(ms(state.last_observation,observed)/1000.0);
        const float width = static_cast<float>(std::max(1,previous.bbox[2]-previous.bbox[0]));
        const float height = static_cast<float>(std::max(1,previous.bbox[3]-previous.bbox[1]));
        const float radius = state.stable_frame_count < 2
            ? std::max(width, config.motion_max_speed_px_s*dt)
            : std::max(width*config.motion_residual_scale, config.motion_max_speed_px_s*dt);
        auto cost = [&](const Detection& d, bool reacquiring=false) {
            const bool class_change = d.cls != previous.cls;
            if (class_change && config.aim_mode != AimMode::Hybrid) return std::numeric_limits<float>::infinity();
            const float wr = std::max(1,d.bbox[2]-d.bbox[0])/width;
            const float hr = std::max(1,d.bbox[3]-d.bbox[1])/height;
            if (!class_change && (wr<0.4F || wr>2.5F || hr<0.4F || hr>2.5F)) return std::numeric_limits<float>::infinity();
            const float move_x=d.x-previous.x+camera_displacement.first;
            const float move_y=d.y-previous.y+camera_displacement.second;
            if(!class_change && std::hypot(move_x,move_y)>std::max(width,config.motion_max_speed_px_s*dt))
                return std::numeric_limits<float>::infinity();
            const float reversal_gate=std::max(1.0F,width*config.motion_residual_scale);
            if(!reacquiring && !class_change && move_x*state.vx+move_y*state.vy<0
                && std::hypot(move_x,move_y)>reversal_gate
                && std::hypot(state.vx,state.vy)*dt>reversal_gate)
                return std::numeric_limits<float>::infinity();
            const float residual = reacquiring ? std::hypot(move_x,move_y)
                : std::hypot(move_x-state.vx*dt,move_y-state.vy*dt);
            const float match_radius = reacquiring ? std::max(width,config.motion_max_speed_px_s*dt) : radius;
            const float limit = class_change ? std::max(match_radius,height) : match_radius;
            return residual <= limit ? residual/std::max(1.0F,limit) : std::numeric_limits<float>::infinity();
        };
        if (config.motion_identity_enable) {
            float best = std::numeric_limits<float>::infinity(), second = best;
            accepted.reset();
            for (const auto& candidate : distinct_candidates) {
                const float score = cost(candidate);
                if (std::isfinite(score)) result.matches.emplace_back(candidate,score);
                if (score < best) { second=best;best=score;accepted=candidate; }
                else if (score < second) second=score;
            }
            if (accepted && second-best < config.motion_match_margin) {
                accepted.reset(); result.ambiguous=true;result.decision="ambiguous_match";
            }
            if(accepted) {
                // A direction/prediction gate may reject the real target during
                // camera-induced reversal. It must not make a reachable rival
                // look uniquely identified. Compare these alternatives in the
                // same displacement-based metric used for reconfirmation.
                const float accepted_displacement=cost(*accepted,true);
                for(const auto& candidate:distinct_candidates) {
                    if(candidate.cls!=previous.cls || std::isfinite(cost(candidate)))continue;
                    const float alternate=cost(candidate,true);
                    if(std::isfinite(alternate) && alternate-accepted_displacement<config.motion_match_margin) {
                        result.matches.emplace_back(candidate,alternate);
                        accepted.reset();result.ambiguous=true;result.decision="ambiguous_motion_change";break;
                    }
                }
            }
        } else if (accepted && !std::isfinite(cost(*accepted))) accepted.reset();
        if (!accepted) {
            // Visible but inconsistent evidence can be a reversal or camera shake.
            // A unique, physically reachable observation starts a new acquisition.
            // It cannot move until a second distinct observation confirms it.
            if (!eligible.empty() && !result.ambiguous) {
                result.ready=false;
                result.decision="inconsistent_motion";
                std::optional<Detection> reacquire;
                float best=std::numeric_limits<float>::infinity(),second=best;
                if (config.motion_identity_enable) {
                    for (const auto& candidate:distinct_candidates) {
                        const float score=cost(candidate,true);
                        if (score<best) {second=best;best=score;reacquire=candidate;}
                        else if(score<second)second=score;
                    }
                    if (reacquire && second-best<config.motion_match_margin) {
                        reacquire.reset();result.ambiguous=true;result.decision="ambiguous_reacquisition";
                    }
                } else if(proposal && std::isfinite(cost(*proposal,true))) reacquire=proposal;
                if (reacquire) {
                    state.reset();
                    state.accepted_detection=reacquire;state.last_selected_bbox=reacquire->bbox;
                    state.last_observation=observed;state.confirmation_started=observed;
                    state.stable_frame_count=1;
                    result.detection=reacquire;result.switched=true;
                    result.decision="reconfirming_motion";
                    return result;
                }
                // A contradicted model must be reconfirmed before output resumes.
                state.confirmed=false;
                state.stable_frame_count=0;
                state.confirmation_started=observed;
            }
            return result;
        }
        result.residual_px = std::hypot(accepted->x-previous.x+camera_displacement.first-state.vx*dt,
            accepted->y-previous.y+camera_displacement.second-state.vy*dt);
        if (!state.confirmed && ms(state.confirmation_started,observed) > config.motion_confirmation_window_ms) {
            state.reset(); new_candidate=true;
        } else {
            // Head/body transitions change the aim point, not target velocity.
            if (accepted->cls == previous.cls) {
                const float raw_x=(accepted->x-previous.x+camera_displacement.first)/dt;
                const float raw_y=(accepted->y-previous.y+camera_displacement.second)/dt;
                const float alpha=state.stable_frame_count<2?1.0F:0.5F;
                state.vx += alpha*(raw_x-state.vx);state.vy += alpha*(raw_y-state.vy);
                const float speed=std::hypot(state.vx,state.vy);
                if (speed>config.motion_max_speed_px_s) {
                    state.vx*=config.motion_max_speed_px_s/speed;state.vy*=config.motion_max_speed_px_s/speed;
                }
            }
        }
    }
    if (!accepted) return result;
    if (new_candidate) { state.confirmation_started=observed;result.switched=true; }
    state.accepted_detection=accepted;
    state.last_selected_bbox=accepted->bbox;
    state.last_observation=observed;
    ++state.stable_frame_count;
    state.confirmed=state.stable_frame_count>=std::max(2,config.detection_dampening_stable_frames);
    result.detection=accepted;result.ready=state.confirmed;
    result.decision=state.confirmed?"accepted_observation":"initial_confirmation";
    return result;
}

DetectionDampeningConfig buildDetectionDampeningConfig(const RuntimeConfig& runtime) {
    DetectionDampeningConfig config{};
    config.enable = runtime.detection_dampening_enable;
    config.stable_frames = std::max(1, runtime.detection_dampening_stable_frames);
    return config;
}

DetectionDampeningResult noteDetectionDampeningSelection(
    DetectionDampeningState& state,
    const DetectionDampeningConfig& config,
    const std::optional<Detection>& detection,
    const bool target_switched) {
    const int required_frames = std::max(1, config.stable_frames);
    if (!config.enable) {
        state.reset();
        return DetectionDampeningResult{
            .ready = true,
            .streak = 0,
            .required_frames = required_frames,
        };
    }

    if (!detection.has_value()) {
        state.reset();
        return DetectionDampeningResult{
            .ready = false,
            .streak = 0,
            .required_frames = required_frames,
        };
    }

    const bool stable_match = !target_switched
        && state.last_selected_bbox.has_value()
        && bboxIou(*state.last_selected_bbox, detection->bbox) >= kDetectionDampeningStableIou;
    state.stable_frame_count = stable_match ? (state.stable_frame_count + 1) : 1;
    state.last_selected_bbox = detection->bbox;

    return DetectionDampeningResult{
        .ready = state.stable_frame_count >= required_frames,
        .streak = state.stable_frame_count,
        .required_frames = required_frames,
    };
}

}  // namespace delta
