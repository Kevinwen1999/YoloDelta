#pragma once
#include "delta/core.hpp"
#include "delta/config.hpp"
#include "delta/runtime_state.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

namespace delta {
inline constexpr float kDispatchVelocityLimitPxS=3200.0F;
// Caller holds SharedState::mutex. Live and replay adapters report the same
// pre-backend command deltas at adapter completion, not physical application.
inline void decayDispatchFeedbackLocked(SharedState& shared) {
    for(auto* velocity : {&shared.ctrl_sent_vx_ema,&shared.ctrl_sent_vy_ema}) {
        const float next=velocity->load(std::memory_order_relaxed)*.92F;
        velocity->store(std::abs(next)<1e-6F?0.0F:next,std::memory_order_relaxed);
    }
}
inline void noteCommandDeliveryLocked(SharedState& shared,const CommandDelivery& delivery) {
    if(!delivery.movement_sent || (delivery.dx==0 && delivery.dy==0)) {
        decayDispatchFeedbackLocked(shared);return;
    }
    const auto update=[](float previous,float sample,float alpha){return previous+(sample-previous)*alpha;};
    if(delivery.command.cmd_generated!=SteadyClock::time_point{} && !delivery.command.synthetic_servo) {
        const float latency=std::max(0.0F,std::chrono::duration<float>(delivery.completed_at-delivery.command.cmd_generated).count());
        shared.cmd_send_latency_ema_s.store(update(shared.cmd_send_latency_ema_s.load(std::memory_order_relaxed),latency,.20F),std::memory_order_relaxed);
    }
    if(shared.ctrl_last_send_tick!=SteadyClock::time_point{}) {
        const float dt=std::max(1e-4F,std::chrono::duration<float>(delivery.completed_at-shared.ctrl_last_send_tick).count());
        const float vx=std::clamp(delivery.dx/dt,-kDispatchVelocityLimitPxS,kDispatchVelocityLimitPxS);
        const float vy=std::clamp(delivery.dy/dt,-kDispatchVelocityLimitPxS,kDispatchVelocityLimitPxS);
        shared.ctrl_sent_vx_ema.store(update(shared.ctrl_sent_vx_ema.load(std::memory_order_relaxed),vx,.30F),std::memory_order_relaxed);
        shared.ctrl_sent_vy_ema.store(update(shared.ctrl_sent_vy_ema.load(std::memory_order_relaxed),vy,.30F),std::memory_order_relaxed);
    }
    shared.ctrl_last_send_tick=delivery.completed_at;
}

// Shared by the real sender and the no-device scheduled-dispatch harness.
inline bool commandFreshAt(const CommandPacket& command, SteadyClock::time_point now,
                           double max_age_ms) {
    if (command.synthetic_recoil) return true;
    const auto origin=command.observation_time!=SteadyClock::time_point{}
        ? command.observation_time : command.frame_ready;
    return origin!=SteadyClock::time_point{} && origin<=now
        && std::chrono::duration<double,std::milli>(now-origin).count()<=max_age_ms
        && (command.expires_at==SteadyClock::time_point{} || now<=command.expires_at);
}
inline SteadyClock::time_point nextDispatchDeadline(SteadyClock::time_point deadline,
    SteadyClock::time_point now, SteadyClock::duration interval) {
    if (deadline==SteadyClock::time_point{}) deadline=now;
    if (deadline<=now) deadline+=interval*((now-deadline)/interval+1);
    return deadline;
}
class SimulatedCommandDispatch {
public:
    explicit SimulatedCommandDispatch(double hz, double max_age_ms)
        : interval_(std::chrono::duration_cast<SteadyClock::duration>(std::chrono::duration<double>(1.0/hz))),max_age_ms_(max_age_ms) {}
    void submit(const CommandPacket& command) { pending_=command; }
    void clear() { pending_.reset(); }
    std::optional<CommandPacket> poll(SteadyClock::time_point now) {
        if (deadline_!=SteadyClock::time_point{} && now<deadline_) return std::nullopt;
        deadline_=nextDispatchDeadline(deadline_,now,interval_);
        auto command=pending_; pending_.reset();
        if (command && !commandFreshAt(*command,now,max_age_ms_)) { ++expired;return std::nullopt; }
        return command;
    }
    int expired=0;
private:
    SteadyClock::duration interval_;
    SteadyClock::time_point deadline_{};
    double max_age_ms_;
    std::optional<CommandPacket> pending_;
};

// Aim-command scheduling shared by the live control loop and closed-loop replay.
// Hotkeys, recoil and physical delivery remain outside this state machine.
struct AimScheduleResult {
    std::optional<CommandPacket> command;
    std::optional<CommandPacket> timed_out_pending;
    bool budget_due=false;
    bool fresh=false;
    bool servo=false;
    bool queued=false;
    bool deferred=false;
    bool zero_servo=false;
};

class AimCommandScheduler {
public:
    void reset() { pending_.reset();next_tick_={};carry_x_=carry_y_=0.0; }
    SteadyClock::time_point deadline(SteadyClock::time_point now, bool active,
                                    SteadyClock::duration idle_wait) {
        if (!active) { next_tick_={};return now+idle_wait; }
        if (next_tick_==SteadyClock::time_point{}) next_tick_=now;
        return next_tick_;
    }
    AimScheduleResult step(SteadyClock::time_point now,SystemClock::time_point system_now,
        std::optional<CommandPacket> incoming,const DisplayRateServoState& servo,
        const RuntimeConfig& runtime,double send_hz,bool target_allowed,bool engaged) {
        AimScheduleResult result;
        const bool sigma=runtime.tracking_strategy==TrackingStrategy::PredictivePid
            && runtime.pid_enable && runtime.predictive_pid_human_motion_enable;
        const bool enabled=runtime.display_rate_servo_enable && !sigma;
        if (!enabled || !target_allowed) reset();
        const bool active=enabled && target_allowed && engaged;
        const double interval_s=1.0/std::clamp(send_hz,1.0,1000.0);
        const auto interval=std::max(SteadyClock::duration{1},
            std::chrono::duration_cast<SteadyClock::duration>(std::chrono::duration<double>(interval_s)));
        if (active && next_tick_==SteadyClock::time_point{}) next_tick_=now;
        if (!active) next_tick_={};
        if (active && now>=next_tick_) {
            result.budget_due=true;
            next_tick_=nextDispatchDeadline(next_tick_,now,interval);
        }
        const auto is_aim=[](const CommandPacket& cmd){return cmd.target_detected && !cmd.synthetic_recoil;};
        if (incoming && is_aim(*incoming) && !target_allowed) incoming.reset();
        if (enabled && target_allowed && incoming && is_aim(*incoming)) {
            pending_=*incoming;
            result.queued=!result.budget_due;
        } else result.command=std::move(incoming);
        if (pending_ && pending_->cmd_generated!=SteadyClock::time_point{}
            && std::chrono::duration<double>(now-pending_->cmd_generated).count()>0.10) {
            result.timed_out_pending=*pending_;pending_.reset();
        }
        if (!result.command && pending_ && result.budget_due && target_allowed) {
            result.command=*pending_;pending_.reset();result.fresh=true;
        }
        result.deferred=!result.command && pending_.has_value();
        if (!result.command && !result.deferred && enabled && target_allowed) {
            const double age=servo.updated_at==SteadyClock::time_point{}
                ? std::numeric_limits<double>::infinity()
                : std::chrono::duration<double>(now-servo.updated_at).count();
            const double max_age=std::max(0.0,runtime.display_rate_servo_max_target_age_ms)/1000.0;
            if (servo.valid && engaged && result.budget_due && age<=max_age && age<=0.10) {
                const double dt=std::min(interval_s,0.05);
                const double move_x=servo.velocity_x*dt+0.5*servo.acceleration_x*dt*dt+carry_x_;
                const double move_y=servo.velocity_y*dt+0.5*servo.acceleration_y*dt*dt+carry_y_;
                const int rounded_x=static_cast<int>(std::lround(move_x));
                const int rounded_y=static_cast<int>(std::lround(move_y));
                const int dx=std::clamp(rounded_x,-runtime.raw_max_step_x,runtime.raw_max_step_x);
                const int dy=std::clamp(rounded_y,-runtime.raw_max_step_y,runtime.raw_max_step_y);
                carry_x_=dx==rounded_x ? move_x-dx : 0.0;
                carry_y_=dy==rounded_y ? move_y-dy : 0.0;
                if (dx || dy) {
                    result.command=CommandPacket{
                        .dx=dx,.dy=dy,.acquire_started=servo.acquire_started,
                        .frame_ready=servo.frame_ready,.capture_done=servo.capture_done,
                        .cmd_generated=now,.generated_at=system_now,.frame_time=servo.frame_time,
                        .capture_time=servo.capture_time,.target_detected=true,
                        .synthetic_recoil=false,.synthetic_servo=true,.trigger_fire=false,
                        .source_id=servo.source_id,.observation_time=servo.observation_time,
                        .expires_at=servo.expires_at};
                    result.servo=true;
                } else result.zero_servo=true;
            } else if (!servo.valid || !engaged || age>max_age) carry_x_=carry_y_=0.0;
        }
        return result;
    }
private:
    std::optional<CommandPacket> pending_;
    SteadyClock::time_point next_tick_{};
    double carry_x_=0.0,carry_y_=0.0;
};
}
