#pragma once
#include "benchmark.hpp"
#include "delta/replay.hpp"
#include <stdexcept>

namespace delta::bench {
template<class Clock> inline std::int64_t clockCount(std::chrono::time_point<Clock> value) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(value.time_since_epoch()).count();
}
template<class Clock> inline auto clockValue(std::int64_t value) {
    return std::chrono::time_point<Clock>(std::chrono::duration_cast<typename Clock::duration>(std::chrono::nanoseconds(value)));
}
inline Json encodeObservation(const ReplayFrame& input) {
    const auto& t=input.toggles;
    if(t.mode!=1 || t.left_hold_engage || t.recoil_tune_fallback || t.left_pressed || t.right_pressed || t.x1_pressed)
        throw std::invalid_argument("Observation recorder only supports the no-device mode-1 simulation toggle state");
    const auto& f=input.frame;
    Json out={{"now_ns",clockCount(input.now)},
        {"capture",{f.capture.left,f.capture.top,f.width,f.height}},
        {"source_id",f.source_id},{"source_present_ns",clockCount(f.source_present)},
        {"acquire_started_ns",clockCount(f.acquire_started)},{"frame_ready_ns",clockCount(f.frame_ready)},
        {"capture_done_ns",clockCount(f.capture_done)},{"frame_time_ns",clockCount(f.frame_time)},
        {"capture_time_ns",clockCount(f.capture_time)},{"used_cached_frame",f.timings.used_cached_frame},
        {"detections",Json::array()},{"dispatch_feedback",Json::array()}};
    for(const auto& d:input.detections)out["detections"].push_back({{"bbox",d.bbox},{"x",d.x},{"y",d.y},{"cls",d.cls},{"conf",d.conf}});
    // Simulation always engages mode 1; reject unsupported toggles on import.
    out["mode"]=input.toggles.mode;
    for(const auto& d:input.dispatch_feedback)out["dispatch_feedback"].push_back({
        {"completed_ns",clockCount(d.completed_at)},{"generated_ns",clockCount(d.command.cmd_generated)},
        {"synthetic_servo",d.command.synthetic_servo},{"dx",d.dx},{"dy",d.dy},{"sent",d.movement_sent}});
    return out;
}
inline ReplayFrame decodeObservation(const Json& value) {
    ReplayFrame out;
    out.now=clockValue<SteadyClock>(value.at("now_ns").get<std::int64_t>());
    const auto box=value.at("capture").get<std::array<int,4>>();
    auto& f=out.frame;f.capture={box[0],box[1],box[2],box[3]};f.width=box[2];f.height=box[3];
    f.source_id=value.at("source_id").get<std::uint64_t>();
    const auto time=[&](const char* name){return clockValue<SteadyClock>(value.at(name).get<std::int64_t>());};
    f.source_present=time("source_present_ns");f.acquire_started=time("acquire_started_ns");
    f.frame_ready=time("frame_ready_ns");f.capture_done=time("capture_done_ns");f.frame_time=time("frame_time_ns");
    f.capture_time=clockValue<SystemClock>(value.at("capture_time_ns").get<std::int64_t>());
    f.timings.used_cached_frame=value.at("used_cached_frame").get<bool>();
    if(value.at("mode").get<int>()!=1)throw std::invalid_argument("Recorded simulation requires mode 1");
    for(const auto& d:value.at("detections"))out.detections.push_back({
        .bbox=d.at("bbox").get<std::array<int,4>>(),.x=d.at("x").get<float>(),.y=d.at("y").get<float>(),
        .cls=d.at("cls").get<int>(),.conf=d.at("conf").get<float>()});
    for(const auto& d:value.at("dispatch_feedback")) {
        CommandDelivery delivery;
        delivery.completed_at=clockValue<SteadyClock>(d.at("completed_ns").get<std::int64_t>());
        delivery.command.cmd_generated=clockValue<SteadyClock>(d.at("generated_ns").get<std::int64_t>());
        delivery.command.synthetic_servo=d.at("synthetic_servo").get<bool>();
        delivery.dx=d.at("dx").get<int>();delivery.dy=d.at("dy").get<int>();delivery.movement_sent=d.at("sent").get<bool>();
        out.dispatch_feedback.push_back(delivery);
    }
    return out;
}
inline Json observationOutcome(const ReplayResult& r) {
    Json selected=Json::array();
    for(const auto& d:r.preview.detections)if(d.selected)selected.push_back({{"bbox",d.bbox},{"cls",d.cls}});
    return {{"command",r.command?Json::array({r.command->dx,r.command->dy}):Json(nullptr)},
        {"selected_detections",selected},{"output_ready",r.output_ready},{"motion_decision",r.motion_decision},
        {"motion_ambiguous",r.motion_ambiguous},{"stale",r.stale_frame},
        {"predicted_point",r.preview.predicted_point?Json::array({r.preview.predicted_point->first,r.preview.predicted_point->second}):Json(nullptr)}};
}
}
