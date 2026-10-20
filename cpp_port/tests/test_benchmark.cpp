#include "benchmark.hpp"
#include "scene_geometry.hpp"
#include "closed_loop_scoring.hpp"
#include "visible_scene.hpp"
#include "observation_replay.hpp"
#include "delta/app.hpp"
#include "delta/detection_dampening.hpp"
#include "delta/dispatch.hpp"
#include "delta/control.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

using delta::bench::Json;
namespace {
void require(bool condition,const char* message) { if (!condition) throw std::runtime_error(message); }
template<class F> void rejects(F action,const char* message) {
    bool rejected=false; try { action(); } catch (const std::exception&) { rejected=true; }
    require(rejected,message);
}

class GapReplay : public delta::ITrackingReplay {
public:
    std::vector<delta::ReplayResult> results;
    bool invalid_time=false;
    std::optional<delta::ReplayFrame> next(const delta::CaptureRegion&) override {
        if (index==15) return std::nullopt;
        delta::ReplayFrame frame;
        frame.now=delta::SteadyClock::time_point(std::chrono::milliseconds(1000+(invalid_time?0:index*10)));
        frame.frame.frame_ready=frame.now-std::chrono::milliseconds(5);
        frame.frame.frame_time=frame.frame.frame_ready;
        frame.frame.capture={280,0,720,720};
        if (index<5) frame.detections.push_back(delta::Detection{.bbox={380,330,410,360},.cls=1,.conf=0.95F});
        ++index; return frame;
    }
    void observe(const delta::ReplayResult& result) override { results.push_back(result); }
private:
    int index=0;
};

class IdentityReplay : public delta::ITrackingReplay {
public:
    std::vector<delta::ReplayResult> results;
    std::optional<delta::ReplayFrame> next(const delta::CaptureRegion&) override {
        if (index==4) return std::nullopt;
        delta::ReplayFrame frame;
        frame.now=delta::SteadyClock::time_point(std::chrono::milliseconds(2000+index*8));
        frame.frame.capture={0,0,1280,720};
        frame.frame.frame_ready=frame.now-std::chrono::milliseconds(1);
        frame.frame.source_id=index<2?1:index;
        frame.frame.source_present=delta::SteadyClock::time_point(std::chrono::milliseconds(index<2?1995:1995+index*8));
        frame.detections.push_back(delta::Detection{.bbox={650+index*24,330,676+index*24,356},.cls=1,.conf=.95F});
        ++index;return frame;
    }
    void observe(const delta::ReplayResult& r) override { results.push_back(r); }
private:int index=0;
};

class ReversalReplay : public delta::ITrackingReplay {
public:
    std::vector<delta::ReplayResult> results;
    std::optional<delta::ReplayFrame> next(const delta::CaptureRegion&) override {
        const std::array<int,4> x={650,674,650,626};
        if(index==x.size())return std::nullopt;
        delta::ReplayFrame frame;
        const auto at=[](int ms){return delta::SteadyClock::time_point(std::chrono::milliseconds(ms));};
        frame.now=at(1005+index*8);frame.frame.frame_ready=at(1001+index*8);
        frame.frame.source_present=at(1000+index*8);frame.frame.source_id=index+1;
        frame.frame.capture={0,0,1280,720};
        frame.detections.push_back(delta::Detection{.bbox={x[index]-13,330,x[index]+13,356},.cls=1,.conf=.95F});
        ++index;return frame;
    }
    void observe(const delta::ReplayResult& r) override {results.push_back(r);}
private:std::size_t index=0;
};

class CameraShiftReplay : public delta::ITrackingReplay {
public:
    bool duplicate_last=false;
    bool excessive_shift=false;
    std::vector<delta::ReplayResult> results;
    std::optional<delta::ReplayFrame> next(const delta::CaptureRegion&) override {
        const std::array<int,3> x={650,697,excessive_shift?777:757},y={343,342,364};
        if(index==x.size())return std::nullopt;
        const auto at=[](int ms){return delta::SteadyClock::time_point(std::chrono::milliseconds(ms));};
        delta::ReplayFrame frame;frame.now=at(2005+index*8);
        frame.frame.frame_ready=at(2001+index*8);frame.frame.source_present=at(2000+index*8);
        frame.frame.source_id=index+1;frame.frame.capture={0,0,1280,720};
        frame.detections.push_back(delta::Detection{.bbox={x[index]-22,y[index]-23,x[index]+23,y[index]+24},.cls=1,.conf=.95F});
        if(duplicate_last && index==2)frame.detections.push_back(frame.detections.front());
        ++index;return frame;
    }
    void observe(const delta::ReplayResult& r) override {results.push_back(r);}
private:std::size_t index=0;
};

class FeedbackReplay : public delta::ITrackingReplay {
public:
    bool future_delivery=false;
    std::vector<delta::ReplayResult> results;
    std::optional<delta::ReplayFrame> next(const delta::CaptureRegion&) override {
        if(index==3)return std::nullopt;
        const auto at=[](int ms){return delta::SteadyClock::time_point(std::chrono::milliseconds(ms));};
        delta::ReplayFrame frame;frame.now=at(1005+index*20);
        frame.frame.frame_ready=frame.now-std::chrono::milliseconds(1);
        frame.frame.capture={0,0,1280,720};
        frame.detections.push_back(delta::Detection{.bbox={680+index*2,330,710+index*2,360},.cls=1,.conf=.95F});
        if(index>0) {
            delta::CommandDelivery delivered;
            delivered.command.cmd_generated=at(985+index*20);
            delivered.completed_at=at(995+index*20);
            if(future_delivery)delivered.completed_at=frame.now+std::chrono::milliseconds(1);
            delivered.dx=10;delivered.dy=-5;delivered.movement_sent=true;
            frame.dispatch_feedback.push_back(delivered);
        }
        ++index;return frame;
    }
    void observe(const delta::ReplayResult& result) override {results.push_back(result);}
private:int index=0;
};

Json scoringFrame(int time,int selected,int cls=1) {
    return {{"time_ms",time},{"dt_ms",10},{"intended_id",1},{"selected_id",selected},{"tracked_id",selected},{"aim_class",cls},
        {"intended_present",true},{"intended_observed",true},{"stale",false},{"command",Json::array({1,0})},
        {"error_normalized",0.5},{"filtered_error_normalized",0.25},{"observation_age_ms",8}};
}
class CropFragmentReplay:public delta::ITrackingReplay {
public:
    std::optional<delta::ReplayFrame> next(const delta::CaptureRegion&)override {
        if(index==2)return {};
        return delta::bench::decodeObservation(inputs.at(index++));
    }
    void observe(const delta::ReplayResult& result)override{results.push_back(result);}
    std::vector<delta::ReplayResult> results;
private:
    std::size_t index=0;
    const Json inputs=Json::parse(R"JSON([
      {"acquire_started_ns":11837764600,"capture":[880,320,800,800],"capture_done_ns":11838983600,"capture_time_ns":1792388482654144200,"detections":[{"bbox":[0,381,29,530],"cls":0,"conf":0.3647775650024414,"x":14.851261138916016,"y":455.8293151855469},{"bbox":[0,380,29,623],"cls":0,"conf":0.2133771777153015,"x":14.655948638916016,"y":501.7427978515625}],"dispatch_feedback":[],"frame_ready_ns":11837789700,"frame_time_ns":11837789700,"mode":1,"now_ns":11850254000,"source_id":112,"source_present_ns":11837733900,"used_cached_frame":false},
      {"acquire_started_ns":11864819300,"capture":[848,288,864,864],"capture_done_ns":11866273100,"capture_time_ns":1792388482681433500,"detections":[{"bbox":[0,440,12,539],"cls":1,"conf":0.12294378876686096,"x":6.309328079223633,"y":489.9271240234375}],"dispatch_feedback":[],"frame_ready_ns":11864875500,"frame_time_ns":11864875500,"mode":1,"now_ns":11877681400,"source_id":114,"source_present_ns":11856082400,"used_cached_frame":false}
    ])JSON");
};
}

int main() {
    {
        const double a=delta::bench::constantAngularTargetAngle(0,6000,1800);
        const double b=delta::bench::constantAngularTargetAngle(100,6000,1800);
        const double c=delta::bench::constantAngularTargetAngle(200,6000,1800);
        require(std::abs((b-a)-(c-b))<1e-9,"Constant-motion scene must not insert speed-dependent reversals");
        require(delta::bench::projectAngularTargetX(3.141592653589793,1800,2560)<0,
            "Targets behind the camera must stay outside the rendered view");
        require(delta::bench::projectAngularTargetX(0,1800,2560)==1280,
            "Aligned target must project to screen center");
    }
    try {
        delta::StaticConfig config; config.screen_w=1280;config.screen_h=720;
        delta::RuntimeConfig runtime;
        {
            delta::DetectionDampeningState state;
            auto policy=runtime;policy.motion_identity_enable=true;policy.motion_stability_enable=true;
            policy.detection_dampening_stable_frames=2;policy.motion_match_margin=.75F;
            policy.motion_residual_scale=.5F;policy.motion_prediction_timeout_ms=40;
            const auto at=[](int ms){return delta::SteadyClock::time_point(std::chrono::milliseconds(ms));};
            const auto head=[](int x){return delta::Detection{.bbox={x-23,100,x+23,147},.x=float(x),.y=123.5F,.cls=1,.conf=.95F};};
            delta::noteMotionSelection(state,policy,head(100),{head(100)},at(1000),at(1001));
            delta::noteMotionSelection(state,policy,head(117),{head(117)},at(1006),at(1007));
            const auto crossing=delta::noteMotionSelection(state,policy,head(77),{head(77),head(184)},at(1024),at(1025));
            require(crossing.ambiguous && !crossing.detection && state.last_observation==at(1006),
                "A physically reachable reversal competitor must prevent a confident identity switch");
        }
        {
            auto cfg=config;cfg.screen_w=2560;cfg.screen_h=1440;
            auto policy=runtime;policy.motion_stability_enable=true;policy.motion_identity_enable=true;
            policy.motion_residual_scale=.5F;policy.motion_prediction_timeout_ms=40;
            policy.motion_match_margin=.75F;policy.motion_prediction_max_displacement_px=30;
            CropFragmentReplay replay;delta::DeltaApp::replay(cfg,policy,replay);
            require(!replay.results.back().output_ready,
                "Crop-clipped detector fragments must not confirm a new motion target");
            const auto at=[](int ms){return delta::SteadyClock::time_point(std::chrono::milliseconds(ms));};
            delta::DetectionDampeningState state;
            policy.detection_dampening_stable_frames=2;
            const delta::CaptureRegion crop{0,0,250,250};
            const delta::Detection full{.bbox={100,100,140,150},.x=120,.y=125,.cls=1,.conf=.9F};
            delta::noteMotionSelection(state,policy,full,{full},at(1000),at(1001),crop);
            delta::noteMotionSelection(state,policy,full,{full},at(1008),at(1009),crop);
            const delta::Detection clipped{.bbox={0,100,12,150},.x=6,.y=125,.cls=1,.conf=.9F};
            const auto predicted=delta::noteMotionSelection(state,policy,clipped,{clipped},at(1016),at(1017),crop);
            require(predicted.ready && !predicted.detection && state.last_observation==at(1008),
                "Clipped evidence must preserve only bounded prediction without refreshing the observation");
            const auto expired=delta::noteMotionSelection(state,policy,clipped,{clipped},at(1050),at(1051),crop);
            require(!expired.ready && !state.confirmed,
                "Repeated clipped evidence must not extend the prediction deadline");
            for(const auto& box:std::vector<std::array<int,4>>{{238,100,249,150},{100,238,150,249}}) {
                delta::DetectionDampeningState edge_state;
                const delta::Detection fragment{.bbox=box,.x=(box[0]+box[2])*.5F,.y=(box[1]+box[3])*.5F,.cls=1,.conf=.9F};
                delta::noteMotionSelection(edge_state,policy,fragment,{fragment},at(1000),at(1001),crop);
                const auto edge=delta::noteMotionSelection(edge_state,policy,fragment,{fragment},at(1008),at(1009),crop);
                require(!edge.ready,"Right/bottom last-pixel detector clipping must not confirm a target");
            }
        }
        {
            const std::vector<delta::bench::SceneIdentityRegions> truth={
                {1,{100,100,200,300},{100,80,125,105}},
                {2,{105,100,205,300},{180,80,205,105}}};
            require(delta::bench::attributeSceneDetection(112,100,1,truth)==1,
                "Head identity must use linked head regions rather than overlapping body unions");
            require(delta::bench::attributeSceneDetection(150,180,0,truth)==-2,
                "Overlapping same-class truth must remain ambiguous");
        }
        {
            FeedbackReplay source;
            for(int i=0;i<3;++i) {
                const auto input=source.next({});
                const auto encoded=delta::bench::encodeObservation(*input);
                require(delta::bench::encodeObservation(delta::bench::decodeObservation(encoded))==encoded,
                    "Observation serialization must preserve exact tracking inputs and delivery feedback");
            }
        }
        {
            constexpr int width=800,height=800;
            for(const std::uint32_t id:{1U,0x12345678U,0xffffffffU}) {
                std::vector<std::uint8_t> pixels(width*height*4,0);
                delta::bench::writeSceneMarker(pixels,width,height,id);
                delta::FramePacket captured;captured.width=320;captured.height=320;
                captured.capture={240,240,320,320};captured.bgr.resize(320*320*3);
                for(int y=0;y<320;++y)for(int x=0;x<320;++x)
                    std::copy_n(pixels.data()+((y+240)*width+x+240)*4,3,captured.bgr.data()+(y*320+x)*3);
                require(delta::bench::readSceneMarker(captured,width,height)==id,
                    "Desktop provenance marker must survive a production-sized BGR crop");
                const int at=((400+140-240+5)*320+(400-48-240+1))*3;
                captured.bgr[at]=captured.bgr[at+1]=captured.bgr[at+2]=pixels[((400+141)*width+353)*4];
                require(!delta::bench::readSceneMarker(captured,width,height),
                    "Torn or corrupt provenance must not produce a valid scene link");
                captured.capture.left=0;
                require(!delta::bench::readSceneMarker(captured,width,height),
                    "A crop excluding the marker must not receive scene provenance");
            }
        }
        {
            auto policy=runtime;policy.detection_dampening_enable=false;policy.target_guard_enable=false;
            FeedbackReplay replay;delta::DeltaApp::replay(config,policy,replay);
            const auto feedback=replay.results.back().dispatch_feedback;
            require(std::abs(feedback.send_latency_s-.0036F)<1e-6F,
                "Replay must feed successful scheduled delivery latency back to the production controller");
            require(std::abs(feedback.sent_velocity_x-150.0F)<1e-3F && std::abs(feedback.sent_velocity_y+75.0F)<1e-3F,
                "Replay must feed delivered movement velocity back to the production controller");
            FeedbackReplay future;future.future_delivery=true;
            rejects([&]{delta::DeltaApp::replay(config,policy,future);},
                "Future delivery feedback must not leak into the current inference step");
        }
        {
            const auto t=[](int ms){return delta::SteadyClock::time_point(std::chrono::milliseconds(ms));};
            delta::SimulatedCommandDispatch dispatch(100,50);
            dispatch.poll(t(1000));
            delta::CommandPacket a; a.dx=1;a.observation_time=t(1000);a.expires_at=t(1030);
            dispatch.submit(a);
            require(!dispatch.poll(t(1005)),"Scheduler must wait for budget");
            a.dx=2;dispatch.submit(a);
            auto sent=dispatch.poll(t(1010));
            require(sent && sent->dx==2,"Dispatch must use newest command");
            dispatch.submit(a);
            require(!dispatch.poll(t(1040)) && dispatch.expired==1,"Dispatch must reject expired prediction even when observation is under age cap");
            a.expires_at={};dispatch.submit(a);
            require(!dispatch.poll(t(1100)) && dispatch.expired==2,"Dispatch must reject old observations");
            require(!dispatch.poll(t(1101)),"Late wake must not create a catch-up burst");
        }
        {
            delta::DetectionDampeningState state;
            auto policy=runtime;policy.motion_stability_enable=true;
            const auto t=[](int ms){return delta::SteadyClock::time_point(std::chrono::milliseconds(ms));};
            const auto box=[](float x){return delta::Detection{.bbox={int(x)-13,0,int(x)+13,26},.x=x,.y=13,.cls=1,.conf=.95F};};
            auto first=delta::noteMotionSelection(state,policy,box(100),{box(100)},t(1000),t(1005));
            require(!first.ready && first.detection.has_value(),"Acquisition estimates before allowing output");
            auto duplicate=delta::noteMotionSelection(state,policy,box(100),{box(100)},t(1000),t(1006));
            require(!duplicate.ready && duplicate.duplicate && state.stable_frame_count==1,"Repeated source frame cannot confirm");
            auto fast=delta::noteMotionSelection(state,policy,box(148),{box(148)},t(1008),t(1013));
            require(fast.ready && fast.detection.has_value(),"Non-overlapping fast target must confirm");
            policy.motion_identity_enable=true;
            auto ambiguous=delta::noteMotionSelection(state,policy,box(195),{box(195),box(197)},t(1016),t(1021));
            require(ambiguous.ambiguous && !ambiguous.detection && state.last_observation==t(1008),"Ambiguity must not refresh or corrupt motion");
            auto expired=delta::noteMotionSelection(state,policy,std::nullopt,{},t(1110),t(1115));
            require(!expired.ready && !state.confirmed,"Prediction deadline must expire without accepted evidence");
            auto reacquire=delta::noteMotionSelection(state,policy,box(200),{box(200)},t(1120),t(1125));
            require(!reacquire.ready,"Expired target must reconfirm");
            delta::noteMotionSelection(state,policy,box(210),{box(210)},t(1130),t(1135));
            auto shaken=delta::noteMotionSelection(state,policy,box(700),{box(700)},t(1140),t(1145));
            require(!shaken.ready && !state.confirmed,"Contradicted motion must withhold output and require reconfirmation");
        }
        const auto encoded=delta::bench::runtimeJson(runtime);
        {
            auto candidate=runtime;candidate.motion_stability_enable=true;candidate.target_guard_enable=false;
            IdentityReplay replay;delta::DeltaApp::replay(config,candidate,replay);
            require(replay.results[0].track.has_value() && !replay.results[0].command,"Pending confirmation must retain an estimate without output");
            require(replay.results[1].stale_frame && !replay.results[1].command,"Same source frame with a new acquisition time must be rejected");
            require(replay.results[2].output_ready && replay.results[2].command.has_value(),"Distinct fast observations should confirm in the production pipeline");
        }
        {
            auto candidate=runtime;
            candidate.motion_stability_enable=true;candidate.motion_identity_enable=true;
            candidate.motion_residual_scale=.5F;candidate.motion_prediction_timeout_ms=40;
            candidate.motion_prediction_max_displacement_px=30;candidate.motion_match_margin=.75F;
            candidate.target_guard_enable=false;
            CameraShiftReplay shift;delta::DeltaApp::replay(config,candidate,shift);
            require(shift.results[1].output_ready,"Fast observations must confirm before camera shift");
            require(shift.results[2].output_ready,"Unique bounded screen motion must remain confirmed through a modest camera shift");
            CameraShiftReplay duplicate;duplicate.duplicate_last=true;
            delta::DeltaApp::replay(config,candidate,duplicate);
            require(!duplicate.results[2].motion_ambiguous,
                "An identical detection repeated in one source frame must not invent a competing target");
            CameraShiftReplay excessive;excessive.excessive_shift=true;
            delta::DeltaApp::replay(config,candidate,excessive);
            require(!excessive.results[2].output_ready,
                "A shift beyond the motion-speed bound must withhold output rather than preserve confirmation");
            ReversalReplay replay;delta::DeltaApp::replay(config,candidate,replay);
            require(replay.results[1].output_ready,"Forward motion must confirm before reversal");
            require(!replay.results[2].output_ready,"Contradicted motion must withhold output");
            require(replay.results[3].output_ready,"Two consistent post-reversal observations must reconfirm without waiting for loss timeout");
        }
        require(delta::bench::runtimeJson(delta::bench::runtimeFromJson(encoded))==encoded,"Runtime snapshot must round-trip");
        rejects([]{delta::bench::runtimeFromJson(Json{{"typo",true}});},"Unknown config keys must fail");
        rejects([]{delta::bench::runtimeFromJson(Json{{"aim_mode",99}});},"Unknown enum must fail");

        runtime.detection_dampening_enable=false;
        runtime.target_guard_enable=false;
        runtime.target_max_lost_frames=100;
        runtime.target_prediction_timeout_ms=25;
        GapReplay gap; delta::DeltaApp::replay(config,runtime,gap);
        require(gap.results.size()==15,"Every replay input must produce exactly one observation");
        require(gap.results[4].command.has_value(),"Stable detections should produce commands");
        require(gap.results[5].command.has_value(),"Short detection gap should predict");
        require(!gap.results[7].command.has_value(),"Millisecond timeout should stop prediction before frame budget");
        require(!gap.results.back().track.has_value(),"Expired tracker should be cleared");
        {
            const auto checks=delta::bench::closedLoopCorrectness(Json::array({
                {{"scene_state_id",1},{"past_loss_deadline",false}},
                {{"scene_state_id",2},{"past_loss_deadline",true}}}));
            require(checks["passed"]==false && checks["violations"].size()==1
                && checks["violations"][0]["scene_state_id"]==2,
                "Expired applied movement must fail correctness without invalidating the captured evidence");
            require(delta::bench::closedLoopCorrectness(Json::array())["passed"]==true,
                "No applied movement must not invent a deadline violation");
        }
        {
            const Json movements=Json::array({{{"selected_truth_id",2},{"aim_class",1}},
                {{"selected_truth_id",1},{"aim_class",1}},{{"selected_truth_id",1},{"aim_class",0}},
                {{"selected_truth_id",-2},{"aim_class",0}},{{"selected_truth_id",2},{"aim_class",0}}});
            const auto identities=delta::bench::scoreMovementIdentity(movements,1);
            require(identities["initial_selection_correct"]==false,"Wrong initial acquisition must be explicit");
            require(identities["wrong_movements_before_intended_acquisition"]==1
                && identities["wrong_movements_after_intended_acquisition"]==1,
                "Initial selection and subsequent identity loss must be scored separately");
            require(identities["identity_switches"]==2 && identities["aim_point_changes"]==1,
                "Ambiguous observations and head/body changes must not invent identity switches");
        }
        {
            auto aim=runtime;aim.head_x_ratio=.8F;aim.head_y_ratio=.2F;aim.body_y_ratio=.7F;
            const std::array<int,4> sprite={84,120,264,370};
            const auto head=delta::bench::projectSceneAimPoint({140,127,186,174},1,sprite,400,400,1,aim);
            const auto body=delta::bench::projectSceneAimPoint({92,128,256,364},0,sprite,400,400,1,aim);
            require(std::abs(head.first-402.8)<1e-3 && std::abs(head.second-291.4)<1e-3,
                "Head truth must honor the configured aim ratios");
            require(std::abs(body.first-400)<1e-3 && std::abs(body.second-448.2)<1e-3,
                "Body truth must honor the configured aim ratio");
        }
        {
            const auto box=delta::bench::projectIdentityBox({10,40,90,200},{30,0,70,40},
                {0,0,100,200},500,500,2);
            require(box==delta::bench::SceneBox{420,300,580,700},"Identity region must project linked labels");
            require(delta::bench::sceneContains(box,500,690),"Reticle inside the body must count independently of aim point");
            require(!delta::bench::sceneContains(box,500,710),"Reticle beyond labeled body must not count as inside");
            require(std::abs(delta::bench::normalizedSceneError(box,580,500,500,500)-.5)<1e-6,
                "Aim error must use labeled identity size rather than sprite padding");
        }
        {
            delta::AimCommandScheduler scheduler;
            delta::RuntimeConfig policy;
            policy.display_rate_servo_enable=true;
            policy.display_rate_servo_hz=100;
            policy.display_rate_servo_max_target_age_ms=35;
            const auto at=[](int ms){return delta::SteadyClock::time_point(std::chrono::milliseconds(1000+ms));};
            delta::DisplayRateServoState servo;
            servo.valid=true;servo.updated_at=at(0);servo.frame_ready=at(0);
            servo.observation_time=at(0);servo.velocity_x=500;
            delta::CommandPacket fresh;
            fresh.dx=2;fresh.target_detected=true;fresh.frame_ready=at(0);
            fresh.observation_time=at(0);fresh.cmd_generated=at(0);
            auto first=scheduler.step(at(0),{},fresh,servo,policy,100,true,true);
            require(first.fresh && first.command && first.command->dx==2,"Fresh command must win first budget");
            auto between=scheduler.step(at(5),{},std::nullopt,servo,policy,100,true,true);
            require(!between.command,"Servo must not bypass a display budget");
            auto extrapolated=scheduler.step(at(10),{},std::nullopt,servo,policy,100,true,true);
            require(extrapolated.servo && extrapolated.command && extrapolated.command->dx==5,
                "Production scheduler must generate servo movement between observations");
            auto expired=scheduler.step(at(40),{},std::nullopt,servo,policy,100,true,true);
            require(!expired.command,"Expired servo observation must not generate movement");
            delta::MouseMovementTransform transform;
            delta::MouseSenderConfig sender;sender.method=delta::MouseOutputMethod::SendInput;
            sender.gain_x=.5F;sender.max_step=2;transform.configure(sender);
            require(transform.apply(1,0)==std::vector<std::pair<int,int>>{{1,0}},"SendInput rounding must be shared");
            require(transform.apply(1,0).empty(),"Fractional carry must be shared");
            require(transform.apply(9,0)==(std::vector<std::pair<int,int>>{{2,0},{2,0},{1,0}}),
                "SendInput step splitting must be shared");
        }
        GapReplay invalid; invalid.invalid_time=true;
        rejects([&]{delta::DeltaApp::replay(config,runtime,invalid);},"Non-increasing logical time must fail");

        auto rows=Json::array({scoringFrame(10,1,0),scoringFrame(20,1,1),scoringFrame(30,2),scoringFrame(40,2)});
        auto score=delta::bench::scoreFrames(rows,150);
        require(score["identity_switches"]==1,"Head/body change is not an identity switch");
        require(score["aim_point_changes"]==1,"Aim-point changes should be recorded separately");
        require(score["wrong_target_ms"]==20,"Wrong identity must be scored using truth identity");
        rows=Json::array({scoringFrame(10,-1)});
        rows[0]["command"]=nullptr;rows[0]["error_normalized"]=nullptr;rows[0]["filtered_error_normalized"]=nullptr;
        score=delta::bench::scoreFrames(rows,150);
        require(score["mean_normalized_error"].is_null(),"No prediction must not yield zero error");
        require(score["prediction_coverage"]==0,"No prediction must yield zero coverage");
        rows=Json::array({scoringFrame(10,1),scoringFrame(20,-1),scoringFrame(30,-1),scoringFrame(40,-1)});
        rows[1]["intended_observed"]=false;rows[2]["intended_observed"]=false;
        score=delta::bench::scoreFrames(rows,5);
        require(score["unrecovered_episodes"]==1,"Unresolved recovery must be counted, not scored as zero time");
        require(score["movement_after_loss_deadline_ms"]==20,"Movement past the evaluation deadline must be counted");

        delta::bench::Options options; options.scenario="combined";
        const auto a=delta::bench::runSuite(config,delta::RuntimeConfig{},options);
        const auto b=delta::bench::runSuite(config,delta::RuntimeConfig{},options);
        require(a==b,"Identical seeds/configuration must yield identical accuracy artifacts");
        require(a["correctness_violations"].empty(),"Production command invariants should hold");
        const auto comparison=delta::bench::compareRuns(a,b);
        require(comparison[0]["candidate_minus_baseline"]["wrong_target_ms"]==0,"Identical comparison must report zero change");
        auto altered=b; altered["runs"][0]["input_hash"]="different";
        rejects([&]{delta::bench::compareRuns(a,altered);},"Mismatched inputs must not be compared");
        options.scenario="stale_cadence";
        const auto stale=delta::bench::runSuite(config,delta::RuntimeConfig{},options);
        require(stale["runs"][0]["metrics"]["stale_frames"].get<int>()>0,"Scenario must exercise production stale-frame rejection");
        for(const auto& frame:stale["runs"][0]["frames"]) if(frame["stale"].get<bool>()) require(frame["command"].is_null(),"Rejected stale input must not emit a command");
        options.scenario="head_body";
        auto hybrid=delta::RuntimeConfig{};hybrid.target_guard_enable=false;hybrid.detection_dampening_enable=false;
        const auto head=delta::bench::runSuite(config,hybrid,options);
        require(head["runs"][0]["metrics"]["identity_switches"]==0,"Same entity head/body transition must preserve truth identity");
        require(head["runs"][0]["metrics"]["aim_point_changes"].get<int>()>0,"Head/body scenario must actually change aim class");
        options.scenario="allowed_switch";
        const auto handoff=delta::bench::runSuite(config,hybrid,options);
        require(handoff["runs"][0]["metrics"]["identity_switches"]==0,"Explicitly permitted handoff must not count as an identity error");
        options.scenario="clean";
        hybrid.predictive_pid_human_motion_enable=true;
        require(delta::bench::runSuite(config,hybrid,options)==delta::bench::runSuite(config,hybrid,options),
            "Production motion shaping must use the replay seed");
        options.suite="speed";options.scenario="high_speed";
        const auto speed=delta::bench::runSuite(config,delta::RuntimeConfig{},options);
        require(speed["runs"].size()==36,"Speed suite must vary velocity, cadence and observation age independently");
        double low_age_error=0,high_age_error=0;
        for(const auto& run:speed["runs"])if(run["speed_px_s"]==1500 && run["cadence_ms"]==4){
            if(run["configured_observation_age_ms"]==5)low_age_error=run["metrics"]["mean_normalized_error"];
            if(run["configured_observation_age_ms"]==30)high_age_error=run["metrics"]["mean_normalized_error"];
        }
        require(high_age_error>low_age_error*2,"High-speed benchmark must expose error caused by stale observations");
        auto malicious=a; malicious["build"]["revision"]="</script><script>throw Error('injection')</script> __DELTA_BASELINE_JSON__";
        delta::bench::writeReport("benchmark-test-report.html",malicious,b);
        std::ifstream html("benchmark-test-report.html"); const std::string content((std::istreambuf_iterator<char>(html)),{});
        require(content.find("<script>throw Error")==std::string::npos,"Report must escape embedded JSON");
        require(content.find("__DELTA_RUN_JSON__")==std::string::npos,"Report data must be embedded");
        std::cout << "Benchmark integration and evaluator checks passed\n";
        return 0;
    } catch(const std::exception& error) { std::cerr<<error.what()<<'\n';return 1; }
}
