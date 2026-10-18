#include "benchmark.hpp"
#include "delta/app.hpp"

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

Json scoringFrame(int time,int selected,int cls=1) {
    return {{"time_ms",time},{"dt_ms",10},{"intended_id",1},{"selected_id",selected},{"tracked_id",selected},{"aim_class",cls},
        {"intended_present",true},{"intended_observed",true},{"stale",false},{"command",Json::array({1,0})},
        {"error_normalized",0.5},{"filtered_error_normalized",0.25},{"observation_age_ms",8}};
}
}

int main() {
    try {
        delta::StaticConfig config; config.screen_w=1280;config.screen_h=720;
        delta::RuntimeConfig runtime;
        const auto encoded=delta::bench::runtimeJson(runtime);
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
