#include "benchmark.hpp"
#include "delta/app.hpp"
#include "benchmark_build.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace delta::bench {
namespace {
constexpr int scenarioVersion = 1;
constexpr int durationMs = 3000;
constexpr int epochMs = 10000; // Zero is a production sentinel, never a replay timestamp.
const std::vector<std::string> names = {"clean", "burst_loss", "noisy_boxes", "false_positives",
    "crossing", "camera_shake", "crop_exit", "stale_cadence", "head_body", "departure", "allowed_switch", "combined"};

SteadyClock::time_point tick(double ms) {
    return SteadyClock::time_point(std::chrono::duration_cast<SteadyClock::duration>(
        std::chrono::duration<double, std::milli>(epochMs + ms)));
}
Json point(float x, float y) { return Json::array({x, y}); }
Json rect(const CaptureRegion& r) { return Json::array({r.left, r.top, r.width, r.height}); }
Json optionalPoint(const std::optional<std::pair<float, float>>& p) {
    return p ? point(p->first, p->second) : Json(nullptr);
}

// A specified integer generator keeps scenario inputs independent of standard
// library distribution implementations and of tracker branching/consumption.
struct Random {
    std::uint64_t state;
    double uniform() {
        state ^= state >> 12; state ^= state << 25; state ^= state >> 27;
        return static_cast<double>((state * 2685821657736338717ULL) >> 11) / 9007199254740992.0;
    }
    int jitter(int amplitude) { return static_cast<int>(uniform() * (2 * amplitude + 1)) - amplitude; }
};
struct Hash {
    std::uint64_t state = 14695981039346656037ULL;
    void add(const std::string& data) { for (const unsigned char c : data) { state ^= c; state *= 1099511628211ULL; } }
    std::string str() const { std::ostringstream s; s << std::hex << std::setw(16) << std::setfill('0') << state; return s.str(); }
};

struct Actor { int id; std::array<int,4> body; std::array<int,4> head; };
struct Scenario {
    std::string name;
    std::uint64_t seed;
    int gap_ms = 150;
    std::string variant = "configured";
    int speed_px_s = 0;
    int cadence_ms = 10;
    int age_ms = 8;
};

std::pair<double,double> camera(const Scenario& s, double t) {
    if ((s.name == "camera_shake" || s.name == "combined") && t >= 0.9 && t < 1.9)
        return {150 * std::sin(t * 73) + (t > 1.3 ? 90 : 0), 95 * std::cos(t * 91)};
    return {0,0};
}

std::vector<Actor> world(const StaticConfig& config, const Scenario& s, double t) {
    const auto [shake_x,shake_y] = camera(s,t);
    double x = config.screen_w / 2.0 + 65 * std::sin(t * 2.1);
    double y = config.screen_h / 2.0 + 28 + 20 * std::sin(t * 3.0);
    if (s.name == "high_speed") {
        // Constant-speed sweeps with abrupt reversals at +/- 260 px. The target
        // stays inside the crop, separating association limits from field-of-view loss.
        const double phase = std::fmod(std::max(0.0,t) * s.speed_px_s + 260.0, 1040.0);
        x = config.screen_w / 2.0 - 260.0 + (phase <= 520.0 ? phase : 1040.0-phase);
        y = config.screen_h / 2.0 + 28;
    }
    if (s.name == "crossing") x = config.screen_w / 2.0 - 120 + t * 80;
    if (s.name == "crop_exit" && t > 0.8 && t < 2.4) x += 800 * std::sin((t - 0.8) / 1.6 * 3.141592653589793);
    const auto actor = [&](int id, double ax, double ay) {
        int cx = static_cast<int>(std::lround(ax + shake_x)), cy = static_cast<int>(std::lround(ay + shake_y));
        return Actor{id, {cx-30,cy-45,cx+30,cy+65}, {cx-13,cy-48,cx+13,cy-22}};
    };
    std::vector<Actor> out;
    if (!(s.name == "departure" && t >= 1.25) && !(s.name == "allowed_switch" && t >= 1.5))
        out.push_back(actor(1,x,y));
    if (s.name == "crossing") out.push_back(actor(2,config.screen_w/2.0 + 120 - t*80,config.screen_h/2.0 + 62));
    if (s.name == "combined") out.push_back(actor(2,config.screen_w/2.0 + 200 - 40*t,config.screen_h/2.0 + 100));
    if (s.name == "allowed_switch" && t >= 1.5) out.push_back(actor(2,config.screen_w/2.0 + 70,config.screen_h/2.0 + 28));
    return out;
}

class ScenarioReplay final : public ITrackingReplay {
public:
    ScenarioReplay(const StaticConfig& config, const RuntimeConfig& runtime, Scenario scenario, bool collect)
        : config_(config), runtime_(runtime), scenario_(std::move(scenario)), rng_{scenario_.seed}, collect_(collect) {}
    std::uint64_t randomSeed() const override { return scenario_.seed; }

    std::optional<ReplayFrame> next(const CaptureRegion& proposed) override {
        if (time_ms_ >= durationMs) return std::nullopt;
        const int cadence[] = {7, 13, 25, 5, 10};
        interval_ms_ = scenario_.name == "stale_cadence" ? cadence[index_ % 5] : scenario_.cadence_ms;
        interval_ms_ = std::min(interval_ms_, durationMs-time_ms_);
        time_ms_ += interval_ms_;
        ++index_;
        const int age_ms = scenario_.name == "stale_cadence" && time_ms_ > 800 && time_ms_ < 2000
            ? (index_ % 4 == 0 ? 90 : index_ % 4 == 1 ? 35 : 8) : scenario_.age_ms;
        const double observation_ms = time_ms_ - age_ms;
        const double t = observation_ms / 1000.0;
        const int intended = scenario_.name == "allowed_switch" && time_ms_ >= 1500 ? 2 : 1;
        // Fixed recorded capture geometry keeps baseline and candidate inputs
        // identical. Production adaptive-crop proposals are recorded separately.
        const int crop_size = std::min({800, config_.screen_w, config_.screen_h});
        const int crop_shift = scenario_.name == "crop_exit" && time_ms_ > 1800 ? 60 : 0;
        CaptureRegion crop{(config_.screen_w-crop_size)/2 + crop_shift, (config_.screen_h-crop_size)/2, crop_size, crop_size};
        ReplayFrame input;
        input.now = tick(time_ms_);
        input.frame.capture = crop;
        input.frame.width = crop.width; input.frame.height = crop.height;
        input.frame.frame_ready = tick(observation_ms);
        input.frame.frame_time = input.frame.frame_ready;
        input.frame.acquire_started = input.frame.frame_ready;
        input.frame.capture_done = input.frame.frame_ready;
        input.frame.capture_time = SystemClock::time_point(std::chrono::duration_cast<SystemClock::duration>(input.frame.frame_ready.time_since_epoch()));
        tagged_ = Json::array();
        auto emit = [&](std::array<int,4> box, int cls, int id, float confidence, int noise) {
            if (box[2] <= crop.left || box[0] >= crop.left+crop.width || box[3] <= crop.top || box[1] >= crop.top+crop.height) return;
            const int jx = noise ? rng_.jitter(noise) : 0, jy = noise ? rng_.jitter(noise) : 0;
            box[0] = std::clamp(box[0]+jx,crop.left,crop.left+crop.width-1);
            box[2] = std::clamp(box[2]+jx,box[0]+1,crop.left+crop.width);
            box[1] = std::clamp(box[1]+jy,crop.top,crop.top+crop.height-1);
            box[3] = std::clamp(box[3]+jy,box[1]+1,crop.top+crop.height);
            Detection detection{.bbox={box[0]-crop.left,box[1]-crop.top,box[2]-crop.left,box[3]-crop.top},.cls=cls,.conf=confidence};
            input.detections.push_back(detection);
            tagged_.push_back({{"bbox",box},{"cls",cls},{"confidence",confidence},{"truth_id",id}});
        };
        bool intended_observed = false;
        const bool burst = (scenario_.name == "burst_loss" || scenario_.name == "combined")
            && observation_ms >= 1100 && observation_ms < 1100+scenario_.gap_ms;
        for (const auto& actor : world(config_,scenario_,t)) {
            if (actor.id == 1 && burst) continue;
            const int noise = scenario_.name == "noisy_boxes" || scenario_.name == "combined" ? 18 : 0;
            float confidence = noise ? static_cast<float>(0.08 + rng_.uniform()*0.88) : 0.95F;
            const auto before = tagged_.size();
            emit(actor.body,0,actor.id,confidence,noise);
            if (!(scenario_.name == "head_body" && t >= 0.9 && t < 1.8)) emit(actor.head,1,actor.id,confidence,noise);
            if (actor.id == intended && tagged_.size() > before) intended_observed = true;
        }
        if ((scenario_.name == "false_positives" || scenario_.name == "combined") && time_ms_ > 900 && time_ms_ < 1900) {
            const int x = config_.screen_w/2 + rng_.jitter(25), y = config_.screen_h/2 + rng_.jitter(25);
            emit({x-12,y-12,x+12,y+12},1,0,0.99F,0);
        }
        row_ = {{"time_ms",time_ms_},{"dt_ms",interval_ms_},{"observation_ms",observation_ms},
            {"observation_age_ms",age_ms},{"capture",rect(crop)},{"proposed_capture",rect(proposed)},
            {"intended_id",intended},{"intended_observed",intended_observed},{"inputs",tagged_},
            {"truth",Json::array()},{"allowed_switches",Json::array()},
            {"external_camera_offset",point(static_cast<float>(camera(scenario_,time_ms_/1000.0).first),static_cast<float>(camera(scenario_,time_ms_/1000.0).second))}};
        if (scenario_.name == "allowed_switch" && time_ms_ >= 1500)
            row_["allowed_switches"].push_back(Json::array({1,2}));
        bool present = false;
        for (const auto& actor : world(config_,scenario_,time_ms_/1000.0)) {
            row_["truth"].push_back({{"id",actor.id},{"body",actor.body},{"head",actor.head}});
            present |= actor.id == intended;
        }
        row_["intended_present"] = present;
        input_hash_.add(Json{{"time",time_ms_},{"age",age_ms},{"crop",rect(crop)},{"detections",tagged_},
            {"truth",row_["truth"]},{"intended",intended},{"allowed_switches",row_["allowed_switches"]}}.dump());
        return input;
    }

    void observe(const ReplayResult& result) override {
        durations.push_back(result.processing_us);
        stages["selection_us"].push_back(result.selection_us);
        stages["tracking_us"].push_back(result.tracking_us);
        stages["prediction_us"].push_back(result.prediction_us);
        stages["control_us"].push_back(result.control_us);
        if (!collect_) return;
        row_["stale"] = result.stale_frame;
        row_["output_ready"] = result.output_ready;
        row_["association_id"] = result.association_id;
        row_["lost_frames"] = result.lost_frames;
        row_["next_crop_size"] = result.next_crop_size;
        row_["predicted"] = optionalPoint(result.preview.predicted_point);
        row_["filtered"] = result.track ? point(result.track->x,result.track->y) : Json(nullptr);
        row_["command"] = result.command ? Json::array({result.command->dx,result.command->dy}) : Json(nullptr);
        row_["detections"] = Json::array();
        int selected_id = -1, selected_class = -1;
        for (const auto& detection : result.preview.detections) {
            row_["detections"].push_back({{"bbox",detection.bbox},{"cls",detection.cls},{"selected",detection.selected}});
            if (!detection.selected) continue;
            selected_class = detection.cls;
            int matches = 0;
            for (const auto& input : tagged_) {
                Detection source;
                source.bbox = input.at("bbox").get<std::array<int,4>>();
                source = scaleDetectionBox(source,runtime_.detection_box_scale,result.preview.capture_region);
                if (input.at("cls") == detection.cls && source.bbox == detection.bbox) {
                    selected_id = input.at("truth_id").get<int>(); ++matches;
                }
            }
            if (matches != 1) selected_id = -2; // Ambiguous evidence, never guess identity by nearest truth.
        }
        if (selected_id != -1) { last_identity_ = selected_id; last_class_ = selected_class; }
        else if (!result.track) { last_identity_ = -1; last_class_ = -1; }
        row_["selected_id"] = selected_id;
        row_["tracked_id"] = last_identity_;
        row_["aim_class"] = last_class_;
        // Fixed evaluator aim policy: head center, body at 15% height. It is not
        // derived from tracker predictions or candidate configuration.
        row_["truth_aim"] = nullptr;
        row_["truth_size"] = nullptr;
        for (const auto& truth : row_["truth"]) if (truth.at("id") == row_.at("intended_id")) {
            const auto box = truth.at(last_class_ == 0 ? "body" : "head").get<std::array<int,4>>();
            row_["truth_aim"] = point((box[0]+box[2])*0.5F,box[1]+(box[3]-box[1])*(last_class_==0 ? 0.15F : 0.5F));
            row_["truth_size"] = point(static_cast<float>(box[2]-box[0]),static_cast<float>(box[3]-box[1]));
        }
        row_["error_normalized"] = nullptr;
        row_["error_px"] = nullptr;
        row_["filtered_error_normalized"] = nullptr;
        for (const auto& [field,metric] : std::vector<std::pair<std::string,std::string>>{
            {"predicted","error_normalized"},{"filtered","filtered_error_normalized"}}) {
            if (!row_[field].is_null() && !row_["truth_aim"].is_null()) {
                const double dx=(row_[field][0].get<double>()-row_["truth_aim"][0].get<double>())/row_["truth_size"][0].get<double>();
                const double dy=(row_[field][1].get<double>()-row_["truth_aim"][1].get<double>())/row_["truth_size"][1].get<double>();
                row_[metric] = std::hypot(dx,dy);
                if(field=="predicted") row_["error_px"]=std::hypot(
                    row_[field][0].get<double>()-row_["truth_aim"][0].get<double>(),
                    row_[field][1].get<double>()-row_["truth_aim"][1].get<double>());
            }
        }
        frames.push_back(row_);
    }
    std::string inputHash() const { return input_hash_.str(); }
    Json frames = Json::array();
    std::vector<double> durations;
    std::map<std::string,std::vector<double>> stages;
private:
    StaticConfig config_;
    RuntimeConfig runtime_;
    Scenario scenario_;
    Random rng_;
    bool collect_;
    int index_=0, time_ms_=0, interval_ms_=0, last_identity_=-1, last_class_=-1;
    Json row_,tagged_;
    Hash input_hash_;
};

Json percentile(std::vector<double> values, double fraction) {
    if (values.empty()) return nullptr;
    std::sort(values.begin(),values.end());
    return values[static_cast<std::size_t>(std::ceil(fraction*values.size()))-1];
}
Json distribution(const std::vector<double>& values) {
    return {{"samples",values.size()},{"p50",percentile(values,0.5)},{"p95",percentile(values,0.95)},{"p99",percentile(values,0.99)},
        {"max",values.empty()?Json(nullptr):Json(*std::max_element(values.begin(),values.end()))}};
}
}

Json scoreFrames(const Json& frames, double loss_deadline_ms) {
    double wrong_ms=0, missing_ms=0, total_present_ms=0, scored_ms=0, weighted_error=0, filtered_scored_ms=0, filtered_error=0;
    double output_after_timeout_ms=0, gap_drift=0;
    int switches=0, allowed_switches=0, aim_switches=0, previous_identity=-1, previous_class=-1;
    int stale_count=0, ambiguous=0, unrecovered=0;
    std::optional<bool> initial_correct;
    bool previous_observed=true;
    std::optional<double> recovery_start, loss_start;
    std::vector<double> errors, pixel_errors, gap_errors, recovery, ages;
    Json events=Json::array();
    std::map<std::string,Json> open_events;
    for (const auto& f : frames) {
        const double time=f.at("time_ms"), dt=f.at("dt_ms");
        const int intended=f.at("intended_id"), selected=f.at("selected_id"), tracked=f.at("tracked_id"), cls=f.at("aim_class");
        const bool present=f.at("intended_present"), observed=f.at("intended_observed"), stale=f.at("stale");
        const bool wrong=tracked >= 0 && tracked != intended;
        const bool missing=present && tracked == -1;
        if (present) total_present_ms+=dt;
        if (wrong) wrong_ms+=dt;
        if (missing) missing_ms+=dt;
        if (tracked == -2) ++ambiguous;
        if (selected >= 0) {
            if (!initial_correct) initial_correct=selected == intended;
            if (previous_identity >= 0 && selected != previous_identity) {
                bool permitted=false;
                for (const auto& pair : f.value("allowed_switches",Json::array()))
                    permitted |= pair[0] == previous_identity && pair[1] == selected;
                if (permitted) ++allowed_switches; else ++switches;
            }
            if (previous_identity == selected && previous_class >= 0 && cls != previous_class) ++aim_switches;
            previous_identity=selected; previous_class=cls;
        }
        if (!f.at("error_normalized").is_null()) {
            const double e=f.at("error_normalized");
            if (!std::isfinite(e)) throw std::runtime_error("Non-finite scored error");
            errors.push_back(e); weighted_error+=e*dt; scored_ms+=dt;
            if (!observed) { gap_drift=std::max(gap_drift,e); gap_errors.push_back(e); }
        }
        if (!f.at("filtered_error_normalized").is_null()) {
            filtered_error+=f.at("filtered_error_normalized").get<double>()*dt; filtered_scored_ms+=dt;
        }
        if(f.contains("error_px") && f["error_px"].is_number())pixel_errors.push_back(f["error_px"].get<double>());
        if (!observed && !loss_start) loss_start=time-dt;
        if (observed && !previous_observed && present) {
            if (recovery_start) ++unrecovered;
            recovery_start=time;
        }
        if (recovery_start && selected == intended) { recovery.push_back(time-*recovery_start); recovery_start.reset(); }
        const bool moves=!f.at("command").is_null() && (f["command"][0] != 0 || f["command"][1] != 0);
        const bool past_timeout=loss_start && time-*loss_start > loss_deadline_ms && !observed && moves;
        if (past_timeout) output_after_timeout_ms+=dt;
        if (observed) loss_start.reset();
        previous_observed=observed;
        if (stale) ++stale_count;
        ages.push_back(f.at("observation_age_ms"));
        for (const auto& [kind,active] : std::vector<std::pair<std::string,bool>>{
            {"wrong_target",wrong},{"no_track",missing},{"stale_observation",stale},{"detection_gap",!observed},
            {"movement_after_loss_deadline",past_timeout},{"ambiguous_identity",tracked==-2}}) {
            if (active) {
                if (!open_events.contains(kind)) open_events[kind]={{"kind",kind},{"start_ms",time-dt},{"end_ms",time}};
                open_events[kind]["end_ms"]=time;
            } else if (open_events.contains(kind)) { events.push_back(open_events[kind]); open_events.erase(kind); }
        }
    }
    if (recovery_start) ++unrecovered;
    for (const auto& [kind,event] : open_events) events.push_back(event);
    std::sort(events.begin(),events.end(),[](const Json& a,const Json& b){return a.at("start_ms") < b.at("start_ms");});
    return {{"wrong_target_ms",wrong_ms},{"identity_switches",switches},{"allowed_identity_switches",allowed_switches},{"aim_point_changes",aim_switches},
        {"initial_selection_correct",initial_correct?Json(*initial_correct):Json(nullptr)},
        {"no_track_ms",missing_ms},{"ambiguous_identity_frames",ambiguous},
        {"prediction_coverage",total_present_ms>0?Json(scored_ms/total_present_ms):Json(nullptr)},
        {"mean_normalized_error",scored_ms>0?Json(weighted_error/scored_ms):Json(nullptr)},
        {"mean_filtered_normalized_error",filtered_scored_ms>0?Json(filtered_error/filtered_scored_ms):Json(nullptr)},
        {"error_distribution",distribution(errors)},{"max_gap_error",gap_errors.empty()?Json(nullptr):Json(gap_drift)},
        {"pixel_error_distribution",distribution(pixel_errors)},
        {"recovery_ms",recovery},{"unrecovered_episodes",unrecovered},
        {"movement_after_loss_deadline_ms",output_after_timeout_ms},{"stale_frames",stale_count},
        {"observation_age_ms",distribution(ages)},{"events",events}};
}

Json runSuite(const StaticConfig& config, const RuntimeConfig& runtime, const Options& options) {
    validateOptions(options);
    if (config.screen_w < 800 || config.screen_h < 600) throw std::invalid_argument("Benchmark screen must be at least 800x600");
    if (!options.scenario.empty() && options.scenario != "high_speed" && std::find(names.begin(),names.end(),options.scenario)==names.end())
        throw std::invalid_argument("Unknown scenario: "+options.scenario);
    Json run={{"schema_version",1},{"scenario_version",scenarioVersion},{"scope","cpp_inference_loop_synthetic_detections"},
        {"suite",options.suite},{"seed",options.seed},{"screen",Json::array({config.screen_w,config.screen_h})},
        {"runtime",runtimeJson(runtime)},{"loss_deadline_ms",options.loss_deadline_ms},{"runs",Json::array()},
        {"static_config",{{"screen_w",config.screen_w},{"screen_h",config.screen_h},{"imgsz",config.imgsz},
            {"capture_crop_size",config.capture_crop_size},{"force_input_size_when_locked",config.force_input_size_when_locked},
            {"tracker_enable",config.tracker_enable}}},
        {"build",{{"revision",DELTA_BENCH_REVISION},{"source_hash",DELTA_BENCH_SOURCE_HASH},{"compiler",DELTA_BENCH_COMPILER},{"configuration",DELTA_BENCH_CONFIGURATION}}},
        {"timing",options.timing},{"correctness_violations",Json::array()}};
    if (const char* cpu=std::getenv("PROCESSOR_IDENTIFIER")) run["build"]["cpu"]=cpu;
    auto execute = [&](Scenario scenario,RuntimeConfig settings) {
        ScenarioReplay replay(config,settings,scenario,true);
        DeltaApp::replay(config,settings,replay);
        Json item={{"key",scenario.name+"/"+std::to_string(scenario.seed)+"/"+std::to_string(scenario.gap_ms)+"/"+scenario.variant},
            {"scenario",scenario.name},{"seed",scenario.seed},{"gap_ms",scenario.gap_ms},{"variant",scenario.variant},
            {"speed_px_s",scenario.speed_px_s},{"cadence_ms",scenario.cadence_ms},{"configured_observation_age_ms",scenario.age_ms},
            {"runtime",runtimeJson(settings)},{"input_hash",replay.inputHash()},{"frames",std::move(replay.frames)}};
        item["metrics"]=scoreFrames(item["frames"],options.loss_deadline_ms);
        if (item["frames"].empty()) throw std::runtime_error("Scenario produced no frames");
        for (const auto& frame : item["frames"]) {
            if (!frame["command"].is_null()) {
                if (std::abs(frame["command"][0].get<int>())>settings.raw_max_step_x
                    || std::abs(frame["command"][1].get<int>())>settings.raw_max_step_y
                    || !frame["output_ready"].get<bool>() || frame["stale"].get<bool>())
                    run["correctness_violations"].push_back({{"key",item["key"]},{"time_ms",frame["time_ms"]},{"reason","command violates output gate or configured step limit"}});
            }
        }
        if (options.timing) {
            // Accuracy recording is not timed. Warm the code once, then collect
            // isolated passes without JSON/report construction in the observer.
            ScenarioReplay warm(config,settings,scenario,false); DeltaApp::replay(config,settings,warm);
            std::vector<double> durations;
            std::map<std::string,std::vector<double>> stages;
            for (int r=0;r<options.repeats;++r) {
                ScenarioReplay sample(config,settings,scenario,false); DeltaApp::replay(config,settings,sample);
                durations.insert(durations.end(),sample.durations.begin(),sample.durations.end());
                for (const auto& [name,values] : sample.stages) stages[name].insert(stages[name].end(),values.begin(),values.end());
            }
            item["performance"]={{"loop_us",distribution(durations)},{"repeats",options.repeats}};
            for (const auto& [name,values] : stages) item["performance"][name]=distribution(values);
        } else item["performance"]=nullptr;
        run["runs"].push_back(std::move(item));
    };
    if (options.suite == "quick") {
        for (const auto& name : names) if (options.scenario.empty() || options.scenario==name)
            execute(Scenario{name,options.seed},runtime);
    } else if (options.suite == "speed") {
        if (!options.scenario.empty() && options.scenario != "high_speed")
            throw std::invalid_argument("The speed suite contains high_speed scenarios");
        for (int speed : {500,1500,3000,6000}) for (int cadence : {4,8,16}) for (int age : {5,15,30}) {
            Scenario scenario{"high_speed",options.seed};
            scenario.speed_px_s=speed;scenario.cadence_ms=cadence;scenario.age_ms=age;
            scenario.variant="speed="+std::to_string(speed)+",cadence="+std::to_string(cadence)+",age="+std::to_string(age);
            execute(scenario,runtime);
        }
    } else {
        for (std::uint64_t seed=options.seed;seed<options.seed+3;++seed)
            for (int gap : {50,150,400}) for (int timeout : {50,150,300}) for (int cap : {50,150})
                for (const auto& name : {"burst_loss","combined"}) {
                    if (!options.scenario.empty() && options.scenario!=name) continue;
                    auto candidate=runtime;
                    candidate.target_prediction_timeout_ms=static_cast<float>(timeout);
                    candidate.raw_max_step_x=cap; candidate.raw_max_step_y=cap;
                    execute(Scenario{name,seed,gap,"timeout="+std::to_string(timeout)+",cap="+std::to_string(cap)},candidate);
                }
    }
    if (run["runs"].empty()) throw std::invalid_argument("Scenario is not part of the selected suite");
    return run;
}

Json compareRuns(const Json& baseline,const Json& candidate) {
    for (const auto* field : {"schema_version","scenario_version","screen","scope","loss_deadline_ms"})
        if (baseline.at(field)!=candidate.at(field)) throw std::invalid_argument(std::string("Incompatible comparison: ")+field);
    if (baseline.at("runs").size()!=candidate.at("runs").size()) throw std::invalid_argument("Comparison requires identical scenario sets");
    std::map<std::string,const Json*> originals;
    for (const auto& item : baseline.at("runs")) {
        if (!originals.emplace(item.at("key").get<std::string>(),&item).second) throw std::invalid_argument("Duplicate baseline scenario key");
    }
    Json out=Json::array();
    for (const auto& item : candidate.at("runs")) {
        const std::string key=item.at("key");
        if (!originals.contains(key) || originals.at(key)->at("input_hash")!=item.at("input_hash"))
            throw std::invalid_argument("Comparison inputs differ for "+key);
        const auto& previous=*originals.at(key);
        Json delta=Json::object();
        for (const auto* metric : {"wrong_target_ms","identity_switches","mean_normalized_error","no_track_ms","prediction_coverage","movement_after_loss_deadline_ms","unrecovered_episodes"}) {
            const auto& a=previous.at("metrics").at(metric); const auto& b=item.at("metrics").at(metric);
            delta[metric]=a.is_number() && b.is_number()?Json(b.get<double>()-a.get<double>()):Json(nullptr);
        }
        out.push_back({{"key",key},{"candidate_minus_baseline",delta}});
        originals.erase(key);
    }
    return out;
}
} // namespace delta::bench
