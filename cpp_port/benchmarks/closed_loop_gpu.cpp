#include "benchmark.hpp"
#include "benchmark_build.hpp"
#include "scene_geometry.hpp"
#include "closed_loop_scoring.hpp"
#include "visible_scene.hpp"
#include "observation_replay.hpp"
#include "delta/app.hpp"
#include "delta/dispatch.hpp"
#include "delta/timing.hpp"

#include <cuda_runtime.h>
#include <windows.h>
#include <wincodec.h>
#include <wrl/client.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <atomic>
#include <mutex>
#include <thread>

namespace {
using delta::bench::Json;
using Microsoft::WRL::ComPtr;
constexpr double pi=3.14159265358979323846;
void check(cudaError_t error,const char* action) {
    if(error!=cudaSuccess)throw std::runtime_error(std::string(action)+": "+cudaGetErrorString(error));
}
void check(HRESULT error,const char* action) {
    if(FAILED(error))throw std::runtime_error(std::string(action)+" failed");
}
struct Image {UINT width=0,height=0;std::vector<std::uint8_t> pixels;};
Image loadImage(const std::filesystem::path& path) {
    ComPtr<IWICImagingFactory> factory;ComPtr<IWICBitmapDecoder> decoder;
    ComPtr<IWICBitmapFrameDecode> frame;ComPtr<IWICFormatConverter> converter;
    check(CoCreateInstance(CLSID_WICImagingFactory,nullptr,CLSCTX_INPROC_SERVER,IID_PPV_ARGS(&factory)),"WIC factory");
    check(factory->CreateDecoderFromFilename(path.c_str(),nullptr,GENERIC_READ,WICDecodeMetadataCacheOnLoad,&decoder),"Image decode");
    check(decoder->GetFrame(0,&frame),"Image frame");
    check(factory->CreateFormatConverter(&converter),"Image converter");
    check(converter->Initialize(frame.Get(),GUID_WICPixelFormat32bppBGRA,WICBitmapDitherTypeNone,nullptr,0,WICBitmapPaletteTypeCustom),"BGRA conversion");
    Image image;check(frame->GetSize(&image.width,&image.height),"Image size");
    if(image.width<1 || image.height<1 || image.width>8192 || image.height>8192)throw std::invalid_argument("Invalid sprite image size");
    image.pixels.resize(static_cast<std::size_t>(image.width)*image.height*4);
    check(converter->CopyPixels(nullptr,image.width*4,static_cast<UINT>(image.pixels.size()),image.pixels.data()),"Image pixels");
    return image;
}
void savePng(const std::filesystem::path& path,const std::vector<std::uint8_t>& pixels,UINT width,UINT height) {
    ComPtr<IWICImagingFactory> factory;ComPtr<IWICStream> stream;
    ComPtr<IWICBitmapEncoder> encoder;ComPtr<IWICBitmapFrameEncode> frame;
    check(CoCreateInstance(CLSID_WICImagingFactory,nullptr,CLSCTX_INPROC_SERVER,IID_PPV_ARGS(&factory)),"PNG factory");
    check(factory->CreateStream(&stream),"PNG stream");
    check(stream->InitializeFromFilename(path.c_str(),GENERIC_WRITE),"PNG output");
    check(factory->CreateEncoder(GUID_ContainerFormatPng,nullptr,&encoder),"PNG encoder");
    check(encoder->Initialize(stream.Get(),WICBitmapEncoderNoCache),"PNG initialize");
    check(encoder->CreateNewFrame(&frame,nullptr),"PNG frame");
    check(frame->Initialize(nullptr),"PNG frame initialize");
    check(frame->SetSize(width,height),"PNG size");
    WICPixelFormatGUID format=GUID_WICPixelFormat32bppBGRA;
    check(frame->SetPixelFormat(&format),"PNG pixel format");
    if(!IsEqualGUID(format,GUID_WICPixelFormat32bppBGRA))throw std::runtime_error("PNG encoder changed pixel format");
    check(frame->WritePixels(height,width*4,static_cast<UINT>(pixels.size()),
        const_cast<BYTE*>(pixels.data())),"PNG pixels");
    check(frame->Commit(),"PNG frame commit");check(encoder->Commit(),"PNG commit");
}
delta::SteadyClock::time_point tick(double ms) {
    return delta::SteadyClock::time_point(std::chrono::duration_cast<delta::SteadyClock::duration>(
        std::chrono::duration<double,std::milli>(10000.0+ms)));
}
delta::SystemClock::time_point systemTick(double ms) {
    return delta::SystemClock::time_point(std::chrono::duration_cast<delta::SystemClock::duration>(
        std::chrono::duration<double,std::milli>(10000.0+ms)));
}
double ms(delta::SteadyClock::time_point time) {
    return std::chrono::duration<double,std::milli>(time-tick(0)).count();
}
std::string fileFingerprint(const std::filesystem::path& path) {
    std::ifstream file(path,std::ios::binary);
    if(!file)throw std::runtime_error("Cannot fingerprint "+path.string());
    std::uint64_t hash=14695981039346656037ULL;
    std::array<char,65536> buffer{};
    while(file.read(buffer.data(),buffer.size()) || file.gcount()>0)
        for(std::streamsize i=0;i<file.gcount();++i) {
            hash^=static_cast<unsigned char>(buffer[static_cast<std::size_t>(i)]);
            hash*=1099511628211ULL;
        }
    std::ostringstream output;output<<std::hex<<std::setw(16)<<std::setfill('0')<<hash;
    return output.str();
}
void checkSourceLabels(const std::filesystem::path& path,UINT width,UINT height,
    const std::array<int,4>& body,const std::array<int,4>& head) {
    std::ifstream file(path);
    if(!file)throw std::runtime_error("Cannot open source labels: "+path.string());
    std::array<bool,2> found{};
    int cls;double x,y,w,h;
    while(file>>cls>>x>>y>>w>>h) {
        if(cls<0 || cls>1)continue;
        const auto& expected=cls==0?body:head;
        const std::array<double,4> actual={
            (x-w*.5)*width,(y-h*.5)*height,(x+w*.5)*width,(y+h*.5)*height};
        bool matches=true;
        for(int i=0;i<4;++i)matches &= std::abs(actual[i]-expected[i])<=2.0;
        found[cls] |= matches;
    }
    if(!found[0] || !found[1])
        throw std::runtime_error("Manifest head/body boxes do not match paired source labels");
}
struct SceneParameters {
    bool visible=false;
    int frames=200;
    double speed=1500,scale=1,cadence_ms=8,inference_delay_ms=5;
    double transport_delay_ms=1,hfov_deg=70,vfov_deg=70,mouse_radians_per_count=.0007;
    std::string scenario="clean";
};
Json distribution(std::vector<double> values) {
    if(values.empty())return {{"samples",0},{"p50",nullptr},{"p95",nullptr},{"p99",nullptr},{"max",nullptr}};
    std::sort(values.begin(),values.end());
    const auto percentile=[&](double q){return values[static_cast<std::size_t>(std::ceil(q*values.size()))-1];};
    return {{"samples",values.size()},{"p50",percentile(.5)},{"p95",percentile(.95)},
        {"p99",percentile(.99)},{"max",values.back()}};
}

class SceneReplay final:public delta::ITrackingReplay {
public:
    SceneReplay(delta::StaticConfig config,delta::RuntimeConfig runtime,Image image,Image background,
        std::array<int,4> box,std::array<int,4> body_box,std::array<int,4> head_box,
        SceneParameters params,std::filesystem::path output)
        :config_(std::move(config)),runtime_(runtime),image_(std::move(image)),background_(std::move(background)),
         box_(box),body_box_(body_box),head_box_(head_box),params_(std::move(params)),output_(std::move(output)) {
        if(box_[0]<0 || box_[1]<0 || box_[2]>static_cast<int>(image_.width)
            || box_[3]>static_cast<int>(image_.height) || box_[2]-box_[0]<8 || box_[3]-box_[1]<8)
            throw std::invalid_argument("Sprite box lies outside image");
        check(cudaSetDevice(config_.onnx_cuda_device_id),"Select inference GPU");
        cudaDeviceProp prop{};check(cudaGetDeviceProperties(&prop,config_.onnx_cuda_device_id),"GPU properties");
        gpu={{"ordinal",config_.onnx_cuda_device_id},{"name",prop.name},{"provider_events",Json::object()}};
        if(config_.onnx_cuda_device_id==1 && std::string(prop.name).find("3070")==std::string::npos)
            throw std::runtime_error("CUDA ordinal 1 is not the intended RTX 3070");
        check(cudaMalloc(&device_,static_cast<std::size_t>(config_.screen_w)*config_.screen_h*4),"Scene GPU buffer");
        full_.resize(static_cast<std::size_t>(config_.screen_w)*config_.screen_h*4);
        background_scaled_.resize(full_.size());
        for(int y=0;y<config_.screen_h;++y)for(int x=0;x<config_.screen_w;++x) {
            const auto source=(static_cast<std::size_t>(y*background_.height/config_.screen_h)*background_.width+x*background_.width/config_.screen_w)*4;
            std::copy_n(background_.pixels.data()+source,4,background_scaled_.data()+(static_cast<std::size_t>(y)*config_.screen_w+x)*4);
        }
        engine_=delta::makeInferenceEngine(config_);
        if(!engine_->supportsGpuInput())throw std::runtime_error("Actual GPU inference required");
        engine_->setModelConfidence(runtime_.model_conf);engine_->warmup();
        render(config_.screen_w/2-(box_[2]-box_[0])/2,
            config_.screen_h/2-(box_[3]-box_[1])/2,box_[2]-box_[0],box_[3]-box_[1],0);
        check(cudaMemcpy(device_,full_.data(),full_.size(),cudaMemcpyHostToDevice),"Audit scene upload");
        delta::GpuFramePacket audit;
        audit.device_ptr=device_;audit.pitch_bytes=config_.screen_w*4;
        audit.width=config_.screen_w;audit.height=config_.screen_h;
        audit.capture={0,0,config_.screen_w,config_.screen_h};
        for(int i=0;i<4;++i)engine_->predictGpu(audit,-1);
        check(cudaDeviceSynchronize(),"Audit inference synchronization");
        const auto profile=engine_->finishProfiling();
        if(!profile.empty()) {
            std::ifstream stream(profile);Json events;stream>>events;
            for(const auto& event:events)if(event.contains("args") && event["args"].contains("provider"))
                gpu["provider_events"][event["args"]["provider"].get<std::string>()]=
                    gpu["provider_events"].value(event["args"]["provider"].get<std::string>(),0)+1;
        }
        if(gpu.value("provider_events",Json::object()).value("CUDAExecutionProvider",0)
            +gpu.value("provider_events",Json::object()).value("TensorrtExecutionProvider",0)==0)
            throw std::runtime_error("GPU execution audit did not find a CUDA/TensorRT node");
        sender_.method=runtime_.mouse_output_method;sender_.gain_x=runtime_.sendinput_gain_x;
        sender_.gain_y=runtime_.sendinput_gain_y;sender_.max_step=runtime_.sendinput_max_step;
        transform_.configure(sender_);
        const double hz=runtime_.display_rate_servo_hz>0?runtime_.display_rate_servo_hz:165.0;
        control_period_ms_=1000.0/std::clamp(hz,1.0,1000.0);
        next_control_ms_=0;
    }
    ~SceneReplay()override {stopVisible();if(device_)cudaFree(device_);}
    void finish() {
        stopVisible();
        for(const auto& saved:saved_frames_)savePng(output_/saved.path,saved.pixels,saved.width,saved.height);
    }
    std::optional<delta::ReplayFrame> next(const delta::CaptureRegion& proposed)override {
        if(params_.visible)return nextVisible(proposed);
        if(index_>=params_.frames)return std::nullopt;
        const double render_ms=params_.cadence_ms*(index_+1);
        advance(render_ms);
        const double render_camera_yaw=camera_yaw_;
        const double render_camera_pitch=camera_pitch_;
        // A perspective projection of one target at fixed depth. The independent
        // target path is a triangle wave, with reversals every 500 projected px.
        const double fx=(config_.screen_w/2.0)/std::tan(params_.hfov_deg*pi/360.0);
        const double target_angle=worldAngle(render_ms,fx);
        const double shake_yaw=externalYaw(render_ms),shake_pitch=externalPitch(render_ms);
        const int center_x=static_cast<int>(std::lround(config_.screen_w/2.0+fx*std::tan(target_angle-camera_yaw_-shake_yaw)));
        const double fy=(config_.screen_h/2.0)/std::tan(params_.vfov_deg*pi/360.0);
        const int center_y=static_cast<int>(std::lround(config_.screen_h/2.0-fy*std::tan(camera_pitch_+shake_pitch)));
        const int width=std::max(8,static_cast<int>(std::lround((box_[2]-box_[0])*params_.scale)));
        const int height=std::max(8,static_cast<int>(std::lround((box_[3]-box_[1])*params_.scale)));
        const int left=center_x-width/2,top=center_y-height/2;
        const int secondary_left=static_cast<int>(std::lround(config_.screen_w/2.0
            +fx*std::tan(-target_angle-camera_yaw_-shake_yaw)))-width/2;
        const auto [aim_x,aim_y]=aimPoint(center_x,center_y,0);
        const auto [head_x,head_y]=aimPoint(center_x,center_y,1);
        const auto identity_box=delta::bench::projectIdentityBox(body_box_,head_box_,box_,center_x,center_y,params_.scale);
        const auto secondary_box=delta::bench::projectIdentityBox(body_box_,head_box_,box_,secondary_left+width/2,center_y,params_.scale);
        const double frame_error=delta::bench::normalizedSceneError(identity_box,aim_x,aim_y,
            config_.screen_w/2.0,config_.screen_h/2.0);
        render(left,top,width,height,render_ms,secondary_left);
        delta::CaptureRegion crop=proposed;
        crop.left=std::clamp(crop.left,0,config_.screen_w-1);
        crop.top=std::clamp(crop.top,0,config_.screen_h-1);
        crop.width=std::clamp(crop.width,1,config_.screen_w-crop.left);
        crop.height=std::clamp(crop.height,1,config_.screen_h-crop.top);
        cropped_.resize(static_cast<std::size_t>(crop.width)*crop.height*4);
        for(int y=0;y<crop.height;++y)
            std::copy_n(full_.data()+(static_cast<std::size_t>(crop.top+y)*config_.screen_w+crop.left)*4,
                static_cast<std::size_t>(crop.width)*4,cropped_.data()+static_cast<std::size_t>(y)*crop.width*4);
        const std::string png="frames/"+std::to_string(index_+1)+".png";
        savePng(output_/png,cropped_,crop.width,crop.height);
        check(cudaSetDevice(config_.onnx_cuda_device_id),"Select inference GPU");
        check(cudaMemcpy(device_,cropped_.data(),cropped_.size(),cudaMemcpyHostToDevice),"Upload rendered scene");
        delta::GpuFramePacket gpu_frame;
        gpu_frame.device_ptr=device_;gpu_frame.pitch_bytes=crop.width*4;gpu_frame.width=crop.width;gpu_frame.height=crop.height;
        gpu_frame.capture=crop;gpu_frame.source_id=++scene_id_;gpu_frame.source_present=tick(render_ms);
        gpu_frame.acquire_started=tick(render_ms+1);gpu_frame.frame_ready=tick(render_ms+1);
        gpu_frame.capture_done=tick(render_ms+1);gpu_frame.frame_time=tick(render_ms);
        gpu_frame.capture_time=systemTick(render_ms);
        const auto wall_start=delta::SteadyClock::now();
        auto inference=engine_->predictGpu(gpu_frame,-1);
        check(cudaDeviceSynchronize(),"Inference synchronization");
        const auto wall_end=delta::SteadyClock::now();
        const std::size_t natural_detections=inference.detections.size();
        const bool injected_drop=params_.scenario=="dropout"
            && render_ms>=400 && render_ms<560;
        if(injected_drop)inference.detections.clear();
        // Logical availability is independent of inference wall time.
        advance(render_ms+params_.inference_delay_ms);
        delta::ReplayFrame out;out.now=tick(render_ms+params_.inference_delay_ms);
        out.frame.capture=crop;out.frame.width=crop.width;out.frame.height=crop.height;
        out.frame.source_id=gpu_frame.source_id;out.frame.source_present=gpu_frame.source_present;
        out.frame.acquire_started=gpu_frame.acquire_started;out.frame.frame_ready=gpu_frame.frame_ready;
        out.frame.capture_done=gpu_frame.capture_done;out.frame.frame_time=gpu_frame.frame_time;
        out.frame.capture_time=gpu_frame.capture_time;out.detections=std::move(inference.detections);
        out.dispatch_feedback=std::move(pending_feedback_);
        pending_feedback_.clear();
        current_={{"scene_state_id",scene_id_},{"world_update_ms",render_ms},
            {"render_submitted_ms",render_ms},{"presented_ms",render_ms},
            {"capture_ms",render_ms+1},{"render_ms",render_ms},
            {"available_ms",render_ms+params_.inference_delay_ms},
            {"camera_yaw_rad",render_camera_yaw},{"camera_pitch_rad",render_camera_pitch},
            {"external_yaw_rad",shake_yaw},{"external_pitch_rad",shake_pitch},
            {"target_center",{center_x,center_y}},
            {"truth_aim",{aim_x,aim_y}},{"truth_body_aim",{aim_x,aim_y}},
            {"truth_head_aim",{head_x,head_y}},{"normalized_aim_error",frame_error},
            {"reticle_within_target",delta::bench::sceneContains(identity_box,config_.screen_w/2.0,config_.screen_h/2.0)},
            {"truth_box",identity_box},
            {"truth_boxes",Json::array({{{"id",1},{"bbox",identity_box}}})},
            {"capture",{crop.left,crop.top,crop.width,crop.height}},{"image",png},
            {"inference_wall_ms",std::chrono::duration<double,std::milli>(wall_end-wall_start).count()},
            {"detections",out.detections.size()},{"natural_detections",natural_detections},
            {"injected_detection_drop",injected_drop},{"applied_movement",applied_since_frame_},
            {"device_output",false},{"scenario",params_.scenario},
            {"target_occluded",params_.scenario=="occlusion" && render_ms>=400 && render_ms<560}};
        if(params_.scenario=="crossing" && render_ms>=40) {
            current_["truth_boxes"].push_back({{"id",2},{"bbox",secondary_box}});
        }
        addClassRegions(current_,center_x,center_y,secondary_left+width/2);
        applied_since_frame_=Json::array();
        ++index_;
        current_["replay_input"]=delta::bench::encodeObservation(out);
        return out;
    }
    void observe(const delta::ReplayResult& result)override {
        std::unique_lock<std::mutex> lock(realtime_mutex_,std::defer_lock);
        if(params_.visible)lock.lock();
        servo_=result.servo;allowed_=result.output_ready;
        if(!result.track) {selected_truth_id_=-1;selected_cls_=-1;}
        for(const auto& detection:result.preview.detections)if(detection.selected) {
            selected_cls_=detection.cls;
            const double x=(detection.bbox[0]+detection.bbox[2])*0.5;
            const double y=(detection.bbox[1]+detection.bbox[3])*0.5;
            std::vector<delta::bench::SceneIdentityRegions> regions;
            for(const auto& truth:current_["truth_boxes"])regions.push_back({truth["id"].get<int>(),
                truth["body_bbox"].get<delta::bench::SceneBox>(),truth["head_bbox"].get<delta::bench::SceneBox>()});
            selected_truth_id_=delta::bench::attributeSceneDetection(x,y,detection.cls,regions);
        }
        const double now=params_.visible?elapsed():params_.cadence_ms*index_+params_.inference_delay_ms;
        selection_by_source_[current_["scene_state_id"].get<std::uint64_t>()]={selected_truth_id_,selected_cls_};
        if(result.command)schedule(now,result.command);
        current_["command"]=result.command?Json::array({result.command->dx,result.command->dy}):Json(nullptr);
        current_["selected"] = result.preview.predicted_point
            ?Json::array({result.preview.predicted_point->first,result.preview.predicted_point->second}):Json(nullptr);
        current_["detector_boxes"]=Json::array();
        for(const auto& detection:result.preview.detections)
            current_["detector_boxes"].push_back({{"bbox",detection.bbox},{"cls",detection.cls},
                {"selected",detection.selected}});
        current_["output_ready"]=result.output_ready;
        current_["selected_truth_id"]=selected_truth_id_;
        current_["selected_class"]=selected_cls_;
        current_["motion_decision"]=result.motion_decision;
        current_["motion_ambiguous"]=result.motion_ambiguous;
        current_["motion_residual_px"]=result.motion_residual_px;
        current_["dispatch_feedback"]={{"send_latency_s",result.dispatch_feedback.send_latency_s},
            {"sent_velocity_x",result.dispatch_feedback.sent_velocity_x},
            {"sent_velocity_y",result.dispatch_feedback.sent_velocity_y}};
        current_["motion_matches"]=Json::array();
        for(const auto& [detection,cost]:result.motion_matches)
            current_["motion_matches"].push_back({{"bbox",detection.bbox},{"class",detection.cls},{"cost",cost}});
        if(selected_cls_==1) {
            current_["truth_aim"]=current_["truth_head_aim"];
            const double x=current_["truth_head_aim"][0].get<double>();
            const double y=current_["truth_head_aim"][1].get<double>();
            const auto box=current_["truth_box"].get<delta::bench::SceneBox>();
            current_["normalized_aim_error"]=delta::bench::normalizedSceneError(box,x,y,
                config_.screen_w/2.0,config_.screen_h/2.0);
        }
        current_["stale"]=result.stale_frame;
        current_["replay_outcome"]=delta::bench::observationOutcome(result);
        current_["tracking_processing_us"]=result.processing_us;
        if(params_.visible)current_["tracking_completed_ms"]=now;
        current_["camera_yaw_after_dispatch_rad"]=camera_yaw_;
        rows.push_back(current_);
    }
    Json gpu,rows=Json::array(),movements=Json::array(),issues=Json::array(),capture_info=Json::object(),dispatch_rejections=Json::array();
private:
    void addClassRegions(Json& state,double x,double y,double secondary_x)const {
        for(auto& truth:state["truth_boxes"]) {
            const double center=truth["id"]==1?x:secondary_x;
            truth["body_bbox"]=delta::bench::projectIdentityBox(body_box_,body_box_,box_,center,y,params_.scale);
            truth["head_bbox"]=delta::bench::projectIdentityBox(head_box_,head_box_,box_,center,y,params_.scale);
        }
    }
    struct SavedFrame {std::string path;std::vector<std::uint8_t> pixels;int width,height;};
    double elapsed()const{return std::chrono::duration<double,std::milli>(delta::SteadyClock::now()-realtime_start_).count();}
    void failVisible(const std::string& reason) {
        issues.push_back(reason);stop_.store(true);
    }
    void stopVisible() {
        stop_.store(true);
        if(render_thread_.joinable())render_thread_.join();
        if(dispatch_thread_.joinable())dispatch_thread_.join();
    }
    Json renderVisible(double at,double yaw,double pitch,std::uint32_t id) {
        const double fx=(config_.screen_w/2.0)/std::tan(params_.hfov_deg*pi/360.0);
        const double fy=(config_.screen_h/2.0)/std::tan(params_.vfov_deg*pi/360.0);
        const double angle=worldAngle(at,fx),shake=externalYaw(at),shake_y=externalPitch(at);
        const int x=static_cast<int>(std::lround(config_.screen_w/2.0+fx*std::tan(angle-yaw-shake)));
        const int y=static_cast<int>(std::lround(config_.screen_h/2.0-fy*std::tan(pitch+shake_y)));
        const int width=std::max(8,static_cast<int>(std::lround((box_[2]-box_[0])*params_.scale)));
        const int height=std::max(8,static_cast<int>(std::lround((box_[3]-box_[1])*params_.scale)));
        const int other=static_cast<int>(std::lround(config_.screen_w/2.0+fx*std::tan(-angle-yaw-shake)));
        render(x-width/2,y-height/2,width,height,at,other-width/2,yaw,pitch);
        delta::bench::writeSceneMarker(full_,config_.screen_w,config_.screen_h,id);
        const auto body=aimPoint(x,y,0),head=aimPoint(x,y,1);
        const auto box=delta::bench::projectIdentityBox(body_box_,head_box_,box_,x,y,params_.scale);
        Json state={{"rendered_scene_id",id},{"world_update_ms",at},{"render_ms",at},
            {"camera_yaw_rad",yaw},{"camera_pitch_rad",pitch},
            {"external_yaw_rad",shake},{"external_pitch_rad",shake_y},
            {"target_center",{x,y}},{"truth_body_aim",{body.first,body.second}},
            {"truth_head_aim",{head.first,head.second}},{"truth_aim",{body.first,body.second}},
            {"truth_box",box},{"truth_boxes",Json::array({{{"id",1},{"bbox",box}}})},
            {"normalized_aim_error",delta::bench::normalizedSceneError(box,body.first,body.second,config_.screen_w/2.0,config_.screen_h/2.0)},
            {"reticle_within_target",delta::bench::sceneContains(box,config_.screen_w/2.0,config_.screen_h/2.0)},
            {"device_output",false},{"scenario",params_.scenario},
            {"target_occluded",params_.scenario=="occlusion" && at>=400 && at<560}};
        if(params_.scenario=="crossing" && at>=40)state["truth_boxes"].push_back({{"id",2},
            {"bbox",delta::bench::projectIdentityBox(body_box_,head_box_,box_,other,y,params_.scale)}});
        addClassRegions(state,x,y,other);
        return state;
    }
    void startVisible() {
        timer_resolution_.emplace(1);
        if(!timer_resolution_->active())throw std::runtime_error("Production Windows timer resolution unavailable");
        display_=delta::bench::captureDisplay(config_);
        capture_info={{"adapter_index",config_.capture_device_idx},{"output_index",config_.capture_output_idx},
            {"adapter",delta::bench::displayUtf8(display_.adapter)},
            {"display",delta::bench::displayUtf8(display_.output)},
            {"render_api","GDI via desktop compositor on configured capture display"},
            {"path","DXGI desktop duplication -> host BGR -> RTX inference"},
            {"requested_timer_resolution_ms",1},
            {"desktop_size",{config_.screen_w,config_.screen_h}},
            {"desktop_origin",{display_.bounds.left,display_.bounds.top}},
            {"latency_scope","DXGI presentation to simulated camera application; excludes physical input delivery"}};
        capture_=delta::makeDefaultCaptureSource(config_);
        capture_->setFreshOnly(runtime_.async_gpu_capture_fresh_only_enable);
        capture_->setCachedFrameTimeoutMs(runtime_.capture_cached_timeout_ms);
        realtime_start_=delta::SteadyClock::now();
        std::cout<<"VISIBLE-SCREEN SIMULATION: no mouse/serial output. Scene occupies the configured capture display.\n";
        render_thread_=std::thread([this] {
            try {
                delta::bench::VisibleSceneWindow window(display_);
                window_.store(window.handle());
                auto deadline=delta::SteadyClock::now();
                std::uint32_t id=0;
                while(!stop_.load()) {
                    const double at=elapsed();double yaw,pitch;
                    {std::lock_guard lock(realtime_mutex_);yaw=camera_yaw_;pitch=camera_pitch_;}
                    auto state=renderVisible(at,yaw,pitch,++id);
                    state["render_submitted_ms"]=elapsed();
                    {std::lock_guard lock(realtime_mutex_);rendered_[id]=state;}
                    window.present(full_,config_.screen_w,config_.screen_h);
                    {std::lock_guard lock(realtime_mutex_);rendered_[id]["presentation_fence_ms"]=elapsed();}
                    deadline+=std::chrono::duration_cast<delta::SteadyClock::duration>(std::chrono::duration<double,std::milli>(params_.cadence_ms));
                    if(deadline<delta::SteadyClock::now())deadline=delta::SteadyClock::now();
                    std::this_thread::sleep_until(deadline);
                }
                window_.store(nullptr);
            } catch(const std::exception& error) {std::lock_guard lock(realtime_mutex_);failVisible(error.what());}
        });
        dispatch_thread_=std::thread([this] {
            try {
                while(!stop_.load()) {
                    double next;
                    {std::lock_guard lock(realtime_mutex_);
                        const double now=elapsed();
                        while(!queued_.empty() && queued_.front().time_ms<=now) {
                            auto event=queued_.front();queued_.erase(queued_.begin());event.time_ms=elapsed();applyMove(event);
                        }
                        if(now>=next_control_ms_) {
                            schedule(now,std::nullopt);next_control_ms_=now+control_period_ms_;
                        }
                        next=next_control_ms_;
                        if(!queued_.empty())next=std::min(next,queued_.front().time_ms);
                    }
                    std::this_thread::sleep_until(realtime_start_+std::chrono::duration_cast<delta::SteadyClock::duration>(std::chrono::duration<double,std::milli>(next)));
                }
            } catch(const std::exception& error) {std::lock_guard lock(realtime_mutex_);failVisible(error.what());}
        });
    }
    std::optional<delta::ReplayFrame> nextVisible(const delta::CaptureRegion& proposed) {
        if(index_>=params_.frames)return std::nullopt;
        try {
            if(!capture_)startVisible();
            const auto timeout=delta::SteadyClock::now()+std::chrono::seconds(5);
            std::optional<delta::FramePacket> captured;Json state;
            while(!stop_.load() && delta::SteadyClock::now()<timeout) {
                if(!window_.load()){std::this_thread::sleep_for(std::chrono::milliseconds(1));continue;}
                captured=capture_->grab(proposed);
                if(!captured)continue;
                const auto marker=delta::bench::readSceneMarker(*captured,config_.screen_w,config_.screen_h);
                if(!marker || !*marker) {
                    if(index_==0)continue;
                    throw std::runtime_error("Captured scene marker is corrupt or outside crop");
                }
                if(*marker==last_rendered_id_)continue;
                if(*marker<last_rendered_id_)throw std::runtime_error("Captured scene IDs moved backward");
                if(!delta::bench::VisibleSceneWindow::alignedAndUnobscured(window_.load(),display_,captured->capture))
                    throw std::runtime_error("Scene window is obscured, moved or capture alignment is invalid");
                {std::lock_guard lock(realtime_mutex_);
                    const auto found=rendered_.find(*marker);
                    if(found==rendered_.end())throw std::runtime_error("Captured scene ID has no render record");
                    state=found->second;
                }
                if(captured->source_id==0 || captured->source_present==delta::SteadyClock::time_point{}
                    || captured->timings.used_cached_frame)throw std::runtime_error("Capture lacks fresh DXGI presentation provenance");
                last_rendered_id_=*marker;break;
            }
            if(stop_.load())return std::nullopt;
            if(state.empty())throw std::runtime_error("Timed out waiting for a linked scene capture");
            const auto wall_start=delta::SteadyClock::now();
            auto inference=engine_->predict(*captured,-1);
            check(cudaDeviceSynchronize(),"Visible inference synchronization");
            const auto wall_end=delta::SteadyClock::now();
            const auto actual_ms=[this](auto time){return std::chrono::duration<double,std::milli>(time-realtime_start_).count();};
            const double present=actual_ms(captured->source_present);
            if(present+0.1<state["render_submitted_ms"].get<double>())throw std::runtime_error("DXGI presentation predates linked render submission");
            const auto natural=inference.detections.size();
            const bool injected=params_.scenario=="dropout" && state["world_update_ms"].get<double>()>=400
                && state["world_update_ms"].get<double>()<560;
            if(injected)inference.detections.clear();
            const std::string png="frames/"+std::to_string(index_+1)+".png";
            std::vector<std::uint8_t> bgra(static_cast<std::size_t>(captured->width)*captured->height*4);
            for(std::size_t i=0;i<captured->bgr.size()/3;++i) {
                std::copy_n(captured->bgr.data()+i*3,3,bgra.data()+i*4);bgra[i*4+3]=255;
            }
            if(saved_bytes_+bgra.size()>1024ULL*1024*1024)throw std::runtime_error("Captured recording exceeds 1 GiB memory budget; use a shorter run");
            saved_bytes_+=bgra.size();saved_frames_.push_back({png,std::move(bgra),captured->width,captured->height});
            // Translate real steady timestamps into the existing replay epoch.
            // Ground truth and the rendered ID stay exclusively in this adapter.
            const auto convert=[&](auto time){return time==delta::SteadyClock::time_point{}?time:tick(actual_ms(time));};
            state["dxgi_source_id"]=captured->source_id;
            state["scene_state_id"]=++scene_id_;
            captured->source_id=scene_id_;captured->source_present=convert(captured->source_present);
            captured->acquire_started=convert(captured->acquire_started);captured->frame_ready=convert(captured->frame_ready);
            captured->capture_done=convert(captured->capture_done);captured->frame_time=convert(captured->frame_time);
            state["presented_ms"]=present;state["capture_ms"]=ms(captured->capture_done);
            state["acquire_started_ms"]=ms(captured->acquire_started);state["frame_ready_ms"]=ms(captured->frame_ready);
            state["inference_started_ms"]=actual_ms(wall_start);state["inference_completed_ms"]=actual_ms(wall_end);
            state["inference_wall_ms"]=std::chrono::duration<double,std::milli>(wall_end-wall_start).count();
            state["capture"]={captured->capture.left,captured->capture.top,captured->width,captured->height};
            state["image"]=png;state["detections"]=inference.detections.size();state["natural_detections"]=natural;
            state["injected_detection_drop"]=injected;
            delta::ReplayFrame out;out.frame=std::move(*captured);out.detections=std::move(inference.detections);
            {std::lock_guard lock(realtime_mutex_);
                out.now=tick(elapsed());out.dispatch_feedback=std::move(pending_feedback_);pending_feedback_.clear();
                state["available_ms"]=ms(out.now);state["applied_movement"]=applied_since_frame_;
                applied_since_frame_=Json::array();current_=std::move(state);
            }
            current_["replay_input"]=delta::bench::encodeObservation(out);
            ++index_;return out;
        } catch(const std::exception& error) {
            std::lock_guard lock(realtime_mutex_);failVisible(error.what());return std::nullopt;
        }
    }
    struct Movement {double time_ms,scheduled_ms,observation_ms,expiry_ms;
        int dx,dy,truth_id,aim_class;std::uint64_t source_id;};
    double externalYaw(double at_ms)const {
        if(params_.scenario!="shake" || at_ms<400 || at_ms>=700)return 0;
        return .05*std::sin(at_ms*.12);
    }
    double externalPitch(double at_ms)const {
        if(params_.scenario!="shake" || at_ms<400 || at_ms>=700)return 0;
        return .03*std::cos(at_ms*.15);
    }
    void render(int left,int top,int width,int height,double at_ms,std::optional<int> secondary_left={},
        std::optional<double> yaw={},std::optional<double> pitch={}) {
        const int W=config_.screen_w,H=config_.screen_h;
        const int pan=static_cast<int>(std::lround(((yaw?*yaw:camera_yaw_)+externalYaw(at_ms))*W*.6));
        const int pan_y=static_cast<int>(std::lround(((pitch?*pitch:camera_pitch_)+externalPitch(at_ms))*H*.6));
        const int shift=((pan%W)+W)%W;
        for(int y=0;y<H;++y) {
            const int sy=(((y+pan_y)%H)+H)%H;
            const auto* source=background_scaled_.data()+static_cast<std::size_t>(sy)*W*4;
            auto* dest=full_.data()+static_cast<std::size_t>(y)*W*4;
            std::copy_n(source+shift*4,static_cast<std::size_t>(W-shift)*4,dest);
            std::copy_n(source,static_cast<std::size_t>(shift)*4,dest+(W-shift)*4);
        }
        // Establish intended identity before introducing the crossing distractor.
        if(params_.scenario=="crossing" && at_ms>=40 && secondary_left)drawSprite(*secondary_left,top,width,height);
        if(params_.scenario=="occlusion" && at_ms>=400 && at_ms<560)return;
        drawSprite(left,top,width,height);
    }
    void drawSprite(int left,int top,int width,int height) {
        const int W=config_.screen_w,H=config_.screen_h;
        const int source_w=box_[2]-box_[0],source_h=box_[3]-box_[1];
        // Feathered ellipse limits copied background around the annotated body.
        // It is an exploratory cutout, not a hand-validated segmentation mask.
        for(int sy=0;sy<height;++sy)for(int sx=0;sx<width;++sx) {
            const int x=left+sx,y=top+sy;if(x<0 || y<0 || x>=W || y>=H)continue;
            const double nx=(sx-width*.5)/(width*.60),ny=(sy-height*.5)/(height*.70);
            const double alpha=std::clamp((1.0-nx*nx-ny*ny)*5.0,0.0,1.0);
            if(alpha<=0)continue;
            const int source_x=box_[0]+sx*source_w/width;
            const int source_y=box_[1]+sy*source_h/height;
            const auto from=(static_cast<std::size_t>(source_y)*image_.width+source_x)*4;
            const auto to=(static_cast<std::size_t>(y)*W+x)*4;
            for(int c=0;c<3;++c)full_[to+c]=static_cast<std::uint8_t>(
                image_.pixels[from+c]*alpha+full_[to+c]*(1.0-alpha));
        }
    }
    void advance(double to_ms) {
        while(next_control_ms_<=to_ms+1e-7 || (!queued_.empty() && queued_.front().time_ms<=to_ms+1e-7)) {
            const bool apply_first=!queued_.empty() && queued_.front().time_ms<=next_control_ms_;
            if(apply_first && queued_.front().time_ms<=to_ms+1e-7) {
                auto event=queued_.front();queued_.erase(queued_.begin());
                applyMove(event);continue;
            }
            if(next_control_ms_>to_ms+1e-7)break;
            schedule(next_control_ms_,std::nullopt);
            next_control_ms_+=control_period_ms_;
        }
        while(!queued_.empty() && queued_.front().time_ms<=to_ms+1e-7) {
            auto event=queued_.front();queued_.erase(queued_.begin());applyMove(event);
        }
    }
    double worldAngle(double at_ms,double fx)const {
        const double phase=std::fmod(at_ms*params_.speed/1000.0,1000.0);
        const double independent_x=phase<=500?phase-250:750-phase;
        return std::atan((120.0+independent_x)/fx);
    }
    std::pair<double,double> aimPoint(double center_x,double center_y,int cls)const {
        const auto& source=cls==1?head_box_:body_box_;
        return delta::bench::projectSceneAimPoint(source,cls,box_,center_x,center_y,params_.scale,runtime_);
    }
    void applyMove(const Movement& event) {
        camera_yaw_+=event.dx*params_.mouse_radians_per_count;
        camera_pitch_+=event.dy*params_.mouse_radians_per_count;
        const double fx=(config_.screen_w/2.0)/std::tan(params_.hfov_deg*pi/360.0);
        const double fy=(config_.screen_h/2.0)/std::tan(params_.vfov_deg*pi/360.0);
        const double body_x=config_.screen_w/2.0+fx*std::tan(
            worldAngle(event.time_ms,fx)-camera_yaw_-externalYaw(event.time_ms));
        const double body_center_y=config_.screen_h/2.0-fy*std::tan(camera_pitch_+externalPitch(event.time_ms));
        const auto identity_box=delta::bench::projectIdentityBox(body_box_,head_box_,box_,body_x,body_center_y,params_.scale);
        const auto [truth_x,truth_y]=aimPoint(body_x,body_center_y,event.aim_class);
        const Json item={{"scene_state_id",event.source_id},{"scheduled_ms",event.scheduled_ms},
            {"applied_ms",event.time_ms},{"observation_ms",event.observation_ms},
            {"observation_to_applied_ms",event.time_ms-event.observation_ms},
            {"dx",event.dx},{"dy",event.dy},{"camera_yaw_rad",camera_yaw_},
            {"camera_pitch_rad",camera_pitch_},{"truth_center",{truth_x,truth_y}},
            {"normalized_aim_error",delta::bench::normalizedSceneError(identity_box,truth_x,truth_y,
                config_.screen_w/2.0,config_.screen_h/2.0)},
            {"within_target_region",delta::bench::sceneContains(identity_box,config_.screen_w/2.0,config_.screen_h/2.0)},
            {"selected_truth_id",event.truth_id},{"aim_class",event.aim_class},
            {"past_loss_deadline",std::isfinite(event.expiry_ms) && event.time_ms>event.expiry_ms}};
        movements.push_back(item);applied_since_frame_.push_back(item);
    }
    void schedule(double at_ms,std::optional<delta::CommandPacket> incoming) {
        const auto result=scheduler_.step(tick(at_ms),systemTick(at_ms),std::move(incoming),servo_,runtime_,
            1000.0/control_period_ms_,allowed_,true);
        if(result.rejected_prediction)dispatch_rejections.push_back({{"source_id",result.rejected_prediction->source_id},
            {"at_ms",at_ms},{"expires_ms",ms(result.rejected_prediction->expires_at)},
            {"reason","prediction_dispatch_budget"}});
        if(!result.command) {
            if(!result.deferred)pending_feedback_.push_back(delta::CommandDelivery{.completed_at=tick(at_ms)});
            return;
        }
        if(!delta::commandFreshAt(*result.command,tick(at_ms),runtime_.command_max_observation_age_ms))return;
        const auto selection=selection_by_source_.find(result.command->source_id);
        if(selection==selection_by_source_.end())throw std::runtime_error("Dispatched command has no source-frame identity record");
        const auto steps=transform_.apply(result.command->dx,result.command->dy);
        pending_feedback_.push_back(delta::CommandDelivery{.command=*result.command,
            .dx=result.command->dx,.dy=result.command->dy,.completed_at=tick(at_ms),.movement_sent=!steps.empty()});
        for(const auto& [dx,dy]:steps)
            queued_.push_back({at_ms+params_.transport_delay_ms,at_ms,ms(result.command->observation_time),
                result.command->expires_at==delta::SteadyClock::time_point{}
                    ?std::numeric_limits<double>::infinity():ms(result.command->expires_at),
                dx,dy,selection->second.first,selection->second.second,result.command->source_id});
    }
    delta::StaticConfig config_;delta::RuntimeConfig runtime_;Image image_,background_;
    std::filesystem::path output_;
    std::array<int,4> box_,body_box_,head_box_;SceneParameters params_;int index_=0;
    double camera_yaw_=0,camera_pitch_=0;
    double control_period_ms_=0,next_control_ms_=0;std::uint64_t scene_id_=0;
    bool allowed_=false;int selected_truth_id_=-1,selected_cls_=-1;
    std::map<std::uint64_t,std::pair<int,int>> selection_by_source_;
    std::vector<delta::CommandDelivery> pending_feedback_;
    delta::DisplayRateServoState servo_{};
    delta::AimCommandScheduler scheduler_;delta::MouseSenderConfig sender_{};
    delta::MouseMovementTransform transform_;
    std::unique_ptr<delta::IInferenceEngine> engine_;void* device_=nullptr;
    std::vector<std::uint8_t> full_,cropped_;std::vector<Movement> queued_;
    Json current_,applied_since_frame_=Json::array();
    std::mutex realtime_mutex_;
    std::atomic_bool stop_{false};std::atomic<HWND> window_{nullptr};
    std::thread render_thread_,dispatch_thread_;
    delta::SteadyClock::time_point realtime_start_{};
    delta::bench::CaptureDisplay display_{};
    std::unique_ptr<delta::ICaptureSource> capture_;
    std::map<std::uint32_t,Json> rendered_;
    std::uint32_t last_rendered_id_=0;
    std::vector<SavedFrame> saved_frames_;
    std::size_t saved_bytes_=0;
    std::vector<std::uint8_t> background_scaled_;
    std::optional<delta::ScopedTimerResolution> timer_resolution_;
};
}

int replayObservations(int argc,char** argv) {
    std::filesystem::path source=argv[2],output,config_file;
    for(int i=3;i<argc;++i) {
        const std::string key=argv[i];if(++i>=argc)throw std::invalid_argument("Missing replay argument");
        if(key=="--output")output=argv[i];else if(key=="--config")config_file=argv[i];
        else throw std::invalid_argument("Unknown replay argument "+key);
    }
    if(output.empty() || std::filesystem::exists(output))throw std::invalid_argument("Replay requires a new --output FILE");
    std::ifstream file(source);Json input;file>>input;
    auto runtime=delta::bench::runtimeFromJson(input.at("runtime"));
    if(!config_file.empty()) {
        std::ifstream override_file(config_file);Json overrides;override_file>>overrides;
        auto merged=input.at("runtime");merged.update(overrides);runtime=delta::bench::runtimeFromJson(merged);
    }
    delta::StaticConfig config;
    const auto viewport=input.at("scene").at("viewport").get<std::array<int,2>>();
    config.screen_w=viewport[0];config.screen_h=viewport[1];
    class Recorded final:public delta::ITrackingReplay {
    public:
        explicit Recorded(const Json& frames):frames_(frames){}
        std::optional<delta::ReplayFrame> next(const delta::CaptureRegion&)override {
            if(index_==frames_.size())return {};
            return delta::bench::decodeObservation(frames_.at(index_++).at("replay_input"));
        }
        void observe(const delta::ReplayResult& result)override {
            auto outcome=delta::bench::observationOutcome(result);
            if(outcome!=frames_.at(index_-1).at("replay_outcome"))++different;
            if(!result.track)identity_=0;
            std::vector<delta::bench::SceneIdentityRegions> regions;
            for(const auto& t:frames_.at(index_-1).at("truth_boxes"))regions.push_back({t.at("id").get<int>(),
                t.at("body_bbox").get<delta::bench::SceneBox>(),t.at("head_bbox").get<delta::bench::SceneBox>()});
            for(const auto& d:result.preview.detections)if(d.selected)identity_=delta::bench::attributeSceneDetection(
                (d.bbox[0]+d.bbox[2])*.5,(d.bbox[1]+d.bbox[3])*.5,d.cls,regions);
            outcome["selected_truth_id"]=identity_;
            if(result.output_ready && identity_>0 && identity_!=1)++wrong_ready;
            if(result.output_ready && identity_<=0)++unknown_ready;
            outcomes.push_back(outcome);
        }
        Json outcomes=Json::array();int different=0,wrong_ready=0,unknown_ready=0;
    private:const Json& frames_;std::size_t index_=0;int identity_=0;
    } replay(input.at("frames"));
    delta::DeltaApp::replay(config,runtime,replay);
    Json result={{"scope","recorded_observations_open_loop"},{"source",source.string()},
        {"recorded_feedback",true},{"device_output",false},{"runtime",delta::bench::runtimeJson(runtime)},
        {"different_outcomes",replay.different},{"wrong_identity_output_ready_observations",replay.wrong_ready},
        {"unknown_identity_output_ready_observations",replay.unknown_ready},{"frames",replay.outcomes}};
    std::ofstream out(output);out<<result.dump(2);if(!out)throw std::runtime_error("Cannot write observation replay");
    std::cout<<"Replayed "<<replay.outcomes.size()<<" observations; "<<replay.different<<" outcomes differ from recording\n";
    return 0;
}
int main(int argc,char** argv) {
    try {
        if(argc>=3 && std::string(argv[1])=="--replay")return replayObservations(argc,argv);
        const std::filesystem::path manifest_path="cpp_port/benchmarks/scenes/delta-backup-first.json";
        std::ifstream manifest_file(manifest_path);
        if(!manifest_file)throw std::runtime_error("Cannot open scene manifest: "+manifest_path.string());
        Json manifest;manifest_file>>manifest;
        std::filesystem::path image=manifest.at("image").get<std::string>();
        std::filesystem::path background=manifest.at("background").get<std::string>();
        const std::filesystem::path source_label=manifest.at("label").get<std::string>();
        const std::filesystem::path background_label=manifest.at("background_label").get<std::string>();
        std::filesystem::path output;
        std::array<int,4> box=manifest.at("sprite_box_xyxy").get<std::array<int,4>>();
        const auto body_box=manifest.at("body_box_xyxy").get<std::array<int,4>>();
        const auto head_box=manifest.at("head_box_xyxy").get<std::array<int,4>>();
        if(!std::filesystem::is_regular_file(source_label))
            throw std::runtime_error("Missing source label: "+source_label.string());
        if(!std::filesystem::is_regular_file(background_label)
            || std::filesystem::file_size(background_label)!=0)
            throw std::runtime_error("Background must have an empty label file: "+background_label.string());
        SceneParameters params;delta::StaticConfig config;delta::RuntimeConfig runtime;
        config.onnx_require_gpu=true;
        for(int i=1;i<argc;++i) {
            const std::string key=argv[i];auto value=[&](){if(++i>=argc)throw std::invalid_argument("Missing "+key);return std::string(argv[i]);};
            if(key=="--mode") {const auto mode=value();if(mode!="controlled" && mode!="visible")throw std::invalid_argument("Mode must be controlled or visible");params.visible=mode=="visible";}
            else if(key=="--image")image=value();else if(key=="--background")background=value();
            else if(key=="--output")output=value();
            else if(key=="--frames")params.frames=std::stoi(value());else if(key=="--speed")params.speed=std::stod(value());
            else if(key=="--scale")params.scale=std::stod(value());
            else if(key=="--screen-width")config.screen_w=std::stoi(value());
            else if(key=="--screen-height")config.screen_h=std::stoi(value());
            else if(key=="--cadence-ms")params.cadence_ms=std::stod(value());
            else if(key=="--inference-delay-ms")params.inference_delay_ms=std::stod(value());
            else if(key=="--transport-delay-ms")params.transport_delay_ms=std::stod(value());
            else if(key=="--hfov")params.hfov_deg=std::stod(value());
            else if(key=="--vfov")params.vfov_deg=std::stod(value());
            else if(key=="--mouse-radians-per-count")params.mouse_radians_per_count=std::stod(value());
            else if(key=="--scenario")params.scenario=value();
            else if(key=="--gpu-device")config.onnx_cuda_device_id=std::stoi(value());
            else if(key=="--provider")config.onnx_provider=value();
            else if(key=="--config"){std::ifstream file(value());if(!file)throw std::runtime_error("Cannot open config");Json json;file>>json;runtime=delta::bench::runtimeFromJson(json);}
            else if(key=="--box") {
                const std::string input=value();if(std::sscanf(input.c_str(),"%d,%d,%d,%d",&box[0],&box[1],&box[2],&box[3])!=4)
                    throw std::invalid_argument("--box needs x1,y1,x2,y2");
            } else if(key=="--help") {std::cout<<"delta_closed_loop_gpu --output NEW_DIRECTORY [--mode controlled|visible] [--image FILE] [--background FILE] [--box x1,y1,x2,y2] [--frames 200] [--speed 1500] [--scale 1] [--cadence-ms 8] [--scenario clean|occlusion|dropout|shake|crossing] [--config FILE] [--gpu-device 1]\n";return 0;}
            else throw std::invalid_argument("Unknown argument "+key);
        }
        if(image.empty() || output.empty() || params.frames<10 || params.frames>10000
            || config.screen_w<640 || config.screen_w>7680 || config.screen_h<640 || config.screen_h>4320
            || !std::isfinite(params.speed) || params.speed<0 || params.speed>6000
            || !std::isfinite(params.scale) || params.scale<.25 || params.scale>2
            || !std::isfinite(params.cadence_ms) || params.cadence_ms<2 || params.cadence_ms>50
            || !std::isfinite(params.inference_delay_ms) || params.inference_delay_ms<1 || params.inference_delay_ms>50
            || (!params.visible && params.inference_delay_ms>params.cadence_ms)
            || !std::isfinite(params.transport_delay_ms) || params.transport_delay_ms<0 || params.transport_delay_ms>50
            || !std::isfinite(params.hfov_deg) || params.hfov_deg<20 || params.hfov_deg>150
            || !std::isfinite(params.vfov_deg) || params.vfov_deg<20 || params.vfov_deg>150
            || !std::isfinite(params.mouse_radians_per_count) || params.mouse_radians_per_count<=0 || params.mouse_radians_per_count>.05
            || (params.scenario!="clean" && params.scenario!="occlusion" && params.scenario!="dropout"
                && params.scenario!="shake" && params.scenario!="crossing"))
            throw std::invalid_argument("Provide image, new output directory, 10..10000 frames and speed 0..6000");
        if(std::filesystem::exists(output))throw std::invalid_argument("Output directory already exists");
        if(params.visible) {
            if(!SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)
                && GetLastError()!=ERROR_ACCESS_DENIED)throw std::runtime_error("Cannot establish physical-pixel DPI coordinates");
            const auto display=delta::bench::captureDisplay(config);
            config.screen_w=display.bounds.right-display.bounds.left;
            config.screen_h=display.bounds.bottom-display.bounds.top;
        }
        std::filesystem::create_directories(output);
        std::filesystem::create_directories(output/"frames");
        check(CoInitializeEx(nullptr,COINIT_MULTITHREADED),"COM initialization");
        config.inference_profile_prefix=(output/"ort-provider-audit").string();
        auto source_image=loadImage(image);
        checkSourceLabels(source_label,source_image.width,source_image.height,body_box,head_box);
        SceneReplay scene(config,runtime,std::move(source_image),loadImage(background),
            box,body_box,head_box,params,output);
        delta::DeltaApp::replay(config,runtime,scene);
        scene.finish();
        Json result={{"schema_version",5},{"scope",params.visible?"visible_screen_gpu_closed_loop_exploratory":"controlled_time_gpu_closed_loop_exploratory"},
            {"mode",params.visible?"visible":"controlled"},{"capture_hardware",scene.capture_info},
            {"device_output",false},{"gpu",scene.gpu},{"frames",scene.rows},{"applied_movements",scene.movements},
            {"dispatch_rejections",scene.dispatch_rejections},
            {"runtime",delta::bench::runtimeJson(runtime)},{"asset_manifest",manifest},
            {"fingerprints",{{"model",fileFingerprint(config.model_path)},
                {"target_image",fileFingerprint(image)},{"background_image",fileFingerprint(background)},
                {"target_label",fileFingerprint(source_label)},
                {"background_label",fileFingerprint(background_label)},
                {"asset_manifest",fileFingerprint(manifest_path)}}},
            {"scene",{{"image",image.string()},{"viewport",{config.screen_w,config.screen_h}},
                {"background",background.string()},{"sprite_box",box},
                {"speed_px_s",params.speed},{"scale",params.scale},{"scenario",params.scenario},
                {"crossing_distractor_visible_after_ms",40},
                {"horizontal_fov_deg",params.hfov_deg},{"vertical_fov_deg",params.vfov_deg},
                {"mouse_radians_per_count",params.mouse_radians_per_count},
                {"render_interval_ms",params.cadence_ms},{"logical_inference_delay_ms",params.visible?Json(nullptr):Json(params.inference_delay_ms)},
                {"simulated_transport_delay_ms",params.transport_delay_ms}}},
            {"build",{{"revision",DELTA_BENCH_REVISION},{"source_hash",DELTA_BENCH_SOURCE_HASH}}}};
        double total_error=0,frame_error=0,wrong_time_ms=0;
        int inside=0,frame_inside=0,wrong=0,past_deadline=0,ready=0,detected=0;
        std::vector<double> inference_times,application_ages;
        Json motion_decisions=Json::object();
        double previous_available_ms=0;
        for(const auto& frame:scene.rows) {
            const double sampled_interval=params.visible?frame["available_ms"].get<double>()-previous_available_ms:params.cadence_ms;
            previous_available_ms=frame["available_ms"].get<double>();
            const auto decision=frame["motion_decision"].get<std::string>();
            motion_decisions[decision]=motion_decisions.value(decision,0)+1;
            frame_error+=frame["normalized_aim_error"].get<double>();
            frame_inside+=frame["reticle_within_target"].get<bool>();
            ready+=frame["output_ready"].get<bool>();detected+=frame["detections"].get<int>()>0;
            inference_times.push_back(frame["inference_wall_ms"].get<double>());
            for(const auto& applied:frame["applied_movement"]) {
                const int id=applied["selected_truth_id"].get<int>();
                if(id>=0 && id!=1) {wrong_time_ms+=sampled_interval;break;}
            }
        }
        for(const auto& movement:scene.movements) {
            total_error+=movement["normalized_aim_error"].get<double>();
            inside+=movement["within_target_region"].get<bool>();
            application_ages.push_back(movement["observation_to_applied_ms"].get<double>());
            wrong+=movement["selected_truth_id"].get<int>()>=0
                && movement["selected_truth_id"].get<int>()!=1;
            past_deadline+=movement["past_loss_deadline"].get<bool>();
        }
        const double frame_count=static_cast<double>(std::max<std::size_t>(1,scene.rows.size()));
        result["summary"]={{"frames",scene.rows.size()},{"applied_movements",scene.movements.size()},
            {"frames_with_detections",detected},{"output_ready_coverage",double(ready)/frame_count},
            {"mean_normalized_error_all_frames",frame_error/frame_count},
            {"fraction_frames_within_target_region",double(frame_inside)/frame_count},
            {"mean_normalized_error_at_application",scene.movements.empty()?Json(nullptr):Json(total_error/scene.movements.size())},
            {"fraction_movements_within_target_region",scene.movements.empty()?Json(nullptr):Json(double(inside)/scene.movements.size())},
            {"movements_with_wrong_selected_target",wrong},{"wrong_target_movement_ms_sampled",wrong_time_ms},
            {"movements_after_loss_deadline",past_deadline},
            {"identity_scoring",delta::bench::scoreMovementIdentity(scene.movements,1)},
            {"motion_decision_counts",motion_decisions},
            {"gpu_inference_wall_ms",distribution(inference_times)},
            {params.visible?"measured_observation_to_applied_ms":"logical_observation_to_applied_ms",distribution(application_ages)}};
        if(params.visible) {
            std::vector<double> render_present,present_capture,capture_inference,inference_tracking,present_intervals;
            std::vector<double> inference_dispatch,tracking_dispatch,dispatch_application,world_application;
            for(const auto& f:scene.rows) {
                render_present.push_back(f["presented_ms"].get<double>()-f["render_submitted_ms"].get<double>());
                present_capture.push_back(f["capture_ms"].get<double>()-f["presented_ms"].get<double>());
                capture_inference.push_back(f["inference_completed_ms"].get<double>()-f["capture_ms"].get<double>());
                inference_tracking.push_back(f["tracking_processing_us"].get<double>()/1000.0);
            }
            for(std::size_t i=1;i<scene.rows.size();++i)present_intervals.push_back(
                scene.rows[i]["presented_ms"].get<double>()-scene.rows[i-1]["presented_ms"].get<double>());
            for(const auto& movement:scene.movements) {
                const auto& source=scene.rows.at(movement["scene_state_id"].get<std::size_t>()-1);
                inference_dispatch.push_back(movement["scheduled_ms"].get<double>()-source["inference_completed_ms"].get<double>());
                tracking_dispatch.push_back(movement["scheduled_ms"].get<double>()-source["tracking_completed_ms"].get<double>());
                dispatch_application.push_back(movement["applied_ms"].get<double>()-movement["scheduled_ms"].get<double>());
                world_application.push_back(movement["applied_ms"].get<double>()-source["world_update_ms"].get<double>());
            }
            result["summary"]["measured_latency_ms"]={{"render_submission_to_dxgi_present",distribution(render_present)},
                {"dxgi_present_to_capture_complete",distribution(present_capture)},
                {"capture_complete_to_inference_complete",distribution(capture_inference)},
                {"tracking_processing",distribution(inference_tracking)},
                {"inference_complete_to_dispatch",distribution(inference_dispatch)},
                {"tracking_complete_to_dispatch",distribution(tracking_dispatch)},
                {"dispatch_to_simulated_application",distribution(dispatch_application)},
                {"world_update_to_simulated_application",distribution(world_application)},
                {"captured_presentation_interval",distribution(present_intervals)}};
        }
        Json validity_reasons=scene.issues;
        if(scene.rows.size()!=static_cast<std::size_t>(params.frames))validity_reasons.push_back("Run did not complete the requested observation count");
        std::size_t expected_id=1;
        for(const auto& frame:scene.rows) {
            if(frame["scene_state_id"].get<std::size_t>()!=expected_id++)
                validity_reasons.push_back("scene-state IDs are not consecutive");
            if(frame["stale"].get<bool>())validity_reasons.push_back("unexpected stale/cached scene frame");
            if(frame["capture_ms"].get<double>()<frame["presented_ms"].get<double>()
                || frame["available_ms"].get<double>()<frame["capture_ms"].get<double>())
                validity_reasons.push_back("invalid frame stage order");
        }
        for(const auto& movement:scene.movements) {
            const auto source_id=movement["scene_state_id"].get<std::size_t>();
            if(source_id<1 || source_id>scene.rows.size())
                validity_reasons.push_back("movement has no source scene frame");
            else if(movement["selected_truth_id"]!=scene.rows[source_id-1]["selected_truth_id"]
                || movement["aim_class"]!=scene.rows[source_id-1]["selected_class"])
                validity_reasons.push_back("movement identity does not match its source frame");
            if(movement["scheduled_ms"].get<double>()>movement["applied_ms"].get<double>()
                || movement["observation_ms"].get<double>()>movement["scheduled_ms"].get<double>())
                validity_reasons.push_back("invalid movement stage order");
        }
        result["validity"]={{"valid",validity_reasons.empty()},{"reasons",validity_reasons},
            {"truth_quality","paired head/body labels checked; approximate ellipse cutout"}};
        result["correctness"]=delta::bench::closedLoopCorrectness(scene.movements);
        std::ofstream output_json(output/"closed-loop.json");output_json<<result.dump(2);
        if(!output_json)throw std::runtime_error("Failed writing result");
        std::string embedded=result.dump();
        std::size_t pos=0;while((pos=embedded.find('<',pos))!=std::string::npos) {embedded.replace(pos,1,"\\u003c");pos+=6;}
        std::ofstream html(output/"report.html");
        html<<R"HTML(<!doctype html><meta charset="utf-8"><title>Delta closed-loop replay</title>
<style>body{margin:0;background:#101820;color:#e9f2f8;font:15px system-ui}main{max-width:1240px;margin:auto;padding:20px}a{color:#81c8ff}canvas{background:#16232d;max-width:100%}section{background:#1b2933;padding:16px;border-radius:12px;margin:12px 0}.row{display:flex;gap:18px;flex-wrap:wrap}label{display:block}input[type=range]{width:100%}code{color:#ffcc82}</style>
<main><h1>Controlled-time GPU closed loop</h1><p>Production detector, tracker, shared aim scheduler and backend movement transform. Simulated camera output; no mouse or serial device output. This exploratory scene uses an approximate cutout and uncalibrated camera sensitivity.</p>
<p id="checks"></p><p><a href="closed-loop.json">Machine-readable result</a></p><section><label>Frame <b id="number"></b><input type="range" id="scrub" min="0" value="0"></label><div class="row"><canvas id="view" width="800" height="800"></canvas><div><p id="state"></p><p id="summary"></p><button id="back">Previous</button> <button id="forward">Next</button></div></div></section>
<section><h2>Aim error at applied movement</h2><canvas id="plot" width="1100" height="240"></canvas></section>
<script id="data" type="application/json">)HTML"<<embedded<<R"HTML(</script><script>
const run=JSON.parse(document.getElementById('data').textContent),rows=run.frames;
document.querySelector('h1').textContent=run.mode==='visible'?'Visible-screen GPU closed loop':'Controlled-time GPU closed loop';
document.getElementById('checks').textContent=`Mode: ${run.mode}. No device output. Evidence valid: ${run.validity.valid}. Correctness checks: ${run.correctness.passed?'passed':'FAILED'}. ${run.validity.reasons.join('; ')}`;
if(run.mode==='visible'){const timing=document.createElement('pre');timing.style.whiteSpace='pre-wrap';timing.textContent='Measured latency (ms): '+JSON.stringify(run.summary.measured_latency_ms,null,2);document.getElementById('checks').after(timing);}
const scrub=document.getElementById('scrub'),view=document.getElementById('view'),ctx=view.getContext('2d');scrub.max=rows.length-1;
let frame=0;const img=new Image();
function box(b,color,line=2){ctx.strokeStyle=color;ctx.lineWidth=line;ctx.strokeRect(b[0],b[1],b[2]-b[0],b[3]-b[1]);}
function draw(){const r=rows[frame],crop=r.capture;document.getElementById('number').textContent=`${frame+1} / ${rows.length}`;scrub.value=frame;
  view.width=crop[2];view.height=crop[3];ctx.drawImage(img,0,0);
  box([r.truth_box[0]-crop[0],r.truth_box[1]-crop[1],r.truth_box[2]-crop[0],r.truth_box[3]-crop[1]],'#42f59b',3);
  for(const d of r.detector_boxes||[])box([d.bbox[0]-crop[0],d.bbox[1]-crop[1],d.bbox[2]-crop[0],d.bbox[3]-crop[1]],d.selected?'#ffbd52':'#67b7ff');
  if(r.selected){ctx.fillStyle='#ffbd52';ctx.beginPath();ctx.arc(r.selected[0]-crop[0],r.selected[1]-crop[1],5,0,Math.PI*2);ctx.fill();}
  document.getElementById('state').innerHTML=`Scene ${r.scene_state_id}<br>Detections ${r.detections}<br>Command ${JSON.stringify(r.command)}<br>Output ready ${r.output_ready}<br>Camera yaw ${r.camera_yaw_rad.toFixed(4)} rad`;
}
function setFrame(n){frame=Math.max(0,Math.min(rows.length-1,n));img.onload=draw;img.src=rows[frame].image;}
scrub.oninput=()=>setFrame(Number(scrub.value));document.getElementById('back').onclick=()=>setFrame(frame-1);document.getElementById('forward').onclick=()=>setFrame(frame+1);
document.getElementById('summary').textContent=`Applied movements: ${run.summary.applied_movements}; mean normalized error (all frames): ${run.summary.mean_normalized_error_all_frames.toFixed(3)}; within target (all frames): ${(run.summary.fraction_frames_within_target_region*100).toFixed(1)}%; output-ready: ${(run.summary.output_ready_coverage*100).toFixed(1)}%`;
const plot=document.getElementById('plot'),p=plot.getContext('2d'),moves=run.applied_movements;p.clearRect(0,0,plot.width,plot.height);p.strokeStyle='#425765';for(let y=0;y<=4;y++){p.beginPath();p.moveTo(0,y*60);p.lineTo(plot.width,y*60);p.stroke();}
if(moves.length){const maxT=Math.max(1,moves[moves.length-1].applied_ms),maxE=Math.max(1,...moves.map(m=>m.normalized_aim_error));p.strokeStyle='#ffbd52';p.lineWidth=2;p.beginPath();moves.forEach((m,i)=>{const x=m.applied_ms/maxT*plot.width,y=plot.height-m.normalized_aim_error/maxE*plot.height;i?p.lineTo(x,y):p.moveTo(x,y)});p.stroke();}
if(rows.length)setFrame(0);
</script></main>)HTML";
        std::cout<<"GPU "<<scene.gpu["name"]<<" frames="<<scene.rows.size()
            <<" applied_moves="<<scene.movements.size()<<" result="<<std::filesystem::absolute(output/"closed-loop.json")<<'\n';
        CoUninitialize();return validity_reasons.empty() && result["correctness"]["passed"].get<bool>()?0:2;
    } catch(const std::exception& error) {std::cerr<<"[closed-loop] "<<error.what()<<'\n';return 1;}
}
