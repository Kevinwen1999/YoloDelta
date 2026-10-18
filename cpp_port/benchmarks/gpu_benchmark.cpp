#include "benchmark.hpp"
#include "benchmark_build.hpp"
#include "delta/app.hpp"

#include <cuda_runtime.h>
#include <windows.h>
#include <wincodec.h>
#include <wrl/client.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>

namespace {
using delta::bench::Json;
using Microsoft::WRL::ComPtr;
void cudaCheck(cudaError_t value,const char* operation) {
    if(value!=cudaSuccess) throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(value));
}
void hrCheck(HRESULT value,const char* operation) {
    if(FAILED(value)) throw std::runtime_error(std::string(operation)+" failed: "+std::to_string(value));
}
double milliseconds(delta::SteadyClock::time_point a,delta::SteadyClock::time_point b) {
    return std::chrono::duration<double,std::milli>(b-a).count();
}
Json distribution(std::vector<double> values) {
    if(values.empty()) return {{"samples",0},{"p50",nullptr},{"p95",nullptr},{"p99",nullptr},{"max",nullptr}};
    std::sort(values.begin(),values.end());
    auto p=[&](double q){return values[static_cast<std::size_t>(std::ceil(q*values.size()))-1];};
    return {{"samples",values.size()},{"p50",p(.5)},{"p95",p(.95)},{"p99",p(.99)},{"max",values.back()}};
}
struct Image {
    UINT width=0,height=0;
    std::vector<std::uint8_t> pixels;
};
Image loadImage(const std::filesystem::path& path) {
    ComPtr<IWICImagingFactory> factory;
    hrCheck(CoCreateInstance(CLSID_WICImagingFactory,nullptr,CLSCTX_INPROC_SERVER,IID_PPV_ARGS(&factory)),"WIC factory");
    ComPtr<IWICBitmapDecoder> decoder; ComPtr<IWICBitmapFrameDecode> frame; ComPtr<IWICFormatConverter> converter;
    hrCheck(factory->CreateDecoderFromFilename(path.c_str(),nullptr,GENERIC_READ,WICDecodeMetadataCacheOnLoad,&decoder),"Image decode");
    hrCheck(decoder->GetFrame(0,&frame),"Image frame");
    hrCheck(factory->CreateFormatConverter(&converter),"Image converter");
    hrCheck(converter->Initialize(frame.Get(),GUID_WICPixelFormat32bppBGRA,WICBitmapDitherTypeNone,nullptr,0,WICBitmapPaletteTypeCustom),"BGRA conversion");
    Image out; hrCheck(frame->GetSize(&out.width,&out.height),"Image size");
    if(out.width<1 || out.height<1 || out.width>8192 || out.height>8192) throw std::invalid_argument("Image dimensions out of range");
    out.pixels.resize(static_cast<std::size_t>(out.width)*out.height*4);
    hrCheck(converter->CopyPixels(nullptr,out.width*4,static_cast<UINT>(out.pixels.size()),out.pixels.data()),"Image pixels");
    return out;
}
struct DeviceBuffer {
    void* data=nullptr;
    ~DeviceBuffer(){if(data)cudaFree(data);}
};

class GpuReplay final : public delta::ITrackingReplay {
public:
    GpuReplay(delta::StaticConfig config,delta::RuntimeConfig runtime,std::filesystem::path image_path,
        int count,double speed,std::filesystem::path output)
        :config_(std::move(config)),runtime_(runtime),image_path_(std::move(image_path)),count_(count),speed_(speed) {
        cudaCheck(cudaSetDevice(config_.onnx_cuda_device_id),"Select inference GPU");
        cudaDeviceProp properties{};
        cudaCheck(cudaGetDeviceProperties(&properties,config_.onnx_cuda_device_id),"GPU properties");
        metadata={{"device_id",config_.onnx_cuda_device_id},{"name",properties.name},
            {"compute_capability",std::to_string(properties.major)+"."+std::to_string(properties.minor)}};
        char pci_bus[32]{};
        cudaCheck(cudaDeviceGetPCIBusId(pci_bus,sizeof(pci_bus),config_.onnx_cuda_device_id),"GPU PCI identity");
        metadata["pci_bus_id"]=pci_bus;
        int driver=0,runtime_version=0;cudaDriverGetVersion(&driver);cudaRuntimeGetVersion(&runtime_version);
        metadata["cuda_driver_version"]=driver;metadata["cuda_runtime_version"]=runtime_version;
        config_.inference_profile_prefix=(output/"ort-provider-audit").string();
        const auto init_start=delta::SteadyClock::now();
        engine_=delta::makeInferenceEngine(config_);
        if(!engine_->supportsGpuInput()) throw std::runtime_error("A working CUDA input/inference path is required: "+std::string(engine_->name()));
        engine_->setModelConfidence(runtime_.model_conf);
        engine_->warmup();
        metadata["session"]=std::string(engine_->name());
        if(image_path_.empty()) {
            capture_=delta::makeDefaultCaptureSource(config_);
            capture_->setGpuConsumerStream(engine_->gpuInputStream());
            capture_->setFreshOnly(false);
            capture_->setCachedFrameTimeoutMs(0);
        } else {
            image_=loadImage(image_path_);
            shifted_=image_.pixels;
            cudaCheck(cudaMalloc(&buffer_.data,image_.pixels.size()),"Fixture GPU buffer");
        }
        metadata["initialization_ms"]=milliseconds(init_start,delta::SteadyClock::now());
        // Warm and audit actual graph execution, then STOP profiling before timing.
        warming_=true;
        for(int i=0;i<12;++i) acquire();
        const auto profile=engine_->finishProfiling();
        if(profile.empty()) throw std::runtime_error("Missing GPU execution profile");
        std::ifstream input(profile);Json events;input>>events;
        Json providers=Json::object();
        for(const auto& event:events) if(event.contains("args") && event["args"].contains("provider")) {
            const std::string provider=event["args"]["provider"];
            providers[provider]=providers.value(provider,0)+1;
        }
        if(providers.value("CUDAExecutionProvider",0)+providers.value("TensorrtExecutionProvider",0)==0)
            throw std::runtime_error("GPU audit failed: no CUDA/TensorRT node execution recorded");
        metadata["audited_node_events_by_provider"]=providers;
        metadata["profile"]=profile;
        metadata["profiling_disabled_for_measurement"]=true;
        warming_=false;index_=0;epoch_=delta::SteadyClock::now();
    }
    ~GpuReplay(){if(capture_)capture_->close();}
    std::optional<delta::ReplayFrame> next(const delta::CaptureRegion&) override {
        if(index_>=count_)return std::nullopt;
        ++index_;
        return acquire();
    }
    void observe(const delta::ReplayResult& result) override {
        row_["tracking_us"]=result.processing_us;
        row_["stale_rejected"]=result.stale_frame;
        row_["output_ready"]=result.output_ready;
        row_["command"]=result.command?Json::array({result.command->dx,result.command->dy}):Json(nullptr);
        row_["predicted"]=result.preview.predicted_point?Json::array({result.preview.predicted_point->first,result.preview.predicted_point->second}):Json(nullptr);
        // Command timestamp is the real post-inference input time in replay. Add
        // measured production-loop cost to expose observation-to-command latency.
        row_["observation_to_loop_complete_ms"]=row_["observation_to_inference_ms"].get<double>()+result.processing_us/1000.0;
        row_["source_to_loop_complete_ms"]=row_["source_to_inference_ms"].get<double>()+result.processing_us/1000.0;
        row_["command_latency_ms"]=result.command?row_["observation_to_loop_complete_ms"]:Json(nullptr);
        rows.push_back(row_);
    }
    Json metadata,rows=Json::array();
private:
    delta::ReplayFrame acquire() {
        delta::ReplayFrame out;
        const auto source_start=delta::SteadyClock::now();
        delta::GpuFramePacket gpu;
        std::optional<delta::FramePacket> cpu;
        bool gpu_capture=false;
        double fixture_shift=0;
        if(capture_) {
            delta::CaptureRegion crop{config_.screen_w/2-320,config_.screen_h/2-320,640,640};
            auto packet=capture_->grabGpu(crop);
            if(packet){gpu=*packet;gpu_capture=true;}
            else cpu=capture_->grab(crop);
            if(!packet && !cpu) throw std::runtime_error("Desktop capture returned no frame");
            if(cpu)out.frame=*cpu;
        } else {
            const double elapsed=warming_?0:milliseconds(epoch_,source_start)/1000;
            // Translate a real validation image, not synthetic detection boxes.
            // Wrap seams and whole-scene motion make this a throughput fixture,
            // not annotated high-speed target accuracy evidence.
            const int shift=static_cast<int>(std::fmod(elapsed*speed_,image_.width));
            fixture_shift=shift;
            for(UINT y=0;y<image_.height;++y) for(UINT x=0;x<image_.width;++x) {
                const auto from=(static_cast<std::size_t>(y)*image_.width+(x+image_.width-shift)%image_.width)*4;
                const auto to=(static_cast<std::size_t>(y)*image_.width+x)*4;
                std::copy_n(image_.pixels.data()+from,4,shifted_.data()+to);
            }
            gpu.width=static_cast<int>(image_.width);gpu.height=static_cast<int>(image_.height);
            gpu.capture={config_.screen_w/2-gpu.width/2,config_.screen_h/2-gpu.height/2,gpu.width,gpu.height};
            gpu.device_ptr=buffer_.data;gpu.pitch_bytes=image_.width*4;
            gpu.frame_ready=source_start;gpu.frame_time=source_start;gpu.acquire_started=source_start;
            gpu.capture_time=delta::SystemClock::now();
            cudaCheck(cudaSetDevice(config_.onnx_cuda_device_id),"Select GPU for upload");
            cudaCheck(cudaMemcpy(buffer_.data,shifted_.data(),shifted_.size(),cudaMemcpyHostToDevice),"Fixture upload");
            gpu.capture_done=delta::SteadyClock::now();gpu_capture=true;
        }
        if(gpu_capture) {
            out.frame.capture=gpu.capture;out.frame.frame_ready=gpu.frame_ready;out.frame.frame_time=gpu.frame_time;
            out.frame.acquire_started=gpu.acquire_started;out.frame.capture_done=gpu.capture_done;
            out.frame.capture_time=gpu.capture_time;
        }
        if(out.frame.frame_ready==delta::SteadyClock::time_point{})out.frame.frame_ready=source_start;
        const auto inference_start=delta::SteadyClock::now();
        auto inference=gpu_capture?engine_->predictGpu(gpu,-1):engine_->predict(*cpu,-1);
        cudaCheck(cudaSetDevice(config_.onnx_cuda_device_id),"Select GPU for synchronization");
        cudaCheck(cudaDeviceSynchronize(),"Inference completion");
        out.now=delta::SteadyClock::now();out.detections=std::move(inference.detections);
        row_={{"frame",index_},{"fixture_shift_px",fixture_shift},{"source_ms",milliseconds(source_start,inference_start)},
            {"inference_wall_ms",milliseconds(inference_start,out.now)},
            {"preprocess_ms",inference.timings.preprocess_ms},{"execute_ms",inference.timings.execute_ms},
            {"postprocess_ms",inference.timings.postprocess_ms},{"detections",out.detections.size()},
            {"gpu_input",gpu_capture},{"cached_capture",cpu?cpu->timings.used_cached_frame:gpu.timings.used_cached_frame},
            {"observation_to_inference_ms",milliseconds(out.frame.frame_ready,out.now)},
            {"source_to_inference_ms",milliseconds(source_start,out.now)}};
        row_["frame_interval_ms"]=(!warming_ && index_>1)?Json(milliseconds(previous_source_,source_start)):Json(nullptr);
        previous_source_=source_start;
        return out;
    }
    delta::StaticConfig config_;
    delta::RuntimeConfig runtime_;
    std::filesystem::path image_path_;
    int count_,index_=0;
    double speed_;
    bool warming_=true;
    std::unique_ptr<delta::IInferenceEngine> engine_;
    std::unique_ptr<delta::ICaptureSource> capture_;
    Image image_;
    std::vector<std::uint8_t> shifted_;
    DeviceBuffer buffer_;
    delta::SteadyClock::time_point epoch_{},previous_source_{};
    Json row_;
};
}

int main(int argc,char** argv) {
    std::cout.setf(std::ios::unitbuf);
    try {
        delta::StaticConfig config;delta::RuntimeConfig runtime;
        config.onnx_require_gpu=true;
        std::filesystem::path image_path,output,config_path;
        bool desktop=false;int frames=200;double speed=1500;
        for(int i=1;i<argc;++i){const std::string arg=argv[i];auto value=[&](){if(++i>=argc)throw std::invalid_argument("Missing "+arg);return std::string(argv[i]);};
            if(arg=="--image")image_path=value();else if(arg=="--desktop")desktop=true;
            else if(arg=="--output")output=value();else if(arg=="--gpu-device")config.onnx_cuda_device_id=std::stoi(value());
            else if(arg=="--capture-adapter")config.capture_device_idx=std::stoi(value());
            else if(arg=="--provider")config.onnx_provider=value();else if(arg=="--model")config.model_path=value();
            else if(arg=="--frames")frames=std::stoi(value());else if(arg=="--speed")speed=std::stod(value());
            else if(arg=="--config")config_path=value();
            else if(arg=="--help"){std::cout<<"delta_gpu_benchmark (--image FILE | --desktop) --output NEW_DIRECTORY [--gpu-device 1] [--provider cuda|tensorrt] [--frames 200] [--speed 1500] [--model FILE] [--config FILE] [--capture-adapter 0]\n";return 0;}
            else throw std::invalid_argument("Unknown argument "+arg);
        }
        if(output.empty() || desktop==!image_path.empty() || frames<10 || frames>10000 || !std::isfinite(speed) || speed<0 || speed>100000)
            throw std::invalid_argument("Provide --image OR --desktop, --output, 10..10000 frames and nonnegative speed");
        if(config.onnx_provider!="auto" && config.onnx_provider!="cuda" && config.onnx_provider!="tensorrt")throw std::invalid_argument("A GPU provider is required");
        if(std::filesystem::exists(output))throw std::invalid_argument("Output directory must be new");
        if(!config_path.empty()){std::ifstream file(config_path);Json json;file>>json;runtime=delta::bench::runtimeFromJson(json);}
        std::filesystem::create_directories(output);
        hrCheck(CoInitializeEx(nullptr,COINIT_MULTITHREADED),"COM initialization");
        {
            GpuReplay source(config,runtime,image_path,frames,speed,output);
            delta::DeltaApp::replay(config,runtime,source);
            Json summary=Json::object();
            for(const auto* key:{"source_ms","inference_wall_ms","preprocess_ms","execute_ms","postprocess_ms","tracking_us",
                "observation_to_loop_complete_ms","source_to_loop_complete_ms","command_latency_ms","frame_interval_ms"}) {
                std::vector<double> values;for(const auto& row:source.rows)if(row[key].is_number())values.push_back(row[key]);
                summary[key]=distribution(values);
            }
            int commands=0,stale=0,detected=0,gpu_inputs=0,cached=0;
            for(const auto& row:source.rows){commands+=!row["command"].is_null();stale+=row["stale_rejected"].get<bool>();
                detected+=row["detections"].get<int>()>0;gpu_inputs+=row["gpu_input"].get<bool>();cached+=row["cached_capture"].get<bool>();}
            Json out={{"schema_version",1},{"scope","GPU source to production tracking/command generation; output delivery excluded"},
                {"gpu",source.metadata},{"summary",summary},{"frames",source.rows},{"runtime",delta::bench::runtimeJson(runtime)},
                {"source",desktop?"desktop":"translated_image"},{"image",image_path.string()},{"image_translation_px_s",speed},
                {"model_requested",config.model_path},{"frames_with_commands",commands},{"stale_rejected",stale},
                {"frames_with_detections",detected},{"gpu_input_frames",gpu_inputs},{"cached_capture_frames",cached},
                {"build",{{"revision",DELTA_BENCH_REVISION},{"source_hash",DELTA_BENCH_SOURCE_HASH},{"configuration",DELTA_BENCH_CONFIGURATION}}}};
            std::ofstream json(output/"gpu.json");json<<out.dump(2);if(!json)throw std::runtime_error("Cannot write GPU results");
            std::ofstream html(output/"report.html");
            html<<"<!doctype html><meta charset='utf-8'><title>Delta GPU latency</title><style>body{font:16px system-ui;background:#101824;color:#e1eaf5;padding:30px;max-width:1100px;margin:auto}td,th{padding:10px;border-bottom:1px solid #405068;text-align:left}a{color:#78bdff}</style><h1>Delta GPU latency</h1><p>Actual ONNX inference and production C++ tracking. Output delivery, physical camera response, and accuracy ground truth are excluded. Profiling is disabled during measurement.</p><p>GPU execution verified by CUDA/TensorRT node events. <a href='gpu.json'>Full measurements and provider evidence</a></p><table><tr><th>Metric</th><th>p50</th><th>p95</th><th>p99</th></tr>";
            for(const auto& [key,value]:summary.items())html<<"<tr><td>"<<key<<"</td><td>"<<value["p50"]<<"</td><td>"<<value["p95"]<<"</td><td>"<<value["p99"]<<"</td></tr>";
            html<<"</table><p>Frames: "<<frames<<"; with detections: "<<detected<<"; commands: "<<commands<<"; stale rejected: "<<stale<<"; GPU input: "<<gpu_inputs<<"; cached capture: "<<cached<<".</p>";
            std::cout<<"GPU: "<<source.metadata["name"]<<"; executed providers: "<<source.metadata["audited_node_events_by_provider"]<<'\n';
            std::cout<<"Frames="<<frames<<" detections="<<detected<<" commands="<<commands<<" stale="<<stale<<" gpu_input="<<gpu_inputs<<'\n';
            std::cout<<summary.dump(2)<<"\nReport: "<<std::filesystem::absolute(output/"report.html")<<'\n';
        }
        CoUninitialize();return 0;
    } catch(const std::exception& error){std::cerr<<"[gpu-benchmark] "<<error.what()<<'\n';return 1;}
}
