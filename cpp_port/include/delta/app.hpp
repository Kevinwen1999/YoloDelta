#pragma once

#include <memory>
#include <thread>

#include "delta/capture.hpp"
#include "delta/config.hpp"
#include "delta/control.hpp"
#include "delta/debug_overlay.hpp"
#include "delta/debug_preview.hpp"
#include "delta/frontend.hpp"
#include "delta/inference.hpp"
#include "delta/recoil.hpp"
#include "delta/runtime_state.hpp"
#include "delta/tracking.hpp"

namespace delta {

struct RuntimePerfWindow;

#if defined(__cpp_lib_jthread) && (__cpp_lib_jthread >= 201911L)
using AppThread = std::jthread;
#else
using AppThread = std::thread;
#endif

class DeltaApp {
public:
    DeltaApp(StaticConfig config, RuntimeConfig runtime);
    ~DeltaApp();

    int run();
    void stop();

private:
    void captureLoop();
    void inferenceLoop();
    void recoilLoop();
    void controlLoop();
    void sideButtonKeySequenceLoop();
    void perfLoop();

    StaticConfig config_{};
    RuntimeConfigStore runtime_store_;
    SharedState shared_{};
    LatestSlot<FramePacket> frame_slot_;
    LatestSlot<GpuFramePacket> gpu_frame_slot_;
    LatestSlot<CommandPacket> command_slot_;
    std::unique_ptr<ICaptureSource> capture_;
    std::unique_ptr<IInferenceEngine> inference_;
    std::unique_ptr<IInputSender> input_sender_;
    std::unique_ptr<RecoilScheduler> recoil_scheduler_;
    std::unique_ptr<DebugPreviewWindow> debug_preview_;
    std::unique_ptr<DebugOverlayWindow> debug_overlay_;
    std::unique_ptr<RuntimeFrontendServer> frontend_;
    std::unique_ptr<RuntimePerfWindow> perf_;
    AppThread capture_thread_;
    AppThread inference_thread_;
    AppThread recoil_thread_;
    AppThread control_thread_;
    AppThread side_button_key_sequence_thread_;
    AppThread perf_thread_;
};

}  // namespace delta
