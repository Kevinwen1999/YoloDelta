#pragma once

#include <memory>
#include <string_view>
#include <vector>

#include "delta/config.hpp"
#include "delta/core.hpp"

namespace delta {

struct InferenceTimings {
    double preprocess_ms = 0.0;
    double execute_ms = 0.0;
    double postprocess_ms = 0.0;
};

struct InferenceResult {
    std::vector<Detection> detections;
    InferenceTimings timings{};
};

enum class GpuCaptureSchedule {
    None,
    Inline,
    InlineTensorRt,
    AsyncLatest,
};

class IInferenceEngine {
public:
    virtual ~IInferenceEngine() = default;
    virtual std::string_view name() const = 0;
    virtual void warmup() = 0;
    virtual void setModelConfidence(float) {}
    virtual void* gpuInputStream() const { return nullptr; }
    virtual GpuCaptureSchedule gpuCaptureSchedule() const { return GpuCaptureSchedule::None; }
    virtual InferenceResult predict(const FramePacket& frame, int target_class) = 0;
    virtual bool supportsGpuInput() const { return false; }
    virtual InferenceResult predictGpu(const GpuFramePacket&, int) { return InferenceResult{}; }
};

class OnnxRuntimeEngine final : public IInferenceEngine {
public:
    explicit OnnxRuntimeEngine(const StaticConfig& config);
    ~OnnxRuntimeEngine() override;

    std::string_view name() const override { return name_; }
    void warmup() override;
    void setModelConfidence(float conf) override;
    void* gpuInputStream() const override;
    GpuCaptureSchedule gpuCaptureSchedule() const override;
    InferenceResult predict(const FramePacket& frame, int target_class) override;
    bool supportsGpuInput() const override;
    InferenceResult predictGpu(const GpuFramePacket& frame, int target_class) override;

private:
    struct Impl;
    StaticConfig config_{};
    std::string name_ = "onnxruntime";
    std::unique_ptr<Impl> impl_;
};

std::unique_ptr<IInferenceEngine> makeInferenceEngine(const StaticConfig& config);

}  // namespace delta
