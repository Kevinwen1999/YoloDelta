#pragma once

#include <memory>
#include <optional>
#include <string_view>

#include "delta/config.hpp"
#include "delta/core.hpp"

namespace delta {

class ICaptureSource {
public:
    virtual ~ICaptureSource() = default;
    virtual std::string_view name() const = 0;
    virtual std::optional<FramePacket> grab(const CaptureRegion& region) = 0;
    virtual std::optional<GpuFramePacket> grabGpu(const CaptureRegion&) { return std::nullopt; }
    virtual void setGpuConsumerStream(void*) {}
    virtual void setCachedFrameTimeoutMs(double) {}
    virtual void setFreshOnly(bool) {}
    virtual void close() = 0;
};

class DesktopDuplicationCapture final : public ICaptureSource {
public:
    explicit DesktopDuplicationCapture(const StaticConfig& config);
    ~DesktopDuplicationCapture() override;

    DesktopDuplicationCapture(const DesktopDuplicationCapture&) = delete;
    DesktopDuplicationCapture& operator=(const DesktopDuplicationCapture&) = delete;
    DesktopDuplicationCapture(DesktopDuplicationCapture&&) noexcept;
    DesktopDuplicationCapture& operator=(DesktopDuplicationCapture&&) noexcept;

    std::string_view name() const override { return "desktop-duplication"; }
    std::optional<FramePacket> grab(const CaptureRegion& region) override;
    std::optional<GpuFramePacket> grabGpu(const CaptureRegion& region) override;
    void setGpuConsumerStream(void* stream) override;
    void setCachedFrameTimeoutMs(double timeout_ms) override;
    void setFreshOnly(bool fresh_only) override;
    void close() override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

std::unique_ptr<ICaptureSource> makeDefaultCaptureSource(const StaticConfig& config);

}  // namespace delta
