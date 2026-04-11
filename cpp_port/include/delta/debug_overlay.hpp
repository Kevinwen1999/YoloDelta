#pragma once

#include <memory>

#include "delta/config.hpp"
#include "delta/debug_preview_types.hpp"

namespace delta {

class DebugOverlayWindow {
public:
    virtual ~DebugOverlayWindow() = default;

    virtual void start() = 0;
    virtual void stop() = 0;
    virtual void setEnabled(bool enabled) = 0;
    virtual void publish(DebugPreviewSnapshot snapshot) = 0;
};

std::unique_ptr<DebugOverlayWindow> makeDebugOverlayWindow(const StaticConfig& config);

}  // namespace delta
