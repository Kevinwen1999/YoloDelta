#pragma once

#include <memory>

#include "delta/config.hpp"
#include "delta/debug_preview_types.hpp"

namespace delta {

class DebugPreviewWindow {
public:
    virtual ~DebugPreviewWindow() = default;

    virtual void start() = 0;
    virtual void stop() = 0;
    virtual void setEnabled(bool enabled) = 0;
    virtual void publish(DebugPreviewSnapshot snapshot) = 0;
};

std::unique_ptr<DebugPreviewWindow> makeDebugPreviewWindow(const StaticConfig& config);

}  // namespace delta
