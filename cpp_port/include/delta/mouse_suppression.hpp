#pragma once

#include <cstdint>
#include <memory>

namespace delta {

struct MouseMoveSuppressionStatus {
    bool supported = false;
    bool active = false;
    std::uint64_t suppressed_count = 0;
};

bool shouldSuppressMouseMoveOnFire(bool enabled, bool left_pressed, bool x1_pressed);

class IMouseMoveSuppressor {
public:
    virtual ~IMouseMoveSuppressor() = default;

    virtual void start() = 0;
    virtual void stop() = 0;
    virtual void setDebugLogging(bool enabled) = 0;
    virtual void setSuppressionActive(bool active) = 0;
    virtual MouseMoveSuppressionStatus snapshot() const = 0;
};

std::unique_ptr<IMouseMoveSuppressor> makeMouseMoveSuppressor();

}  // namespace delta
