#include "delta/capture_focus.hpp"

namespace delta {

std::pair<int, int> selectCaptureFocus(
    const bool freeze_to_center,
    const bool target_found,
    const std::pair<int, int> screen_center,
    const std::pair<int, int> tracked_focus) {
    if (freeze_to_center || !target_found) {
        return screen_center;
    }
    return tracked_focus;
}

}  // namespace delta
