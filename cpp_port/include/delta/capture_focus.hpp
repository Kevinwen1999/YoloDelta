#pragma once

#include <utility>

namespace delta {

std::pair<int, int> selectCaptureFocus(
    bool freeze_to_center,
    bool target_found,
    std::pair<int, int> screen_center,
    std::pair<int, int> tracked_focus);

}  // namespace delta
