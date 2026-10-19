#pragma once

#include <optional>
#include <string>
#include <vector>

#include "delta/core.hpp"
#include "delta/debug_preview_types.hpp"
#include "delta/tracking.hpp"
#include "delta/runtime_state.hpp"

namespace delta {

// Input to the SAME inference loop used by the live application. Coordinates in
// detections are capture-local, just like IInferenceEngine output. No truth IDs
// enter this interface. Logical time must increase strictly, starting above zero.
struct ReplayFrame {
    SteadyClock::time_point now{};
    FramePacket frame{};
    std::vector<Detection> detections;
    std::vector<CommandDelivery> dispatch_feedback;
    ToggleState toggles{.mode = 1};
};

struct ReplayResult {
    DebugPreviewSnapshot preview{};
    std::optional<CommandPacket> command;
    DisplayRateServoState servo{};
    DispatchFeedbackSnapshot dispatch_feedback{};
    std::optional<TrackerState> track;
    TrackerDiagnostics tracker_diagnostics{};
    bool stale_frame = false;
    bool output_ready = false;
    bool motion_ambiguous = false;
    std::string motion_decision = "legacy_policy";
    float motion_residual_px = 0;
    std::vector<std::pair<Detection,float>> motion_matches;
    int association_id = -1;
    int lost_frames = 0;
    int next_crop_size = 0;
    // Wall-clock cost of the executed inference-loop stages. Excludes source,
    // observer, report generation, capture, neural inference and output delivery.
    double processing_us = 0.0;
    double selection_us = 0.0;
    double tracking_us = 0.0;
    double prediction_us = 0.0;
    double control_us = 0.0;
};

class ITrackingReplay {
public:
    virtual ~ITrackingReplay() = default;
    // The production capture policy proposes a crop. An adapter may use a fixed
    // recorded crop instead; the actual crop must be stored in frame.capture.
    virtual std::optional<ReplayFrame> next(const CaptureRegion& proposed_crop) = 0;
    virtual void observe(const ReplayResult& result) = 0;
    virtual std::uint64_t randomSeed() const { return 1; }
};

} // namespace delta
