#pragma once
#include "benchmark.hpp"
#include <optional>

namespace delta::bench {
inline Json closedLoopCorrectness(const Json& movements) {
    Json violations=Json::array();
    for(const auto& movement:movements) {
        if(movement.at("past_loss_deadline").get<bool>())
            violations.push_back({{"kind","movement_after_loss_deadline"},
                {"scene_state_id",movement.at("scene_state_id")}});
    }
    return {{"passed",violations.empty()},{"violations",violations}};
}

inline Json scoreMovementIdentity(const Json& movements,int intended_id) {
    std::optional<int> initial,previous,previous_class;
    bool acquired=false;
    int wrong_before=0,wrong_after=0,switches=0,aim_changes=0;
    for(const auto& movement:movements) {
        const int id=movement.at("selected_truth_id").get<int>();
        if(id<0)continue;
        const int cls=movement.at("aim_class").get<int>();
        if(!initial)initial=id;
        if(id==intended_id)acquired=true;
        else if(acquired)++wrong_after;
        else ++wrong_before;
        if(previous && *previous!=id)++switches;
        if(previous && *previous==id && previous_class && *previous_class!=cls)++aim_changes;
        previous=id;previous_class=cls;
    }
    return {{"initial_applied_identity",initial?Json(*initial):Json(nullptr)},
        {"initial_selection_correct",initial?Json(*initial==intended_id):Json(nullptr)},
        {"intended_target_acquired",acquired},
        {"wrong_movements_before_intended_acquisition",wrong_before},
        {"wrong_movements_after_intended_acquisition",wrong_after},
        {"identity_switches",switches},{"aim_point_changes",aim_changes}};
}
}
