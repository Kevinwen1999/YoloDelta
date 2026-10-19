#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>
#include "delta/tracking.hpp"

namespace delta::bench {
using SceneBox = std::array<double,4>;
struct SceneIdentityRegions {int id;SceneBox body,head;};
inline int attributeSceneDetection(double x,double y,int cls,const std::vector<SceneIdentityRegions>& truth) {
    if(cls!=0 && cls!=1)return 0;
    int identity=0,matches=0;
    for(const auto& t:truth) {
        const auto& region=cls==1?t.head:t.body;
        if(x>=region[0] && x<=region[2] && y>=region[1] && y<=region[3]) {identity=t.id;++matches;}
    }
    return matches>1?-2:identity;
}
inline std::pair<double,double> projectSceneAimPoint(const std::array<int,4>& annotation,
    int cls,const std::array<int,4>& sprite,double center_x,double center_y,double scale,
    const RuntimeConfig& runtime) {
    const Detection truth{.bbox=annotation,.cls=cls};
    const auto [x,y]=detectionAimPoint(truth,runtime.body_y_ratio,runtime.head_x_ratio,runtime.head_y_ratio);
    return {center_x+(x-(sprite[0]+sprite[2])*.5)*scale,
        center_y+(y-(sprite[1]+sprite[3])*.5)*scale};
}
inline SceneBox projectIdentityBox(const std::array<int,4>& body,
    const std::array<int,4>& head,const std::array<int,4>& sprite,
    double center_x,double center_y,double scale) {
    const double origin_x=center_x-(sprite[0]+sprite[2])*.5*scale;
    const double origin_y=center_y-(sprite[1]+sprite[3])*.5*scale;
    return {origin_x+std::min(body[0],head[0])*scale,
        origin_y+std::min(body[1],head[1])*scale,
        origin_x+std::max(body[2],head[2])*scale,
        origin_y+std::max(body[3],head[3])*scale};
}
inline bool sceneContains(const SceneBox& box,double x,double y) {
    return x>=box[0] && x<=box[2] && y>=box[1] && y<=box[3];
}
inline double normalizedSceneError(const SceneBox& box,double aim_x,double aim_y,
    double reticle_x,double reticle_y) {
    return std::hypot((aim_x-reticle_x)/(box[2]-box[0]),
        (aim_y-reticle_y)/(box[3]-box[1]));
}
}
