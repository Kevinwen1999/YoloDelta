#pragma once

#include <string>

#include "delta/core.hpp"

namespace delta::gpu {

enum class TensorElementType {
    Float32,
    Float16,
};

bool preprocessBgraToNchw(
    const GpuFramePacket& src,
    void* dst_tensor,
    int dst_width,
    int dst_height,
    TensorElementType dst_type,
    void* stream,
    std::string* error_message);

}  // namespace delta::gpu
