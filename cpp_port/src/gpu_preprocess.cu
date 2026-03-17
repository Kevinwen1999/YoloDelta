#include "delta/gpu_preprocess.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace delta::gpu {

namespace {

template <typename T>
__device__ inline T toTensorValue(float value);

template <>
__device__ inline float toTensorValue<float>(float value) {
    return value;
}

template <>
__device__ inline __half toTensorValue<__half>(float value) {
    return __float2half_rn(value);
}

template <typename T>
__global__ void bgraToNchwKernel(
    const uchar4* src,
    std::size_t src_pitch_bytes,
    const int src_width,
    const int src_height,
    T* dst,
    const int dst_width,
    const int dst_height) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= dst_width || y >= dst_height) {
        return;
    }

    const int sx = min((x * src_width) / max(dst_width, 1), src_width - 1);
    const int sy = min((y * src_height) / max(dst_height, 1), src_height - 1);
    const auto* src_row = reinterpret_cast<const uchar4*>(
        reinterpret_cast<const std::uint8_t*>(src) + (static_cast<std::size_t>(sy) * src_pitch_bytes));
    const uchar4 pixel = src_row[sx];
    constexpr float scale = 1.0f / 255.0f;

    const std::size_t plane = static_cast<std::size_t>(dst_width) * static_cast<std::size_t>(dst_height);
    const std::size_t idx = static_cast<std::size_t>(y) * static_cast<std::size_t>(dst_width) + static_cast<std::size_t>(x);
    dst[idx] = toTensorValue<T>(static_cast<float>(pixel.z) * scale);
    dst[idx + plane] = toTensorValue<T>(static_cast<float>(pixel.y) * scale);
    dst[idx + plane * 2ULL] = toTensorValue<T>(static_cast<float>(pixel.x) * scale);
}

bool launchTypedKernel(
    const GpuFramePacket& src,
    void* dst_tensor,
    const int dst_width,
    const int dst_height,
    const TensorElementType dst_type,
    cudaStream_t stream,
    std::string* error_message) {
    const dim3 block(16, 16);
    const dim3 grid(
        static_cast<unsigned int>((dst_width + block.x - 1) / block.x),
        static_cast<unsigned int>((dst_height + block.y - 1) / block.y));

    if (dst_type == TensorElementType::Float16) {
        bgraToNchwKernel<<<grid, block, 0, stream>>>(
            static_cast<const uchar4*>(src.device_ptr),
            src.pitch_bytes,
            src.width,
            src.height,
            static_cast<__half*>(dst_tensor),
            dst_width,
            dst_height);
    } else {
        bgraToNchwKernel<<<grid, block, 0, stream>>>(
            static_cast<const uchar4*>(src.device_ptr),
            src.pitch_bytes,
            src.width,
            src.height,
            static_cast<float*>(dst_tensor),
            dst_width,
            dst_height);
    }

    if (const cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
        if (error_message != nullptr) {
            *error_message = cudaGetErrorString(err);
        }
        return false;
    }
    return true;
}

}  // namespace

bool preprocessBgraToNchw(
    const GpuFramePacket& src,
    void* dst_tensor,
    const int dst_width,
    const int dst_height,
    const TensorElementType dst_type,
    void* stream,
    std::string* error_message) {
    if (src.device_ptr == nullptr || dst_tensor == nullptr || src.width <= 0 || src.height <= 0 || dst_width <= 0 || dst_height <= 0) {
        if (error_message != nullptr) {
            *error_message = "Invalid GPU preprocess arguments.";
        }
        return false;
    }
    if (src.pixel_format != PixelFormat::Bgra8) {
        if (error_message != nullptr) {
            *error_message = "Only BGRA8 GPU frame packets are currently supported.";
        }
        return false;
    }

    return launchTypedKernel(src, dst_tensor, dst_width, dst_height, dst_type, static_cast<cudaStream_t>(stream), error_message);
}

}  // namespace delta::gpu
