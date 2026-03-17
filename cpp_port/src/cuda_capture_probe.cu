#include <windows.h>

#include <d3d11.h>
#include <dxgi1_2.h>
#include <wrl/client.h>

#include <cuda_d3d11_interop.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cwchar>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "delta/core.hpp"

namespace {

using Microsoft::WRL::ComPtr;

constexpr int kDefaultCropSize = 640;
constexpr int kDefaultFrames = 5;
constexpr int kDefaultTimeoutMs = 250;

struct ProbeOptions {
    int width = kDefaultCropSize;
    int height = kDefaultCropSize;
    int frames = kDefaultFrames;
    int timeout_ms = kDefaultTimeoutMs;
    int left = -1;
    int top = -1;
};

class ScopeExit {
public:
    explicit ScopeExit(std::function<void()> fn) : fn_(std::move(fn)) {}
    ~ScopeExit() {
        if (fn_) {
            fn_();
        }
    }

    ScopeExit(const ScopeExit&) = delete;
    ScopeExit& operator=(const ScopeExit&) = delete;

private:
    std::function<void()> fn_;
};

std::string wideToUtf8(const wchar_t* value) {
    if (value == nullptr || value[0] == L'\0') {
        return {};
    }
    const int wide_len = static_cast<int>(std::wcslen(value));
    const int size = WideCharToMultiByte(CP_UTF8, 0, value, wide_len, nullptr, 0, nullptr, nullptr);
    if (size <= 0) {
        return {};
    }
    std::string result(static_cast<std::size_t>(size), '\0');
    WideCharToMultiByte(
        CP_UTF8,
        0,
        value,
        wide_len,
        result.data(),
        size,
        nullptr,
        nullptr
    );
    return result;
}

std::string hresultMessage(HRESULT hr) {
    char* buffer = nullptr;
    const DWORD flags = FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS;
    const DWORD count = FormatMessageA(
        flags,
        nullptr,
        static_cast<DWORD>(hr),
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        reinterpret_cast<LPSTR>(&buffer),
        0,
        nullptr
    );

    std::ostringstream oss;
    oss << "HRESULT 0x" << std::hex << std::uppercase << static_cast<std::uint32_t>(hr);
    if (count > 0 && buffer != nullptr) {
        std::string text(buffer, buffer + count);
        while (!text.empty() && (text.back() == '\r' || text.back() == '\n')) {
            text.pop_back();
        }
        if (!text.empty()) {
            oss << " (" << text << ")";
        }
    }
    if (buffer != nullptr) {
        LocalFree(buffer);
    }
    return oss.str();
}

void checkHr(HRESULT hr, std::string_view what) {
    if (FAILED(hr)) {
        throw std::runtime_error(std::string(what) + " failed: " + hresultMessage(hr));
    }
}

void checkCuda(cudaError_t status, std::string_view what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(what) + " failed: " + cudaGetErrorString(status)
        );
    }
}

int parsePositiveInt(const std::string& text, std::string_view name) {
    const int value = std::stoi(text);
    if (value <= 0) {
        throw std::runtime_error(std::string(name) + " must be > 0");
    }
    return value;
}

ProbeOptions parseArgs(int argc, char** argv) {
    ProbeOptions options{};
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto readValue = [&](std::string_view flag) -> std::string {
            if (arg == flag) {
                if (i + 1 >= argc) {
                    throw std::runtime_error("Missing value for " + std::string(flag));
                }
                ++i;
                return argv[i];
            }
            const std::string prefix = std::string(flag) + "=";
            if (arg.rfind(prefix, 0) == 0) {
                return arg.substr(prefix.size());
            }
            return {};
        };

        if (const std::string value = readValue("--width"); !value.empty()) {
            options.width = parsePositiveInt(value, "width");
            continue;
        }
        if (const std::string value = readValue("--height"); !value.empty()) {
            options.height = parsePositiveInt(value, "height");
            continue;
        }
        if (const std::string value = readValue("--frames"); !value.empty()) {
            options.frames = parsePositiveInt(value, "frames");
            continue;
        }
        if (const std::string value = readValue("--timeout-ms"); !value.empty()) {
            options.timeout_ms = parsePositiveInt(value, "timeout-ms");
            continue;
        }
        if (const std::string value = readValue("--left"); !value.empty()) {
            options.left = std::stoi(value);
            continue;
        }
        if (const std::string value = readValue("--top"); !value.empty()) {
            options.top = std::stoi(value);
            continue;
        }
        if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: delta_cuda_capture_probe "
                << "[--width N] [--height N] [--frames N] [--timeout-ms N] "
                << "[--left N --top N]\n";
            std::exit(0);
        }
        throw std::runtime_error("Unknown argument: " + arg);
    }
    return options;
}

std::uint64_t fnv1a64(const std::vector<std::uint8_t>& bytes) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const std::uint8_t value : bytes) {
        hash ^= static_cast<std::uint64_t>(value);
        hash *= 1099511628211ULL;
    }
    return hash;
}

__global__ void bgraToBgrKernel(
    const uchar4* src,
    std::size_t src_pitch,
    std::uint8_t* dst,
    std::size_t dst_pitch,
    int width,
    int height
) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= width || y >= height) {
        return;
    }

    const auto* src_row = reinterpret_cast<const uchar4*>(
        reinterpret_cast<const std::uint8_t*>(src) + (static_cast<std::size_t>(y) * src_pitch)
    );
    auto* dst_row = reinterpret_cast<std::uint8_t*>(
        reinterpret_cast<std::uint8_t*>(dst) + (static_cast<std::size_t>(y) * dst_pitch)
    );

    const uchar4 pixel = src_row[x];
    const std::size_t offset = static_cast<std::size_t>(x) * 3ULL;
    dst_row[offset + 0] = pixel.x;
    dst_row[offset + 1] = pixel.y;
    dst_row[offset + 2] = pixel.z;
}

class DesktopDuplicationCudaProbe {
public:
    explicit DesktopDuplicationCudaProbe(ProbeOptions options) : options_(options) {
        initializeDxgi();
        initializeCuda();
        initializeDevice();
        initializeDuplication();
        crop_ = resolveCropRegion();
        createInteropTexture();
        allocateCudaBuffers();
    }

    ~DesktopDuplicationCudaProbe() {
        releaseCudaResources();
    }

    void run() {
        std::cout
            << "[probe] adapter=" << adapter_name_
            << " output=" << output_name_
            << " desktop=" << output_width_ << "x" << output_height_
            << " crop=(" << crop_.left << "," << crop_.top
            << " " << crop_.width << "x" << crop_.height << ")"
            << " cudaDevice=" << cuda_device_
            << "\n";

        int captured = 0;
        int attempts = 0;
        const int max_attempts = std::max(options_.frames * 20, 40);
        while (captured < options_.frames) {
            ++attempts;
            if (captureOne(captured + 1)) {
                ++captured;
                continue;
            }
            if (attempts >= max_attempts) {
                throw std::runtime_error("Timed out waiting for desktop duplication frames.");
            }
        }
    }

private:
    void initializeDxgi() {
        ComPtr<IDXGIFactory1> factory;
        checkHr(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "CreateDXGIFactory1");
        checkHr(factory->EnumAdapters1(0, &adapter_), "EnumAdapters1(0)");

        DXGI_ADAPTER_DESC1 adapter_desc{};
        checkHr(adapter_->GetDesc1(&adapter_desc), "IDXGIAdapter1::GetDesc1");
        adapter_name_ = wideToUtf8(adapter_desc.Description);

        checkHr(adapter_->EnumOutputs(0, &output_), "EnumOutputs(0)");
        checkHr(output_.As(&output1_), "Query IDXGIOutput1");
        checkHr(output_->GetDesc(&output_desc_), "IDXGIOutput::GetDesc");

        output_name_ = wideToUtf8(output_desc_.DeviceName);
        output_width_ = output_desc_.DesktopCoordinates.right - output_desc_.DesktopCoordinates.left;
        output_height_ = output_desc_.DesktopCoordinates.bottom - output_desc_.DesktopCoordinates.top;
        if (output_width_ <= 0 || output_height_ <= 0) {
            throw std::runtime_error("Desktop output dimensions are invalid.");
        }
    }

    void initializeCuda() {
        checkCuda(cudaD3D11GetDevice(&cuda_device_, adapter_.Get()), "cudaD3D11GetDevice");
        checkCuda(cudaSetDevice(cuda_device_), "cudaSetDevice");
        checkCuda(cudaStreamCreate(&stream_), "cudaStreamCreate");
    }

    void initializeDevice() {
        constexpr D3D_FEATURE_LEVEL levels[] = {
            D3D_FEATURE_LEVEL_11_1,
            D3D_FEATURE_LEVEL_11_0,
        };
        D3D_FEATURE_LEVEL selected = D3D_FEATURE_LEVEL_11_0;
        const UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
        checkHr(
            D3D11CreateDevice(
                adapter_.Get(),
                D3D_DRIVER_TYPE_UNKNOWN,
                nullptr,
                flags,
                levels,
                static_cast<UINT>(std::size(levels)),
                D3D11_SDK_VERSION,
                &device_,
                &selected,
                &context_
            ),
            "D3D11CreateDevice"
        );
    }

    void initializeDuplication() {
        checkHr(output1_->DuplicateOutput(device_.Get(), &duplication_), "IDXGIOutput1::DuplicateOutput");
    }

    delta::CaptureRegion resolveCropRegion() const {
        const int width = delta::clamp(options_.width, 1, output_width_);
        const int height = delta::clamp(options_.height, 1, output_height_);
        const int default_left = std::max(0, (output_width_ - width) / 2);
        const int default_top = std::max(0, (output_height_ - height) / 2);

        const int left = delta::clamp(
            options_.left >= 0 ? options_.left : default_left,
            0,
            output_width_ - width
        );
        const int top = delta::clamp(
            options_.top >= 0 ? options_.top : default_top,
            0,
            output_height_ - height
        );
        delta::CaptureRegion region{};
        region.left = left;
        region.top = top;
        region.width = width;
        region.height = height;
        return region;
    }

    void createInteropTexture() {
        D3D11_TEXTURE2D_DESC desc{};
        desc.Width = static_cast<UINT>(crop_.width);
        desc.Height = static_cast<UINT>(crop_.height);
        desc.MipLevels = 1;
        desc.ArraySize = 1;
        desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        desc.SampleDesc.Count = 1;
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        desc.CPUAccessFlags = 0;
        desc.MiscFlags = 0;

        checkHr(device_->CreateTexture2D(&desc, nullptr, &interop_texture_), "CreateTexture2D(interop)");
        checkCuda(
            cudaGraphicsD3D11RegisterResource(
                &cuda_resource_,
                interop_texture_.Get(),
                cudaGraphicsRegisterFlagsNone
            ),
            "cudaGraphicsD3D11RegisterResource"
        );
    }

    void allocateCudaBuffers() {
        host_bgr_.assign(static_cast<std::size_t>(crop_.width) * static_cast<std::size_t>(crop_.height) * 3ULL, 0U);
        checkCuda(
            cudaMallocPitch(
                &bgra_device_,
                &bgra_pitch_,
                static_cast<std::size_t>(crop_.width) * sizeof(uchar4),
                static_cast<std::size_t>(crop_.height)
            ),
            "cudaMallocPitch(bgra)"
        );
        checkCuda(
            cudaMallocPitch(
                &bgr_device_,
                &bgr_pitch_,
                static_cast<std::size_t>(crop_.width) * 3ULL,
                static_cast<std::size_t>(crop_.height)
            ),
            "cudaMallocPitch(bgr)"
        );
    }

    void releaseCudaResources() {
        if (cuda_resource_ != nullptr) {
            cudaGraphicsUnregisterResource(cuda_resource_);
            cuda_resource_ = nullptr;
        }
        if (bgra_device_ != nullptr) {
            cudaFree(bgra_device_);
            bgra_device_ = nullptr;
        }
        if (bgr_device_ != nullptr) {
            cudaFree(bgr_device_);
            bgr_device_ = nullptr;
        }
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
        interop_texture_.Reset();
        duplication_.Reset();
        context_.Reset();
        device_.Reset();
        output1_.Reset();
        output_.Reset();
        adapter_.Reset();
    }

    bool captureOne(int frame_number) {
        DXGI_OUTDUPL_FRAME_INFO frame_info{};
        ComPtr<IDXGIResource> frame_resource;

        const auto acquire_start = std::chrono::steady_clock::now();
        const HRESULT acquire_hr = duplication_->AcquireNextFrame(
            static_cast<UINT>(options_.timeout_ms),
            &frame_info,
            &frame_resource
        );
        if (acquire_hr == DXGI_ERROR_WAIT_TIMEOUT) {
            std::cout << "[probe] timeout waiting for frame\n";
            return false;
        }
        checkHr(acquire_hr, "AcquireNextFrame");
        bool frame_acquired = true;
        ScopeExit release_frame([&]() {
            if (frame_acquired) {
                duplication_->ReleaseFrame();
            }
        });

        ComPtr<ID3D11Texture2D> frame_texture;
        checkHr(frame_resource.As(&frame_texture), "QueryInterface(ID3D11Texture2D)");

        D3D11_BOX source_box{};
        source_box.left = static_cast<UINT>(crop_.left);
        source_box.top = static_cast<UINT>(crop_.top);
        source_box.front = 0;
        source_box.right = static_cast<UINT>(crop_.left + crop_.width);
        source_box.bottom = static_cast<UINT>(crop_.top + crop_.height);
        source_box.back = 1;

        const auto d3d_start = std::chrono::steady_clock::now();
        context_->CopySubresourceRegion(
            interop_texture_.Get(),
            0,
            0,
            0,
            0,
            frame_texture.Get(),
            0,
            &source_box
        );
        context_->Flush();
        const auto d3d_end = std::chrono::steady_clock::now();
        checkHr(duplication_->ReleaseFrame(), "ReleaseFrame");
        frame_acquired = false;

        bool mapped = false;
        auto unmap_cuda = ScopeExit([&]() {
            if (mapped && cuda_resource_ != nullptr) {
                cudaGraphicsUnmapResources(1, &cuda_resource_, stream_);
            }
        });

        const auto cuda_start = std::chrono::steady_clock::now();
        checkCuda(cudaGraphicsMapResources(1, &cuda_resource_, stream_), "cudaGraphicsMapResources");
        mapped = true;

        cudaArray_t mapped_array = nullptr;
        checkCuda(
            cudaGraphicsSubResourceGetMappedArray(&mapped_array, cuda_resource_, 0, 0),
            "cudaGraphicsSubResourceGetMappedArray"
        );
        checkCuda(
            cudaMemcpy2DFromArrayAsync(
                bgra_device_,
                bgra_pitch_,
                mapped_array,
                0,
                0,
                static_cast<std::size_t>(crop_.width) * sizeof(uchar4),
                static_cast<std::size_t>(crop_.height),
                cudaMemcpyDeviceToDevice,
                stream_
            ),
            "cudaMemcpy2DFromArrayAsync"
        );

        const dim3 block(16, 16);
        const dim3 grid(
            static_cast<unsigned int>((crop_.width + block.x - 1) / block.x),
            static_cast<unsigned int>((crop_.height + block.y - 1) / block.y)
        );
        bgraToBgrKernel<<<grid, block, 0, stream_>>>(
            static_cast<const uchar4*>(bgra_device_),
            bgra_pitch_,
            static_cast<std::uint8_t*>(bgr_device_),
            bgr_pitch_,
            crop_.width,
            crop_.height
        );
        checkCuda(cudaGetLastError(), "bgraToBgrKernel launch");

        checkCuda(
            cudaMemcpy2DAsync(
                host_bgr_.data(),
                static_cast<std::size_t>(crop_.width) * 3ULL,
                bgr_device_,
                bgr_pitch_,
                static_cast<std::size_t>(crop_.width) * 3ULL,
                static_cast<std::size_t>(crop_.height),
                cudaMemcpyDeviceToHost,
                stream_
            ),
            "cudaMemcpy2DAsync(download)"
        );
        checkCuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
        const auto cuda_end = std::chrono::steady_clock::now();

        const auto acquire_ms = std::chrono::duration<double, std::milli>(d3d_start - acquire_start).count();
        const auto d3d_ms = std::chrono::duration<double, std::milli>(d3d_end - d3d_start).count();
        const auto cuda_ms = std::chrono::duration<double, std::milli>(cuda_end - cuda_start).count();
        const std::uint64_t checksum = fnv1a64(host_bgr_);

        const std::uint8_t b = host_bgr_.empty() ? 0U : host_bgr_[0];
        const std::uint8_t g = host_bgr_.size() > 1 ? host_bgr_[1] : 0U;
        const std::uint8_t r = host_bgr_.size() > 2 ? host_bgr_[2] : 0U;

        std::cout
            << "[probe] frame=" << frame_number
            << " acquire=" << std::fixed << std::setprecision(2) << acquire_ms << "ms"
            << " d3d-copy=" << d3d_ms << "ms"
            << " cuda=" << cuda_ms << "ms"
            << " firstBGR=(" << static_cast<int>(b) << "," << static_cast<int>(g) << "," << static_cast<int>(r) << ")"
            << " checksum=0x" << std::hex << checksum << std::dec
            << "\n";

        return true;
    }

    ProbeOptions options_{};
    delta::CaptureRegion crop_{};
    ComPtr<IDXGIAdapter1> adapter_;
    ComPtr<IDXGIOutput> output_;
    ComPtr<IDXGIOutput1> output1_;
    ComPtr<ID3D11Device> device_;
    ComPtr<ID3D11DeviceContext> context_;
    ComPtr<IDXGIOutputDuplication> duplication_;
    ComPtr<ID3D11Texture2D> interop_texture_;
    DXGI_OUTPUT_DESC output_desc_{};
    std::string adapter_name_;
    std::string output_name_;
    int output_width_ = 0;
    int output_height_ = 0;
    int cuda_device_ = -1;
    cudaGraphicsResource_t cuda_resource_ = nullptr;
    cudaStream_t stream_ = nullptr;
    void* bgra_device_ = nullptr;
    std::size_t bgra_pitch_ = 0;
    void* bgr_device_ = nullptr;
    std::size_t bgr_pitch_ = 0;
    std::vector<std::uint8_t> host_bgr_;
};

}  // namespace

int main(int argc, char** argv) {
    try {
        const ProbeOptions options = parseArgs(argc, argv);
        DesktopDuplicationCudaProbe probe(options);
        probe.run();
        std::cout << "[probe] D3D11/CUDA interop capture path is operational.\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[probe] ERROR: " << ex.what() << "\n";
        return 1;
    }
}
