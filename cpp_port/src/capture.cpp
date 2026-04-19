#include "delta/capture.hpp"

#include <windows.h>

#include <d3d11.h>
#include <dxgi1_2.h>
#include <wrl/client.h>

#if defined(DELTA_WITH_CUDA_PIPELINE)
#include <cuda_d3d11_interop.h>
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cwchar>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace delta {

namespace {

using Microsoft::WRL::ComPtr;

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
    WideCharToMultiByte(CP_UTF8, 0, value, wide_len, result.data(), size, nullptr, nullptr);
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

double secondsSince(const SteadyClock::time_point since, const SteadyClock::time_point now) {
    return std::chrono::duration<double>(now - since).count();
}

std::optional<double> envDouble(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return std::nullopt;
    }
    try {
        return std::stod(value);
    } catch (...) {
        return std::nullopt;
    }
}

UINT dxgiTimeoutMs(const double timeout_ms) {
    if (!std::isfinite(timeout_ms) || timeout_ms <= 0.0) {
        return 0;
    }
    constexpr double kMaxDxgiTimeoutMs = static_cast<double>(std::numeric_limits<UINT>::max());
    return static_cast<UINT>(std::min(std::ceil(timeout_ms), kMaxDxgiTimeoutMs));
}

#if defined(DELTA_WITH_CUDA_PIPELINE)
void checkCuda(cudaError_t status, std::string_view what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + " failed: " + cudaGetErrorString(status));
    }
}

void CUDART_CB markGpuSlotCopyComplete(void* user_data) {
    auto* copy_pending = static_cast<std::shared_ptr<std::atomic_bool>*>(user_data);
    if (copy_pending != nullptr) {
        (*copy_pending)->store(false, std::memory_order_release);
        delete copy_pending;
    }
}
#endif

CaptureRegion clampRegionToDesktop(const CaptureRegion& requested, int desktop_width, int desktop_height) {
    const int width = clamp(requested.width, 1, desktop_width);
    const int height = clamp(requested.height, 1, desktop_height);
    const int left = clamp(requested.left, 0, desktop_width - width);
    const int top = clamp(requested.top, 0, desktop_height - height);
    return CaptureRegion{
        .left = left,
        .top = top,
        .width = width,
        .height = height,
    };
}

}  // namespace

struct DesktopDuplicationCapture::Impl {
    explicit Impl(StaticConfig cfg) : config(std::move(cfg)) {}

    double effectiveAcquireTimeoutMs() const {
        double timeout_ms = std::max(0.0, static_cast<double>(config.capture_timeout_ms));
        if (const auto override_ms = envDouble("DELTA_CAPTURE_TIMEOUT_MS"); override_ms.has_value()) {
            timeout_ms = std::max(0.0, *override_ms);
        }
        if (config.capture_video_mode && has_cached_desktop && cached_desktop != nullptr) {
            timeout_ms = std::max(0.0, cached_frame_timeout_ms.load(std::memory_order_relaxed));
            if (const auto override_ms = envDouble("DELTA_CAPTURE_CACHED_TIMEOUT_MS"); override_ms.has_value()) {
                timeout_ms = std::max(0.0, *override_ms);
            }
        }
        return timeout_ms;
    }

    void initialize() {
        if (initialized) {
            return;
        }

        ComPtr<IDXGIFactory1> factory;
        checkHr(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "CreateDXGIFactory1");
        checkHr(factory->EnumAdapters1(config.capture_device_idx, &adapter), "EnumAdapters1");

        DXGI_ADAPTER_DESC1 adapter_desc{};
        checkHr(adapter->GetDesc1(&adapter_desc), "IDXGIAdapter1::GetDesc1");
        adapter_name = wideToUtf8(adapter_desc.Description);

        checkHr(adapter->EnumOutputs(config.capture_output_idx, &output), "EnumOutputs");
        checkHr(output.As(&output1), "QueryInterface(IDXGIOutput1)");
        checkHr(output->GetDesc(&output_desc), "IDXGIOutput::GetDesc");

        output_name = wideToUtf8(output_desc.DeviceName);
        desktop_width = output_desc.DesktopCoordinates.right - output_desc.DesktopCoordinates.left;
        desktop_height = output_desc.DesktopCoordinates.bottom - output_desc.DesktopCoordinates.top;
        if (desktop_width <= 0 || desktop_height <= 0) {
            throw std::runtime_error("Desktop output has invalid dimensions.");
        }

        constexpr D3D_FEATURE_LEVEL levels[] = {
            D3D_FEATURE_LEVEL_11_1,
            D3D_FEATURE_LEVEL_11_0,
        };
        D3D_FEATURE_LEVEL selected = D3D_FEATURE_LEVEL_11_0;
        const UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
        checkHr(
            D3D11CreateDevice(
                adapter.Get(),
                D3D_DRIVER_TYPE_UNKNOWN,
                nullptr,
                flags,
                levels,
                static_cast<UINT>(std::size(levels)),
                D3D11_SDK_VERSION,
                &device,
                &selected,
                &context
            ),
            "D3D11CreateDevice"
        );

        recreateDuplication();
        initialized = true;

        if (config.debug_log) {
            std::cout
                << "[capture] desktop duplication ready on "
                << adapter_name << " " << output_name
                << " (" << desktop_width << "x" << desktop_height << ")\n";
        }
    }

    void recreateDuplication() {
        duplication.Reset();
        checkHr(output1->DuplicateOutput(device.Get(), &duplication), "IDXGIOutput1::DuplicateOutput");
    }

    void close() {
#if defined(DELTA_WITH_CUDA_PIPELINE)
        releaseCudaResources();
#endif
        crop_staging.Reset();
        cached_desktop.Reset();
        duplication.Reset();
        context.Reset();
        device.Reset();
        output1.Reset();
        output.Reset();
        adapter.Reset();
        initialized = false;
        has_cached_desktop = false;
        cached_desktop_frame_ready = {};
        desktop_width = 0;
        desktop_height = 0;
    }

    void ensureCropStaging(const CaptureRegion& region) {
        if (
            crop_staging != nullptr
            && crop_region.width == region.width
            && crop_region.height == region.height
        ) {
            return;
        }

        D3D11_TEXTURE2D_DESC desc{};
        desc.Width = static_cast<UINT>(region.width);
        desc.Height = static_cast<UINT>(region.height);
        desc.MipLevels = 1;
        desc.ArraySize = 1;
        desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        desc.SampleDesc.Count = 1;
        desc.Usage = D3D11_USAGE_STAGING;
        desc.BindFlags = 0;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        desc.MiscFlags = 0;

        crop_staging.Reset();
        checkHr(device->CreateTexture2D(&desc, nullptr, &crop_staging), "CreateTexture2D(crop staging)");
        crop_region = region;
    }

    void ensureCachedDesktop(ID3D11Texture2D* frame_texture) {
        if (!config.capture_video_mode || frame_texture == nullptr) {
            return;
        }

        D3D11_TEXTURE2D_DESC source_desc{};
        frame_texture->GetDesc(&source_desc);
        if (
            cached_desktop != nullptr
            && cached_desktop_width == static_cast<int>(source_desc.Width)
            && cached_desktop_height == static_cast<int>(source_desc.Height)
            && cached_desktop_format == source_desc.Format
        ) {
            return;
        }

        D3D11_TEXTURE2D_DESC desc = source_desc;
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = 0;
        desc.CPUAccessFlags = 0;
        desc.MiscFlags = 0;

        cached_desktop.Reset();
        checkHr(device->CreateTexture2D(&desc, nullptr, &cached_desktop), "CreateTexture2D(cached desktop)");
        cached_desktop_width = static_cast<int>(desc.Width);
        cached_desktop_height = static_cast<int>(desc.Height);
        cached_desktop_format = desc.Format;
    }

#if defined(DELTA_WITH_CUDA_PIPELINE)
    static constexpr std::size_t kGpuFrameBufferCount = 3;

    struct GpuFrameBufferSlot {
        ComPtr<ID3D11Texture2D> interop_texture;
        cudaGraphicsResource_t resource = nullptr;
        void* bgra_device = nullptr;
        std::size_t bgra_pitch = 0;
        cudaEvent_t ready_event = nullptr;
        CaptureRegion region{};
        std::shared_ptr<std::atomic_bool> copy_pending = std::make_shared<std::atomic_bool>(false);
        std::shared_ptr<std::atomic_bool> in_use = std::make_shared<std::atomic_bool>(false);
    };

    void initializeCudaInterop() {
        if (cuda_ready) {
            return;
        }

        checkCuda(cudaD3D11GetDevice(&cuda_device, adapter.Get()), "cudaD3D11GetDevice");
        checkCuda(cudaSetDevice(cuda_device), "cudaSetDevice");
        if (consumer_cuda_stream == nullptr) {
            checkCuda(cudaStreamCreateWithFlags(&cuda_stream, cudaStreamNonBlocking), "cudaStreamCreate");
        }
        cuda_ready = true;
    }

    cudaStream_t activeCudaStream() const {
        return consumer_cuda_stream != nullptr ? static_cast<cudaStream_t>(consumer_cuda_stream) : cuda_stream;
    }

    bool gpuSlotDimensionsMatch(const CaptureRegion& region) const {
        return gpu_slots_initialized
            && gpu_slots_region.width == region.width
            && gpu_slots_region.height == region.height;
    }

    bool gpuSlotCopyPendingLocked(GpuFrameBufferSlot& slot) {
        return slot.copy_pending && slot.copy_pending->load(std::memory_order_acquire);
    }

    bool gpuSlotBusyLocked(GpuFrameBufferSlot& slot) {
        if (slot.in_use && slot.in_use->load(std::memory_order_acquire)) {
            return true;
        }
        return gpuSlotCopyPendingLocked(slot);
    }

    bool anyGpuSlotInUseLocked() {
        for (const auto& slot : gpu_slots) {
            if (slot.in_use && slot.in_use->load(std::memory_order_acquire)) {
                return true;
            }
        }
        for (auto& slot : gpu_slots) {
            if (gpuSlotCopyPendingLocked(slot)) {
                return true;
            }
        }
        return false;
    }

    void destroyGpuSlot(GpuFrameBufferSlot& slot) {
        if (slot.resource != nullptr) {
            cudaGraphicsUnregisterResource(slot.resource);
            slot.resource = nullptr;
        }
        if (slot.bgra_device != nullptr) {
            cudaFree(slot.bgra_device);
            slot.bgra_device = nullptr;
        }
        if (slot.ready_event != nullptr) {
            cudaEventDestroy(slot.ready_event);
            slot.ready_event = nullptr;
        }
        slot.interop_texture.Reset();
        slot.bgra_pitch = 0;
        slot.region = {};
        if (slot.copy_pending) {
            slot.copy_pending->store(false, std::memory_order_release);
        }
        if (slot.in_use) {
            slot.in_use->store(false, std::memory_order_release);
        }
    }

    void destroyGpuSlotsLocked() {
        for (auto& slot : gpu_slots) {
            destroyGpuSlot(slot);
        }
        gpu_slots_initialized = false;
        gpu_slots_region = {};
        next_gpu_slot = 0;
    }

    void createGpuSlot(GpuFrameBufferSlot& slot, const CaptureRegion& region) {
        destroyGpuSlot(slot);

        D3D11_TEXTURE2D_DESC desc{};
        desc.Width = static_cast<UINT>(region.width);
        desc.Height = static_cast<UINT>(region.height);
        desc.MipLevels = 1;
        desc.ArraySize = 1;
        desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        desc.SampleDesc.Count = 1;
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        desc.CPUAccessFlags = 0;
        desc.MiscFlags = 0;
        checkHr(device->CreateTexture2D(&desc, nullptr, &slot.interop_texture), "CreateTexture2D(cuda interop)");
        checkCuda(
            cudaGraphicsD3D11RegisterResource(&slot.resource, slot.interop_texture.Get(), cudaGraphicsRegisterFlagsNone),
            "cudaGraphicsD3D11RegisterResource");
        checkCuda(
            cudaMallocPitch(
                &slot.bgra_device,
                &slot.bgra_pitch,
                static_cast<std::size_t>(region.width) * sizeof(std::uint32_t),
                static_cast<std::size_t>(region.height)),
            "cudaMallocPitch(cuda bgra)");
        checkCuda(cudaEventCreateWithFlags(&slot.ready_event, cudaEventDisableTiming), "cudaEventCreate(capture ready)");
        slot.region = region;
        slot.in_use->store(false, std::memory_order_release);
    }

    GpuFrameBufferSlot* acquireGpuSlot(const CaptureRegion& region) {
        initializeCudaInterop();

        std::lock_guard<std::mutex> lock(gpu_slots_mutex);
        if (!gpuSlotDimensionsMatch(region)) {
            if (anyGpuSlotInUseLocked()) {
                return nullptr;
            }
            destroyGpuSlotsLocked();
            for (auto& slot : gpu_slots) {
                createGpuSlot(slot, region);
            }
            gpu_slots_region = region;
            gpu_slots_initialized = true;
        }

        for (std::size_t offset = 0; offset < gpu_slots.size(); ++offset) {
            const std::size_t index = (next_gpu_slot + offset) % gpu_slots.size();
            auto& slot = gpu_slots[index];
            if (gpuSlotBusyLocked(slot)) {
                continue;
            }
            bool expected = false;
            if (slot.in_use->compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
                next_gpu_slot = (index + 1U) % gpu_slots.size();
                return &slot;
            }
        }
        return nullptr;
    }

    std::shared_ptr<void> makeGpuSlotLease(GpuFrameBufferSlot& slot) {
        auto in_use = slot.in_use;
        return std::shared_ptr<void>(nullptr, [in_use](void*) {
            in_use->store(false, std::memory_order_release);
        });
    }

    void releaseCudaResources() {
        const cudaStream_t stream_to_sync = cuda_ready ? activeCudaStream() : nullptr;
        if (stream_to_sync != nullptr) {
            cudaStreamSynchronize(stream_to_sync);
        }

        {
            std::lock_guard<std::mutex> lock(gpu_slots_mutex);
            if (anyGpuSlotInUseLocked()) {
                return;
            }
            destroyGpuSlotsLocked();
        }

        if (cuda_stream != nullptr) {
            cudaStreamDestroy(cuda_stream);
            cuda_stream = nullptr;
        }
        cuda_device = -1;
        cuda_ready = false;
    }

    std::optional<GpuFramePacket> copyRegionToGpuPacket(
        ID3D11Texture2D* source_texture,
        const CaptureRegion& region,
        const SteadyClock::time_point acquire_started,
        const SteadyClock::time_point frame_ready,
        const bool used_cached_frame) {
        GpuFrameBufferSlot* slot = acquireGpuSlot(region);
        if (slot == nullptr) {
            return std::nullopt;
        }
        std::shared_ptr<void> slot_lease = makeGpuSlotLease(*slot);

        CaptureTimings timings{};
        timings.used_cached_frame = used_cached_frame;

        D3D11_BOX source_box{};
        source_box.left = static_cast<UINT>(region.left);
        source_box.top = static_cast<UINT>(region.top);
        source_box.front = 0;
        source_box.right = static_cast<UINT>(region.left + region.width);
        source_box.bottom = static_cast<UINT>(region.top + region.height);
        source_box.back = 1;

        const auto d3d_copy_start = SteadyClock::now();
        context->CopySubresourceRegion(
            slot->interop_texture.Get(),
            0,
            0,
            0,
            0,
            source_texture,
            0,
            &source_box);
        const auto d3d_copy_end = SteadyClock::now();
        timings.d3d_copy_s = secondsSince(d3d_copy_start, d3d_copy_end);

        const auto d3d_sync_start = SteadyClock::now();
        context->Flush();
        const auto d3d_sync_end = SteadyClock::now();
        timings.d3d_sync_s = secondsSince(d3d_sync_start, d3d_sync_end);

        const cudaStream_t stream = activeCudaStream();

        bool mapped = false;
        ScopeExit unmap_cuda([&]() {
            if (mapped && slot->resource != nullptr) {
                cudaGraphicsUnmapResources(1, &slot->resource, stream);
            }
        });

        const auto cuda_copy_start = SteadyClock::now();
        checkCuda(cudaGraphicsMapResources(1, &slot->resource, stream), "cudaGraphicsMapResources");
        mapped = true;

        cudaArray_t mapped_array = nullptr;
        checkCuda(
            cudaGraphicsSubResourceGetMappedArray(&mapped_array, slot->resource, 0, 0),
            "cudaGraphicsSubResourceGetMappedArray");
        checkCuda(
            cudaMemcpy2DFromArrayAsync(
                slot->bgra_device,
                slot->bgra_pitch,
                mapped_array,
                0,
                0,
                static_cast<std::size_t>(region.width) * sizeof(std::uint32_t),
                static_cast<std::size_t>(region.height),
                cudaMemcpyDeviceToDevice,
                stream),
            "cudaMemcpy2DFromArrayAsync(cuda bgra)");
        checkCuda(cudaGraphicsUnmapResources(1, &slot->resource, stream), "cudaGraphicsUnmapResources");
        mapped = false;
        checkCuda(cudaEventRecord(slot->ready_event, stream), "cudaEventRecord(capture ready)");
        slot->copy_pending->store(true, std::memory_order_release);
        std::unique_ptr<std::shared_ptr<std::atomic_bool>> copy_pending_holder =
            std::make_unique<std::shared_ptr<std::atomic_bool>>(slot->copy_pending);
        const cudaError_t host_status = cudaLaunchHostFunc(stream, markGpuSlotCopyComplete, copy_pending_holder.get());
        if (host_status != cudaSuccess) {
            slot->copy_pending->store(false, std::memory_order_release);
            checkCuda(host_status, "cudaLaunchHostFunc(capture ready)");
        }
        copy_pending_holder.release();
        timings.cuda_copy_s = secondsSince(cuda_copy_start, SteadyClock::now());

        GpuFramePacket packet{};
        packet.device_ptr = slot->bgra_device;
        packet.pitch_bytes = slot->bgra_pitch;
        packet.width = region.width;
        packet.height = region.height;
        packet.pixel_format = PixelFormat::Bgra8;
        packet.capture = region;
        packet.acquire_started = acquire_started;
        packet.frame_ready = frame_ready;
        packet.capture_done = SteadyClock::now();
        packet.frame_time = frame_ready;
        packet.capture_time = SystemClock::now();
        if (used_cached_frame) {
            timings.cached_reuse_s = secondsSince(frame_ready, packet.capture_done);
        }
        packet.timings = timings;
        packet.cuda_stream = stream;
        packet.ready_event = slot->ready_event;
        packet.lifetime = std::move(slot_lease);
        return packet;
    }
#endif

    FramePacket copyRegionToPacket(
        ID3D11Texture2D* source_texture,
        const CaptureRegion& region,
        const SteadyClock::time_point acquire_started,
        const SteadyClock::time_point frame_ready,
        const bool used_cached_frame) {
        ensureCropStaging(region);
        CaptureTimings timings{};
        timings.used_cached_frame = used_cached_frame;

        D3D11_BOX source_box{};
        source_box.left = static_cast<UINT>(region.left);
        source_box.top = static_cast<UINT>(region.top);
        source_box.front = 0;
        source_box.right = static_cast<UINT>(region.left + region.width);
        source_box.bottom = static_cast<UINT>(region.top + region.height);
        source_box.back = 1;

        const auto d3d_copy_start = SteadyClock::now();
        context->CopySubresourceRegion(
            crop_staging.Get(),
            0,
            0,
            0,
            0,
            source_texture,
            0,
            &source_box
        );
        timings.d3d_copy_s = secondsSince(d3d_copy_start, SteadyClock::now());

        const auto cpu_copy_start = SteadyClock::now();
        D3D11_MAPPED_SUBRESOURCE mapped{};
        checkHr(context->Map(crop_staging.Get(), 0, D3D11_MAP_READ, 0, &mapped), "Map(crop staging)");
        ScopeExit unmap([&]() { context->Unmap(crop_staging.Get(), 0); });

        FramePacket packet{};
        packet.width = region.width;
        packet.height = region.height;
        packet.capture = region;
        packet.acquire_started = acquire_started;
        packet.frame_ready = frame_ready;
        packet.bgr.resize(static_cast<std::size_t>(region.width) * static_cast<std::size_t>(region.height) * 3ULL);

        const auto* src_base = static_cast<const std::uint8_t*>(mapped.pData);
        for (int y = 0; y < region.height; ++y) {
            const auto* src_row = src_base + (static_cast<std::size_t>(y) * mapped.RowPitch);
            auto* dst_row = packet.bgr.data() + (static_cast<std::size_t>(y) * static_cast<std::size_t>(region.width) * 3ULL);
            for (int x = 0; x < region.width; ++x) {
                const std::size_t src = static_cast<std::size_t>(x) * 4ULL;
                const std::size_t dst = static_cast<std::size_t>(x) * 3ULL;
                dst_row[dst + 0] = src_row[src + 0];
                dst_row[dst + 1] = src_row[src + 1];
                dst_row[dst + 2] = src_row[src + 2];
            }
        }

        packet.capture_done = SteadyClock::now();
        packet.frame_time = frame_ready;
        packet.capture_time = SystemClock::now();
        timings.cpu_copy_s = secondsSince(cpu_copy_start, packet.capture_done);
        if (used_cached_frame) {
            timings.cached_reuse_s = secondsSince(frame_ready, packet.capture_done);
        }
        packet.timings = timings;
        return packet;
    }

    std::optional<FramePacket> grab(const CaptureRegion& requested_region) {
        initialize();
        const CaptureRegion region = clampRegionToDesktop(requested_region, desktop_width, desktop_height);
        const auto acquire_started = SteadyClock::now();

        DXGI_OUTDUPL_FRAME_INFO frame_info{};
        ComPtr<IDXGIResource> frame_resource;
        HRESULT hr = duplication->AcquireNextFrame(
            dxgiTimeoutMs(effectiveAcquireTimeoutMs()),
            &frame_info,
            &frame_resource
        );

        if (hr == DXGI_ERROR_ACCESS_LOST) {
            recreateDuplication();
            hr = duplication->AcquireNextFrame(
                dxgiTimeoutMs(effectiveAcquireTimeoutMs()),
                &frame_info,
                &frame_resource
            );
        }
        const auto frame_ready = SteadyClock::now();

        if (hr == DXGI_ERROR_WAIT_TIMEOUT) {
            if (!config.capture_video_mode || !has_cached_desktop || cached_desktop == nullptr) {
                return std::nullopt;
            }
            const SteadyClock::time_point cached_frame_ready = cached_desktop_frame_ready == SteadyClock::time_point{}
                ? frame_ready
                : cached_desktop_frame_ready;
            FramePacket packet = copyRegionToPacket(cached_desktop.Get(), region, acquire_started, cached_frame_ready, true);
            packet.timings.acquire_s = secondsSince(acquire_started, frame_ready);
            return packet;
        }

        checkHr(hr, "AcquireNextFrame");

        bool frame_acquired = true;
        ScopeExit release_frame([&]() {
            if (frame_acquired && duplication != nullptr) {
                duplication->ReleaseFrame();
            }
        });

        ComPtr<ID3D11Texture2D> frame_texture;
        checkHr(frame_resource.As(&frame_texture), "QueryInterface(ID3D11Texture2D)");

        ID3D11Texture2D* source_texture = frame_texture.Get();
        if (config.capture_video_mode) {
            // Keep one full desktop copy so repeated grab() calls can crop different
            // regions even when Desktop Duplication has no fresh frame available.
            // This preserves Python's "video mode" behavior while keeping the future
            // CUDA/D3D11 interop optimization localized to this module.
            ensureCachedDesktop(frame_texture.Get());
            context->CopyResource(cached_desktop.Get(), frame_texture.Get());
            source_texture = cached_desktop.Get();
            has_cached_desktop = true;
            cached_desktop_frame_ready = frame_ready;
        }

        FramePacket packet = copyRegionToPacket(source_texture, region, acquire_started, frame_ready, false);
        packet.timings.acquire_s = secondsSince(acquire_started, frame_ready);
        checkHr(duplication->ReleaseFrame(), "ReleaseFrame");
        frame_acquired = false;
        return packet;
    }

#if defined(DELTA_WITH_CUDA_PIPELINE)
    std::optional<GpuFramePacket> grabGpu(const CaptureRegion& requested_region) {
        if (gpu_capture_disabled) {
            return std::nullopt;
        }

        initialize();
        const CaptureRegion region = clampRegionToDesktop(requested_region, desktop_width, desktop_height);
        const auto acquire_started = SteadyClock::now();

        DXGI_OUTDUPL_FRAME_INFO frame_info{};
        ComPtr<IDXGIResource> frame_resource;
        HRESULT hr = duplication->AcquireNextFrame(
            dxgiTimeoutMs(effectiveAcquireTimeoutMs()),
            &frame_info,
            &frame_resource);

        if (hr == DXGI_ERROR_ACCESS_LOST) {
            recreateDuplication();
            hr = duplication->AcquireNextFrame(
                dxgiTimeoutMs(effectiveAcquireTimeoutMs()),
                &frame_info,
                &frame_resource);
        }
        const auto frame_ready = SteadyClock::now();

        if (hr == DXGI_ERROR_WAIT_TIMEOUT) {
            if (!config.capture_video_mode || !has_cached_desktop || cached_desktop == nullptr) {
                return std::nullopt;
            }
            const SteadyClock::time_point cached_frame_ready = cached_desktop_frame_ready == SteadyClock::time_point{}
                ? frame_ready
                : cached_desktop_frame_ready;
            std::optional<GpuFramePacket> packet =
                copyRegionToGpuPacket(cached_desktop.Get(), region, acquire_started, cached_frame_ready, true);
            if (!packet.has_value()) {
                return std::nullopt;
            }
            packet->timings.acquire_s = secondsSince(acquire_started, frame_ready);
            return packet;
        }

        checkHr(hr, "AcquireNextFrame");

        bool frame_acquired = true;
        ScopeExit release_frame([&]() {
            if (frame_acquired && duplication != nullptr) {
                duplication->ReleaseFrame();
            }
        });

        ComPtr<ID3D11Texture2D> frame_texture;
        checkHr(frame_resource.As(&frame_texture), "QueryInterface(ID3D11Texture2D)");

        ID3D11Texture2D* source_texture = frame_texture.Get();
        if (config.capture_video_mode) {
            ensureCachedDesktop(frame_texture.Get());
            context->CopyResource(cached_desktop.Get(), frame_texture.Get());
            source_texture = cached_desktop.Get();
            has_cached_desktop = true;
            cached_desktop_frame_ready = frame_ready;
        }

        std::optional<GpuFramePacket> packet = copyRegionToGpuPacket(source_texture, region, acquire_started, frame_ready, false);
        if (!packet.has_value()) {
            return std::nullopt;
        }
        packet->timings.acquire_s = secondsSince(acquire_started, frame_ready);
        checkHr(duplication->ReleaseFrame(), "ReleaseFrame");
        frame_acquired = false;
        return packet;
    }
#endif

    StaticConfig config{};
    bool initialized = false;
    bool has_cached_desktop = false;
    SteadyClock::time_point cached_desktop_frame_ready{};
    int desktop_width = 0;
    int desktop_height = 0;
    int cached_desktop_width = 0;
    int cached_desktop_height = 0;
    DXGI_FORMAT cached_desktop_format = DXGI_FORMAT_UNKNOWN;
    CaptureRegion crop_region{};
    ComPtr<IDXGIAdapter1> adapter;
    ComPtr<IDXGIOutput> output;
    ComPtr<IDXGIOutput1> output1;
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    ComPtr<IDXGIOutputDuplication> duplication;
    ComPtr<ID3D11Texture2D> cached_desktop;
    ComPtr<ID3D11Texture2D> crop_staging;
    std::atomic<double> cached_frame_timeout_ms{1.0};
#if defined(DELTA_WITH_CUDA_PIPELINE)
    bool cuda_ready = false;
    int cuda_device = -1;
    cudaStream_t cuda_stream = nullptr;
    void* consumer_cuda_stream = nullptr;
    std::array<GpuFrameBufferSlot, kGpuFrameBufferCount> gpu_slots{};
    CaptureRegion gpu_slots_region{};
    bool gpu_slots_initialized = false;
    std::size_t next_gpu_slot = 0;
    std::mutex gpu_slots_mutex;
    bool gpu_capture_disabled = false;
    bool gpu_capture_failure_logged = false;
#endif
    DXGI_OUTPUT_DESC output_desc{};
    std::string adapter_name;
    std::string output_name;
};

DesktopDuplicationCapture::DesktopDuplicationCapture(const StaticConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}

DesktopDuplicationCapture::~DesktopDuplicationCapture() = default;

DesktopDuplicationCapture::DesktopDuplicationCapture(DesktopDuplicationCapture&&) noexcept = default;

DesktopDuplicationCapture& DesktopDuplicationCapture::operator=(DesktopDuplicationCapture&&) noexcept = default;

std::optional<FramePacket> DesktopDuplicationCapture::grab(const CaptureRegion& region) {
    if (!impl_) {
        return std::nullopt;
    }

    try {
        return impl_->grab(region);
    } catch (const std::exception& ex) {
        if (impl_->config.debug_log) {
            std::cerr << "[capture] " << ex.what() << "\n";
        }
        return std::nullopt;
    }
}

std::optional<GpuFramePacket> DesktopDuplicationCapture::grabGpu(const CaptureRegion& region) {
    if (!impl_) {
        return std::nullopt;
    }

#if defined(DELTA_WITH_CUDA_PIPELINE)
    try {
        return impl_->grabGpu(region);
    } catch (const std::exception& ex) {
        impl_->releaseCudaResources();
        impl_->gpu_capture_disabled = true;
        if (impl_->config.debug_log && !impl_->gpu_capture_failure_logged) {
            std::cerr << "[capture] " << ex.what() << "\n";
            std::cerr << "[capture] GPU interop disabled for this run.\n";
            impl_->gpu_capture_failure_logged = true;
        }
        return std::nullopt;
    }
#else
    (void)region;
    return std::nullopt;
#endif
}

void DesktopDuplicationCapture::setGpuConsumerStream(void* stream) {
#if defined(DELTA_WITH_CUDA_PIPELINE)
    if (impl_) {
        impl_->consumer_cuda_stream = stream;
    }
#else
    (void)stream;
#endif
}

void DesktopDuplicationCapture::setCachedFrameTimeoutMs(const double timeout_ms) {
    if (impl_) {
        impl_->cached_frame_timeout_ms.store(std::max(0.0, timeout_ms), std::memory_order_relaxed);
    }
}

void DesktopDuplicationCapture::close() {
    if (impl_) {
        impl_->close();
    }
}

std::unique_ptr<ICaptureSource> makeDefaultCaptureSource(const StaticConfig& config) {
    return std::make_unique<DesktopDuplicationCapture>(config);
}

}  // namespace delta
