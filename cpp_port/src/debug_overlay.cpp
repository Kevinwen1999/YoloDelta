#include "delta/debug_overlay.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

#if defined(_WIN32)
#include <d3d11.h>
#include <dxgi.h>
#include <windows.h>
#include <wrl/client.h>

#include "imgui.h"
#include "imgui_impl_dx11.h"
#include "imgui_impl_win32.h"

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam);
#endif

namespace delta {

namespace {

#if defined(_WIN32)

using Microsoft::WRL::ComPtr;

constexpr wchar_t kOverlayClassName[] = L"DeltaDebugOverlayWindow";
constexpr wchar_t kOverlayWindowTitle[] = L"Delta Detection Overlay";
constexpr UINT kOverlaySyncMessage = WM_APP + 2;
constexpr UINT kOverlayRefreshMs = 16;

#ifndef WDA_EXCLUDEFROMCAPTURE
#define WDA_EXCLUDEFROMCAPTURE 0x00000011
#endif

std::string formatLastErrorMessage(const DWORD error) {
    if (error == 0) {
        return "ok";
    }

    LPSTR buffer = nullptr;
    const DWORD flags = FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS;
    const DWORD size = FormatMessageA(
        flags,
        nullptr,
        error,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        reinterpret_cast<LPSTR>(&buffer),
        0,
        nullptr);
    std::string message = "Win32 error " + std::to_string(error);
    if (size > 0 && buffer != nullptr) {
        message += ": ";
        message.append(buffer, buffer + size);
        while (!message.empty() && (message.back() == '\r' || message.back() == '\n')) {
            message.pop_back();
        }
    }
    if (buffer != nullptr) {
        LocalFree(buffer);
    }
    return message;
}

class Win32DebugOverlayWindow final : public DebugOverlayWindow {
public:
    explicit Win32DebugOverlayWindow(StaticConfig config)
        : config_(std::move(config)) {}

    ~Win32DebugOverlayWindow() override {
        stop();
    }

    void start() override {
        std::unique_lock<std::mutex> lock(window_mutex_);
        if (thread_.joinable() || init_failed_) {
            return;
        }
        ready_ = false;
        stopping_.store(false, std::memory_order_relaxed);
        thread_ = std::thread([this]() { threadMain(); });
        ready_cv_.wait(lock, [this]() { return ready_; });
    }

    void stop() override {
        HWND hwnd = nullptr;
        {
            std::lock_guard<std::mutex> lock(window_mutex_);
            if (!thread_.joinable()) {
                return;
            }
            stopping_.store(true, std::memory_order_relaxed);
            hwnd = hwnd_;
        }
        if (hwnd != nullptr) {
            PostMessageW(hwnd, WM_CLOSE, 0, 0);
        }
        if (thread_.joinable()) {
            thread_.join();
        }
        {
            std::lock_guard<std::mutex> lock(window_mutex_);
            hwnd_ = nullptr;
            ready_ = false;
            stopping_.store(false, std::memory_order_relaxed);
        }
        {
            std::lock_guard<std::mutex> lock(snapshot_mutex_);
            snapshot_ = {};
            sequence_ = 0;
        }
    }

    void setEnabled(const bool enabled) override {
        {
            std::lock_guard<std::mutex> lock(snapshot_mutex_);
            enabled_ = enabled;
            if (!enabled_) {
                snapshot_ = {};
                sequence_ = 0;
            }
        }

        HWND hwnd = nullptr;
        {
            std::lock_guard<std::mutex> lock(window_mutex_);
            hwnd = hwnd_;
        }
        if (hwnd != nullptr) {
            PostMessageW(hwnd, kOverlaySyncMessage, 0, 0);
        }
    }

    void publish(DebugPreviewSnapshot snapshot) override {
        {
            std::lock_guard<std::mutex> lock(snapshot_mutex_);
            if (!enabled_ || init_failed_) {
                return;
            }
            snapshot.sequence = ++sequence_;
            snapshot_ = std::move(snapshot);
        }

        HWND hwnd = nullptr;
        {
            std::lock_guard<std::mutex> lock(window_mutex_);
            hwnd = hwnd_;
        }
        if (hwnd != nullptr) {
            PostMessageW(hwnd, kOverlaySyncMessage, 0, 0);
        }
    }

private:
    static LRESULT CALLBACK windowProcSetup(HWND hwnd, UINT message, WPARAM wparam, LPARAM lparam) {
        Win32DebugOverlayWindow* self = nullptr;
        if (message == WM_NCCREATE) {
            const auto* create = reinterpret_cast<const CREATESTRUCTW*>(lparam);
            self = static_cast<Win32DebugOverlayWindow*>(create->lpCreateParams);
            SetWindowLongPtrW(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(self));
        } else {
            self = reinterpret_cast<Win32DebugOverlayWindow*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));
        }
        if (self != nullptr) {
            return self->windowProc(hwnd, message, wparam, lparam);
        }
        return DefWindowProcW(hwnd, message, wparam, lparam);
    }

    LRESULT windowProc(HWND hwnd, UINT message, WPARAM wparam, LPARAM lparam) {
        if (imgui_initialized_ && ImGui_ImplWin32_WndProcHandler(hwnd, message, wparam, lparam)) {
            return 1;
        }

        switch (message) {
        case WM_NCCREATE:
            return DefWindowProcW(hwnd, message, wparam, lparam);
        case WM_CREATE:
            syncWindowState(hwnd);
            return 0;
        case WM_MOUSEACTIVATE:
            return MA_NOACTIVATE;
        case WM_ERASEBKGND:
            return 1;
        case WM_SIZE:
            if (device_ && wparam != SIZE_MINIMIZED) {
                pending_width_ = LOWORD(lparam);
                pending_height_ = HIWORD(lparam);
            }
            return 0;
        case kOverlaySyncMessage:
            syncWindowState(hwnd);
            return 0;
        case WM_CLOSE:
            if (!stopping_.load(std::memory_order_relaxed)) {
                ShowWindow(hwnd, SW_HIDE);
                return 0;
            }
            DestroyWindow(hwnd);
            return 0;
        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
        default:
            return DefWindowProcW(hwnd, message, wparam, lparam);
        }
    }

    void threadMain() {
        const HINSTANCE instance = GetModuleHandleW(nullptr);
        WNDCLASSEXW wc{};
        wc.cbSize = sizeof(wc);
        wc.lpfnWndProc = &Win32DebugOverlayWindow::windowProcSetup;
        wc.hInstance = instance;
        wc.hCursor = LoadCursorW(nullptr, MAKEINTRESOURCEW(32512));
        wc.lpszClassName = kOverlayClassName;
        const ATOM class_atom = RegisterClassExW(&wc);
        if (class_atom == 0 && GetLastError() != ERROR_CLASS_ALREADY_EXISTS && config_.debug_log) {
            std::cerr << "[debug_overlay] RegisterClassExW failed: " << formatLastErrorMessage(GetLastError()) << "\n";
        }

        HWND hwnd = CreateWindowExW(
            WS_EX_TOPMOST | WS_EX_TOOLWINDOW | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE,
            kOverlayClassName,
            kOverlayWindowTitle,
            WS_POPUP,
            0,
            0,
            std::max(1, config_.screen_w),
            std::max(1, config_.screen_h),
            nullptr,
            nullptr,
            instance,
            this);
        if (hwnd == nullptr) {
            if (config_.debug_log) {
                std::cerr << "[debug_overlay] CreateWindowExW failed: " << formatLastErrorMessage(GetLastError()) << "\n";
            }
            init_failed_ = true;
            signalReady(nullptr);
            return;
        }

        if (!initializeD3D(hwnd) || !initializeImGui(hwnd)) {
            init_failed_ = true;
            shutdownImGui();
            cleanupGraphics();
            DestroyWindow(hwnd);
            signalReady(nullptr);
            return;
        }

        signalReady(hwnd);

        bool running = true;
        while (running) {
            MSG msg{};
            while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
                if (msg.message == WM_QUIT) {
                    running = false;
                    break;
                }
                TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
            if (!running) {
                break;
            }

            if (pending_width_ > 0 && pending_height_ > 0) {
                resizeSwapChain();
            }

            DebugPreviewSnapshot snapshot{};
            bool enabled = false;
            {
                std::lock_guard<std::mutex> lock(snapshot_mutex_);
                enabled = enabled_;
                snapshot = snapshot_;
            }

            if (!enabled) {
                WaitMessage();
                continue;
            }

            renderFrame(snapshot);
            std::this_thread::sleep_for(std::chrono::milliseconds(kOverlayRefreshMs));
        }

        shutdownImGui();
        cleanupGraphics();
        {
            std::lock_guard<std::mutex> lock(window_mutex_);
            hwnd_ = nullptr;
        }
    }

    void signalReady(HWND hwnd) {
        {
            std::lock_guard<std::mutex> lock(window_mutex_);
            hwnd_ = hwnd;
            ready_ = true;
        }
        ready_cv_.notify_all();
    }

    bool initializeD3D(HWND hwnd) {
        DXGI_SWAP_CHAIN_DESC swap_chain_desc{};
        swap_chain_desc.BufferCount = 2;
        swap_chain_desc.BufferDesc.Width = std::max(1, config_.screen_w);
        swap_chain_desc.BufferDesc.Height = std::max(1, config_.screen_h);
        swap_chain_desc.BufferDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        swap_chain_desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        swap_chain_desc.OutputWindow = hwnd;
        swap_chain_desc.SampleDesc.Count = 1;
        swap_chain_desc.Windowed = TRUE;
        swap_chain_desc.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

        UINT device_flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
        const D3D_FEATURE_LEVEL feature_levels[] = {
            D3D_FEATURE_LEVEL_11_0,
            D3D_FEATURE_LEVEL_10_1,
            D3D_FEATURE_LEVEL_10_0,
        };
        D3D_FEATURE_LEVEL feature_level{};
        const HRESULT hr = D3D11CreateDeviceAndSwapChain(
            nullptr,
            D3D_DRIVER_TYPE_HARDWARE,
            nullptr,
            device_flags,
            feature_levels,
            static_cast<UINT>(sizeof(feature_levels) / sizeof(feature_levels[0])),
            D3D11_SDK_VERSION,
            &swap_chain_desc,
            swap_chain_.GetAddressOf(),
            device_.GetAddressOf(),
            &feature_level,
            context_.GetAddressOf());
        if (FAILED(hr)) {
            if (config_.debug_log) {
                std::cerr << "[debug_overlay] D3D11CreateDeviceAndSwapChain failed: 0x"
                          << std::hex << static_cast<unsigned long>(hr) << std::dec << "\n";
            }
            return false;
        }

        return createRenderTarget();
    }

    bool createRenderTarget() {
        render_target_.Reset();
        if (!swap_chain_) {
            return false;
        }

        ComPtr<ID3D11Texture2D> back_buffer;
        const HRESULT hr = swap_chain_->GetBuffer(0, IID_PPV_ARGS(back_buffer.GetAddressOf()));
        if (FAILED(hr)) {
            if (config_.debug_log) {
                std::cerr << "[debug_overlay] swap-chain GetBuffer failed: 0x"
                          << std::hex << static_cast<unsigned long>(hr) << std::dec << "\n";
            }
            return false;
        }

        const HRESULT rtv_hr = device_->CreateRenderTargetView(back_buffer.Get(), nullptr, render_target_.GetAddressOf());
        if (FAILED(rtv_hr)) {
            if (config_.debug_log) {
                std::cerr << "[debug_overlay] CreateRenderTargetView failed: 0x"
                          << std::hex << static_cast<unsigned long>(rtv_hr) << std::dec << "\n";
            }
            return false;
        }
        return true;
    }

    bool initializeImGui(HWND hwnd) {
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGui::StyleColorsDark();
        ImGuiIO& io = ImGui::GetIO();
        io.IniFilename = nullptr;
        io.LogFilename = nullptr;

        if (!ImGui_ImplWin32_Init(hwnd)) {
            if (config_.debug_log) {
                std::cerr << "[debug_overlay] ImGui_ImplWin32_Init failed\n";
            }
            ImGui::DestroyContext();
            return false;
        }
        if (!ImGui_ImplDX11_Init(device_.Get(), context_.Get())) {
            if (config_.debug_log) {
                std::cerr << "[debug_overlay] ImGui_ImplDX11_Init failed\n";
            }
            ImGui_ImplWin32_Shutdown();
            ImGui::DestroyContext();
            return false;
        }

        imgui_initialized_ = true;
        syncWindowState(hwnd);
        return true;
    }

    void shutdownImGui() {
        if (!imgui_initialized_) {
            return;
        }
        ImGui_ImplDX11_Shutdown();
        ImGui_ImplWin32_Shutdown();
        ImGui::DestroyContext();
        imgui_initialized_ = false;
    }

    void cleanupGraphics() {
        render_target_.Reset();
        swap_chain_.Reset();
        context_.Reset();
        device_.Reset();
        pending_width_ = 0;
        pending_height_ = 0;
    }

    void resizeSwapChain() {
        if (!swap_chain_) {
            pending_width_ = 0;
            pending_height_ = 0;
            return;
        }

        const UINT width = pending_width_;
        const UINT height = pending_height_;
        pending_width_ = 0;
        pending_height_ = 0;

        render_target_.Reset();
        const HRESULT hr = swap_chain_->ResizeBuffers(0, width, height, DXGI_FORMAT_UNKNOWN, 0);
        if (FAILED(hr)) {
            if (config_.debug_log) {
                std::cerr << "[debug_overlay] ResizeBuffers failed: 0x"
                          << std::hex << static_cast<unsigned long>(hr) << std::dec << "\n";
            }
            return;
        }
        createRenderTarget();
    }

    void syncWindowState(HWND hwnd) {
        bool enabled = false;
        {
            std::lock_guard<std::mutex> lock(snapshot_mutex_);
            enabled = enabled_;
        }

        if (enabled) {
            SetLayeredWindowAttributes(hwnd, RGB(0, 0, 0), 255, LWA_ALPHA);
            SetWindowPos(
                hwnd,
                HWND_TOPMOST,
                0,
                0,
                std::max(1, config_.screen_w),
                std::max(1, config_.screen_h),
                SWP_NOACTIVATE | SWP_SHOWWINDOW);
            ShowWindow(hwnd, SW_SHOWNOACTIVATE);
            if (imgui_initialized_) {
                ImGui_ImplWin32_EnableAlphaCompositing(hwnd);
            }
            const BOOL affinity_ok = SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE);
            if (!affinity_ok && config_.debug_log && !affinity_warning_logged_) {
                affinity_warning_logged_ = true;
                std::cout << "[debug_overlay] capture affinity unavailable: "
                          << formatLastErrorMessage(GetLastError()) << "\n";
            }
        } else {
            ShowWindow(hwnd, SW_HIDE);
        }
    }

    void drawDetections(const DebugPreviewSnapshot& snapshot) {
        if (!snapshot.active) {
            return;
        }

        ImDrawList* draw_list = ImGui::GetForegroundDrawList();
        for (const auto& detection : snapshot.detections) {
            const float x1 = static_cast<float>(detection.bbox[0]);
            const float y1 = static_cast<float>(detection.bbox[1]);
            const float x2 = static_cast<float>(detection.bbox[2]);
            const float y2 = static_cast<float>(detection.bbox[3]);
            if (x2 <= x1 || y2 <= y1) {
                continue;
            }

            const ImU32 color = detection.selected ? IM_COL32(111, 231, 150, 255) : IM_COL32(243, 181, 95, 235);
            const float thickness = detection.selected ? 1.5F : 1.0F;
            draw_list->AddRect(ImVec2(x1, y1), ImVec2(x2, y2), color, 0.0F, 0, thickness);

            char label[64]{};
            std::snprintf(label, sizeof(label), "c%d %.2f%s", detection.cls, detection.conf, detection.selected ? " *" : "");
            const ImVec2 text_pos(x1 + 4.0F, std::max(2.0F, y1 - 20.0F));
            const ImVec2 text_size = ImGui::CalcTextSize(label);
            draw_list->AddRectFilled(
                ImVec2(text_pos.x - 3.0F, text_pos.y - 2.0F),
                ImVec2(text_pos.x + text_size.x + 3.0F, text_pos.y + text_size.y + 2.0F),
                IM_COL32(8, 12, 15, 190));
            draw_list->AddText(text_pos, color, label);
        }
    }

    void drawCaptureRegion(const DebugPreviewSnapshot& snapshot) {
        if (snapshot.capture_region.width <= 0 || snapshot.capture_region.height <= 0) {
            return;
        }

        const float left = static_cast<float>(snapshot.capture_region.left);
        const float top = static_cast<float>(snapshot.capture_region.top);
        const float right = left + static_cast<float>(snapshot.capture_region.width);
        const float bottom = top + static_cast<float>(snapshot.capture_region.height);
        if (right <= left || bottom <= top) {
            return;
        }

        ImDrawList* draw_list = ImGui::GetForegroundDrawList();
        const ImU32 color = IM_COL32(99, 186, 255, 220);
        draw_list->AddRect(ImVec2(left, top), ImVec2(right, bottom), color, 0.0F, 0, 1.0F);

        char label[48]{};
        std::snprintf(label, sizeof(label), "crop %dx%d", snapshot.capture_region.width, snapshot.capture_region.height);
        const ImVec2 text_pos(left + 4.0F, std::max(2.0F, top - 20.0F));
        const ImVec2 text_size = ImGui::CalcTextSize(label);
        draw_list->AddRectFilled(
            ImVec2(text_pos.x - 3.0F, text_pos.y - 2.0F),
            ImVec2(text_pos.x + text_size.x + 3.0F, text_pos.y + text_size.y + 2.0F),
            IM_COL32(8, 12, 15, 190));
        draw_list->AddText(text_pos, color, label);
    }

    void drawGuardRegion(const DebugPreviewSnapshot& snapshot) {
        if (!snapshot.guard_region.has_value()) {
            return;
        }

        const float left = static_cast<float>(snapshot.guard_region->left);
        const float top = static_cast<float>(snapshot.guard_region->top);
        const float right = left + static_cast<float>(snapshot.guard_region->width);
        const float bottom = top + static_cast<float>(snapshot.guard_region->height);
        if (right <= left || bottom <= top) {
            return;
        }

        ImDrawList* draw_list = ImGui::GetForegroundDrawList();
        const ImU32 color = IM_COL32(255, 122, 122, 220);
        draw_list->AddRect(ImVec2(left, top), ImVec2(right, bottom), color, 0.0F, 0, 1.0F);

        char label[48]{};
        std::snprintf(label, sizeof(label), "guard %dx%d", snapshot.guard_region->width, snapshot.guard_region->height);
        const ImVec2 text_pos(left + 4.0F, std::max(2.0F, top - 20.0F));
        const ImVec2 text_size = ImGui::CalcTextSize(label);
        draw_list->AddRectFilled(
            ImVec2(text_pos.x - 3.0F, text_pos.y - 2.0F),
            ImVec2(text_pos.x + text_size.x + 3.0F, text_pos.y + text_size.y + 2.0F),
            IM_COL32(8, 12, 15, 190));
        draw_list->AddText(text_pos, color, label);
    }

    void drawLeadTelemetry(const DebugPreviewSnapshot& snapshot) {
        if (!snapshot.predicted_point.has_value()) {
            return;
        }

        ImDrawList* draw_list = ImGui::GetForegroundDrawList();
        const ImVec2 predicted(
            snapshot.predicted_point->first,
            snapshot.predicted_point->second);
        const ImU32 predicted_color = snapshot.lead_active
            ? IM_COL32(255, 146, 96, 240)
            : IM_COL32(255, 122, 122, 220);
        draw_list->AddCircleFilled(predicted, 4.0F, IM_COL32(8, 12, 15, 220), 12);
        draw_list->AddCircle(predicted, 4.0F, predicted_color, 12, 1.5F);

        if (snapshot.lead_active && snapshot.detected_point.has_value()) {
            const ImVec2 detected(
                snapshot.detected_point->first,
                snapshot.detected_point->second);
            const ImU32 detected_color = snapshot.detected_point_stale
                ? IM_COL32(112, 172, 112, 180)
                : IM_COL32(111, 231, 150, 235);
            draw_list->AddCircleFilled(detected, 4.0F, IM_COL32(8, 12, 15, 220), 12);
            draw_list->AddCircle(detected, 4.0F, detected_color, 12, 1.5F);
            draw_list->AddLine(
                detected,
                predicted,
                snapshot.detected_point_stale ? IM_COL32(160, 138, 108, 180) : IM_COL32(255, 184, 112, 220),
                1.2F);
        }

        if (snapshot.lead_active && snapshot.lead_time_s.has_value()) {
            char label[48]{};
            std::snprintf(
                label,
                sizeof(label),
                snapshot.detected_point_stale ? "lead %.0fms stale" : "lead %.0fms",
                *snapshot.lead_time_s * 1000.0F);
            const ImVec2 text_pos(predicted.x + 8.0F, predicted.y - 18.0F);
            const ImVec2 text_size = ImGui::CalcTextSize(label);
            draw_list->AddRectFilled(
                ImVec2(text_pos.x - 3.0F, text_pos.y - 2.0F),
                ImVec2(text_pos.x + text_size.x + 3.0F, text_pos.y + text_size.y + 2.0F),
                IM_COL32(8, 12, 15, 190));
            draw_list->AddText(
                text_pos,
                snapshot.detected_point_stale ? IM_COL32(205, 164, 126, 230) : IM_COL32(255, 214, 156, 235),
                label);
        }
    }

    void drawScreenCenterMarker() {
        const ImVec2 display_size = ImGui::GetIO().DisplaySize;
        if (display_size.x <= 0.0F || display_size.y <= 0.0F) {
            return;
        }

        ImDrawList* draw_list = ImGui::GetForegroundDrawList();
        const ImVec2 center(display_size.x * 0.5F, display_size.y * 0.5F);
        draw_list->AddCircleFilled(center, 2.5F, IM_COL32(8, 12, 15, 220), 8);
        draw_list->AddCircleFilled(center, 1.5F, IM_COL32(255, 255, 255, 230), 8);
    }

    void renderFrame(const DebugPreviewSnapshot& snapshot) {
        if (!device_ || !context_ || !render_target_) {
            return;
        }

        ImGui_ImplDX11_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();
        drawScreenCenterMarker();
        drawCaptureRegion(snapshot);
        drawGuardRegion(snapshot);
        drawDetections(snapshot);
        drawLeadTelemetry(snapshot);
        ImGui::Render();

        constexpr float clear_color[4] = {0.0F, 0.0F, 0.0F, 0.0F};
        ID3D11RenderTargetView* render_target = render_target_.Get();
        context_->OMSetRenderTargets(1, &render_target, nullptr);
        context_->ClearRenderTargetView(render_target, clear_color);
        ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
        swap_chain_->Present(1, 0);
    }

    StaticConfig config_{};
    std::mutex snapshot_mutex_;
    DebugPreviewSnapshot snapshot_{};
    bool enabled_ = false;
    std::uint64_t sequence_ = 0;

    std::mutex window_mutex_;
    std::condition_variable ready_cv_;
    bool ready_ = false;
    std::atomic<bool> stopping_{false};
    HWND hwnd_ = nullptr;
    std::thread thread_;
    bool init_failed_ = false;
    bool affinity_warning_logged_ = false;

    ComPtr<ID3D11Device> device_;
    ComPtr<ID3D11DeviceContext> context_;
    ComPtr<IDXGISwapChain> swap_chain_;
    ComPtr<ID3D11RenderTargetView> render_target_;
    bool imgui_initialized_ = false;
    UINT pending_width_ = 0;
    UINT pending_height_ = 0;
};

class NullDebugOverlayWindow final : public DebugOverlayWindow {
public:
    void start() override {}
    void stop() override {}
    void setEnabled(bool) override {}
    void publish(DebugPreviewSnapshot) override {}
};

#else

class NullDebugOverlayWindow final : public DebugOverlayWindow {
public:
    void start() override {}
    void stop() override {}
    void setEnabled(bool) override {}
    void publish(DebugPreviewSnapshot) override {}
};

#endif

}  // namespace

std::unique_ptr<DebugOverlayWindow> makeDebugOverlayWindow(const StaticConfig& config) {
#if defined(_WIN32)
    return std::make_unique<Win32DebugOverlayWindow>(config);
#else
    (void)config;
    return std::make_unique<NullDebugOverlayWindow>();
#endif
}

}  // namespace delta
