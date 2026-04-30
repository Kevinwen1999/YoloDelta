#include "delta/debug_preview.hpp"

#include <atomic>
#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <mutex>
#include <thread>
#include <utility>

#if defined(_WIN32)
#include <windows.h>
#endif

namespace delta {

namespace {

#if defined(_WIN32)

constexpr wchar_t kPreviewClassName[] = L"DeltaDebugPreviewWindow";
constexpr wchar_t kPreviewWindowTitle[] = L"Delta Debug Preview";
constexpr UINT_PTR kPreviewTimerId = 1;
constexpr UINT kPreviewRefreshMs = 33;
constexpr UINT kPreviewSyncMessage = WM_APP + 1;
constexpr int kPreviewMargin = 16;
constexpr int kPreviewHeaderHeight = 54;
constexpr int kPreviewOverviewWidth = 92;
constexpr int kPreviewOverviewHeight = 36;

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

class Win32DebugPreviewWindow final : public DebugPreviewWindow {
public:
    explicit Win32DebugPreviewWindow(StaticConfig config)
        : config_(std::move(config)) {}

    ~Win32DebugPreviewWindow() override {
        stop();
    }

    void start() override {
        std::unique_lock<std::mutex> lock(window_mutex_);
        if (thread_.joinable()) {
            return;
        }
        ready_ = false;
        stopping_.store(false, std::memory_order_relaxed);
        thread_ = std::thread([this]() { threadMain(); });
        ready_cv_.wait(lock, [this]() { return ready_ || !thread_.joinable(); });
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
            PostMessageW(hwnd, kPreviewSyncMessage, 0, 0);
        }
    }

    void publish(DebugPreviewSnapshot snapshot) override {
        {
            std::lock_guard<std::mutex> lock(snapshot_mutex_);
            if (!enabled_) {
                return;
            }
            snapshot.sequence = ++sequence_;
            snapshot_ = std::move(snapshot);
        }
    }

private:
    struct PreviewLayout {
        RECT view{0, 0, 0, 0};
        float scale = 1.0F;
        int origin_x = 0;
        int origin_y = 0;
    };

    class ScopedSelectObject {
    public:
        ScopedSelectObject(HDC dc, HGDIOBJ object) : dc_(dc), previous_(SelectObject(dc, object)) {}
        ~ScopedSelectObject() {
            if (dc_ != nullptr && previous_ != nullptr) {
                SelectObject(dc_, previous_);
            }
        }

        ScopedSelectObject(const ScopedSelectObject&) = delete;
        ScopedSelectObject& operator=(const ScopedSelectObject&) = delete;

    private:
        HDC dc_ = nullptr;
        HGDIOBJ previous_ = nullptr;
    };

    class ScopedDeleteObject {
    public:
        explicit ScopedDeleteObject(HGDIOBJ object) : object_(object) {}
        ~ScopedDeleteObject() {
            if (object_ != nullptr) {
                DeleteObject(object_);
            }
        }

        ScopedDeleteObject(const ScopedDeleteObject&) = delete;
        ScopedDeleteObject& operator=(const ScopedDeleteObject&) = delete;

    private:
        HGDIOBJ object_ = nullptr;
    };

    static LRESULT CALLBACK windowProcSetup(HWND hwnd, UINT message, WPARAM wparam, LPARAM lparam) {
        Win32DebugPreviewWindow* self = nullptr;
        if (message == WM_NCCREATE) {
            const auto* create = reinterpret_cast<const CREATESTRUCTW*>(lparam);
            self = static_cast<Win32DebugPreviewWindow*>(create->lpCreateParams);
            SetWindowLongPtrW(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(self));
        } else {
            self = reinterpret_cast<Win32DebugPreviewWindow*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));
        }
        if (self != nullptr) {
            return self->windowProc(hwnd, message, wparam, lparam);
        }
        return DefWindowProcW(hwnd, message, wparam, lparam);
    }

    LRESULT windowProc(HWND hwnd, UINT message, WPARAM wparam, LPARAM lparam) {
        switch (message) {
        case WM_NCCREATE:
            return DefWindowProcW(hwnd, message, wparam, lparam);
        case WM_CREATE:
            syncWindowState(hwnd);
            return 0;
        case WM_TIMER:
            if (wparam == kPreviewTimerId) {
                InvalidateRect(hwnd, nullptr, FALSE);
            }
            return 0;
        case WM_ERASEBKGND:
            return 1;
        case kPreviewSyncMessage:
            syncWindowState(hwnd);
            InvalidateRect(hwnd, nullptr, FALSE);
            return 0;
        case WM_CLOSE:
            if (!stopping_.load(std::memory_order_relaxed)) {
                ShowWindow(hwnd, SW_HIDE);
                KillTimer(hwnd, kPreviewTimerId);
                return 0;
            }
            DestroyWindow(hwnd);
            return 0;
        case WM_DESTROY:
            KillTimer(hwnd, kPreviewTimerId);
            PostQuitMessage(0);
            return 0;
        case WM_PAINT:
            paint(hwnd);
            return 0;
        default:
            return DefWindowProcW(hwnd, message, wparam, lparam);
        }
    }

    void threadMain() {
        const HINSTANCE instance = GetModuleHandleW(nullptr);
        WNDCLASSEXW wc{};
        wc.cbSize = sizeof(wc);
        wc.lpfnWndProc = &Win32DebugPreviewWindow::windowProcSetup;
        wc.hInstance = instance;
        wc.hCursor = LoadCursorW(nullptr, MAKEINTRESOURCEW(32512));
        wc.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_WINDOW + 1);
        wc.lpszClassName = kPreviewClassName;
        const ATOM class_atom = RegisterClassExW(&wc);
        if (class_atom == 0 && GetLastError() != ERROR_CLASS_ALREADY_EXISTS) {
            if (config_.debug_log) {
                std::cerr << "[debug_preview] RegisterClassExW failed: " << formatLastErrorMessage(GetLastError()) << "\n";
            }
        }

        const int client_width = defaultClientWidth();
        const int client_height = defaultClientHeight();
        RECT rect{0, 0, client_width, client_height};
        AdjustWindowRectEx(&rect, WS_OVERLAPPEDWINDOW, FALSE, WS_EX_TOPMOST | WS_EX_TOOLWINDOW);
        const int width = rect.right - rect.left;
        const int height = rect.bottom - rect.top;
        RECT work_area{};
        if (!SystemParametersInfoW(SPI_GETWORKAREA, 0, &work_area, 0)) {
            work_area.left = 0;
            work_area.top = 0;
            work_area.right = GetSystemMetrics(SM_CXSCREEN);
            work_area.bottom = GetSystemMetrics(SM_CYSCREEN);
        }
        const int x = std::max(work_area.left, work_area.right - width - 24);
        const int y = std::max(work_area.top, work_area.top + 72);

        HWND hwnd = CreateWindowExW(
            WS_EX_TOPMOST | WS_EX_TOOLWINDOW,
            kPreviewClassName,
            kPreviewWindowTitle,
            WS_OVERLAPPEDWINDOW,
            x,
            y,
            width,
            height,
            nullptr,
            nullptr,
            instance,
            this);

        if (hwnd == nullptr && config_.debug_log) {
            std::cerr << "[debug_preview] CreateWindowExW failed: " << formatLastErrorMessage(GetLastError()) << "\n";
        } else if (hwnd != nullptr && config_.debug_log) {
            std::cout << "[debug_preview] window created at (" << x << ", " << y << ") size "
                      << width << "x" << height << "\n";
        }

        {
            std::lock_guard<std::mutex> lock(window_mutex_);
            hwnd_ = hwnd;
            ready_ = true;
        }
        ready_cv_.notify_all();

        if (hwnd == nullptr) {
            return;
        }

        MSG msg{};
        while (GetMessageW(&msg, nullptr, 0, 0) > 0) {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }

        {
            std::lock_guard<std::mutex> lock(window_mutex_);
            hwnd_ = nullptr;
        }
    }

    int defaultClientWidth() const {
        return clamp(static_cast<int>(effectiveCaptureCropSize(config_) * 0.65F), 280, 520);
    }

    int defaultClientHeight() const {
        return defaultClientWidth() + 88;
    }

    void syncWindowState(HWND hwnd) {
        bool enabled = false;
        {
            std::lock_guard<std::mutex> lock(snapshot_mutex_);
            enabled = enabled_;
        }
        if (enabled) {
            SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW);
            ShowWindow(hwnd, SW_SHOWNORMAL);
            UpdateWindow(hwnd);
            SetTimer(hwnd, kPreviewTimerId, kPreviewRefreshMs, nullptr);
            const BOOL affinity_ok = SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE);
            if (config_.debug_log) {
                std::cout << "[debug_preview] visible";
                if (!affinity_ok) {
                    std::cout << " (capture affinity unavailable: " << formatLastErrorMessage(GetLastError()) << ")";
                }
                std::cout << "\n";
            }
        } else {
            KillTimer(hwnd, kPreviewTimerId);
            ShowWindow(hwnd, SW_HIDE);
            if (config_.debug_log) {
                std::cout << "[debug_preview] hidden\n";
            }
        }
    }

    PreviewLayout computeLayout(const RECT& client, const CaptureRegion& capture_region) const {
        PreviewLayout layout{};
        if (capture_region.width <= 0 || capture_region.height <= 0) {
            layout.view = client;
            layout.origin_x = client.left;
            layout.origin_y = client.top;
            return layout;
        }

        RECT content{
            client.left + kPreviewMargin,
            client.top + kPreviewHeaderHeight,
            client.right - kPreviewMargin,
            client.bottom - kPreviewMargin,
        };
        const int content_width = std::max<int>(1, static_cast<int>(content.right - content.left));
        const int content_height = std::max<int>(1, static_cast<int>(content.bottom - content.top));
        const float scale_x = static_cast<float>(content_width) / static_cast<float>(capture_region.width);
        const float scale_y = static_cast<float>(content_height) / static_cast<float>(capture_region.height);
        layout.scale = std::max(0.01F, std::min(scale_x, scale_y));
        const int view_width = std::max(1, static_cast<int>(capture_region.width * layout.scale));
        const int view_height = std::max(1, static_cast<int>(capture_region.height * layout.scale));
        layout.origin_x = content.left + ((content_width - view_width) / 2);
        layout.origin_y = content.top + ((content_height - view_height) / 2);
        layout.view = RECT{
            layout.origin_x,
            layout.origin_y,
            layout.origin_x + view_width,
            layout.origin_y + view_height,
        };
        return layout;
    }

    RECT mapBox(const PreviewLayout& layout, const CaptureRegion& capture_region, const std::array<int, 4>& bbox) const {
        const float x1 = static_cast<float>(bbox[0] - capture_region.left);
        const float y1 = static_cast<float>(bbox[1] - capture_region.top);
        const float x2 = static_cast<float>(bbox[2] - capture_region.left);
        const float y2 = static_cast<float>(bbox[3] - capture_region.top);
        RECT out{};
        out.left = layout.origin_x + static_cast<int>(x1 * layout.scale);
        out.top = layout.origin_y + static_cast<int>(y1 * layout.scale);
        out.right = layout.origin_x + static_cast<int>(x2 * layout.scale);
        out.bottom = layout.origin_y + static_cast<int>(y2 * layout.scale);
        return out;
    }

    POINT mapPoint(const PreviewLayout& layout, const CaptureRegion& capture_region, const float screen_x, const float screen_y) const {
        POINT pt{};
        pt.x = layout.origin_x + static_cast<int>((screen_x - static_cast<float>(capture_region.left)) * layout.scale);
        pt.y = layout.origin_y + static_cast<int>((screen_y - static_cast<float>(capture_region.top)) * layout.scale);
        return pt;
    }

    void drawMarker(HDC dc, const POINT pt, const COLORREF color, const int radius) const {
        HPEN pen = CreatePen(PS_SOLID, 2, color);
        ScopedDeleteObject pen_cleanup(pen);
        ScopedSelectObject select_pen(dc, pen);
        MoveToEx(dc, pt.x - radius, pt.y, nullptr);
        LineTo(dc, pt.x + radius + 1, pt.y);
        MoveToEx(dc, pt.x, pt.y - radius, nullptr);
        LineTo(dc, pt.x, pt.y + radius + 1);
    }

    void drawLeadLabel(HDC dc, const POINT pt, const COLORREF color, const float lead_time_s, const bool stale) const {
        char label[48]{};
        std::snprintf(label, sizeof(label), stale ? "lead %.0fms stale" : "lead %.0fms", lead_time_s * 1000.0F);
        SetBkMode(dc, TRANSPARENT);
        SetTextColor(dc, color);
        TextOutA(dc, pt.x + 8, pt.y - 14, label, static_cast<int>(std::strlen(label)));
    }

    void drawBackground(HDC dc, const RECT& client, const PreviewLayout& layout) const {
        HBRUSH bg = CreateSolidBrush(RGB(11, 16, 20));
        ScopedDeleteObject bg_cleanup(bg);
        FillRect(dc, &client, bg);

        HBRUSH panel = CreateSolidBrush(RGB(18, 26, 31));
        ScopedDeleteObject panel_cleanup(panel);
        FillRect(dc, &layout.view, panel);

        HPEN border_pen = CreatePen(PS_SOLID, 1, RGB(66, 85, 96));
        ScopedDeleteObject border_cleanup(border_pen);
        ScopedSelectObject select_pen(dc, border_pen);
        ScopedSelectObject select_brush(dc, GetStockObject(HOLLOW_BRUSH));
        Rectangle(dc, layout.view.left, layout.view.top, layout.view.right, layout.view.bottom);
    }

    RECT computeOverviewRect(const RECT& client) const {
        const int right = client.right - kPreviewMargin;
        const int left = std::max(kPreviewMargin, right - kPreviewOverviewWidth);
        const int top = 12;
        const int bottom = std::min(kPreviewHeaderHeight - 6, top + kPreviewOverviewHeight);
        return RECT{left, top, right, bottom};
    }

    void drawCaptureOverview(HDC dc, const DebugPreviewSnapshot& snapshot, const RECT& client) const {
        const RECT overview = computeOverviewRect(client);
        if (overview.right <= overview.left || overview.bottom <= overview.top) {
            return;
        }

        HBRUSH panel = CreateSolidBrush(RGB(12, 18, 22));
        ScopedDeleteObject panel_cleanup(panel);
        FillRect(dc, &overview, panel);

        HPEN border_pen = CreatePen(PS_SOLID, 1, RGB(58, 74, 84));
        ScopedDeleteObject border_cleanup(border_pen);
        ScopedSelectObject select_border_pen(dc, border_pen);
        ScopedSelectObject select_border_brush(dc, GetStockObject(HOLLOW_BRUSH));
        Rectangle(dc, overview.left, overview.top, overview.right, overview.bottom);

        const int screen_width = std::max(1, config_.screen_w);
        const int screen_height = std::max(1, config_.screen_h);
        RECT inner{
            overview.left + 4,
            overview.top + 4,
            overview.right - 4,
            overview.bottom - 4,
        };
        const int inner_width = std::max(1, static_cast<int>(inner.right - inner.left));
        const int inner_height = std::max(1, static_cast<int>(inner.bottom - inner.top));
        const float scale_x = static_cast<float>(inner_width) / static_cast<float>(screen_width);
        const float scale_y = static_cast<float>(inner_height) / static_cast<float>(screen_height);
        const float scale = std::max(0.01F, std::min(scale_x, scale_y));
        const int screen_rect_width = std::max(1, static_cast<int>(screen_width * scale));
        const int screen_rect_height = std::max(1, static_cast<int>(screen_height * scale));
        RECT screen_rect{
            inner.left + ((inner_width - screen_rect_width) / 2),
            inner.top + ((inner_height - screen_rect_height) / 2),
            inner.left + ((inner_width - screen_rect_width) / 2) + screen_rect_width,
            inner.top + ((inner_height - screen_rect_height) / 2) + screen_rect_height,
        };

        HPEN screen_pen = CreatePen(PS_SOLID, 1, RGB(86, 105, 116));
        ScopedDeleteObject screen_pen_cleanup(screen_pen);
        ScopedSelectObject select_screen_pen(dc, screen_pen);
        Rectangle(dc, screen_rect.left, screen_rect.top, screen_rect.right, screen_rect.bottom);

        if (snapshot.capture_region.width > 0 && snapshot.capture_region.height > 0) {
            RECT crop_rect{
                screen_rect.left + static_cast<int>(static_cast<float>(snapshot.capture_region.left) * scale),
                screen_rect.top + static_cast<int>(static_cast<float>(snapshot.capture_region.top) * scale),
                screen_rect.left + static_cast<int>(static_cast<float>(snapshot.capture_region.left + snapshot.capture_region.width) * scale),
                screen_rect.top + static_cast<int>(static_cast<float>(snapshot.capture_region.top + snapshot.capture_region.height) * scale),
            };
            crop_rect.right = std::max(crop_rect.left + 1, crop_rect.right);
            crop_rect.bottom = std::max(crop_rect.top + 1, crop_rect.bottom);

            HPEN crop_pen = CreatePen(PS_SOLID, 1, RGB(99, 186, 255));
            ScopedDeleteObject crop_pen_cleanup(crop_pen);
            ScopedSelectObject select_crop_pen(dc, crop_pen);
            Rectangle(dc, crop_rect.left, crop_rect.top, crop_rect.right, crop_rect.bottom);
        }

        const POINT center{
            screen_rect.left + static_cast<int>(static_cast<float>(snapshot.screen_center.first) * scale),
            screen_rect.top + static_cast<int>(static_cast<float>(snapshot.screen_center.second) * scale),
        };
        drawMarker(dc, center, RGB(229, 229, 229), 2);
    }

    void drawStatusText(HDC dc, const DebugPreviewSnapshot& snapshot, const RECT& client) const {
        SetBkMode(dc, TRANSPARENT);
        SetTextColor(dc, RGB(230, 241, 246));
        ScopedSelectObject select_font(dc, GetStockObject(DEFAULT_GUI_FONT));

        char line1[160]{};
        char line2[160]{};
        if (!snapshot.active) {
            std::snprintf(line1, sizeof(line1), "Preview enabled | waiting for tracking activity");
            std::snprintf(line2, sizeof(line2), "Move/engage or enable trigger monitor to populate boxes");
        } else {
            std::snprintf(
                line1,
                sizeof(line1),
                "Capture %dx%d | detections %zu | target %s",
                snapshot.capture_region.width,
                snapshot.capture_region.height,
                snapshot.detections.size(),
                snapshot.target_found ? "locked" : "none");
            std::snprintf(
                line2,
                sizeof(line2),
                "Class %d | speed %.1f px/s | geometry-only preview",
                snapshot.target_cls,
                snapshot.target_speed);
        }
        const RECT overview = computeOverviewRect(client);
        const int text_right = std::max(kPreviewMargin + 120, static_cast<int>(overview.left) - 12);
        RECT line1_rect{kPreviewMargin, 11, text_right, 28};
        RECT line2_rect{kPreviewMargin, 29, text_right, 46};
        DrawTextA(dc, line1, -1, &line1_rect, DT_SINGLELINE | DT_END_ELLIPSIS | DT_NOPREFIX);
        SetTextColor(dc, RGB(143, 167, 177));
        DrawTextA(dc, line2, -1, &line2_rect, DT_SINGLELINE | DT_END_ELLIPSIS | DT_NOPREFIX);
        drawCaptureOverview(dc, snapshot, client);
    }

    void drawDetections(HDC dc, const DebugPreviewSnapshot& snapshot, const PreviewLayout& layout) const {
        SetBkMode(dc, TRANSPARENT);
        ScopedSelectObject select_font(dc, GetStockObject(DEFAULT_GUI_FONT));
        for (const auto& detection : snapshot.detections) {
            const RECT box = mapBox(layout, snapshot.capture_region, detection.bbox);
            const COLORREF color = detection.selected ? RGB(111, 231, 150) : RGB(243, 181, 95);
            const int thickness = 1;
            HPEN pen = CreatePen(PS_SOLID, thickness, color);
            ScopedDeleteObject pen_cleanup(pen);
            ScopedSelectObject select_pen(dc, pen);
            ScopedSelectObject select_brush(dc, GetStockObject(HOLLOW_BRUSH));
            Rectangle(dc, box.left, box.top, box.right, box.bottom);

            char label[64]{};
            std::snprintf(label, sizeof(label), "c%d %.2f%s", detection.cls, detection.conf, detection.selected ? " *" : "");
            SetTextColor(dc, color);
            TextOutA(
                dc,
                box.left + 2,
                std::max<int>(kPreviewHeaderHeight, static_cast<int>(box.top - 16)),
                label,
                static_cast<int>(std::strlen(label)));
        }
    }

    void drawGuardRegion(HDC dc, const DebugPreviewSnapshot& snapshot, const PreviewLayout& layout) const {
        if (!snapshot.guard_region.has_value()) {
            return;
        }

        const auto& guard = *snapshot.guard_region;
        const RECT box = mapBox(
            layout,
            snapshot.capture_region,
            std::array<int, 4>{guard.left, guard.top, guard.left + guard.width, guard.top + guard.height});
        HPEN pen = CreatePen(PS_SOLID, 1, RGB(255, 122, 122));
        ScopedDeleteObject pen_cleanup(pen);
        ScopedSelectObject select_pen(dc, pen);
        ScopedSelectObject select_brush(dc, GetStockObject(HOLLOW_BRUSH));
        Rectangle(dc, box.left, box.top, box.right, box.bottom);
    }

    void drawGuides(HDC dc, const DebugPreviewSnapshot& snapshot, const PreviewLayout& layout) const {
        if (snapshot.capture_region.width <= 0 || snapshot.capture_region.height <= 0) {
            return;
        }

        const POINT center = mapPoint(
            layout,
            snapshot.capture_region,
            static_cast<float>(snapshot.screen_center.first),
            static_cast<float>(snapshot.screen_center.second));
        drawMarker(dc, center, RGB(229, 229, 229), 7);

        if (snapshot.locked_point.has_value()) {
            const POINT locked = mapPoint(layout, snapshot.capture_region, snapshot.locked_point->first, snapshot.locked_point->second);
            drawMarker(dc, locked, RGB(99, 186, 255), 6);
        }
        if (snapshot.detected_point.has_value()) {
            const POINT detected = mapPoint(layout, snapshot.capture_region, snapshot.detected_point->first, snapshot.detected_point->second);
            drawMarker(dc, detected, snapshot.detected_point_stale ? RGB(112, 172, 112) : RGB(111, 231, 150), 6);
        }
        if (snapshot.kalman_filtered_point.has_value()) {
            const POINT filtered = mapPoint(
                layout,
                snapshot.capture_region,
                snapshot.kalman_filtered_point->first,
                snapshot.kalman_filtered_point->second);
            drawMarker(dc, filtered, RGB(172, 128, 255), 6);
        }
        if (snapshot.kalman_predicted_point.has_value()) {
            const POINT kalman_predicted = mapPoint(
                layout,
                snapshot.capture_region,
                snapshot.kalman_predicted_point->first,
                snapshot.kalman_predicted_point->second);
            drawMarker(dc, kalman_predicted, RGB(255, 102, 204), 5);
            if (snapshot.kalman_filtered_point.has_value()) {
                const POINT filtered = mapPoint(
                    layout,
                    snapshot.capture_region,
                    snapshot.kalman_filtered_point->first,
                    snapshot.kalman_filtered_point->second);
                HPEN pen = CreatePen(PS_SOLID, 1, RGB(202, 146, 255));
                ScopedDeleteObject pen_cleanup(pen);
                ScopedSelectObject select_pen(dc, pen);
                MoveToEx(dc, filtered.x, filtered.y, nullptr);
                LineTo(dc, kalman_predicted.x, kalman_predicted.y);
            }
        }
        if (snapshot.predicted_point.has_value()) {
            const POINT predicted = mapPoint(layout, snapshot.capture_region, snapshot.predicted_point->first, snapshot.predicted_point->second);
            drawMarker(dc, predicted, snapshot.lead_active ? RGB(255, 146, 96) : RGB(255, 122, 122), 6);
            if (snapshot.lead_active && snapshot.detected_point.has_value()) {
                const POINT detected = mapPoint(layout, snapshot.capture_region, snapshot.detected_point->first, snapshot.detected_point->second);
                HPEN pen = CreatePen(PS_SOLID, 1, snapshot.detected_point_stale ? RGB(160, 138, 108) : RGB(255, 184, 112));
                ScopedDeleteObject pen_cleanup(pen);
                ScopedSelectObject select_pen(dc, pen);
                MoveToEx(dc, detected.x, detected.y, nullptr);
                LineTo(dc, predicted.x, predicted.y);
            }
            if (snapshot.lead_active && snapshot.lead_time_s.has_value()) {
                drawLeadLabel(
                    dc,
                    predicted,
                    snapshot.detected_point_stale ? RGB(205, 164, 126) : RGB(255, 214, 156),
                    *snapshot.lead_time_s,
                    snapshot.detected_point_stale);
            }
        }
    }

    void paint(HWND hwnd) {
        PAINTSTRUCT ps{};
        HDC hdc = BeginPaint(hwnd, &ps);

        RECT client{};
        GetClientRect(hwnd, &client);
        const int width = std::max<int>(1, static_cast<int>(client.right - client.left));
        const int height = std::max<int>(1, static_cast<int>(client.bottom - client.top));

        HDC buffer_dc = CreateCompatibleDC(hdc);
        HBITMAP buffer_bitmap = CreateCompatibleBitmap(hdc, width, height);
        ScopedDeleteObject bitmap_cleanup(buffer_bitmap);
        ScopedSelectObject select_bitmap(buffer_dc, buffer_bitmap);

        DebugPreviewSnapshot snapshot{};
        {
            std::lock_guard<std::mutex> lock(snapshot_mutex_);
            snapshot = snapshot_;
        }

        const PreviewLayout layout = computeLayout(client, snapshot.capture_region);
        drawBackground(buffer_dc, client, layout);
        drawStatusText(buffer_dc, snapshot, client);
        drawGuides(buffer_dc, snapshot, layout);
        drawGuardRegion(buffer_dc, snapshot, layout);
        drawDetections(buffer_dc, snapshot, layout);

        BitBlt(hdc, 0, 0, width, height, buffer_dc, 0, 0, SRCCOPY);
        DeleteDC(buffer_dc);
        EndPaint(hwnd, &ps);
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
};

class NullDebugPreviewWindow final : public DebugPreviewWindow {
public:
    void start() override {}
    void stop() override {}
    void setEnabled(bool) override {}
    void publish(DebugPreviewSnapshot) override {}
};

#else

class NullDebugPreviewWindow final : public DebugPreviewWindow {
public:
    void start() override {}
    void stop() override {}
    void setEnabled(bool) override {}
    void publish(DebugPreviewSnapshot) override {}
};

#endif

}  // namespace

std::unique_ptr<DebugPreviewWindow> makeDebugPreviewWindow(const StaticConfig& config) {
#if defined(_WIN32)
    return std::make_unique<Win32DebugPreviewWindow>(config);
#else
    (void)config;
    return std::make_unique<NullDebugPreviewWindow>();
#endif
}

}  // namespace delta
