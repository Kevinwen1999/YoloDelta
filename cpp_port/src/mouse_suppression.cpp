#include "delta/mouse_suppression.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <thread>

#if defined(_WIN32)
#include <windows.h>

#ifndef LLMHF_LOWER_IL_INJECTED
#define LLMHF_LOWER_IL_INJECTED 0x00000002
#endif
#endif

namespace delta {

bool shouldSuppressMouseMoveOnFire(const bool enabled, const bool left_pressed, const bool x1_pressed) {
    return enabled && (left_pressed || x1_pressed);
}

namespace {

class NullMouseMoveSuppressor final : public IMouseMoveSuppressor {
public:
    void start() override {}
    void stop() override {}
    void setDebugLogging(const bool enabled) override {
        (void)enabled;
    }
    void setSuppressionActive(const bool active) override {
        (void)active;
    }
    MouseMoveSuppressionStatus snapshot() const override {
        return {};
    }
};

#if defined(_WIN32)

class Win32MouseMoveSuppressor final : public IMouseMoveSuppressor {
public:
    Win32MouseMoveSuppressor() = default;

    ~Win32MouseMoveSuppressor() override {
        stop();
    }

    void start() override {
        std::unique_lock<std::mutex> lock(mutex_);
        if (thread_.joinable()) {
            return;
        }
        ready_ = false;
        thread_ = std::thread([this]() { threadMain(); });
        ready_cv_.wait(lock, [this]() { return ready_; });
    }

    void stop() override {
        setSuppressionActive(false);

        DWORD thread_id = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            thread_id = thread_id_;
        }
        if (thread_id != 0) {
            PostThreadMessageW(thread_id, WM_QUIT, 0, 0);
        }
        if (thread_.joinable()) {
            thread_.join();
        }

        supported_.store(false, std::memory_order_relaxed);
        active_requested_.store(false, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            ready_ = false;
            thread_id_ = 0;
        }
    }

    void setDebugLogging(const bool enabled) override {
        debug_logging_.store(enabled, std::memory_order_relaxed);
    }

    void setSuppressionActive(const bool active) override {
        const bool previous = active_requested_.exchange(active, std::memory_order_relaxed);
        if (previous != active
            && supported_.load(std::memory_order_relaxed)
            && debug_logging_.load(std::memory_order_relaxed)) {
            std::cout << "[mouse_suppression] " << (active ? "ACTIVE" : "IDLE") << "\n";
        }
    }

    MouseMoveSuppressionStatus snapshot() const override {
        const bool supported = supported_.load(std::memory_order_relaxed);
        return MouseMoveSuppressionStatus{
            .supported = supported,
            .active = supported && active_requested_.load(std::memory_order_relaxed),
            .suppressed_count = suppressed_count_.load(std::memory_order_relaxed),
        };
    }

private:
    static LRESULT CALLBACK hookProc(const int code, const WPARAM wparam, const LPARAM lparam) {
        if (code < 0) {
            return CallNextHookEx(nullptr, code, wparam, lparam);
        }
        Win32MouseMoveSuppressor* self = instance_.load(std::memory_order_acquire);
        if (self == nullptr) {
            return CallNextHookEx(nullptr, code, wparam, lparam);
        }
        return self->handleHook(code, wparam, lparam);
    }

    LRESULT handleHook(const int code, const WPARAM wparam, const LPARAM lparam) {
        if (!supported_.load(std::memory_order_relaxed) || !active_requested_.load(std::memory_order_relaxed)) {
            return CallNextHookEx(nullptr, code, wparam, lparam);
        }
        const auto* info = reinterpret_cast<const MSLLHOOKSTRUCT*>(lparam);
        const DWORD flags = info == nullptr ? 0UL : info->flags;
        if ((flags & LLMHF_INJECTED) != 0 || (flags & LLMHF_LOWER_IL_INJECTED) != 0) {
            return CallNextHookEx(nullptr, code, wparam, lparam);
        }
        if (wparam == WM_MOUSEMOVE) {
            suppressed_count_.fetch_add(1, std::memory_order_relaxed);
            return 1;
        }
        return CallNextHookEx(nullptr, code, wparam, lparam);
    }

    void threadMain() {
        MSG msg{};
        PeekMessageW(&msg, nullptr, 0, 0, PM_NOREMOVE);

        {
            std::lock_guard<std::mutex> lock(mutex_);
            thread_id_ = GetCurrentThreadId();
        }

        HHOOK hook = SetWindowsHookExW(WH_MOUSE_LL, &Win32MouseMoveSuppressor::hookProc, nullptr, 0);
        if (hook == nullptr) {
            if (debug_logging_.load(std::memory_order_relaxed)) {
                std::cout << "[mouse_suppression] SetWindowsHookExW failed: " << GetLastError() << "\n";
            }
            supported_.store(false, std::memory_order_relaxed);
            active_requested_.store(false, std::memory_order_relaxed);
            {
                std::lock_guard<std::mutex> lock(mutex_);
                ready_ = true;
                thread_id_ = 0;
            }
            ready_cv_.notify_all();
            return;
        }

        instance_.store(this, std::memory_order_release);
        supported_.store(true, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            ready_ = true;
        }
        ready_cv_.notify_all();

        BOOL result = 0;
        while ((result = GetMessageW(&msg, nullptr, 0, 0)) > 0) {
        }
        if (result == -1 && debug_logging_.load(std::memory_order_relaxed)) {
            std::cout << "[mouse_suppression] GetMessageW failed: " << GetLastError() << "\n";
        }

        instance_.store(nullptr, std::memory_order_release);
        if (!UnhookWindowsHookEx(hook) && debug_logging_.load(std::memory_order_relaxed)) {
            std::cout << "[mouse_suppression] UnhookWindowsHookEx failed: " << GetLastError() << "\n";
        }
        supported_.store(false, std::memory_order_relaxed);
        active_requested_.store(false, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            thread_id_ = 0;
        }
    }

    mutable std::mutex mutex_;
    std::condition_variable ready_cv_;
    std::thread thread_;
    DWORD thread_id_ = 0;
    bool ready_ = false;
    std::atomic<bool> supported_{false};
    std::atomic<bool> active_requested_{false};
    std::atomic<bool> debug_logging_{false};
    std::atomic<std::uint64_t> suppressed_count_{0};
    static std::atomic<Win32MouseMoveSuppressor*> instance_;
};

std::atomic<Win32MouseMoveSuppressor*> Win32MouseMoveSuppressor::instance_{nullptr};

#endif

}  // namespace

std::unique_ptr<IMouseMoveSuppressor> makeMouseMoveSuppressor() {
#if defined(_WIN32)
    return std::make_unique<Win32MouseMoveSuppressor>();
#else
    return std::make_unique<NullMouseMoveSuppressor>();
#endif
}

}  // namespace delta
