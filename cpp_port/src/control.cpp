#include "delta/control.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <thread>

#if defined(_WIN32)
#include <windows.h>
#endif

namespace delta {

#if defined(_WIN32)
namespace {

bool sendMouseClickTap(const DWORD down_flag, const DWORD up_flag, const double hold_s) {
    INPUT down{};
    down.type = INPUT_MOUSE;
    down.mi.dwFlags = down_flag;

    INPUT up{};
    up.type = INPUT_MOUSE;
    up.mi.dwFlags = up_flag;

    if (SendInput(1, &down, sizeof(INPUT)) != 1) {
        return false;
    }
    if (hold_s > 0.0) {
        std::this_thread::sleep_for(std::chrono::duration<double>(hold_s));
    }
    return SendInput(1, &up, sizeof(INPUT)) == 1;
}

}  // namespace
#endif

InputSnapshot Win32HotkeySource::poll() const {
    InputSnapshot snapshot{};
#if defined(_WIN32)
    auto pressed = [](int vk) -> bool {
        return (GetAsyncKeyState(vk) & 0x8000) != 0;
    };

    snapshot.insert_pressed = pressed(VK_INSERT);
    snapshot.left_pressed = pressed(VK_LBUTTON);
    snapshot.right_pressed = pressed(VK_RBUTTON);
    snapshot.f4_pressed = pressed(VK_F4);
    snapshot.x1_pressed = pressed(VK_XBUTTON1);
    snapshot.x2_pressed = pressed(VK_XBUTTON2);
    snapshot.f5_pressed = pressed(VK_F5);
    snapshot.f6_pressed = pressed(VK_F6);
    snapshot.f7_pressed = pressed(VK_F7);
    snapshot.f8_pressed = pressed(VK_F8);
#endif
    return snapshot;
}

void SendInputMouseSender::configure(const MouseSenderConfig& config) {
    config_.gain_x = config.gain_x;
    config_.gain_y = config.gain_y;
    config_.max_step = std::max(1, config.max_step);
}

bool SendInputMouseSender::sendRelative(int dx, int dy) {
    if (dx == 0 && dy == 0) {
        return true;
    }
#if defined(_WIN32)
    const float move_x = (static_cast<float>(dx) * config_.gain_x) + frac_x_;
    const float move_y = (static_cast<float>(dy) * config_.gain_y) + frac_y_;
    int send_x = static_cast<int>(std::lround(move_x));
    int send_y = static_cast<int>(std::lround(move_y));
    frac_x_ = move_x - static_cast<float>(send_x);
    frac_y_ = move_y - static_cast<float>(send_y);

    bool sent_any = false;
    while (send_x != 0 || send_y != 0) {
        const int step_x = std::clamp(send_x, -config_.max_step, config_.max_step);
        const int step_y = std::clamp(send_y, -config_.max_step, config_.max_step);
        INPUT input{};
        input.type = INPUT_MOUSE;
        input.mi.dx = step_x;
        input.mi.dy = step_y;
        input.mi.dwFlags = MOUSEEVENTF_MOVE;
        if (SendInput(1, &input, sizeof(INPUT)) != 1) {
            return false;
        }
        sent_any = true;
        send_x -= step_x;
        send_y -= step_y;
    }
    return sent_any;
#else
    (void)dx;
    (void)dy;
    return false;
#endif
}

bool SendInputMouseSender::clickLeft(double hold_s) {
    return sendLeftClickTap(hold_s);
}

bool sendLeftClickTap(const double hold_s) {
#if defined(_WIN32)
    return sendMouseClickTap(MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP, hold_s);
#else
    (void)hold_s;
    return false;
#endif
}

bool sendRightClickTap(const double hold_s) {
#if defined(_WIN32)
    return sendMouseClickTap(MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP, hold_s);
#else
    (void)hold_s;
    return false;
#endif
}

bool isLeftHoldEngageSatisfied(
    const bool left_hold_engage,
    const LeftHoldEngageButton engage_button,
    const bool left_pressed,
    const bool right_pressed,
    const bool x1_pressed) {
    if (!left_hold_engage) {
        return true;
    }
    switch (engage_button) {
    case LeftHoldEngageButton::Left:
        return left_pressed;
    case LeftHoldEngageButton::X1:
        return x1_pressed;
    case LeftHoldEngageButton::Both:
        // "Both" is the legacy stored value; intended behavior is that
        // left, right, or the X1 side button satisfies engage when selected.
        return left_pressed || right_pressed || x1_pressed;
    case LeftHoldEngageButton::Right:
    default:
        return right_pressed;
    }
}

void playToggleBeep(const int frequency_hz, const int duration_ms) {
#if defined(_WIN32)
    const int safe_frequency = std::clamp(frequency_hz, 37, 32767);
    const int safe_duration = std::max(1, duration_ms);
    Beep(static_cast<DWORD>(safe_frequency), static_cast<DWORD>(safe_duration));
#else
    (void)frequency_hz;
    (void)duration_ms;
#endif
}

bool sendVirtualKeyTap(const std::uint16_t virtual_key, const double hold_ms) {
#if defined(_WIN32)
    INPUT key_down{};
    key_down.type = INPUT_KEYBOARD;
    key_down.ki.wVk = static_cast<WORD>(virtual_key);

    INPUT key_up = key_down;
    key_up.ki.dwFlags = KEYEVENTF_KEYUP;

    if (SendInput(1, &key_down, sizeof(INPUT)) != 1) {
        return false;
    }
    if (hold_ms > 0.0) {
        std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(hold_ms));
    }
    return SendInput(1, &key_up, sizeof(INPUT)) == 1;
#else
    (void)virtual_key;
    (void)hold_ms;
    return false;
#endif
}

std::unique_ptr<IInputSender> makeInputSender() {
    return std::make_unique<SendInputMouseSender>();
}

}  // namespace delta
