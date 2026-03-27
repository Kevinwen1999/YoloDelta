#pragma once

#include <cstdint>
#include <memory>
#include <string_view>

#include "delta/config.hpp"

namespace delta {

struct InputSnapshot {
    bool insert_pressed = false;
    bool left_pressed = false;
    bool right_pressed = false;
    bool f4_pressed = false;
    bool x1_pressed = false;
    bool x2_pressed = false;
    bool f5_pressed = false;
    bool f6_pressed = false;
    bool f7_pressed = false;
    bool f8_pressed = false;
};

struct MouseSenderConfig {
    float gain_x = 1.0F;
    float gain_y = 1.0F;
    int max_step = 127;
};

class Win32HotkeySource {
public:
    InputSnapshot poll() const;
};

class IInputSender {
public:
    virtual ~IInputSender() = default;
    virtual std::string_view name() const = 0;
    virtual void configure(const MouseSenderConfig& config) = 0;
    virtual bool sendRelative(int dx, int dy) = 0;
    virtual bool clickLeft(double hold_s) = 0;
};

class SendInputMouseSender final : public IInputSender {
public:
    std::string_view name() const override { return "sendinput"; }
    void configure(const MouseSenderConfig& config) override;
    bool sendRelative(int dx, int dy) override;
    bool clickLeft(double hold_s) override;

private:
    MouseSenderConfig config_{};
    float frac_x_ = 0.0F;
    float frac_y_ = 0.0F;
};

bool isLeftHoldEngageSatisfied(
    bool left_hold_engage,
    LeftHoldEngageButton engage_button,
    bool left_pressed,
    bool right_pressed);

void playToggleBeep(int frequency_hz, int duration_ms = 100);

bool sendVirtualKeyTap(std::uint16_t virtual_key, int hold_ms = 0);
bool sendLeftClickTap(double hold_s = 0.0);
bool sendRightClickTap(double hold_s = 0.0);

std::unique_ptr<IInputSender> makeInputSender();

}  // namespace delta
