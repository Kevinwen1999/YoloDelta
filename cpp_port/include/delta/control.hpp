#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
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
    MouseOutputMethod method = MouseOutputMethod::SendInput;
    std::string serial_port = "COM7";
    int serial_baud = 921600;
    std::uint64_t reconnect_token = 0;
    float gain_x = 1.0F;
    float gain_y = 1.0F;
    int max_step = 127;
};

enum class InputSenderState {
    Ready,
    Connecting,
    Connected,
    Disconnected,
};

inline const char* inputSenderStateName(const InputSenderState state) {
    switch (state) {
    case InputSenderState::Connecting: return "connecting";
    case InputSenderState::Connected: return "connected";
    case InputSenderState::Disconnected: return "disconnected";
    case InputSenderState::Ready:
    default: return "ready";
    }
}

struct InputSenderStatus {
    MouseOutputMethod method = MouseOutputMethod::SendInput;
    InputSenderState state = InputSenderState::Ready;
    bool serial_connected = false;
    std::string error;
};

std::array<std::uint8_t, 6> encodeSerialMouseFrame(int dx, int dy);

class ISerialMouseTransport {
public:
    virtual ~ISerialMouseTransport() = default;
    virtual bool open(const std::string& port, int baud, std::string& error) = 0;
    virtual void close() = 0;
    virtual bool isOpen() const = 0;
    virtual bool write(const std::uint8_t* data, std::size_t size, std::string& error) = 0;
};

std::unique_ptr<ISerialMouseTransport> makeSerialMouseTransport();

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
    virtual InputSenderStatus status() const = 0;
};

class SendInputMouseSender final : public IInputSender {
public:
    std::string_view name() const override { return "sendinput"; }
    void configure(const MouseSenderConfig& config) override;
    bool sendRelative(int dx, int dy) override;
    bool clickLeft(double hold_s) override;
    InputSenderStatus status() const override { return {}; }

private:
    MouseSenderConfig config_{};
    float frac_x_ = 0.0F;
    float frac_y_ = 0.0F;
};

class SerialMouseSender final : public IInputSender {
public:
    explicit SerialMouseSender(std::unique_ptr<ISerialMouseTransport> transport = makeSerialMouseTransport());
    ~SerialMouseSender() override;

    std::string_view name() const override { return "serial"; }
    void configure(const MouseSenderConfig& config) override;
    bool sendRelative(int dx, int dy) override;
    bool clickLeft(double hold_s) override;
    InputSenderStatus status() const override { return status_; }

private:
    void disconnect(std::string error = {});

    std::unique_ptr<ISerialMouseTransport> transport_;
    std::string port_;
    int baud_ = 921600;
    std::uint64_t reconnect_token_ = 0;
    InputSenderStatus status_{MouseOutputMethod::Serial, InputSenderState::Disconnected, false, {}};
};

class SwitchableMouseSender final : public IInputSender {
public:
    SwitchableMouseSender();
    SwitchableMouseSender(std::unique_ptr<IInputSender> sendinput, std::unique_ptr<IInputSender> serial);

    std::string_view name() const override;
    void configure(const MouseSenderConfig& config) override;
    bool sendRelative(int dx, int dy) override;
    bool clickLeft(double hold_s) override;
    InputSenderStatus status() const override;

private:
    MouseOutputMethod method_ = MouseOutputMethod::SendInput;
    std::unique_ptr<IInputSender> sendinput_;
    std::unique_ptr<IInputSender> serial_;
};

bool isLeftHoldEngageSatisfied(
    bool left_hold_engage,
    LeftHoldEngageButton engage_button,
    bool left_pressed,
    bool right_pressed,
    bool x1_pressed);

void playToggleBeep(int frequency_hz, int duration_ms = 100);

bool sendVirtualKeyTap(std::uint16_t virtual_key, double hold_ms = 0.0);
bool sendLeftClickTap(double hold_s = 0.0);
bool sendRightClickTap(double hold_s = 0.0);

std::unique_ptr<IInputSender> makeInputSender();

}  // namespace delta
