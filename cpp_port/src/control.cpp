#include "delta/control.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <sstream>
#include <thread>

#if defined(_WIN32)
#include <windows.h>
#endif

namespace delta {

namespace {
// Rate limit for silently retrying a dropped serial connection from the
// control loop, so a persistently absent/unresponsive device can't turn
// sendRelative() into a tight retry loop that starves hotkey polling.
constexpr std::chrono::milliseconds kSerialAutoReconnectInterval{1000};
}  // namespace

#if defined(_WIN32)
namespace {

constexpr UINT kMouseInputBatchLimit = 64U;

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

std::string win32ErrorMessage(const char* operation, const DWORD code = GetLastError()) {
    char* message = nullptr;
    const DWORD length = FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr,
        code,
        0,
        reinterpret_cast<char*>(&message),
        0,
        nullptr);
    std::ostringstream oss;
    oss << operation << " failed (Win32 " << code << ')';
    if (length > 0 && message != nullptr) {
        std::string detail(message, length);
        while (!detail.empty() && (detail.back() == '\r' || detail.back() == '\n' || detail.back() == ' ')) {
            detail.pop_back();
        }
        if (!detail.empty()) {
            oss << ": " << detail;
        }
    }
    if (message != nullptr) {
        LocalFree(message);
    }
    return oss.str();
}

std::wstring serialDevicePath(const std::string& port) {
    const std::string prefix = R"(\\.\)";
    const std::string normalized = port.rfind(prefix, 0) == 0 ? port : prefix + port;
    return std::wstring(normalized.begin(), normalized.end());
}

class Win32SerialMouseTransport final : public ISerialMouseTransport {
public:
    ~Win32SerialMouseTransport() override {
        close();
    }

    bool open(const std::string& port, const int baud, std::string& error) override {
        close();
        if (port.empty()) {
            error = "Serial port must not be empty.";
            return false;
        }
        if (baud <= 0) {
            error = "Serial baud must be positive.";
            return false;
        }

        handle_ = CreateFileW(
            serialDevicePath(port).c_str(),
            GENERIC_WRITE,
            0,
            nullptr,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL,
            nullptr);
        if (handle_ == INVALID_HANDLE_VALUE) {
            error = win32ErrorMessage("Opening serial port");
            return false;
        }

        auto fail = [this, &error](const char* operation) {
            error = win32ErrorMessage(operation);
            close();
            return false;
        };

        SetupComm(handle_, 4096, 4096);
        DCB dcb{};
        dcb.DCBlength = sizeof(dcb);
        if (!GetCommState(handle_, &dcb)) {
            return fail("GetCommState");
        }
        dcb.BaudRate = static_cast<DWORD>(baud);
        dcb.ByteSize = 8;
        dcb.Parity = NOPARITY;
        dcb.StopBits = ONESTOPBIT;
        dcb.fBinary = TRUE;
        dcb.fParity = FALSE;
        dcb.fOutxCtsFlow = FALSE;
        dcb.fOutxDsrFlow = FALSE;
        dcb.fDtrControl = DTR_CONTROL_DISABLE;
        dcb.fDsrSensitivity = FALSE;
        dcb.fOutX = FALSE;
        dcb.fInX = FALSE;
        dcb.fRtsControl = RTS_CONTROL_DISABLE;
        if (!SetCommState(handle_, &dcb)) {
            return fail("SetCommState");
        }

        COMMTIMEOUTS timeouts{};
        timeouts.WriteTotalTimeoutConstant = 500;
        if (!SetCommTimeouts(handle_, &timeouts)) {
            return fail("SetCommTimeouts");
        }
        EscapeCommFunction(handle_, CLRDTR);
        EscapeCommFunction(handle_, CLRRTS);
        PurgeComm(handle_, PURGE_TXABORT | PURGE_RXABORT | PURGE_TXCLEAR | PURGE_RXCLEAR);

        std::this_thread::sleep_for(std::chrono::seconds(2));
        error.clear();
        return true;
    }

    void close() override {
        if (handle_ != INVALID_HANDLE_VALUE) {
            CloseHandle(handle_);
            handle_ = INVALID_HANDLE_VALUE;
        }
    }

    bool isOpen() const override {
        return handle_ != INVALID_HANDLE_VALUE;
    }

    bool write(const std::uint8_t* data, const std::size_t size, std::string& error) override {
        if (!isOpen()) {
            error = "Serial port is not open.";
            return false;
        }
        if (size > static_cast<std::size_t>(std::numeric_limits<DWORD>::max())) {
            error = "Serial write is too large.";
            return false;
        }
        DWORD written = 0;
        if (!WriteFile(handle_, data, static_cast<DWORD>(size), &written, nullptr)) {
            error = win32ErrorMessage("Serial write");
            return false;
        }
        if (written != size) {
            error = "Serial write incomplete (" + std::to_string(written) + "/" + std::to_string(size) + " bytes).";
            return false;
        }
        error.clear();
        return true;
    }

private:
    HANDLE handle_ = INVALID_HANDLE_VALUE;
};

}  // namespace
#endif

#if !defined(_WIN32)
namespace {

class UnsupportedSerialMouseTransport final : public ISerialMouseTransport {
public:
    bool open(const std::string&, int, std::string& error) override {
        error = "Serial mouse output is supported on Windows only.";
        return false;
    }
    void close() override {}
    bool isOpen() const override { return false; }
    bool write(const std::uint8_t*, std::size_t, std::string& error) override {
        error = "Serial mouse output is supported on Windows only.";
        return false;
    }
};

}  // namespace
#endif

std::array<std::uint8_t, 6> encodeSerialMouseFrame(const int dx, const int dy) {
    constexpr std::uint8_t sync = 0xAA;
    const auto x = static_cast<std::uint16_t>(static_cast<std::int16_t>(std::clamp(dx, -32768, 32767)));
    const auto y = static_cast<std::uint16_t>(static_cast<std::int16_t>(std::clamp(dy, -32768, 32767)));
    const auto dxh = static_cast<std::uint8_t>((x >> 8U) & 0xFFU);
    const auto dxl = static_cast<std::uint8_t>(x & 0xFFU);
    const auto dyh = static_cast<std::uint8_t>((y >> 8U) & 0xFFU);
    const auto dyl = static_cast<std::uint8_t>(y & 0xFFU);
    return {sync, dxh, dxl, dyh, dyl, static_cast<std::uint8_t>(dxh ^ dxl ^ dyh ^ dyl)};
}

std::unique_ptr<ISerialMouseTransport> makeSerialMouseTransport() {
#if defined(_WIN32)
    return std::make_unique<Win32SerialMouseTransport>();
#else
    return std::make_unique<UnsupportedSerialMouseTransport>();
#endif
}

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

    std::array<INPUT, kMouseInputBatchLimit> inputs{};
    UINT input_count = 0;
    bool sent_any = false;
    auto flush_inputs = [&inputs, &input_count]() -> bool {
        if (input_count == 0) {
            return true;
        }
        const UINT count = input_count;
        input_count = 0;
        return SendInput(count, inputs.data(), sizeof(INPUT)) == count;
    };

    while (send_x != 0 || send_y != 0) {
        const int step_x = std::clamp(send_x, -config_.max_step, config_.max_step);
        const int step_y = std::clamp(send_y, -config_.max_step, config_.max_step);
        INPUT& input = inputs[input_count++];
        input = {};
        input.type = INPUT_MOUSE;
        input.mi.dx = step_x;
        input.mi.dy = step_y;
        input.mi.dwFlags = MOUSEEVENTF_MOVE;
        sent_any = true;
        send_x -= step_x;
        send_y -= step_y;

        if (input_count == kMouseInputBatchLimit && !flush_inputs()) {
            return false;
        }
    }
    if (!sent_any) {
        return false;
    }
    return flush_inputs();
#else
    (void)dx;
    (void)dy;
    return false;
#endif
}

bool SendInputMouseSender::clickLeft(double hold_s) {
    return sendLeftClickTap(hold_s);
}

SerialMouseSender::SerialMouseSender(std::unique_ptr<ISerialMouseTransport> transport)
    : transport_(std::move(transport)) {}

SerialMouseSender::~SerialMouseSender() {
    if (reconnect_worker_.joinable()) {
        reconnect_worker_.join();
    }
    disconnect();
}

void SerialMouseSender::setStatus(InputSenderStatus status) {
    std::lock_guard<std::mutex> lock(status_mutex_);
    status_ = std::move(status);
}

InputSenderStatus SerialMouseSender::status() const {
    std::lock_guard<std::mutex> lock(status_mutex_);
    return status_;
}

void SerialMouseSender::disconnect(std::string error) {
    std::lock_guard<std::mutex> lock(transport_mutex_);
    if (transport_) {
        transport_->close();
    }
    setStatus({MouseOutputMethod::Serial, InputSenderState::Disconnected, false, std::move(error)});
}

void SerialMouseSender::configure(const MouseSenderConfig& config) {
    // A background auto-reconnect attempt (see maybeStartAutoReconnect()) may be
    // using transport_/port_/baud_ right now. Reap it first so this explicit,
    // user-driven (re)configure never races it. By the time reconnect_worker_ is
    // joinable and we get here, any in-flight attempt has either already finished
    // or is about to — this does not wait on a fresh multi-second connect.
    if (reconnect_worker_.joinable()) {
        reconnect_worker_.join();
    }

    std::lock_guard<std::mutex> lock(transport_mutex_);
    const bool settings_changed = config.serial_port != port_ || config.serial_baud != baud_;
    const bool reconnect_requested = config.reconnect_token != reconnect_token_;
    port_ = config.serial_port;
    baud_ = config.serial_baud;
    reconnect_token_ = config.reconnect_token;

    if (config.method != MouseOutputMethod::Serial) {
        if (transport_) {
            transport_->close();
        }
        setStatus({MouseOutputMethod::Serial, InputSenderState::Disconnected, false, {}});
        return;
    }
    if (!settings_changed && !reconnect_requested && transport_ && transport_->isOpen()) {
        setStatus({MouseOutputMethod::Serial, InputSenderState::Connected, true, {}});
        return;
    }

    if (transport_) {
        transport_->close();
    }
    setStatus({MouseOutputMethod::Serial, InputSenderState::Connecting, false, {}});
    next_auto_reconnect_attempt_ = std::chrono::steady_clock::now() + kSerialAutoReconnectInterval;
    std::string error;
    if (!transport_ || !transport_->open(port_, baud_, error)) {
        setStatus({MouseOutputMethod::Serial, InputSenderState::Disconnected, false,
                   error.empty() ? "Failed to open serial mouse output." : std::move(error)});
        return;
    }
    setStatus({MouseOutputMethod::Serial, InputSenderState::Connected, true, {}});
}

void SerialMouseSender::maybeStartAutoReconnect() {
    const auto now = std::chrono::steady_clock::now();
    if (now < next_auto_reconnect_attempt_) {
        return;
    }
    next_auto_reconnect_attempt_ = now + kSerialAutoReconnectInterval;

    bool expected = false;
    if (!reconnect_in_progress_.compare_exchange_strong(expected, true)) {
        return;  // an attempt is already running
    }
    // reconnect_in_progress_ was false, so any previous worker already finished
    // and released transport_mutex_ before clearing that flag (see below) — this
    // join reaps a completed thread, it does not wait on a fresh connect.
    if (reconnect_worker_.joinable()) {
        reconnect_worker_.join();
    }
    reconnect_worker_ = std::thread([this]() {
        {
            std::lock_guard<std::mutex> lock(transport_mutex_);
            if (transport_ && !transport_->isOpen() && !port_.empty()) {
                std::string error;
                if (transport_->open(port_, baud_, error)) {
                    setStatus({MouseOutputMethod::Serial, InputSenderState::Connected, true, {}});
                } else {
                    setStatus({MouseOutputMethod::Serial, InputSenderState::Disconnected, false,
                               error.empty() ? "Failed to reconnect serial mouse output." : error});
                }
            }
        }
        reconnect_in_progress_ = false;
    });
}

bool SerialMouseSender::sendRelative(const int dx, const int dy) {
    if (dx == 0 && dy == 0) {
        return true;
    }
    // try_lock, not lock: if a background reconnect attempt currently owns
    // transport_ (including a multi-second open()), this must fail fast rather
    // than block the caller — sendRelative() runs on the real-time control loop
    // thread, which also polls hotkeys (including the panic-stop key) and paces
    // the triggerbot. Blocking it here previously caused exactly the "locks up
    // issuing left click constantly" symptom this whole reconnect path exists
    // to prevent.
    std::unique_lock<std::mutex> lock(transport_mutex_, std::try_to_lock);
    if (!lock.owns_lock()) {
        return false;
    }

    if (!transport_ || !transport_->isOpen()) {
        // A previous write/open failure disconnected the transport. Without
        // reconnecting, the sender stays dead forever after a single transient
        // hiccup: aim correction silently stops while the crosshair sits frozen
        // on whatever it last saw, which lets the triggerbot fire nonstop on a
        // target it can no longer track off of. Kick off a backoff-paced retry
        // on a background thread instead of giving up for good.
        {
            std::lock_guard<std::mutex> slock(status_mutex_);
            if (status_.error.empty()) {
                status_.error = "Serial port is not open.";
            }
            status_.state = InputSenderState::Disconnected;
            status_.serial_connected = false;
        }
        maybeStartAutoReconnect();
        return false;
    }
    const auto frame = encodeSerialMouseFrame(dx, dy);
    std::string error;
    if (!transport_->write(frame.data(), frame.size(), error)) {
        transport_->close();
        setStatus({MouseOutputMethod::Serial, InputSenderState::Disconnected, false,
                   error.empty() ? "Serial mouse write failed." : std::move(error)});
        return false;
    }
    return true;
}

bool SerialMouseSender::clickLeft(double) {
    return false;
}

SwitchableMouseSender::SwitchableMouseSender()
    : SwitchableMouseSender(
        std::make_unique<SendInputMouseSender>(),
        std::make_unique<SerialMouseSender>()) {}

SwitchableMouseSender::SwitchableMouseSender(
    std::unique_ptr<IInputSender> sendinput,
    std::unique_ptr<IInputSender> serial)
    : sendinput_(std::move(sendinput)),
      serial_(std::move(serial)) {}

std::string_view SwitchableMouseSender::name() const {
    const IInputSender* selected = method_ == MouseOutputMethod::Serial ? serial_.get() : sendinput_.get();
    return selected ? selected->name() : "none";
}

void SwitchableMouseSender::configure(const MouseSenderConfig& config) {
    method_ = config.method;
    if (sendinput_) {
        sendinput_->configure(config);
    }
    if (serial_) {
        serial_->configure(config);
    }
}

bool SwitchableMouseSender::sendRelative(const int dx, const int dy) {
    IInputSender* selected = method_ == MouseOutputMethod::Serial ? serial_.get() : sendinput_.get();
    return selected && selected->sendRelative(dx, dy);
}

bool SwitchableMouseSender::clickLeft(const double hold_s) {
    return sendinput_ && sendinput_->clickLeft(hold_s);
}

InputSenderStatus SwitchableMouseSender::status() const {
    const IInputSender* selected = method_ == MouseOutputMethod::Serial ? serial_.get() : sendinput_.get();
    if (!selected) {
        return {method_, InputSenderState::Disconnected, false, "Mouse output backend is unavailable."};
    }
    InputSenderStatus result = selected->status();
    result.method = method_;
    return result;
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
    return std::make_unique<SwitchableMouseSender>();
}

}  // namespace delta
