#include <winsock2.h>
#include <ws2tcpip.h>

#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>

#include "delta/frontend.hpp"

namespace {
std::string request(unsigned short port, const std::string& path, const std::string& body = {}) {
    SOCKET socket_handle = INVALID_SOCKET;
    for (int attempt = 0; attempt < 100; ++attempt) {
        socket_handle = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        if (connect(socket_handle, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) break;
        closesocket(socket_handle);
        socket_handle = INVALID_SOCKET;
        Sleep(10);
    }
    if (socket_handle == INVALID_SOCKET) throw std::runtime_error("Frontend did not start");
    DWORD timeout = 2000;
    setsockopt(socket_handle, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<char*>(&timeout), sizeof(timeout));
    const std::string message = (body.empty() ? "GET " : "POST ") + path +
                                " HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nContent-Type: "
                                "application/json\r\nContent-Length: " +
                                std::to_string(body.size()) + "\r\n\r\n" + body;
    for (std::size_t sent = 0; sent < message.size();) {
        const int count = send(socket_handle, message.data() + sent, int(message.size() - sent), 0);
        if (count <= 0) {
            closesocket(socket_handle);
            throw std::runtime_error("Frontend request send failed");
        }
        sent += count;
    }
    std::string response;
    char buffer[4096];
    int count;
    while ((count = recv(socket_handle, buffer, sizeof(buffer), 0)) > 0) response.append(buffer, count);
    closesocket(socket_handle);
    return response;
}
}  // namespace
int main() try {
    WSADATA wsa{};
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) throw std::runtime_error("WSA startup failed");
    SOCKET reservation = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (bind(reservation, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)))
        throw std::runtime_error("Port reservation failed");
    int length = sizeof(addr);
    getsockname(reservation, reinterpret_cast<sockaddr*>(&addr), &length);
    delta::StaticConfig config;
    config.frontend_port = ntohs(addr.sin_port);
    closesocket(reservation);
    delta::RuntimeConfigStore store;
    delta::SharedState shared;
    delta::RuntimeFrontendServer server(config, store, shared);
    server.start();
    const auto page = request(config.frontend_port, "/");
    for (const auto* key :
         {"calibrated_control_enable", "calibrated_pixels_per_count_x", "calibrated_pixels_per_count_y",
          "calibrated_response_time_s", "calibrated_response_delay_ms", "legacy_recoil_output_enable"})
        if (page.find(std::string("k:\"") + key + "\"") == std::string::npos)
            throw std::runtime_error("Missing frontend control");
    if (store.snapshot().legacy_recoil_output_enable)
        throw std::runtime_error("Legacy recoil output must default off");
    const auto result = request(
        config.frontend_port, "/api/pid",
        R"({"legacy_recoil_output_enable":true,"calibrated_control_enable":true,"calibrated_pixels_per_count_x":0.1,"calibrated_pixels_per_count_y":0.2,"calibrated_response_time_s":0.06,"calibrated_response_delay_ms":9})");
    if (result.find("200 OK") == std::string::npos) throw std::runtime_error("Configuration apply failed");
    const auto payload = nlohmann::json::parse(result.substr(result.find("\r\n\r\n") + 4));
    if (!store.snapshot().legacy_recoil_output_enable ||
        payload["config"]["legacy_recoil_output_enable"] != true ||
        !store.snapshot().calibrated_control_enable ||
        payload["config"]["calibrated_pixels_per_count_x"] != .1 ||
        payload["config"]["calibrated_pixels_per_count_y"] != .2 ||
        payload["config"]["calibrated_response_time_s"] != .06 ||
        payload["config"]["calibrated_response_delay_ms"] != 9)
        throw std::runtime_error("Frontend changes must reach runtime and round-trip through the API");
    for (const auto* invalid :
         {R"({"calibrated_pixels_per_count_x":0})", R"({"calibrated_response_time_s":0.001})",
          R"({"calibrated_response_delay_ms":51})"})
        if (request(config.frontend_port, "/api/pid", invalid).find("400 Bad Request") == std::string::npos)
            throw std::runtime_error("Invalid calibration must be rejected");
    if (request(config.frontend_port, "/api/pid", R"({"legacy_recoil_output_enable":false})").find("200 OK") == std::string::npos ||
        store.snapshot().legacy_recoil_output_enable)
        throw std::runtime_error("Legacy recoil output toggle must disable live compensation");
    const auto recoil_result = request(config.frontend_port, "/api/recoil",
        R"({"legacy_recoil_output_enable":true})");
    if (recoil_result.find("200 OK") == std::string::npos || !store.snapshot().legacy_recoil_output_enable)
        throw std::runtime_error("The recoil API must share the direct legacy output setting");
    const auto recoil_payload = nlohmann::json::parse(recoil_result.substr(recoil_result.find("\r\n\r\n") + 4));
    if (recoil_payload["config"]["legacy_recoil_output_enable"] != true ||
        recoil_payload["status"]["legacy_recoil_output_enable"] != true)
        throw std::runtime_error("Recoil status must expose enabled direct output even when F7 is off");
    if (request(config.frontend_port, "/api/recoil", R"({"legacy_recoil_output_enable":"true"})").find("400 Bad Request") == std::string::npos)
        throw std::runtime_error("The recoil API must reject invalid toggle values");
    server.stop();
    WSACleanup();
    std::cout << "Calibrated control frontend checks passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
