#pragma once

#include <atomic>
#include <memory>
#include <thread>

#include "delta/config.hpp"
#include "delta/runtime_state.hpp"

namespace delta {

class RuntimeFrontendServer {
public:
    RuntimeFrontendServer(const StaticConfig& config, RuntimeConfigStore& config_store, SharedState& shared_state);
    ~RuntimeFrontendServer();

    void start();
    void stop();

private:
    void serve();

    StaticConfig config_;
    RuntimeConfigStore& config_store_;
    SharedState& shared_state_;
    std::atomic<bool> running_{false};
    std::thread thread_;
};

std::unique_ptr<RuntimeFrontendServer> makeRuntimeFrontend(const StaticConfig& config, RuntimeConfigStore& config_store, SharedState& shared_state);

}  // namespace delta
