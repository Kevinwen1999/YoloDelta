#include "delta/app.hpp"
#include "delta/config.hpp"

#include <exception>
#include <iostream>

int main() {
    std::cout.setf(std::ios::unitbuf);
    try {
        delta::StaticConfig config{};
        delta::RuntimeConfig runtime{};
        delta::DeltaApp app(config, runtime);
        return app.run();
    } catch (const std::exception& error) {
        std::cerr << "[fatal] " << error.what() << "\n";
        return 1;
    }
}
