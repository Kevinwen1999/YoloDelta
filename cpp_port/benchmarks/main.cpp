#include "benchmark.hpp"

#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
using delta::bench::Json;
Json read(const std::filesystem::path& path) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("Cannot read "+path.string());
    Json value; stream >> value; return value;
}
void write(const std::filesystem::path& path,const Json& value) {
    std::ofstream stream(path);
    if (!stream) throw std::runtime_error("Cannot write "+path.string());
    stream << value.dump(2) << '\n';
    if (!stream) throw std::runtime_error("Failed to finish "+path.string());
}
}

int main(int argc,char** argv) {
    try {
        delta::bench::Options options;
        delta::StaticConfig screen;
        delta::RuntimeConfig runtime;
        std::filesystem::path output="cpp_port/benchmark-results/latest", baseline_path, config_path, dump_path;
        for (int i=1;i<argc;++i) {
            const std::string arg=argv[i];
            auto value=[&]() -> std::string { if (++i>=argc) throw std::invalid_argument("Missing value for "+arg); return argv[i]; };
            if (arg=="--help") {
                std::cout << "delta_benchmark [--suite quick|sweep|speed|speed_sizes] [--scenario NAME] [--seed N]\n"
                    "  [--config runtime.json] [--output DIRECTORY] [--baseline run.json]\n"
                    "  [--timing] [--repeats N] [--loss-deadline-ms N]\n"
                    "  [--write-default-config PATH]\n"
                    "Output directories must not already contain run.json or report.html.\n"
                    "Exit 0: completed; 1: harness/input error; 2: correctness violation.\n"
                    "Quality differences are informational until thresholds are calibrated.\n";
                return 0;
            } else if (arg=="--suite") options.suite=value();
            else if (arg=="--scenario") options.scenario=value();
            else if (arg=="--seed") options.seed=std::stoull(value());
            else if (arg=="--repeats") options.repeats=std::stoi(value());
            else if (arg=="--loss-deadline-ms") options.loss_deadline_ms=std::stod(value());
            else if (arg=="--output") output=value();
            else if (arg=="--baseline") baseline_path=value();
            else if (arg=="--config") config_path=value();
            else if (arg=="--write-default-config") dump_path=value();
            else if (arg=="--timing") options.timing=true;
            else throw std::invalid_argument("Unknown option: "+arg);
        }
        delta::bench::validateOptions(options);
        if (!dump_path.empty()) {
            if (std::filesystem::exists(dump_path)) throw std::runtime_error("Refusing to replace configuration snapshot");
            write(dump_path,delta::bench::runtimeJson(runtime));
            std::cout << "Saved current compiled defaults to " << dump_path << '\n'; return 0;
        }
        if (!config_path.empty()) runtime=delta::bench::runtimeFromJson(read(config_path));
        Json baseline=nullptr;
        if (!baseline_path.empty()) baseline=read(baseline_path);
        if (std::filesystem::exists(output/"run.json") || std::filesystem::exists(output/"report.html"))
            throw std::runtime_error("Choose a new output directory; benchmark artifacts are immutable");
        std::cout << "Running " << options.suite << " C++ replay suite" << (options.timing?" with isolated timing passes":"") << "...\n";
        auto run=delta::bench::runSuite(screen,runtime,options);
        if (!baseline.is_null()) run["comparison"]=delta::bench::compareRuns(baseline,run);
        std::filesystem::create_directories(output);
        write(output/"run.json",run);
        delta::bench::writeReport(output/"report.html",run,baseline);
        std::cout << "Scenarios: " << run["runs"].size() << "; correctness violations: " << run["correctness_violations"].size() << '\n';
        for (const auto& item : run["runs"]) {
            const auto& m=item["metrics"];
            std::cout << item["key"].get<std::string>() << ": wrong=" << m["wrong_target_ms"] << "ms switches=" << m["identity_switches"]
                << " error=" << m["mean_normalized_error"] << " coverage=" << m["prediction_coverage"] << '\n';
        }
        std::cout << "Report: " << std::filesystem::absolute(output/"report.html") << '\n';
        return run["correctness_violations"].empty()?0:2;
    } catch (const std::exception& error) {
        std::cerr << "[benchmark] " << error.what() << '\n'; return 1;
    }
}
