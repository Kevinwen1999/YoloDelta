#include "benchmark.hpp"
#include "benchmark_report.hpp"
#include <fstream>
#include <stdexcept>

namespace delta::bench {
void writeReport(const std::filesystem::path& path,const Json& run,const Json& baseline) {
    std::string html=benchmarkReportTemplate;
    // JSON is embedded as data, never executable JavaScript. Escaping '<' also
    // prevents embedded text from terminating its script element.
    auto escape=[](const Json& data) {
        const auto raw=data.dump(); std::string escaped; escaped.reserve(raw.size());
        for (char c : raw) { if (c=='<') escaped+="\\u003c"; else escaped+=c; }
        return escaped;
    };
    const std::string run_token="__DELTA_RUN_JSON__", baseline_token="__DELTA_BASELINE_JSON__";
    const auto run_offset=html.find(run_token), baseline_offset=html.find(baseline_token);
    if (run_offset==std::string::npos || baseline_offset==std::string::npos || run_offset>=baseline_offset)
        throw std::runtime_error("Missing or reordered report template tokens");
    // Replace back-to-front at ORIGINAL offsets: strings inside user data may
    // themselves contain a token, and must never be interpreted as placeholders.
    html.replace(baseline_offset,baseline_token.size(),escape(baseline));
    html.replace(run_offset,run_token.size(),escape(run));
    std::ofstream stream(path,std::ios::binary);
    stream << html;
    if (!stream) throw std::runtime_error("Cannot write report "+path.string());
}
}
