#include "delta/inference.hpp"
#include "delta/gpu_preprocess.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#endif

#if defined(DELTA_WITH_CUDA_PIPELINE)
#include <cuda_runtime.h>
#endif

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"

namespace delta {
namespace fs = std::filesystem;

namespace {

struct FlatTensor {
    std::vector<float> data;
    std::vector<int64_t> shape;
};

struct Candidate {
    std::array<float, 4> box{};
    int cls = -1;
    float conf = 0.0F;
};

enum class ProviderPreference {
    Cpu,
    Cuda,
    TensorRt,
};

double ms(const SteadyClock::time_point a, const SteadyClock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

std::optional<std::string> envVar(const char* key) {
    if (const char* value = std::getenv(key); value && *value) {
        return std::string(value);
    }
    return std::nullopt;
}

bool fileExists(const fs::path& path) {
    std::error_code ec;
    return fs::exists(path, ec) && fs::is_regular_file(path, ec);
}

bool directoryExists(const fs::path& path) {
    std::error_code ec;
    return fs::exists(path, ec) && fs::is_directory(path, ec);
}

std::string lowerCopy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

bool envFlag(const char* key) {
    if (auto value = envVar(key); value.has_value()) {
        const std::string lowered = lowerCopy(*value);
        return lowered == "1" || lowered == "true" || lowered == "yes" || lowered == "on";
    }
    return false;
}

bool hasDllPrefix(const fs::path& dir, const std::string& prefix) {
    if (!directoryExists(dir)) return false;
    std::error_code ec;
    for (const auto& entry : fs::directory_iterator(dir, ec)) {
        if (ec || !entry.is_regular_file(ec)) continue;
        const std::string name = lowerCopy(entry.path().filename().string());
        if (name.rfind(lowerCopy(prefix), 0) == 0 && entry.path().extension() == ".dll") {
            return true;
        }
    }
    return false;
}

#if defined(DELTA_WITH_CUDA_PIPELINE)
void checkCuda(cudaError_t status, std::string_view what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + " failed: " + cudaGetErrorString(status));
    }
}
#endif

void addAncestorCandidates(std::vector<fs::path>& out, fs::path start, const std::vector<fs::path>& suffixes) {
    for (int depth = 0; depth < 6 && !start.empty(); ++depth) {
        for (const auto& suffix : suffixes) {
            out.push_back(start / suffix);
        }
        const fs::path parent = start.parent_path();
        if (parent.empty() || parent == start) {
            break;
        }
        start = parent;
    }
}

fs::path exeDir() {
#if defined(_WIN32)
    std::wstring buffer(MAX_PATH, L'\0');
    DWORD size = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (size == 0) {
        return {};
    }
    buffer.resize(size);
    return fs::path(buffer).parent_path();
#else
    return {};
#endif
}

float iou(const std::array<float, 4>& a, const std::array<float, 4>& b) {
    const float x1 = std::max(a[0], b[0]);
    const float y1 = std::max(a[1], b[1]);
    const float x2 = std::min(a[2], b[2]);
    const float y2 = std::min(a[3], b[3]);
    const float inter = std::max(0.0F, x2 - x1) * std::max(0.0F, y2 - y1);
    const float area_a = std::max(0.0F, a[2] - a[0]) * std::max(0.0F, a[3] - a[1]);
    const float area_b = std::max(0.0F, b[2] - b[0]) * std::max(0.0F, b[3] - b[1]);
    const float denom = area_a + area_b - inter;
    return denom > 1e-6F ? inter / denom : 0.0F;
}

std::vector<size_t> nms(
    const std::vector<std::array<float, 4>>& boxes,
    const std::vector<float>& scores,
    const float iou_thresh,
    const size_t max_det) {
    std::vector<size_t> order(scores.size());
    std::iota(order.begin(), order.end(), size_t{0});
    std::sort(order.begin(), order.end(), [&](size_t l, size_t r) { return scores[l] > scores[r]; });
    std::vector<size_t> keep;
    while (!order.empty() && keep.size() < max_det) {
        const size_t cur = order.front();
        keep.push_back(cur);
        std::vector<size_t> next;
        for (size_t i = 1; i < order.size(); ++i) {
            if (iou(boxes[cur], boxes[order[i]]) <= iou_thresh) {
                next.push_back(order[i]);
            }
        }
        order.swap(next);
    }
    return keep;
}

void resizeNearest(
    const std::uint8_t* src,
    const int src_w,
    const int src_h,
    std::vector<std::uint8_t>& dst,
    const int dst_w,
    const int dst_h) {
    dst.resize(static_cast<size_t>(dst_w) * static_cast<size_t>(dst_h) * 3U);
    for (int y = 0; y < dst_h; ++y) {
        const int sy = std::min((y * src_h) / std::max(1, dst_h), src_h - 1);
        for (int x = 0; x < dst_w; ++x) {
            const int sx = std::min((x * src_w) / std::max(1, dst_w), src_w - 1);
            const size_t so = (static_cast<size_t>(sy) * static_cast<size_t>(src_w) + static_cast<size_t>(sx)) * 3U;
            const size_t doff = (static_cast<size_t>(y) * static_cast<size_t>(dst_w) + static_cast<size_t>(x)) * 3U;
            dst[doff + 0U] = src[so + 0U];
            dst[doff + 1U] = src[so + 1U];
            dst[doff + 2U] = src[so + 2U];
        }
    }
}

FlatTensor toFlatTensor(const Ort::Value& value) {
    FlatTensor out{};
    auto info = value.GetTensorTypeAndShapeInfo();
    out.shape = info.GetShape();
    const size_t count = info.GetElementCount();
    out.data.resize(count);
    if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        const float* ptr = value.GetTensorData<float>();
        out.data.assign(ptr, ptr + count);
    } else if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        const Ort::Float16_t* ptr = value.GetTensorData<Ort::Float16_t>();
        for (size_t i = 0; i < count; ++i) {
            out.data[i] = static_cast<float>(ptr[i]);
        }
    } else {
        throw std::runtime_error("Unsupported ONNX output tensor type.");
    }
    return out;
}

Detection makeDetection(const Candidate& c) {
    Detection d{};
    d.bbox = {static_cast<int>(c.box[0]), static_cast<int>(c.box[1]), static_cast<int>(c.box[2]), static_cast<int>(c.box[3])};
    d.x = 0.5F * (c.box[0] + c.box[2]);
    d.y = 0.5F * (c.box[1] + c.box[3]);
    d.cls = c.cls;
    d.conf = c.conf;
    return d;
}

}  // namespace

struct OnnxRuntimeEngine::Impl {
    explicit Impl(StaticConfig cfg) : config(std::move(cfg)) { init(); }
    ~Impl() {
        session.reset();
        device_memory_info.reset();
        memory_info.reset();
        allocator.reset();
        env.reset();
#if defined(DELTA_WITH_CUDA_PIPELINE)
        if (gpu_input_buffer != nullptr) {
            cudaFree(gpu_input_buffer);
            gpu_input_buffer = nullptr;
        }
        if (ort_stream != nullptr) {
            cudaStreamDestroy(ort_stream);
            ort_stream = nullptr;
        }
#endif
#if defined(_WIN32)
        for (const auto cookie : dll_dir_cookies) {
            if (cookie) {
                RemoveDllDirectory(cookie);
            }
        }
#endif
    }

    StaticConfig config{};
    std::string name = "onnxruntime[unavailable]";
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::AllocatorWithDefaultOptions> allocator;
    std::unique_ptr<Ort::MemoryInfo> memory_info;
    std::unique_ptr<Ort::MemoryInfo> device_memory_info;
    std::unique_ptr<Ort::Session> session;
    fs::path runtime_root;
    fs::path model_path;
    fs::path cuda_bin_dir;
    fs::path tensorrt_lib_dir;
    std::string input_name;
    std::vector<std::string> output_names;
    std::vector<const char*> output_name_ptrs;
    std::array<int64_t, 4> input_shape{1, 3, 0, 0};
    ONNXTensorElementDataType input_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    int input_w = 0;
    int input_h = 0;
    size_t input_tensor_bytes = 0;
    std::vector<std::uint8_t> resized_bgr;
    std::vector<float> input_f32;
    std::vector<Ort::Float16_t> input_f16;
    bool session_uses_gpu = false;
    bool gpu_input_ready = false;
    float model_conf_override = -1.0F;
#if defined(DELTA_WITH_CUDA_PIPELINE)
    void* gpu_input_buffer = nullptr;
    cudaStream_t ort_stream = nullptr;
#endif
#if defined(_WIN32)
    HMODULE runtime_module = nullptr;
    std::vector<DLL_DIRECTORY_COOKIE> dll_dir_cookies;
#endif

    void warmup() {
        if (!session) return;
        FramePacket frame{};
        frame.width = input_w;
        frame.height = input_h;
        frame.bgr.assign(static_cast<size_t>(input_w) * static_cast<size_t>(input_h) * 3U, 0U);
        (void)predict(frame, 0);
    }

    InferenceResult predict(const FramePacket& frame, const int target_class) {
        InferenceResult result{};
        if (!session || frame.bgr.empty() || frame.width <= 0 || frame.height <= 0) {
            return result;
        }
        const auto t0 = SteadyClock::now();
        preprocess(frame);
        const auto t1 = SteadyClock::now();
        result.timings.preprocess_ms = ms(t0, t1);
        Ort::RunOptions run_options;
        Ort::Value input = makeInput();
        const char* input_names[] = {input_name.c_str()};
        const auto t2 = SteadyClock::now();
        std::vector<Ort::Value> outputs = session->Run(run_options, input_names, &input, 1, output_name_ptrs.data(), output_name_ptrs.size());
        const auto t3 = SteadyClock::now();
        result.timings.execute_ms = ms(t2, t3);
        const auto t4 = SteadyClock::now();
        if (!outputs.empty()) {
            if (config.onnx_output_has_nms) {
                if (auto decoded = decodeNms(toFlatTensor(outputs.front()), frame.width, frame.height, target_class); decoded.has_value()) {
                    result.detections = std::move(decoded.value());
                    result.timings.postprocess_ms = ms(t4, SteadyClock::now());
                    return result;
                }
            }
            for (const auto& out : outputs) {
                if (auto decoded = decodeNms(toFlatTensor(out), frame.width, frame.height, target_class); decoded.has_value()) {
                    result.detections = std::move(decoded.value());
                    result.timings.postprocess_ms = ms(t4, SteadyClock::now());
                    return result;
                }
            }
            result.detections = decodeRaw(toFlatTensor(outputs.front()), frame.width, frame.height, target_class);
        }
        result.timings.postprocess_ms = ms(t4, SteadyClock::now());
        return result;
    }

    InferenceResult predictGpu(const GpuFramePacket& frame, const int target_class);

    void init();
    ProviderPreference requestedProvider() const;
    fs::path findRuntimeRoot() const;
    fs::path findModelPath() const;
    fs::path findCudaBinDir() const;
    fs::path findTensorRtLibDir() const;
    void configureRuntimeEnvironment(const fs::path& root);
    void loadRuntime(const fs::path& root);
    void createSession(bool use_tensorrt);
    void cacheMetadata();
    void initializeGpuInput();
    void preprocess(const FramePacket& frame);
    Ort::Value makeInput();
    Ort::Value makeGpuInput(const GpuFramePacket& frame);
    std::optional<std::vector<Detection>> decodeNms(const FlatTensor& tensor, int fw, int fh, int target_class) const;
    std::vector<Detection> decodeRaw(const FlatTensor& tensor, int fw, int fh, int target_class) const;
    bool wantsCuda() const { return requestedProvider() != ProviderPreference::Cpu; }
    bool wantsTensorRt() const { return requestedProvider() == ProviderPreference::TensorRt; }
    bool classMatch(int cls, int target_class) const { return target_class < 0 || cls == target_class; }
    float effectiveModelConf() const {
        if (model_conf_override >= 0.0F) {
            return std::clamp(model_conf_override, 0.0F, 1.0F);
        }
        return std::clamp(config.conf, 0.0F, 1.0F);
    }
};

ProviderPreference OnnxRuntimeEngine::Impl::requestedProvider() const {
    const auto parse = [](std::string value) -> std::optional<ProviderPreference> {
        value = lowerCopy(std::move(value));
        if (value == "cpu") return ProviderPreference::Cpu;
        if (value == "cuda" || value == "gpu") return ProviderPreference::Cuda;
        if (value == "tensorrt" || value == "trt") return ProviderPreference::TensorRt;
        return std::nullopt;
    };
    if (auto env_provider = envVar("DELTA_ONNX_PROVIDER"); env_provider.has_value()) {
        if (const auto parsed = parse(*env_provider); parsed.has_value()) {
            return *parsed;
        }
    }
    if (lowerCopy(config.inference_device) != "cuda") {
        return ProviderPreference::Cpu;
    }
    if (!config.onnx_provider.empty() && lowerCopy(config.onnx_provider) != "auto") {
        if (const auto parsed = parse(config.onnx_provider); parsed.has_value()) {
            return *parsed;
        }
    }
    return config.onnx_use_tensorrt ? ProviderPreference::TensorRt : ProviderPreference::Cuda;
}

void OnnxRuntimeEngine::Impl::init() {
    runtime_root = findRuntimeRoot();
    model_path = findModelPath();
    loadRuntime(runtime_root);
    env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "delta-native");
    allocator = std::make_unique<Ort::AllocatorWithDefaultOptions>();
    memory_info = std::make_unique<Ort::MemoryInfo>(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
#if defined(DELTA_WITH_CUDA_PIPELINE)
    if (wantsCuda()) {
        checkCuda(cudaSetDevice(config.onnx_cuda_device_id), "cudaSetDevice");
        checkCuda(cudaStreamCreate(&ort_stream), "cudaStreamCreate(onnx)");
    }
#endif
    try {
        createSession(wantsTensorRt());
    } catch (const std::exception& e) {
        if (wantsTensorRt()) {
            std::cout << "[inference] TensorRT EP init failed; retrying with CUDA EP only. reason: " << e.what() << "\n";
            try {
                createSession(false);
            } catch (const std::exception& cuda_error) {
                if (config.onnx_require_gpu || envFlag("DELTA_ONNX_REQUIRE_GPU")) {
                    throw;
                }
                std::cout << "[inference] CUDA EP init failed; falling back to CPU. reason: " << cuda_error.what() << "\n";
                config.inference_device = "cpu";
                config.onnx_provider = "cpu";
                createSession(false);
            }
        } else if (wantsCuda()) {
            if (config.onnx_require_gpu || envFlag("DELTA_ONNX_REQUIRE_GPU")) {
                throw;
            }
            std::cout << "[inference] CUDA EP init failed; falling back to CPU. reason: " << e.what() << "\n";
            config.inference_device = "cpu";
            config.onnx_provider = "cpu";
            createSession(false);
        } else {
            throw;
        }
    }
    cacheMetadata();
    initializeGpuInput();
}

fs::path OnnxRuntimeEngine::Impl::findRuntimeRoot() const {
    std::vector<fs::path> candidates;
    if (auto env_root = envVar("DELTA_ONNXRUNTIME_ROOT"); env_root) candidates.emplace_back(*env_root);
    if (!config.onnxruntime_root.empty()) candidates.emplace_back(config.onnxruntime_root);
    candidates.push_back(exeDir());
    candidates.push_back(exeDir() / "onnxruntime");
    addAncestorCandidates(
        candidates,
        exeDir(),
        {fs::path("runtime") / "onnxruntime", fs::path("cpp_port") / "runtime" / "onnxruntime"});
    candidates.push_back(fs::current_path());
    candidates.push_back(fs::current_path() / "onnxruntime");
    candidates.push_back(fs::current_path() / "runtime" / "onnxruntime");
    addAncestorCandidates(
        candidates,
        fs::current_path(),
        {fs::path("runtime") / "onnxruntime", fs::path("cpp_port") / "runtime" / "onnxruntime"});
    if (auto local = envVar("LOCALAPPDATA"); local) {
        const fs::path py_root = fs::path(*local) / "Programs" / "Python";
        std::error_code ec;
        if (fs::exists(py_root, ec)) {
            for (const auto& entry : fs::directory_iterator(py_root, ec)) {
                candidates.push_back(entry.path() / "Lib" / "site-packages" / "onnxruntime");
                candidates.push_back(entry.path() / "Lib" / "site-packages" / "onnxruntime" / "capi");
            }
        }
    }
    for (const auto& candidate : candidates) {
        if (fileExists(candidate / "onnxruntime.dll")) return candidate;
    }
    throw std::runtime_error("Unable to find onnxruntime.dll. Set DELTA_ONNXRUNTIME_ROOT or copy it into cpp_port/runtime/onnxruntime.");
}

fs::path OnnxRuntimeEngine::Impl::findModelPath() const {
    std::vector<fs::path> candidates;
    if (auto env_model = envVar("DELTA_MODEL_PATH"); env_model) candidates.emplace_back(*env_model);
    if (!config.model_path.empty()) {
        candidates.emplace_back(config.model_path);
        fs::path onnx_path(config.model_path);
        if (onnx_path.extension() != ".onnx") candidates.push_back(onnx_path.replace_extension(".onnx"));
    }
    candidates.push_back(fs::current_path() / "models" / "best.onnx");
    addAncestorCandidates(
        candidates,
        exeDir(),
        {fs::path("models") / "best.onnx", fs::path("cpp_port") / "models" / "best.onnx"});
    addAncestorCandidates(
        candidates,
        fs::current_path(),
        {fs::path("models") / "best.onnx", fs::path("cpp_port") / "models" / "best.onnx"});
    for (const auto& candidate : candidates) {
        if (fileExists(candidate)) return candidate;
    }
    throw std::runtime_error("Unable to find an ONNX model. Set DELTA_MODEL_PATH or copy best.onnx into cpp_port/models.");
}

fs::path OnnxRuntimeEngine::Impl::findCudaBinDir() const {
    std::vector<fs::path> candidates;
    if (auto env_root = envVar("DELTA_CUDA_ROOT"); env_root) candidates.emplace_back(*env_root);
    if (!config.cuda_root.empty()) candidates.emplace_back(config.cuda_root);
    if (auto env_root = envVar("CUDA_PATH"); env_root) candidates.emplace_back(*env_root);
    if (auto env_root = envVar("CUDA_PATH_V12_4"); env_root) candidates.emplace_back(*env_root);
    if (auto env_root = envVar("ProgramFiles"); env_root) {
        const fs::path base = fs::path(*env_root) / "NVIDIA GPU Computing Toolkit" / "CUDA";
        std::error_code ec;
        if (directoryExists(base)) {
            for (const auto& entry : fs::directory_iterator(base, ec)) {
                if (!ec && entry.is_directory(ec)) {
                    candidates.push_back(entry.path());
                }
            }
        }
    }
    for (const auto& candidate : candidates) {
        if (hasDllPrefix(candidate, "cudart64_")) return candidate;
        if (hasDllPrefix(candidate / "bin", "cudart64_")) return candidate / "bin";
    }
    return {};
}

fs::path OnnxRuntimeEngine::Impl::findTensorRtLibDir() const {
    std::vector<fs::path> candidates;
    if (auto env_root = envVar("DELTA_TENSORRT_ROOT"); env_root) candidates.emplace_back(*env_root);
    if (!config.tensorrt_root.empty()) candidates.emplace_back(config.tensorrt_root);
    candidates.push_back(fs::current_path() / "TensorRT");
    candidates.push_back(fs::current_path() / "TensorRT-10.9.0.34");
    addAncestorCandidates(candidates, exeDir(), {fs::path("TensorRT"), fs::path("TensorRT-10.9.0.34"), fs::path("cpp_port") / "TensorRT-10.9.0.34"});
    if (auto system_drive = envVar("SystemDrive"); system_drive) {
        const fs::path sdk_root = fs::path(*system_drive + std::string("\\SDKs"));
        std::error_code ec;
        if (directoryExists(sdk_root)) {
            for (const auto& entry : fs::directory_iterator(sdk_root, ec)) {
                if (!ec && entry.is_directory(ec) && lowerCopy(entry.path().filename().string()).find("tensorrt") != std::string::npos) {
                    candidates.push_back(entry.path());
                }
            }
        }
    }
    for (const auto& candidate : candidates) {
        if (hasDllPrefix(candidate, "nvinfer_")) return candidate;
        if (hasDllPrefix(candidate / "lib", "nvinfer_")) return candidate / "lib";
    }
    return {};
}

void OnnxRuntimeEngine::Impl::configureRuntimeEnvironment(const fs::path& root) {
#if defined(_WIN32)
    dll_dir_cookies.clear();
    SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_USER_DIRS);
    const auto add_dir = [&](const fs::path& dir, const char* label) {
        if (!directoryExists(dir)) return;
        if (const DLL_DIRECTORY_COOKIE cookie = AddDllDirectory(dir.c_str()); cookie) {
            dll_dir_cookies.push_back(cookie);
            if (config.debug_log) {
                std::cout << "[inference] Added " << label << " DLL directory: " << dir.string() << "\n";
            }
        } else if (config.debug_log) {
            std::cout << "[inference] Failed to add " << label << " DLL directory: " << dir.string() << "\n";
        }
    };
    add_dir(root, "onnxruntime");
    if (wantsCuda()) {
        cuda_bin_dir = findCudaBinDir();
        if (!cuda_bin_dir.empty()) {
            add_dir(cuda_bin_dir, "cuda");
        } else if (config.debug_log) {
            std::cout << "[inference] CUDA runtime directory not found; GPU provider load will rely on PATH.\n";
        }
    }
    if (wantsTensorRt()) {
        tensorrt_lib_dir = findTensorRtLibDir();
        if (!tensorrt_lib_dir.empty()) {
            add_dir(tensorrt_lib_dir, "tensorrt");
        } else if (config.debug_log) {
            std::cout << "[inference] TensorRT library directory not found; TensorRT EP may fail to initialize.\n";
        }
    }
    if (config.debug_log) {
        std::cout << "[inference] Runtime root: " << root.string() << "\n";
        std::cout << "[inference] Model path: " << model_path.string() << "\n";
    }
#else
    (void)root;
#endif
}

void OnnxRuntimeEngine::Impl::loadRuntime(const fs::path& root) {
#if defined(_WIN32)
    configureRuntimeEnvironment(root);
    const fs::path dll = root / "onnxruntime.dll";
    runtime_module = LoadLibraryExW(dll.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!runtime_module) throw std::runtime_error("Failed to load onnxruntime.dll from " + dll.string());
    using GetApiBaseFn = const OrtApiBase*(ORT_API_CALL*)();
    const auto get_api_base = reinterpret_cast<GetApiBaseFn>(GetProcAddress(runtime_module, "OrtGetApiBase"));
    if (!get_api_base) throw std::runtime_error("Failed to resolve OrtGetApiBase.");
    const OrtApi* api = get_api_base()->GetApi(ORT_API_VERSION);
    if (!api) throw std::runtime_error("The ONNX Runtime DLL does not support the requested C API version.");
    Ort::InitApi(api);
#else
    (void)root;
    throw std::runtime_error("Dynamic ONNX Runtime loading is implemented for Windows only.");
#endif
}

void OnnxRuntimeEngine::Impl::createSession(const bool use_tensorrt) {
    Ort::SessionOptions so;
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    so.EnableMemPattern();
    so.EnableCpuMemArena();
    std::vector<std::string> providers;
    session_uses_gpu = false;
    if (wantsCuda()) {
        if (use_tensorrt) {
            Ort::TensorRTProviderOptions trt;
            fs::path trt_cache = config.tensorrt_cache_dir.empty()
                ? model_path.parent_path() / "trt_cache"
                : fs::path(config.tensorrt_cache_dir);
            std::error_code ec;
            fs::create_directories(trt_cache, ec);
            trt.Update({
                {"device_id", std::to_string(config.onnx_cuda_device_id)},
                {"trt_fp16_enable", config.onnx_trt_fp16 ? "1" : "0"},
                {"trt_engine_cache_enable", "1"},
                {"trt_engine_cache_path", trt_cache.string()},
                {"trt_cuda_graph_enable", config.onnx_trt_cuda_graph_enable ? "1" : "0"},
            });
#if defined(DELTA_WITH_CUDA_PIPELINE)
            if (ort_stream != nullptr) {
                trt.UpdateWithValue("user_compute_stream", ort_stream);
            }
#endif
            so.AppendExecutionProvider_TensorRT_V2(*trt);
            providers.push_back("TensorrtExecutionProvider");
            session_uses_gpu = true;
        }
        Ort::CUDAProviderOptions cuda;
        cuda.Update({
            {"device_id", std::to_string(config.onnx_cuda_device_id)},
            {"arena_extend_strategy", "kNextPowerOfTwo"},
            {"cudnn_conv_algo_search", "EXHAUSTIVE"},
            {"do_copy_in_default_stream", "1"},
            {"cudnn_conv_use_max_workspace", "1"},
            {"enable_cuda_graph", config.onnx_enable_cuda_graph ? "1" : "0"},
        });
#if defined(DELTA_WITH_CUDA_PIPELINE)
        if (ort_stream != nullptr) {
            cuda.UpdateWithValue("user_compute_stream", ort_stream);
        }
#endif
        so.AppendExecutionProvider_CUDA_V2(*cuda);
        providers.push_back("CUDAExecutionProvider");
        session_uses_gpu = true;
    }
    providers.push_back("CPUExecutionProvider");
    session = std::make_unique<Ort::Session>(*env, model_path.c_str(), so);
    name = "onnxruntime[" + [&]() {
        std::string joined;
        for (size_t i = 0; i < providers.size(); ++i) {
            if (i) joined += ",";
            joined += providers[i];
        }
        return joined;
    }() + "]";
}

void OnnxRuntimeEngine::Impl::cacheMetadata() {
    if (session->GetInputCount() == 0) throw std::runtime_error("The ONNX model has no inputs.");
    {
        auto alloc = session->GetInputNameAllocated(0, *allocator);
        input_name = alloc.get() ? alloc.get() : "images";
    }
    {
        auto type_info = session->GetInputTypeInfo(0);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        const auto shape = tensor_info.GetShape();
        if (shape.size() < 4) throw std::runtime_error("Expected an NCHW input tensor.");
        input_type = tensor_info.GetElementType();
        input_h = shape[2] > 0 ? static_cast<int>(shape[2]) : config.imgsz;
        input_w = shape[3] > 0 ? static_cast<int>(shape[3]) : config.imgsz;
        input_shape = {1, 3, input_h, input_w};
    }
    const size_t pixels = static_cast<size_t>(input_w) * static_cast<size_t>(input_h);
    input_tensor_bytes = pixels * 3U * (input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ? sizeof(Ort::Float16_t) : sizeof(float));
    resized_bgr.resize(pixels * 3U);
    if (input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        input_f16.resize(pixels * 3U);
    } else if (input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        input_f32.resize(pixels * 3U);
    } else {
        throw std::runtime_error("Only float32 and float16 ONNX inputs are supported.");
    }
    output_names.clear();
    output_name_ptrs.clear();
    for (size_t i = 0; i < session->GetOutputCount(); ++i) {
        auto alloc = session->GetOutputNameAllocated(i, *allocator);
        output_names.push_back(alloc.get() ? alloc.get() : ("output_" + std::to_string(i)));
    }
    for (const auto& name_value : output_names) output_name_ptrs.push_back(name_value.c_str());
    if ((input_w != config.imgsz || input_h != config.imgsz) && config.debug_log) {
        std::cout << "[inference] ONNX model input is fixed at " << input_w << "x" << input_h
                  << "; using that instead of IMGSZ=" << config.imgsz << ".\n";
    }
}

void OnnxRuntimeEngine::Impl::initializeGpuInput() {
    gpu_input_ready = false;
    if (!session_uses_gpu || !wantsCuda()) {
        return;
    }
#if defined(DELTA_WITH_CUDA_PIPELINE)
    checkCuda(cudaSetDevice(config.onnx_cuda_device_id), "cudaSetDevice");
    if (gpu_input_buffer == nullptr && input_tensor_bytes > 0) {
        checkCuda(cudaMalloc(&gpu_input_buffer, input_tensor_bytes), "cudaMalloc(onnx input)");
    }
    device_memory_info = std::make_unique<Ort::MemoryInfo>("Cuda", OrtDeviceAllocator, config.onnx_cuda_device_id, OrtMemTypeDefault);
    gpu_input_ready = gpu_input_buffer != nullptr;
    if (config.debug_log && gpu_input_ready) {
        std::cout << "[inference] GPU input path ready (" << input_w << "x" << input_h
                  << ", bytes=" << input_tensor_bytes << ").\n";
    }
#else
    if (config.debug_log) {
        std::cout << "[inference] Session is GPU-backed, but delta_native was built without DELTA_WITH_CUDA_PIPELINE.\n";
    }
#endif
}

void OnnxRuntimeEngine::Impl::preprocess(const FramePacket& frame) {
    const std::uint8_t* source = frame.bgr.data();
    if (!(config.onnx_skip_resize_if_match && frame.width == input_w && frame.height == input_h)) {
        resizeNearest(source, frame.width, frame.height, resized_bgr, input_w, input_h);
        source = resized_bgr.data();
    }
    const size_t pixels = static_cast<size_t>(input_w) * static_cast<size_t>(input_h);
    constexpr float scale = 1.0F / 255.0F;
    if (input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        auto* r = input_f16.data();
        auto* g = r + pixels;
        auto* b = g + pixels;
        for (size_t i = 0; i < pixels; ++i) {
            const size_t off = i * 3U;
            r[i] = Ort::Float16_t(static_cast<float>(source[off + 2U]) * scale);
            g[i] = Ort::Float16_t(static_cast<float>(source[off + 1U]) * scale);
            b[i] = Ort::Float16_t(static_cast<float>(source[off + 0U]) * scale);
        }
    } else {
        auto* r = input_f32.data();
        auto* g = r + pixels;
        auto* b = g + pixels;
        for (size_t i = 0; i < pixels; ++i) {
            const size_t off = i * 3U;
            r[i] = static_cast<float>(source[off + 2U]) * scale;
            g[i] = static_cast<float>(source[off + 1U]) * scale;
            b[i] = static_cast<float>(source[off + 0U]) * scale;
        }
    }
}

Ort::Value OnnxRuntimeEngine::Impl::makeInput() {
    if (input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        return Ort::Value::CreateTensor<Ort::Float16_t>(*memory_info, input_f16.data(), input_f16.size(), input_shape.data(), input_shape.size());
    }
    return Ort::Value::CreateTensor<float>(*memory_info, input_f32.data(), input_f32.size(), input_shape.data(), input_shape.size());
}

Ort::Value OnnxRuntimeEngine::Impl::makeGpuInput(const GpuFramePacket& frame) {
#if defined(DELTA_WITH_CUDA_PIPELINE)
    if (!gpu_input_ready || gpu_input_buffer == nullptr || !device_memory_info) {
        throw std::runtime_error("GPU input path is not initialized.");
    }

    if (frame.cuda_stream != nullptr && frame.cuda_stream != ort_stream) {
        checkCuda(cudaStreamSynchronize(static_cast<cudaStream_t>(frame.cuda_stream)), "cudaStreamSynchronize(capture)");
    }
    std::string error;
    const gpu::TensorElementType tensor_type = input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
        ? gpu::TensorElementType::Float16
        : gpu::TensorElementType::Float32;
    if (!gpu::preprocessBgraToNchw(frame, gpu_input_buffer, input_w, input_h, tensor_type, ort_stream, &error)) {
        throw std::runtime_error("GPU preprocess failed: " + error);
    }
    return Ort::Value::CreateTensor(*device_memory_info, gpu_input_buffer, input_tensor_bytes, input_shape.data(), input_shape.size(), input_type);
#else
    (void)frame;
    throw std::runtime_error("GPU input path is unavailable in this build.");
#endif
}

InferenceResult OnnxRuntimeEngine::Impl::predictGpu(const GpuFramePacket& frame, const int target_class) {
    InferenceResult result{};
    if (!session || !gpu_input_ready || frame.device_ptr == nullptr || frame.width <= 0 || frame.height <= 0) {
        return result;
    }

    const auto t0 = SteadyClock::now();
    Ort::Value input = makeGpuInput(frame);
    const auto t1 = SteadyClock::now();
    result.timings.preprocess_ms = ms(t0, t1);

    Ort::RunOptions run_options;
    Ort::IoBinding binding(*session);
    binding.BindInput(input_name.c_str(), input);
    for (const auto& name_value : output_names) {
        binding.BindOutput(name_value.c_str(), *memory_info);
    }

    const auto t2 = SteadyClock::now();
    session->Run(run_options, binding);
    binding.SynchronizeOutputs();
    const auto t3 = SteadyClock::now();
    result.timings.execute_ms = ms(t2, t3);

    const auto t4 = SteadyClock::now();
    std::vector<Ort::Value> outputs = binding.GetOutputValues();
    if (!outputs.empty()) {
        if (config.onnx_output_has_nms) {
            if (auto decoded = decodeNms(toFlatTensor(outputs.front()), frame.width, frame.height, target_class); decoded.has_value()) {
                result.detections = std::move(decoded.value());
                result.timings.postprocess_ms = ms(t4, SteadyClock::now());
                return result;
            }
        }
        for (const auto& out : outputs) {
            if (auto decoded = decodeNms(toFlatTensor(out), frame.width, frame.height, target_class); decoded.has_value()) {
                result.detections = std::move(decoded.value());
                result.timings.postprocess_ms = ms(t4, SteadyClock::now());
                return result;
            }
        }
        result.detections = decodeRaw(toFlatTensor(outputs.front()), frame.width, frame.height, target_class);
    }
    result.timings.postprocess_ms = ms(t4, SteadyClock::now());
    return result;
}

std::optional<std::vector<Detection>> OnnxRuntimeEngine::Impl::decodeNms(
    const FlatTensor& tensor,
    const int fw,
    const int fh,
    const int target_class) const {
    int rows = 0;
    int cols = 0;
    if (tensor.shape.size() == 3 && tensor.shape[0] == 1) {
        rows = static_cast<int>(tensor.shape[1]);
        cols = static_cast<int>(tensor.shape[2]);
    } else if (tensor.shape.size() == 2) {
        rows = static_cast<int>(tensor.shape[0]);
        cols = static_cast<int>(tensor.shape[1]);
    } else {
        return std::nullopt;
    }
    if (rows <= 0 || cols < 6) return std::nullopt;
    std::vector<Candidate> candidates;
    float coord_max = 0.0F;
    for (int row = 0; row < rows; ++row) {
        const size_t base = static_cast<size_t>(row) * static_cast<size_t>(cols);
        const float conf = tensor.data[base + 4U];
        const int cls = static_cast<int>(tensor.data[base + 5U]);
        if (conf < effectiveModelConf() || !classMatch(cls, target_class)) continue;
        Candidate c{};
        c.box = {tensor.data[base + 0U], tensor.data[base + 1U], tensor.data[base + 2U], tensor.data[base + 3U]};
        c.cls = cls;
        c.conf = conf;
        coord_max = std::max(coord_max, std::max({c.box[0], c.box[1], c.box[2], c.box[3]}));
        candidates.push_back(c);
    }
    if (candidates.empty()) return std::vector<Detection>{};
    if (coord_max <= static_cast<float>(std::max(input_w, input_h) + 8)) {
        const float sx = static_cast<float>(fw) / static_cast<float>(input_w);
        const float sy = static_cast<float>(fh) / static_cast<float>(input_h);
        for (auto& c : candidates) {
            c.box[0] *= sx; c.box[2] *= sx;
            c.box[1] *= sy; c.box[3] *= sy;
        }
    }
    for (auto& c : candidates) {
        c.box[0] = std::clamp(c.box[0], 0.0F, static_cast<float>(fw - 1));
        c.box[2] = std::clamp(c.box[2], 0.0F, static_cast<float>(fw - 1));
        c.box[1] = std::clamp(c.box[1], 0.0F, static_cast<float>(fh - 1));
        c.box[3] = std::clamp(c.box[3], 0.0F, static_cast<float>(fh - 1));
    }
    std::sort(candidates.begin(), candidates.end(), [](const auto& l, const auto& r) { return l.conf > r.conf; });
    if (!config.onnx_output_has_nms) {
        std::vector<std::array<float, 4>> boxes;
        std::vector<float> scores;
        for (const auto& c : candidates) { boxes.push_back(c.box); scores.push_back(c.conf); }
        const auto keep = nms(boxes, scores, config.onnx_nms_iou, static_cast<size_t>(config.max_detections));
        std::vector<Detection> dets;
        for (size_t idx : keep) dets.push_back(makeDetection(candidates[idx]));
        return dets;
    }
    std::vector<Detection> dets;
    for (size_t i = 0; i < candidates.size() && static_cast<int>(i) < config.max_detections; ++i) dets.push_back(makeDetection(candidates[i]));
    return dets;
}

std::vector<Detection> OnnxRuntimeEngine::Impl::decodeRaw(
    const FlatTensor& tensor,
    const int fw,
    const int fh,
    const int target_class) const {
    int src_rows = 0;
    int src_cols = 0;
    if (tensor.shape.size() == 3 && tensor.shape[0] == 1) {
        src_rows = static_cast<int>(tensor.shape[1]);
        src_cols = static_cast<int>(tensor.shape[2]);
    } else if (tensor.shape.size() == 2) {
        src_rows = static_cast<int>(tensor.shape[0]);
        src_cols = static_cast<int>(tensor.shape[1]);
    } else {
        return {};
    }
    if (src_rows <= 0 || src_cols < 6) return {};
    const bool transposed = src_rows < src_cols && src_rows <= 128;
    const int rows = transposed ? src_cols : src_rows;
    const int cols = transposed ? src_rows : src_cols;
    const int class_count = cols - 4;
    if (class_count <= 0) return {};
    if (target_class >= class_count) return {};
    auto at = [&](int row, int col) -> float {
        if (!transposed) return tensor.data[static_cast<size_t>(row) * static_cast<size_t>(cols) + static_cast<size_t>(col)];
        return tensor.data[static_cast<size_t>(col) * static_cast<size_t>(src_cols) + static_cast<size_t>(row)];
    };
    std::vector<Candidate> candidates;
    for (int row = 0; row < rows; ++row) {
        int cls = 0;
        float conf = at(row, 4);
        if (target_class >= 0 && config.onnx_force_target_class_decode) {
            cls = target_class;
            conf = at(row, 4 + cls);
        } else {
            for (int c = 1; c < class_count; ++c) {
                const float score = at(row, 4 + c);
                if (score > conf) { conf = score; cls = c; }
            }
            if (!classMatch(cls, target_class)) continue;
        }
        if (conf < effectiveModelConf()) continue;
        float cx = at(row, 0), cy = at(row, 1), bw = std::max(1.0F, at(row, 2)), bh = std::max(1.0F, at(row, 3));
        if (std::max({cx, cy, bw, bh}) <= 2.0F) {
            cx *= static_cast<float>(input_w); cy *= static_cast<float>(input_h);
            bw *= static_cast<float>(input_w); bh *= static_cast<float>(input_h);
        }
        Candidate c{};
        c.box = {
            std::clamp((cx - (bw * 0.5F)) * (static_cast<float>(fw) / static_cast<float>(input_w)), 0.0F, static_cast<float>(fw - 1)),
            std::clamp((cy - (bh * 0.5F)) * (static_cast<float>(fh) / static_cast<float>(input_h)), 0.0F, static_cast<float>(fh - 1)),
            std::clamp((cx + (bw * 0.5F)) * (static_cast<float>(fw) / static_cast<float>(input_w)), 0.0F, static_cast<float>(fw - 1)),
            std::clamp((cy + (bh * 0.5F)) * (static_cast<float>(fh) / static_cast<float>(input_h)), 0.0F, static_cast<float>(fh - 1)),
        };
        c.cls = cls;
        c.conf = conf;
        candidates.push_back(c);
    }
    if (candidates.empty()) return {};
    std::sort(candidates.begin(), candidates.end(), [](const auto& l, const auto& r) { return l.conf > r.conf; });
    if (static_cast<int>(candidates.size()) > config.onnx_topk_pre_nms) candidates.resize(static_cast<size_t>(config.onnx_topk_pre_nms));
    std::vector<std::array<float, 4>> boxes;
    std::vector<float> scores;
    for (const auto& c : candidates) { boxes.push_back(c.box); scores.push_back(c.conf); }
    const auto keep = nms(boxes, scores, config.onnx_nms_iou, static_cast<size_t>(config.max_detections));
    std::vector<Detection> dets;
    for (size_t idx : keep) dets.push_back(makeDetection(candidates[idx]));
    return dets;
}

OnnxRuntimeEngine::OnnxRuntimeEngine(const StaticConfig& config) : config_(config) {
    try {
        impl_ = std::make_unique<Impl>(config_);
        name_ = impl_->name;
    } catch (const std::exception& e) {
        name_ = "onnxruntime[unavailable]";
        if (config_.debug_log) std::cout << "[inference] " << e.what() << "\n";
    }
}

OnnxRuntimeEngine::~OnnxRuntimeEngine() = default;

void OnnxRuntimeEngine::warmup() {
    if (impl_) impl_->warmup();
}

void OnnxRuntimeEngine::setModelConfidence(const float conf) {
    if (impl_) {
        impl_->model_conf_override = std::clamp(conf, 0.0F, 1.0F);
    }
}

void* OnnxRuntimeEngine::gpuInputStream() const {
#if defined(DELTA_WITH_CUDA_PIPELINE)
    return impl_ ? impl_->ort_stream : nullptr;
#else
    return nullptr;
#endif
}

InferenceResult OnnxRuntimeEngine::predict(const FramePacket& frame, const int target_class) {
    return impl_ ? impl_->predict(frame, target_class) : InferenceResult{};
}

bool OnnxRuntimeEngine::supportsGpuInput() const {
    return impl_ ? impl_->gpu_input_ready : false;
}

InferenceResult OnnxRuntimeEngine::predictGpu(const GpuFramePacket& frame, const int target_class) {
    return impl_ ? impl_->predictGpu(frame, target_class) : InferenceResult{};
}

std::unique_ptr<IInferenceEngine> makeInferenceEngine(const StaticConfig& config) {
    return std::make_unique<OnnxRuntimeEngine>(config);
}

}  // namespace delta
