cmake_minimum_required(VERSION 3.25)

project(DeltaNative LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

option(DELTA_ENABLE_OPENCV "Enable OpenCV-backed capture and image processing." OFF)
option(DELTA_ENABLE_ONNXRUNTIME "Enable ONNX Runtime-backed inference." OFF)
option(DELTA_ENABLE_HTTP_FRONTEND "Enable the runtime HTTP tuning frontend." OFF)
option(DELTA_ENABLE_CUDA_D3D11_INTEROP "Build the native D3D11/CUDA capture probe." OFF)
option(DELTA_ENABLE_CUDA_PIPELINE "Enable CUDA-backed capture + preprocess support in delta_native." OFF)

add_library(delta_runtime STATIC
    src/app.cpp
    src/capture.cpp
    src/capture_focus.cpp
    src/calibrated_control.cpp
    src/control.cpp
    src/debug_overlay.cpp
    src/debug_preview.cpp
    src/detection_dampening.cpp
    src/frontend.cpp
    src/inference.cpp
    src/mouse_suppression.cpp
    src/predictive_pid.cpp
    src/recoil.cpp
    src/recoil_aim_offset.cpp
    src/sigma_drift.cpp
    src/target_association.cpp
    src/target_guard.cpp
    src/target_lead.cpp
    src/triggerbot.cpp
    src/tracking.cpp
    third_party/imgui/imgui.cpp
    third_party/imgui/imgui_draw.cpp
    third_party/imgui/imgui_tables.cpp
    third_party/imgui/imgui_widgets.cpp
    third_party/imgui/imgui_impl_dx11.cpp
    third_party/imgui/imgui_impl_win32.cpp
)

add_executable(delta_native src/main.cpp)
target_link_libraries(delta_native PRIVATE delta_runtime)

target_include_directories(delta_runtime
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/imgui
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/nlohmann/include
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime/include
)

if(WIN32)
    target_compile_definitions(delta_runtime PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN)
    target_link_libraries(delta_runtime PUBLIC d3d11 dxgi dwmapi gdi32 user32 winmm ws2_32)
endif()

if(DELTA_ENABLE_CUDA_PIPELINE OR DELTA_ENABLE_CUDA_D3D11_INTEROP)
    if(NOT WIN32)
        message(FATAL_ERROR "CUDA support currently requires Windows.")
    endif()

    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

if(DELTA_ENABLE_CUDA_PIPELINE)
    target_sources(delta_runtime PRIVATE src/gpu_preprocess.cu)
    target_include_directories(delta_runtime PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_compile_definitions(delta_runtime PUBLIC DELTA_WITH_CUDA_PIPELINE=1)
    set_target_properties(delta_runtime PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_link_libraries(delta_runtime PUBLIC CUDA::cudart)
endif()

if(DELTA_ENABLE_OPENCV)
    find_package(OpenCV REQUIRED COMPONENTS core imgproc video tracking)
    target_link_libraries(delta_runtime PUBLIC ${OpenCV_LIBS})
    target_compile_definitions(delta_runtime PUBLIC DELTA_WITH_OPENCV=1)
endif()

if(DELTA_ENABLE_ONNXRUNTIME)
    message(STATUS "DELTA_ENABLE_ONNXRUNTIME is ON. Add your ONNX Runtime import logic here.")
    target_compile_definitions(delta_runtime PUBLIC DELTA_WITH_ONNXRUNTIME=1)
endif()

if(DELTA_ENABLE_HTTP_FRONTEND)
    message(STATUS "DELTA_ENABLE_HTTP_FRONTEND is ON. Add your preferred HTTP library here.")
    target_compile_definitions(delta_runtime PUBLIC DELTA_WITH_HTTP_FRONTEND=1)
endif()

if(DELTA_ENABLE_CUDA_D3D11_INTEROP)
    add_executable(delta_cuda_capture_probe
        src/cuda_capture_probe.cu
    )

    target_include_directories(delta_cuda_capture_probe
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CUDAToolkit_INCLUDE_DIRS}
    )

    target_compile_definitions(delta_cuda_capture_probe
        PRIVATE
            NOMINMAX
            WIN32_LEAN_AND_MEAN
    )

    set_target_properties(delta_cuda_capture_probe PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CUDA_SEPARABLE_COMPILATION ON
    )

    target_link_libraries(delta_cuda_capture_probe
        PRIVATE
            d3d11
            dxgi
            ${CUDA_CUDART}
    )
endif()

enable_testing()

# Replay links the exact same implementation as delta_native. Device adapters
# are never constructed by DeltaApp::replay; GPU/model assets are not required.
file(GLOB_RECURSE BENCH_FINGERPRINT_FILES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/delta/*"
    "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/*")
list(APPEND BENCH_FINGERPRINT_FILES "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${BENCH_FINGERPRINT_FILES})
set(BENCH_HASH_INPUT "")
foreach(BENCH_FILE IN LISTS BENCH_FINGERPRINT_FILES)
    file(SHA256 "${BENCH_FILE}" BENCH_FILE_HASH)
    file(RELATIVE_PATH BENCH_RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "${BENCH_FILE}")
    string(APPEND BENCH_HASH_INPUT "${BENCH_RELATIVE}:${BENCH_FILE_HASH}\n")
endforeach()
string(SHA256 BENCH_SOURCE_HASH "${BENCH_HASH_INPUT}")
execute_process(COMMAND git rev-parse HEAD WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_VARIABLE BENCH_REVISION OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/report.html" BENCH_REPORT_HTML)
configure_file(benchmarks/benchmark_build.hpp.in generated/benchmark_build.hpp @ONLY)
configure_file(benchmarks/benchmark_report.hpp.in generated/benchmark_report.hpp @ONLY)

add_library(delta_benchmark_support STATIC benchmarks/benchmark.cpp benchmarks/config.cpp benchmarks/report.cpp)
target_link_libraries(delta_benchmark_support PUBLIC delta_runtime)
target_include_directories(delta_benchmark_support PUBLIC benchmarks PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_compile_definitions(delta_benchmark_support PRIVATE DELTA_BENCH_CONFIGURATION="$<CONFIG>")
add_executable(delta_benchmark benchmarks/main.cpp)
target_link_libraries(delta_benchmark PRIVATE delta_benchmark_support)
add_executable(delta_control_response benchmarks/control_response_main.cpp benchmarks/control_response.cpp)
target_link_libraries(delta_control_response PRIVATE delta_benchmark_support)
add_executable(delta_control_response_tests tests/test_control_response.cpp benchmarks/control_response.cpp)
target_link_libraries(delta_control_response_tests PRIVATE delta_benchmark_support)
add_test(NAME delta_control_response_tests COMMAND delta_control_response_tests)
if(WIN32)
    add_executable(delta_control_frontend_tests tests/test_control_frontend.cpp)
    target_link_libraries(delta_control_frontend_tests PRIVATE delta_runtime)
    add_test(NAME delta_control_frontend_tests COMMAND delta_control_frontend_tests)
    set_tests_properties(delta_control_frontend_tests PROPERTIES TIMEOUT 10)
endif()
if(DELTA_ENABLE_CUDA_PIPELINE)
    add_executable(delta_gpu_benchmark benchmarks/gpu_benchmark.cpp)
    target_link_libraries(delta_gpu_benchmark PRIVATE delta_benchmark_support CUDA::cudart windowscodecs ole32)
    target_include_directories(delta_gpu_benchmark PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
    target_compile_definitions(delta_gpu_benchmark PRIVATE DELTA_BENCH_CONFIGURATION="$<CONFIG>")
    add_executable(delta_closed_loop_gpu benchmarks/closed_loop_gpu.cpp)
    target_link_libraries(delta_closed_loop_gpu PRIVATE delta_benchmark_support CUDA::cudart windowscodecs ole32)
    target_include_directories(delta_closed_loop_gpu PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
    target_compile_definitions(delta_closed_loop_gpu PRIVATE DELTA_BENCH_CONFIGURATION="$<CONFIG>")
endif()
add_executable(delta_benchmark_tests tests/test_benchmark.cpp)
target_link_libraries(delta_benchmark_tests PRIVATE delta_benchmark_support)
add_test(NAME delta_benchmark_tests COMMAND delta_benchmark_tests)
add_test(NAME delta_benchmark_reject_unknown_option COMMAND delta_benchmark --unknown-option)
set_tests_properties(delta_benchmark_reject_unknown_option PROPERTIES WILL_FAIL TRUE)

# Keep checked STL algorithms in a standalone executable: its runtime/debug
# settings do not cross the production library's ABI boundary.
add_executable(delta_dispatch_tests tests/test_dispatch.cpp)
target_sources(delta_dispatch_tests PRIVATE src/calibrated_control.cpp)
target_include_directories(delta_dispatch_tests PRIVATE include)
if(MSVC)
    target_compile_definitions(delta_dispatch_tests PRIVATE _DEBUG _ITERATOR_DEBUG_LEVEL=2)
    set_property(TARGET delta_dispatch_tests PROPERTY MSVC_RUNTIME_LIBRARY MultiThreadedDebugDLL)
endif()
add_test(NAME delta_dispatch_tests COMMAND delta_dispatch_tests)
set_tests_properties(delta_dispatch_tests PROPERTIES TIMEOUT 5)

# Legacy tests are local/ignored in this repository. Register the ones present;
# the versioned benchmark integration test above is always built.
function(delta_add_existing_test name)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_${name}.cpp")
        return()
    endif()
    add_executable(delta_${name}_tests ${ARGN} tests/test_${name}.cpp)
    target_include_directories(delta_${name}_tests PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/nlohmann/include")
    if(WIN32 AND name STREQUAL "control")
        target_compile_definitions(delta_${name}_tests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
        target_link_libraries(delta_${name}_tests PRIVATE user32)
    endif()
    add_test(NAME delta_${name}_tests COMMAND delta_${name}_tests)
endfunction()

delta_add_existing_test(recoil src/recoil.cpp)
delta_add_existing_test(recoil_aim_offset src/recoil_aim_offset.cpp)
delta_add_existing_test(tracking src/tracking.cpp)
delta_add_existing_test(predictive_pid src/predictive_pid.cpp)
delta_add_existing_test(sigma_drift src/sigma_drift.cpp)
delta_add_existing_test(capture_focus src/capture_focus.cpp)
delta_add_existing_test(target_association src/target_association.cpp)
delta_add_existing_test(target_guard src/target_guard.cpp src/tracking.cpp)
delta_add_existing_test(target_lead src/target_lead.cpp)
delta_add_existing_test(triggerbot src/triggerbot.cpp)
delta_add_existing_test(mouse_suppression src/mouse_suppression.cpp)
delta_add_existing_test(control src/control.cpp)
delta_add_existing_test(detection_dampening src/detection_dampening.cpp)
