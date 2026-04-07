# Delta C++ Port Scaffold

This directory is the native port workspace for moving `testNonKal1.py` to C++ on Windows.

It is intentionally scoped as a safe starting point:

- It does not modify the current Python workflow.
- It standardizes the runtime target around ONNX Runtime instead of Ultralytics/PyTorch.
- It assumes Windows `SendInput` for mouse output instead of the current driver/socket backends.
- It keeps the same high-level pipeline: capture -> inference -> tracking/PID -> output.

## Planned Dependencies

- CMake 3.25+
- MSVC with C++20 support
- Windows 10/11 SDK
- OpenCV for image operations and optional tracker compatibility
- ONNX Runtime GPU for inference
- Optional HTTP/JSON libraries for the runtime tuning frontend

## Suggested Build Flow

```powershell
cd cpp_port
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

The current app is no longer just a stub. The base build and the CUDA build both run the native runtime loop, and the CUDA build keeps capture + preprocess + model input on the GPU.

## Inference Assets

For native inference work, the cleanest setup is to keep the model and runtime assets in native-only folders instead of depending on the Python site-packages layout at debug time.

Recommended layout:

```text
cpp_port/
  models/
    best.onnx
    trt_cache/
  runtime/
    onnxruntime/
      onnxruntime.dll
      onnxruntime_providers_shared.dll
      ...optional CUDA/TensorRT provider DLLs...
```

What the current C++ inference path does:

- It loads ONNX Runtime dynamically at runtime instead of requiring a separate C++ SDK install.
- It looks for the runtime in this order:
  - `DELTA_ONNXRUNTIME_ROOT`
  - `StaticConfig.onnxruntime_root`
  - `cpp_port/runtime/onnxruntime`
  - the current Python `onnxruntime` wheel
- It looks for the model in this order:
  - `DELTA_MODEL_PATH`
  - `StaticConfig.model_path`
  - `cpp_port/models/best.onnx`
- For GPU inference, it can also resolve SDK/runtime directories from:
  - `DELTA_ONNX_PROVIDER` with `tensorrt`, `cuda`, or `cpu`
  - `DELTA_CUDA_ROOT`
  - `DELTA_TENSORRT_ROOT`
  - `DELTA_CAPTURE_CROP_SIZE` to capture a larger square region and resize it down to model input
  - `DELTA_ONNX_REQUIRE_GPU=1` to fail instead of falling back to CPU
- On Windows it adds the resolved ONNX Runtime, CUDA, and TensorRT folders to the process DLL search path before ONNX Runtime starts, so Visual Studio launches do not depend on shell state alone.

Practical recommendation:

- Keep using your existing `runs\\detect\\...\\best.onnx` path while we port behavior.
- If you want the native app to be self-contained in Visual Studio, copy the ONNX model into `cpp_port/models`.
- For the runtime DLLs, prefer `cpp_port/runtime/onnxruntime`.
- For TensorRT, either keep the extracted SDK under `cpp_port/TensorRT-10.9.0.34` or point `DELTA_TENSORRT_ROOT` at your external install.

Current status on this machine:

- The native ONNX path is working.
- The TensorRT + CUDA ONNX Runtime provider path is initializing successfully from both Visual Studio build folders.
- In the CUDA build, `delta_native` now runs `DXGI crop -> CUDA BGRA buffer -> CUDA preprocess -> ONNX Runtime GPU input`.
- Output decode/NMS is still CPU-side; that is the next inference-side GPU step.

## Visual Studio Workflow

The easiest way to develop this in Visual Studio is as a CMake project, not by hand-maintaining a `.sln` file.

1. Open Visual Studio 2022.
2. Use `File -> Open -> Folder...` and select `cpp_port`.
3. Wait for CMake to detect the presets from `CMakePresets.json`.
4. Pick one of these configure presets in the toolbar:
   - `vs2022-x64` for the base native scaffold
   - `vs2022-cuda-probe` for the CUDA-powered native pipeline plus the D3D11/CUDA interop probe
5. Build from `Build -> Build All` or by selecting a build preset.
6. Set the startup target in the CMake targets view:
   - `delta_native` for the scaffold app
   - `delta_cuda_capture_probe` for the CUDA probe
7. Set command-line arguments in `Debug -> Debug and Launch Settings for CMake` when needed.

Suggested probe args:

```text
--frames 3 --width 640 --height 640
```

The `vs2022-cuda-probe` preset now also builds `delta_native` with the CUDA pipeline enabled.

## Native Control Path

- The native port does not use the Python socket mouse backend.
- `src/control.cpp` uses Win32 hotkey polling and `SendInput` directly for relative mouse movement and left-click injection.
- The low-level sender now mirrors the Python GHUB-style behavior more closely by keeping fractional movement remainders and splitting large moves into bounded steps.
- `delta_native` now starts the full native runtime by default: capture, inference, raw/raw-delta/legacy_pid tracking, frontend, and `SendInput` control.
- `StaticConfig.imgsz` stays the inference input size. `StaticConfig.capture_crop_size` controls the square desktop crop size, and `0` keeps it locked to `imgsz`.
- Use `Insert` to stop the app.
- `XBUTTON2` toggles mode, `F4` toggles head/body class, `F5` toggles the async `XBUTTON1`-held configurable sequence loop, `F6` toggles left-hold engage, `F7` toggles recoil fallback, and `F8` toggles triggerbot.
- Toggle beeps now mirror the Python version.
- Periodic `[PERF]` logs are emitted from the native runtime for benchmarking.

Current scope:

- The native runtime currently exposes `raw`, `raw_delta`, and `legacy_pid` tracking paths.
- The HTTP frontend serves `http://127.0.0.1:8765/` with `/api/pid` and `/api/pid/status` backed by the live native runtime state.

If Visual Studio says the CUDA toolset is missing:

- make sure the CUDA toolkit was installed with Visual Studio integration
- confirm `CUDA_PATH` is set in your environment
- use the `vs2022-cuda-probe` preset, which seeds `CMAKE_GENERATOR_TOOLSET=cuda=$penv{CUDA_PATH}`

Command-line preset flow:

```powershell
cd cpp_port
& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --preset vs2022-cuda-probe
& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build --preset delta-native-cuda-debug
```

## CUDA Interop Probe

There is also an optional prototype target for validating a real GPU-native capture path:

- Desktop Duplication acquires the desktop frame into a D3D11 texture
- `CopySubresourceRegion` copies only the requested crop into a CUDA-registered D3D11 texture
- CUDA maps that texture, copies it to a device buffer, and converts `BGRA -> BGR` on the GPU

Build it with:

```powershell
cd cpp_port
cmake -S . -B build-cuda -G "Visual Studio 17 2022" -A x64 -T "cuda=$env:CUDA_PATH" -DDELTA_ENABLE_CUDA_D3D11_INTEROP=ON
cmake --build build-cuda --config Release --target delta_cuda_capture_probe
```

Run it with:

```powershell
.\build-cuda\Release\delta_cuda_capture_probe.exe --frames 3 --width 640 --height 640
```

This probe is intentionally narrow: it proves the D3D11/CUDA interop path and produces a CUDA-resident `BGR` crop, but it does not yet wire that buffer into ONNX Runtime I/O binding or the full app pipeline.

## Porting Entry Points

If you are starting to move behavior from `testNonKal1.py` into C++, these are the main files to work in:

- `src/capture.cpp`
  Port the DXGI capture path from the Python capture classes.
- `src/inference.cpp`
  Port the ONNX Runtime preprocessing and inference path.
- `src/control.cpp`
  Keep the Win32 input/output path here.
- `src/app.cpp`
  This now owns the full native runtime loop plus the `raw`, `raw_delta`, and `legacy_pid` behavior branches.

## Design Intent

- `include/delta/core.hpp`
  Core data structures shared across modules.
- `include/delta/config.hpp`
  Static config and runtime-tunable config.
- `include/delta/runtime_state.hpp`
  Shared state and queue semantics that mirror the Python pipeline.
- `include/delta/capture.hpp`
  Capture interfaces and the Desktop Duplication target for v1.
- `include/delta/inference.hpp`
  ONNX Runtime-first inference seam.
- `include/delta/tracking.hpp`
  PID, target tracking, and sticky-target selection seam.
- `include/delta/control.hpp`
  Input polling and `SendInput` output seam.
- `include/delta/frontend.hpp`
  Runtime tuning frontend seam.
- `include/delta/app.hpp`
  Top-level orchestration.
- `docs/migration_plan.md`
  Detailed migration plan and exact Python-to-C++ mapping.

## Important Constraint

`SendInput` is acceptable for the port, but it is not behaviorally identical to a driver-backed mouse path. If the target app filters or treats injected input differently, that remains a runtime limitation even after the C++ rewrite.
