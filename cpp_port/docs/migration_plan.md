# C++ Migration Plan For `testNewKal1.py`

## Portability Verdict

The script is portable to C++, but the port should be treated as a subsystem rewrite rather than source translation.

The architecture already maps well to native code:

- capture backends are isolated behind classes
- inference is isolated behind backend classes
- tracking and PID are custom logic, not Python-specific
- runtime tuning is already a typed config store plus HTTP frontend
- the runtime is already organized as bounded queues and dedicated threads

The main non-portable area is the Python `UltralyticsBackend`. The native runtime should standardize on ONNX Runtime.

## v1 Scope

Implement in C++:

- Desktop Duplication capture
- ONNX Runtime inference
- detection decode and NMS
- sticky target selection
- observed-motion tracker and Kalman tracker
- PID control
- `SendInput` mouse output
- hotkeys and toggles
- runtime tuning HTTP server
- performance logging

Defer from v1:

- Python `UltralyticsBackend`
- `mss` fallback
- `dxcam` monkey-patching compatibility layer
- GHUB/socket/`pydirectinput` mouse output backends

## Repo Layout

```text
cpp_port/
  CMakeLists.txt
  README.md
  docs/
    migration_plan.md
  include/
    delta/
      app.hpp
      capture.hpp
      config.hpp
      control.hpp
      core.hpp
      frontend.hpp
      inference.hpp
      runtime_state.hpp
      tracking.hpp
  src/
    app.cpp
    control.cpp
    main.cpp
```

## Dependency Strategy

Required for the real port:

- Windows SDK
- OpenCV
- ONNX Runtime GPU

Recommended:

- `nlohmann/json`
- `cpp-httplib` or Boost.Beast

Rationale:

- OpenCV already matches the Python implementation for resize/color conversion and offers tracker compatibility if needed.
- ONNX Runtime gives the cleanest parity path from the existing ONNX backend.
- HTTP/JSON should stay lightweight because the frontend is already browser-based and simple.

## Exact Python To C++ Mapping

| Python Area | Python Symbol | C++ Target | Notes |
| --- | --- | --- | --- |
| Config constants | module globals | `delta::StaticConfig` | Move all fixed defaults into one typed config object. |
| Runtime-tunable PID/tracking values | `RuntimePIDConfig` | `delta::RuntimeConfigStore` | Keep version/reset-token behavior. |
| Shared runtime status | `state` dict | `delta::SharedState` | Replace string-key dict access with typed fields. |
| Queue semantics | `queue.Queue(maxsize=1)` | `delta::LatestSlot<T>` | Preserve overwrite-latest behavior. |
| Capture backends | `MSSCaptureSource`, `DXGISyncCaptureSource`, `DXGIAsyncCaptureSource` | `delta::ICaptureSource`, `delta::DesktopDuplicationCapture` | Start with one native DXGI implementation. |
| ONNX backend | `OnnxRuntimeBackend` | `delta::OnnxRuntimeEngine` | Port first and treat as primary runtime engine. |
| Ultralytics backend | `UltralyticsBackend` | none in v1 | Keep Python-only for training/export/debug. |
| PID controller | `PIDController` | `delta::PIDController` | Port nearly verbatim. |
| Kalman tracker | `KalmanTargetTracker` | `delta::KalmanTargetTracker` | Use OpenCV C++ or a native matrix impl. |
| EMA/raw tracker | `ObservedMotionTracker` | `delta::ObservedMotionTracker` | Port nearly verbatim. |
| Sticky target scoring | `pick_sticky_target` | `delta::pickStickyTarget` | Preserve exact tie-breaking rules. |
| Mouse output | `MouseMoveClient` | `delta::SendInputMouseSender` | Replace GHUB/socket logic with `SendInput`. |
| Input polling | `keyboard`, `pynput`, `GetAsyncKeyState` | `delta::Win32HotkeySource` | Use Win32 polling or hooks only. |
| HTTP frontend | `ThreadingHTTPServer` + HTML string | `delta::RuntimeFrontendServer` | Reuse the same endpoints and frontend payload shape. |
| Main orchestration | `main()` and nested loops | `delta::DeltaApp` | Turn closures into explicit member functions. |

## Phase Plan

### Phase 1: Freeze The Reference Runtime

Deliverables:

- stable ONNX model artifact
- saved runtime config snapshot
- sample perf logs
- replay dataset of captured frames and expected `dx/dy`

Exit criteria:

- Python behavior is reproducible enough to compare against C++

### Phase 2: Port Pure Logic

Deliverables:

- core types
- clamp/EMA helpers
- IOU/NMS helpers
- PID controller
- tracker implementations
- sticky-target selector

Exit criteria:

- unit tests pass against Python-generated fixtures

### Phase 3: Port Inference

Deliverables:

- ONNX Runtime engine
- provider selection and warmup
- resize/color conversion
- decode path for NMS and raw outputs

Exit criteria:

- identical class filtering and numerically close decoded boxes/confidences

### Phase 4: Port Capture

Deliverables:

- Desktop Duplication capture source
- crop-following logic
- frame timestamping and stale-frame rejection

Exit criteria:

- stable frame delivery at expected rates

### Phase 5: Port Output And Input Control

Deliverables:

- `SendInput` output path
- input polling for `Insert`, `F6`, `F7`, left click, `XBUTTON1`, `XBUTTON2`
- mode/aim/recoil/engage toggles

Exit criteria:

- toggle semantics and emitted `dx/dy` match Python behavior

### Phase 6: Port The Runtime Frontend

Deliverables:

- `/api/pid`
- `/api/pid/status`
- `/api/pid/reset`
- browser UI serving the same HTML/JS or a direct native equivalent

Exit criteria:

- live tuning works without restarting the C++ app

### Phase 7: Integrate The Full Pipeline

Deliverables:

- capture thread
- inference thread
- control thread
- perf thread
- safe shutdown and reset handling

Exit criteria:

- end-to-end runtime is stable under repeated toggling and target loss

## `SendInput` Notes

Recommended v1 approach:

- use relative motion with `MOUSEEVENTF_MOVE`
- preserve step clamping and integer rounding
- apply recoil compensation in the control loop, not inside the sender

Known risk:

- `SendInput` can behave differently from driver-backed movement because Windows pointer settings and target-app input handling may differ. This does not block the port, but it can change feel.

## First Real Implementation Order

1. `core.hpp`, `config.hpp`, `runtime_state.hpp`
2. `tracking.hpp`
3. `control.cpp`
4. `inference.hpp` implementation
5. `capture.hpp` implementation
6. `app.cpp` end-to-end loops
7. `frontend.hpp` implementation

## Acceptance Criteria

- C++ runtime loads the ONNX model and warms up successfully
- capture, inference, and control threads stay alive for long runs
- target selection and loss behavior match Python within tolerance
- `F6` and `F7` runtime behavior matches Python
- runtime config changes apply without restart
- `SendInput` path is good enough to replace the current Python output path for testing

