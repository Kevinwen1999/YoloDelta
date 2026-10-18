# Tracking benchmark

The first milestone evaluates the **production C++ inference loop** using
synthetic detections and explicit logical timestamps. `DeltaApp::replay` runs
the same selection, guard, association, dampening, tracking, prediction, PID and
command-generation code as `delta_native`. It constructs no capture, inference,
input, UI or network adapters and starts no worker threads. Commands are recorded.

## Run it

From the repository root, on Windows with Visual Studio 2022 C++ tools:

```powershell
.\cpp_port\benchmarks\run.ps1 -Timing
```

This configures a separate Release build, runs the benchmark integration checks,
and creates a timestamped directory under `cpp_port/benchmark-results/`. Open
`report.html` directly in a browser; no server or internet connection is needed.
The companion `run.json` contains all inputs, truth, frame-level results,
metrics, failure intervals and configuration for programmatic inspection.

The output directory must be new: existing `run.json` or `report.html` files are
never replaced by the runner. This protects baseline evidence.

To reproduce the frozen initial configuration and compare a later run:

```powershell
.\cpp_port\benchmarks\run.ps1 -Config .\cpp_port\benchmarks\baseline-config.json -Output .\cpp_port\benchmark-results\my-baseline -Timing
.\cpp_port\benchmarks\run.ps1 -Baseline .\cpp_port\benchmark-results\my-baseline\run.json -Output .\cpp_port\benchmark-results\my-candidate -Timing
```

The first command measures the frozen **configuration on the code currently
built**. To preserve a code baseline, retain its original run artifact; rerunning
an old configuration on new code is not the same thing. Each run records the
Git revision, a SHA-256 of working source files (including uncommitted code),
compiler, build configuration, screen dimensions and CPU identifier. Do not
compare hardware timings across machines or build configurations as regressions.

The checked-in configuration snapshots the working production defaults when this
benchmark was introduced, including the user's serial port setting. It is never
used to open devices. It is not a snapshot of settings in a running tuning UI.

The initial measured metrics and build provenance are preserved in
[`benchmark-baseline.json`](benchmark-baseline.json). Its full local replay is
`cpp_port/benchmark-results/baseline/run.json`; retain that file for comparison.

Runtime config files can be complete snapshots or partial overrides. Unknown
fields and invalid types fail. Enums use their C++ integer values:

| Field | Values |
| --- | --- |
| `tracking_strategy` | 0 raw, 1 Kalman legacy alias, 2 EMA legacy alias, 3 DEMA legacy alias, 4 raw+velocity, 5 legacy PID, 6 predictive PID |
| `aim_mode` | 0 head, 1 body, 2 hybrid |
| `mouse_output_method` | 0 SendInput, 1 serial (neither is instantiated) |
| `recoil_mode` | 0 legacy, 1 advanced profile |
| `left_hold_engage_button` | 0 left, 1 right, 2 X1, 3 both |

Keep `benchmarks/runtime_fields.inc` synchronized when adding `RuntimeConfig`
fields; all settings must be saved, including fields unused by a scenario.

## Scenarios and sweeps

The quick suite runs 12 three-second scenarios at nominally 100 Hz:

| Scenario | Disturbance |
| --- | --- |
| `clean` | Smooth motion and direction changes |
| `burst_loss` | Consecutive missed detections |
| `noisy_boxes` | Seeded localization and confidence noise |
| `false_positives` | High-confidence distractors near screen center |
| `crossing` | Two distinct targets crossing nearby |
| `camera_shake` | Common camera translation, oscillation and an abrupt displacement |
| `crop_exit` | Target exits and re-enters the recorded crop; crop origin also changes |
| `stale_cadence` | Uneven processing cadence and observations aged 8, 35 or 90 ms |
| `head_body` | Head detections disappear while the same entity's body remains |
| `departure` | Target leaves permanently |
| `allowed_switch` | Scenario explicitly permits an identity handoff |
| `combined` | Shake, burst loss, box/confidence noise and competing detections |

```powershell
.\cpp_port\benchmarks\run.ps1 -Suite sweep
.\cpp_port\benchmarks\run.ps1 -Scenario camera_shake -Timing
```

The broad suite runs `burst_loss` and `combined` over three seeds, three gap
durations (50/150/400 ms), three production prediction timeouts (50/150/300 ms)
and two per-command movement caps (50/150 px). These are experimental ranges,
not approved quality thresholds. The existing frame budget and dampening still
apply: a timeout permits prediction only while other production gates allow it.
For example, the initial default enables dampening, which resets prediction on
misses; timeout changes alone may therefore have no effect. Compare a deliberate
config override with dampening disabled to investigate that interaction.

`target_prediction_timeout_ms` is an optional production age cap measured from
the last accepted observation. Zero preserves the old frame-budget behavior.
It is available through the runtime JSON endpoint as well as benchmark configs.
The existing `raw_max_step_x/y` bound each generated command, not cumulative
movement over a whole dropout. Sweeps do not simulate physical camera response.

Direct executable options:

```powershell
.\cpp_port\build-benchmark\Release\delta_benchmark.exe --help
```

`--seed`, `--loss-deadline-ms`, `--repeats` and `--write-default-config` are
available through the executable. The default evaluation loss deadline is
150 ms; it is a visible provisional scoring policy, separate from production
settings. No claim is made that this is an acceptable operational threshold.

## Reading the evidence

### High-speed targets and GPU latency

```powershell
.\cpp_port\benchmarks\run.ps1 -Suite speed -Timing
.\cpp_port\benchmarks\run-gpu.ps1 -Image .\valid\images\111_png_jpg.rf.9fc3828c35fd338fa4e2f363e8658fcb.jpg
.\cpp_port\benchmarks\run-gpu.ps1 -Desktop
```

The deterministic `speed` suite covers 36 combinations: 500/1500/3000/6000 px/s,
4/8/16 ms observation intervals, and 5/15/30 ms observation ages. Targets traverse
the crop and reverse abruptly, staying visible. It measures error against current
truth, with pixel-error percentiles as well as target-normalized error and coverage.
These velocities are experimental stress levels, not a promised tracking envelope.

The separate GPU runner builds Release CUDA code for this workstation's Ampere
GPUs (compute capability 8.6). It executes the real model, GPU preprocessing,
decoding and production tracking without opening a movement-output device.
It calls production warmup, audits actual CUDA/TensorRT node execution, ends ONNX
profiling, then records hardware latency distributions. A missing GPU path or
missing GPU execution evidence fails the run. `gpu.json` records the actual GPU
name, PCI identity, provider evidence, configuration, timings and per-frame results.

CUDA ordinal **0 is the RTX 3090 on this workstation**, and ordinal 1 is the
RTX 3070; this differs from `nvidia-smi` ordering. The intended application GPU is
the RTX 3070 (ordinal 1). The runner follows the compiled application selection
unless `-GpuDevice` is explicitly provided. Always inspect the recorded name when
moving machines. Earlier RTX 3090 measurements are historical comparisons, not
the baseline for the intended RTX 3070 configuration.
TensorRT caches are separated by GPU model to avoid reusing plans from the other
card. Native startup fails if GPU execution is required but session creation fails.
TensorRT graph mode defaults off because this model/runtime fails graph warmup;
the measured non-graph TensorRT path remains GPU accelerated. CUDA EP graph mode
is independently available and passed testing.
Use `-GpuDevice 0 -Provider cuda` to explicitly test the other card/provider.

Image mode translates a real validation image at `-Speed` pixels/second (default
1500). It includes CPU image translation and GPU upload in source latency. This
is a **throughput fixture, not detector accuracy evaluation**: wrap boundaries,
whole-image motion and absence of motion blur do not reproduce real fast motion.
Desktop mode uses the real capture adapter, records cached frames and whether
GPU capture was available, and falls back to host capture when required.

`inference_wall_ms` includes GPU completion; `tracking_us` measures CPU tracking
and command generation. `observation_to_loop_complete_ms` includes observation
age, inference and tracking. `command_latency_ms` contains only frames that
generated commands; inspect detection, command and stale-rejection counts beside
it. `frame_interval_ms` records actual benchmark cadence. The source-to-loop
measurement excludes output delivery, servo scheduling and physical response;
it is not a full screen-to-movement latency measurement. GPU tests run separately
from deterministic replay, never concurrently with other performance runs.

Truth is generated independently of detections. Truth IDs never enter the
production interface. Head and body share an entity ID. The evaluator uses
source provenance to identify selected boxes; ambiguous matches are explicitly
marked rather than guessed from proximity. Production association IDs are
recorded as diagnostics and never treated as truth.

Wrong-target time is weighted by frame duration. Unpermitted identity switches
and head/body aim-point changes are separate counters. Initial selection is
scored separately. Permitted handoffs do not count as identity errors, although
time spent on the old target after the handoff still counts as wrong-target time.

Position error is `sqrt((dx/width)^2 + (dy/height)^2)` against **current-time**
truth, with a fixed evaluator aim policy (head center; body at 15% height).
Both filtered and predicted position errors are saved. Means are time-weighted;
percentiles are frame-sampled nearest-rank statistics. Missing predictions yield
`null`, not zero. Always compare error alongside prediction coverage and missing
track duration. Gap drift is `null` if no gap prediction was scored.

Recovery starts when observations of the intended target return and ends on its
first correct selection. Unresolved recovery episodes remain counted; permanent
departures do not invent a reacquisition time. Failure intervals have start/end
timestamps and are clickable in the viewer. JSON exposes every underlying frame.

Comparison requires identical scenario keys, input/truth hashes, screen geometry,
schema, scenario version and evaluation deadline. It reports per-metric deltas;
there is no combined score or automatic quality threshold. Current correctness
checks reject malformed input and commands violating output gates or configured
step limits. The integration tests additionally verify determinism, timeout
expiry, stale rejection, evaluator accounting and escaped report data.

`--timing` runs a warmup and separate measurement passes. Wall-clock timings cover
only the executed inference-loop work, including diagnostic snapshot creation;
scenario generation, result scoring and report construction occur outside the
measured interval. Selection, tracking, prediction and PID stage percentiles are
also recorded. Frames that exit early contribute zero to unexecuted stages.
Logical observation age is reported separately from CPU processing time.

## Scope and next milestones

Version one uses fixed recorded crops so baseline and candidate receive identical
inputs. It records the production adaptive-crop proposal but does not feed that
proposal back into the next observation. It tests coordinate handling and crop
exits; it does not validate adaptive capture performance.

The capture/inference adapters, worker scheduling, latest-slot frame replacement,
display-rate servo, input sender, physical output and recoil thread are excluded.
Camera shake here changes coordinates; motion blur and detector degradation need
real imagery. External camera motion is recorded separately from commands, which
do not affect future frames.

Next milestones remain:

1. Annotated real-video replay, held-out evaluation clips, detector accuracy and
   real capture-to-output timing.
2. A camera-response model driven by commands, then closed-loop following error,
   overshoot and oscillation evaluation.

Measured high-speed findings and limitations are recorded in
[`high-speed-findings.md`](high-speed-findings.md).

The legacy unit tests are ignored/local in the current repository; CMake registers
those present. The new benchmark integration tests are versioned. During initial
validation, two legacy tests failed on existing defaults: predictive PID expected
disabled dead-zone ratios, and control expected SendInput as the startup output.
The committed defaults already specify nonzero ratios and serial output. Those
unrelated defaults have been preserved.
