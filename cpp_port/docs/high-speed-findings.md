# High-speed tracking measurements

The user subsequently restored the intended RTX 3070 configuration (CUDA ordinal
1). The runner now follows that compiled selection unless explicitly overridden.
RTX 3090 results below describe the historical measurement configuration; a new
RTX 3070 baseline is required before evaluating the proposed tracking fixes.

Measured locally on 2026-10-18 using Release C++ production replay. Full artifacts
are under `cpp_port/benchmark-results/`; they are deliberately ignored by Git.

## What the original benchmark established

The original quick suite measured CPU selection/tracking/command generation at
roughly 1.9 microseconds median in the clean scenario. It did not measure capture,
neural inference, output delivery, or physical following. The combined disturbance
case produced 980 ms of wrong-target selection and 13 identity switches. Unit-test
success therefore did not imply robust tracking.

## High-speed coverage is currently limited by detection dampening

The new deterministic suite independently varies target velocity, observation
interval and observation age. All 36 scenarios pass command correctness checks.
Representative results at 5 ms observation age:

| Speed | Observation interval | Default prediction coverage | Dampening disabled |
| --- | --- | --- | --- |
| 1500 px/s | 8 ms | 99.73% | 100% |
| 1500 px/s | 16 ms | 5.07% | 100% |
| 3000 px/s | 4 ms | 99.87% | 100% |
| 3000 px/s | 8 ms | 2.93% | 100% |
| 6000 px/s | 4 ms | 3.07% | 100% |

The paired configuration changes only `detection_dampening_enable`. The gate
requires consecutive boxes with IoU >= 0.2. The synthetic head is 26 px wide;
24 px displacement per observation is already enough to break that requirement.
This explains the abrupt coverage collapse despite perfect detections.

Artifacts: `speed-default/report.html`, `speed-no-dampening/report.html` (paired
comparison), and `quick-no-dampening/report.html` (broader behavior comparison).
Disabling the gate restores coverage but does not fix identity selection: the
combined case still has 980 ms wrong-target time and 13 switches. Production
dampening remains enabled. The next tracking improvement should use motion-aware
identity continuity to establish stability, tested against noisy detections,
crossing targets and bursts of missing detections before choosing defaults.

## Observation age dominates spatial lag

At 3000 px/s and 4 ms intervals, median position errors are 15, 45 and 90 pixels
for 5, 15 and 30 ms observation ages. Coverage remains 99.87% in these cases, so
the error is not hidden by missing predictions. With the current configuration,
fresh observations matter much more than microsecond-level CPU optimizations.

These are synthetic position estimates, not measured physical following. There
is no motion blur, detector miss, actuator response or closed-loop camera model
in this suite. GPU image throughput is measured separately; annotated fast-motion
footage and complete capture-to-output latency remain necessary.

## GPU inference and capture are now exercised

Actual ONNX profiling verifies TensorRT execution on the RTX 3090, with profiling
disabled before measurement. The model is `train4/weights/best.onnx`, fixed
416x416 input. A 1000-frame translated-image fixture at 3000 px/s measured:

| RTX 3090 provider | Inference p50 / p95 | Source-to-loop p50 / p95 |
| --- | --- | --- |
| TensorRT FP16 | 2.07 / 2.41 ms | 2.72 / 3.07 ms |
| CUDA | 3.31 / 3.73 ms | 3.96 / 4.44 ms |

These are single local warmed runs, not universal latency guarantees or a
detector-accuracy comparison. Actual wall-time motion yields different sampled
images between providers. TensorRT produced detections on 980/1000 frames and
commands on 715; CUDA produced detections on 982 and commands on only 4.
This makes command counts essential context for latency: fast inference alone
does not establish effective following. The stability gate, sampling and detector
outputs need annotated analysis before attributing differences between providers.

Artifacts: `gpu-rtx3090-trt-validated/gpu.json` and
`gpu-rtx3090-cuda-validated/gpu.json`. The initial TensorRT graph request failed
warmup; the production fallback rebuilt a working non-graph TensorRT session.
Defaults now request that working mode directly, and select CUDA device 0 (the
3090, despite its different `nvidia-smi` ordinal). Plans are cached per GPU model.

The final runner/configuration was rebuilt and rerun end to end:
`gpu-rtx3090-final/report.html`. It passed integration checks and audited 13
TensorRT execution events before disabling profiling. Across 1000 measured
frames, inference was 2.06 ms median / 2.45 ms p95; source-to-loop was 2.72 / 3.14
ms (p99 3.45 ms). CPU tracking was 4.5 microseconds median / 6.1 microseconds p95.
983 frames contained detections; 695 generated commands; none were stale-rejected.
There was no graph-warmup error with the final settings.

The desktop smoke test verified direct CUDA capture on the RTX 3090's
2560x1440, 165 Hz display. All 200 inputs used GPU capture, but 178 reused cached
frames on a mostly static desktop. Source-to-loop median was 2.47 ms, while
observation-age-inclusive median was 14.20 ms (p95 31.59 ms). There were no target
detections or commands. This verifies capture plumbing, **not** high-speed
screen tracking or fresh-frame FPS. Artifact: `gpu-desktop-rtx3090/gpu.json`.
The desktop benchmark explicitly allows cached frames; production defaults
request fresh frames. These results do not establish a production caching defect.
