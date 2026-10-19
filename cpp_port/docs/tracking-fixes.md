# Motion continuity and freshness implementation

The approved policy is implemented behind separate motion-stability and identity
flags. Both remain **off by default** pending review of quality tradeoffs. Fresh
source timestamps, duplicate rejection and dispatch-age checks are active in the
normal pipeline. The intended inference GPU remains the RTX 3070, CUDA ordinal 1.

## Behavior

`noteMotionSelection` maintains the last accepted observation, velocity, distinct
confirmation count and confirmation window. It uses actual elapsed observation
time rather than loop frequency. New targets require at least two distinct,
consistent observations; continued targets may move without box overlap.
Estimation proceeds while output is gated. A candidate's class and size must
remain compatible with the current aim policy. Head/body transitions do not
inject their aim-point displacement into velocity.

The identity variant evaluates candidates against predicted position, records
plausible boxes and costs in `motion_matches`, and requires a best-versus-runner-up
margin. Ambiguous observations never update velocity or restart the prediction
deadline. Prediction is bounded by elapsed time, estimated displacement and the
existing per-command movement limits. Visible contradictory motion withholds
output and requires confirmation again. Timeout clears continuity state;
reacquisition requires new confirmation.

This is a geometry-based continuity model, not appearance recognition or a camera
motion estimator. It does not establish correctness when multiple targets are
geometrically indistinguishable. The original initial selection and head/body
candidate preferences are retained.

## Freshness and dispatch

DXGI `LastPresentTime` supplies source identity and presentation time. A QPC/steady
clock anchor converts that time into the pipeline clock; capture completion stays
separate. Cached and pointer-only updates retain source identity. Missing source
timestamps fall back to acquisition time and cannot establish presentation age.

The pipeline rejects repeated/non-increasing observations and checks age before
and after inference. Commands and servo state carry the last accepted observation
and prediction deadline. The actual sender checks freshness again immediately
before sending. Default maximum observation age is 50 ms, independently adjustable
through `command_max_observation_age_ms`; it is an initial policy, not a measured
acceptable latency target.

`SimulatedCommandDispatch` shares deadline advancement and freshness checks with
the production control loop. Replay reports include scheduled dispatch events,
observation-age percentiles and expired-command counts at a configurable rate
(165 Hz when unspecified). The simulation covers latest-command replacement,
budget waits and expiry; it excludes the servo's continuous extrapolation, output
device delivery and physical camera response.

The GPU runner follows production fresh-frame settings. Cached diagnostics can
be requested with runtime overrides; they must not be interpreted as production
fresh-frame behavior. The RTX 3070 inference / RTX 3090 capture pairing uses host
memory because interop cannot cross those physical GPUs. It still runs neural
inference on the selected RTX 3070.

## Reproduce the experiments

```powershell
.\cpp_port\benchmarks\run.ps1 -Suite quick -Config .\cpp_port\benchmarks\motion-stability-config.json
.\cpp_port\benchmarks\run.ps1 -Suite speed_sizes -Config .\cpp_port\benchmarks\motion-identity-config.json
.\cpp_port\benchmarks\run.ps1 -Suite speed_sizes -Config .\cpp_port\benchmarks\motion-compensation-config.json
.\cpp_port\benchmarks\run-gpu.ps1 -Image .\valid\images\111_png_jpg.rf.9fc3828c35fd338fa4e2f363e8658fcb.jpg -Frames 1000 -Speed 3000 -Config .\cpp_port\benchmarks\motion-identity-config.json
```

`speed_sizes` covers 108 combinations of 500/1500/3000/6000 px/s, 4/8/16 ms
intervals, 5/15/30 ms observation ages and 0.5/1/2 target-size scales. The original
36-case `speed` suite remains available. Compare output-ready coverage and
wrong-target command time alongside estimate coverage: withheld output is not
successful following.

The saved candidate uses a 40 ms prediction deadline, 30 px maximum predicted
displacement, 0.5 box-width residual scale and 0.75 matching margin. These are
experimental parameters selected from 36 quick-suite configurations, not approved
production thresholds. The compensation profile separately enables the existing
latency-aware lead, bounded to 50 ms and one box width; it remains disabled in the
default configuration.

Integration checks exercise fast non-overlapping observations, estimation before
confirmation, duplicate source IDs with changing capture times, ambiguity without
deadline extension, timeout/reconfirmation, contradictory shake, scheduled command
replacement, command expiry and prevention of catch-up dispatch bursts.

## Validation results (2026-10-18)

The final stability, identity and compensation profiles each completed the 12-case
quick suite and 108-case size/speed suite: 360 cases with no harness correctness
violations. CPU and CUDA benchmark integration checks passed (2/2 each), and the
native CUDA application built successfully. These checks do not imply acceptable
tracking quality.

Matched deterministic comparisons against the default policy show:

| Case / metric | Default | Identity candidate |
| --- | ---: | ---: |
| 1500 px/s, 16 ms cadence, 5 ms age, normal size: output-ready coverage | 5.1% | 89.9% |
| 3000 px/s, 8 ms cadence, 5 ms age, normal size: output-ready coverage | 2.9% | 90.7% |
| Combined disturbances: identity switches | 13 | 5 |
| Combined disturbances: wrong-target time | 980 ms | 1060 ms |
| Combined disturbances: wrong-target command time | 200 ms | 250 ms |

The candidate improves motion continuity but fails the agreed no-regression
criterion. Both motion flags therefore remain disabled by default. Compensation
does not resolve the combined identity failure; its combined normalized error is
2.176 versus 2.152 without compensation. Further work should use recorded match
diagnostics to distinguish incorrect acceptance from incorrect reacquisition,
then validate on additional seeds before promoting thresholds.

An isolated 1000-frame translated-image run at 3000 px/s verified TensorRT on the
RTX 3070 (1000 GPU inputs). Median inference was 3.214 ms; source-to-loop latency
was 3.897 ms median, 4.273 ms p95 and 4.594 ms p99. Tracking alone was 5.6 us median
and 20.3 us p99. The candidate generated commands on 809 frames versus 8 in the
earlier default run, with similar median latency (3.882 ms previously). These
real-time runs sample different translated images; command counts demonstrate
continuity, not a paired accuracy comparison. Latency excludes physical output.

A separate 20-frame desktop smoke run used RTX 3070 inference with host capture
from the RTX 3090 display. All source IDs were distinct, with no cached or stale
frames. It verifies capture plumbing, not target accuracy or sustained latency.

Local artifacts under `cpp_port/benchmark-results/` (ignored by Git):

- `fixes-review-identity-speed_sizes/report.html`: paired speed/size comparison.
- `fixes-review-identity-quick/report.html`: disturbance regressions and playback.
- `fixes-review-gpu-3070/report.html`: final GPU timing run.
- `fixes-before-3070-idle/gpu.json`: isolated earlier GPU baseline.
- `fixes-fresh-desktop-3070/gpu.json`: fresh desktop capture smoke run.
