# Sensitivity, following lag and oscillation

The production C++ controller reproduces the reported tradeoff with perfect detections: a 1500 px/s target, observations every 16 ms, 12 ms observation delay and 1 ms simulated transport. A 46 px target is observed on a linear camera plant. Acquisition dampening, guard and adaptive crop are disabled only in this isolation harness; it still executes `DeltaApp::replay`, the production controller, `AimCommandScheduler` and the selected backend's numeric transform. It never opens a mouse or serial device.

## Reproduction and diagnosis

`delta_control_response --output NEW.json --gain 0.1` gives approximately 140 px median absolute error after the first 500 ms. At gain 1.28 it gives approximately 58 px median error and 147 crossings of a +/-2 px deadband. Both measurements use the same default controller and delays. The regression command is `ctest --test-dir cpp_port/build-cuda-benchmark -C Release -R delta_control_response_tests --output-on-failure`; before the opt-in implementation it failed on the low-gain 140 px error.

The ranked hypotheses were delayed feedback, camera contamination of velocity, overlapping frame/servo corrections, and reversal/ramp behavior. Changing one setting at a time showed:

- Turning latency compensation on reduced low-gain median error to 59 px but destabilized the high-gain run enough to lose the target. It is not a sufficient fix.
- Turning the servo off removed large high-gain oscillation, but low-gain median lag increased to 243 px.
- Disabling the derivative term reduced high-gain error, but repeated excursions remained. Disabling reverse suppression did not change that run.
- The predictive velocity equation adds `previous_output` (requested mouse counts) to a change in screen pixels. This assumes a one-pixel response per count, ignores whether the requested correction was delivered, and omits additional continuous servo movement. The scheduler then combines per-frame PID corrections with velocity/acceleration integration on other ticks. Those assumptions do not describe a camera whose sensitivity changes.

## Experimental correction

The existing default remains unchanged. Within Predictive PID, `calibrated_control_enable` selects a separate rate controller when PID and Display Rate Servo are enabled. It uses the existing target selection, aim-point preferences, confirmation, ambiguity handling and freshness checks. It does not consult simulator ground truth.

The controller estimates target motion from source-timed observations and adapter-completed movement history. For each axis, with calibrated response `g` in screen pixels per **pre-backend command count**:

```
target_velocity = (change_in_observed_error + g * delivered_counts_between_frames) / frame_dt
current_error = observed_error + filtered_target_velocity * observation_age
                - g * delivered_counts_since_frame
command_rate = (filtered_target_velocity + current_error / response_time) / g
```

The same compensation now reaches motion matching: displacement between source observations adds the intervening delivered camera movement before speed, reversal and residual checks. Detection boxes and crop checks remain in screen coordinates. This prevents the controller's own catch-up movement from being mistaken for a target reversal. A three-frame production replay failed before this correction and now keeps the confirmed target.

Loss extrapolation and the matching prediction limit bound **remaining screen movement after subtracting delivered camera movement**, rather than bounding world displacement first. The latter ordering can invent a reverse correction while the camera is already following correctly. A second regression failed on a stationary screen error during a 50 ms observation gap with 50 px of target/camera movement; it now preserves that screen error. Neither correction estimates external shake or uses future target motion.

Following velocity supplies the ongoing movement needed for a moving target; position feedback gradually removes the residual center error. The shared scheduler integrates this one count rate, with fractional carry and movement limits, on every send tick. It does not add a separate per-frame correction. Calibration/mode changes discard pending output and require a newly updated servo state. Invalid calibration withholds movement. Missing observations cannot refresh the accepted observation deadline; the existing motion prediction displacement limit bounds the extrapolated position. This path also applies a maximum prediction age of 100 ms, shortened by the configured loss/freshness policy.

The delivered-history timestamps mean **adapter completion**, not physical application. The current simulator has a known transport delay; real device transport, application response, acceleration or nonlinear sensitivity still need measurement. Calibration includes output backend gain, application sensitivity, field of view and resolution. Recalibrate after changing these. A moving target alone cannot identify both unknown target velocity and unknown camera gain, so this implementation does not pretend to infer sensitivity automatically.

There is also an output-capacity limit: following speed requires approximately `target_speed / pixels_per_count` counts per second, while dispatch and per-command limits cap available counts per second. Extremely low sensitivity can make a requested speed unreachable even with correct control. Increasing PID gains cannot remove that limit.

## Controls and scope

The frontend's **Calibrated Control** group exposes all five new runtime settings:

| Setting | Meaning |
| --- | --- |
| `calibrated_control_enable` | Experimental opt-in; default false |
| `calibrated_pixels_per_count_x/y` | Measured positive screen displacement per command count, .001 to 100 |
| `calibrated_response_time_s` | Residual center-error convergence time, .02 to 1 s; default .08 |
| `calibrated_response_delay_ms` | Estimated completed-command-to-visible-response delay, 0 to 50 ms; default 0 |

The rate path also uses the existing Predictive Velocity Alpha and Max Velocity settings (velocity is capped at 10000 px/s), servo rate/age, raw command limits, and detection/loss policies. Its controller replaces predictive PID gains, Target Lead, Kalman servo velocity and human-motion shaping; applying them together would reintroduce multiple predictions or output paths. This milestone validates tracking, not firing/recoil interactions or physical device delivery.

Head/body aim-point changes reanchor the position history while retaining established translational velocity. They do not become a sudden velocity sample or an identity change. Both a stationary head/body transition through production replay and moving-target reanchoring have regression coverage.

`benchmarks/calibrated-simulation-config.json` is a **simulation-specific experiment**, using 40 ms correction time, Velocity Alpha .5, a 2560x1440 viewport, 70 degree horizontal/vertical FOV and .00021875 radians/count. Its .39988144/.22493331 pixel/count calibration is derived from that camera model. It is not a calibrated setting for the user's application. The higher-sensitivity simulation (.0007 rad/count) uses 1.27962061/.71978660 px/count instead.

```powershell
& cpp_port/benchmarks/run-closed-loop.ps1 -MouseRadiansPerCount .00021875 `
    -CadenceMs 8 -InferenceDelayMs 5 -Frames 200 `
    -Config cpp_port/benchmarks/calibrated-simulation-config.json
```

The first GPU experiments held the controller implementation fixed while varying one parameter. Reducing correction time from 80 to 40 ms improved high-sensitivity following. Increasing the existing velocity alpha from .23 to .5 then improved low-sensitivity reversals, but gave up some high-sensitivity improvement. Final paired runs use the same candidate parameters across sensitivities, rather than choosing the best setting for each scene.

Response delay is a separate experiment. It shifts the completed-command history used for between-frame camera compensation and extends the controller's response horizon; it does not alter raw source timestamps, confirmation times, or the last-accepted-observation loss deadline. Dispatch reserves the greater of one send interval and the configured response delay before an explicit deadline. Calibration/backend changes invalidate the prior rate and motion confirmation. Capture/inference age is already measured separately and must not be added again to this delay.

The minimized delayed-response reproduction uses a stationary target initially 100 px off-center, 24 ms simulated transport, 16 ms observations, 12 ms processing age, .4 px/count, 40 ms correction time and Velocity Alpha .5. Without response alignment, steady median error is 77.2 px, with 64 large center crossings. Setting response delay to the known 24 ms reduces median error to approximately zero and maximum sampled steady error to 1.2 px. This regression failed before the history alignment and now passes through production control and dispatch.

`calibrated-visible-simulation-config.json` adds an estimated 9 ms response delay, a 12000 px/s apparent-motion gate and a 60 ms experimental loss timeout for the current visible simulator. The delay comes from roughly 6.5 ms frame-state-to-presentation time plus about 2 ms simulated application time. These are measured simulation parameters, not physical application calibration. A visible nominal-6000 transition implied about 9400 px/s over presentation timestamps, exceeding the older 8000 gate: variable rendering/presentation intervals matter as well as nominal speed. The wider gate is an experiment and must also be checked against crossing/identity failures. The controlled-time profile retains zero response delay and the original gate.

```powershell
& cpp_port/benchmarks/run-closed-loop.ps1 -Mode visible -Scenario linear -Speed 6000 `
    -MouseRadiansPerCount .00021875 -Frames 100 `
    -Config cpp_port/benchmarks/calibrated-visible-simulation-config.json
```

The historical `clean` scene is a triangle wave with a reversal every 500 reference pixels: at 6000 px/s, it reverses every 83 ms. It therefore mixes speed and reversal difficulty. The new `linear` scenario isolates constant angular motion, with speed specified as the equivalent pixel velocity at screen center. Targets behind the perspective camera stay outside the rendered view instead of wrapping through the tangent projection. Constant angular motion does not imply constant pixel velocity far from screen center. Both motion models remain useful and are labeled separately in results.

Observation cadence, processing age and loss timeout must be evaluated together. At 16 ms cadence and 12 ms processing age, one missing observation makes the next available accepted source at least 44 ms old relative to the prior accepted source. A 40 ms loss policy deliberately expires before that recovery. At 6000 px/s, 12 ms also represents 72 px of target movement before unseen direction changes can be observed. Neither timeout calibration nor this information delay can be solved by increasing controller gain.

## Regression boundary

The controller test checks gains .1, .4, 1.28 and 2: steady following, stationary settling, reversal recovery, changed observation cadence and a long detection gap. It also checks that camera-only motion does not become target velocity, invalid calibration stops output, and SendInput numeric gains .5 and 2 are included in calibration. The dispatch test checks that frame corrections are not added twice and that changing calibration invalidates the previous rate. The frontend HTTP test verifies rendered controls, applying/reading settings and rejecting invalid values.

At the default experimental response time, the perfect-detection constant-speed test reduces median error to 2.0–2.8 px across those gains, with maximum sampled steady error 6.0–6.5 px and zero excursions crossing both sides of a +/-12 px band. Small center crossings remain: discrete display-rate camera steps create visible sampling ripple. This is a controller-isolation result, not a claim of 2 px accuracy with a real detector or physical device.

## GPU evidence and remaining limits

The final review is saved locally at `benchmark-results/calibrated-control-review-20261019/index.html`, with the before/after controller plots, paired visible replay, machine-readable summaries and loss-timeout sweep. The tracking-comparison binary (before the subsequent left-click output-stage fix below) completed 14 visible runs and two controlled GPU runs: 1600 observations, one source fingerprint, audited RTX 3070 inference on CUDA ordinal 1, and no device output. All pass evidence validity and deadline correctness. Capture remains on RTX 3090 / DISPLAY5 through host memory, with a 2560x1440, 165 Hz display.

The final visible experiment uses the 60 ms timeout, estimated 9 ms response delay and 12000 px/s apparent-motion gate. The baseline is the unchanged default configuration. Each main run has 100 observations:

| Visible case | Output ready: default / candidate | Frames within target: default / candidate | Attributed wrong movements: default / candidate |
| --- | ---: | ---: | ---: |
| Lower sensitivity, crossing 1500 px/s | 79% / 90% | 82% / 90% | 0 / 0 |
| Higher sensitivity, crossing 1500 px/s | 56% / 88% | 69% / 97% | 3 / 0 |
| Lower sensitivity, constant 6000 px/s | 0% / 97% | 1% / 79% | 0 / 0 |
| Higher sensitivity, constant 6000 px/s | 0% / 99% | 2% / 89% | 0 / 0 |

An ablation keeps the final motion settings and disables only calibrated control: output readiness falls to 4% / 13% and frames within the target region to 3% / 5% at lower / higher sensitivity. This supports the controller change rather than the wider motion gate alone.

Two further constant-6000 repeats per sensitivity give 93–99% output-ready observations and 79–89% frames within the target region across all six candidate runs. No attributed wrong-target or post-deadline movements occur in the final candidate runs. The main crossing pair still contains 33 ambiguous/unattributed movements; these cannot be called proven correct following. The region metric covers the linked target identity, not a claim that every frame hits the head aim point.

For the lower/higher-sensitivity fast runs, presentation-to-first-applied-movement latency is respectively 12.4 / 11.4 ms median and 18.4 / 15.3 ms p95, across 90 / 94 source observations with output. These are first responses to presented observations, excluding physical device delivery. All applied-movement observation ages, including later servo ticks holding the same observation, have p95 values 27.3 / 24.9 ms. Reporting only the first response would conceal aging held predictions; reporting only all servo ticks would confuse observation age with initial dispatch delay.

Tracking processing p95 is .45 / .39 ms in those runs; inference wall p95 is 9.09 / 4.57 ms. Captured presentation intervals have medians around 18 ms despite the requested 8 ms renderer schedule. This is a limit of the current visible test path and does not establish the live application's maximum capture rate. The two latest controlled-time fast runs, with 8 ms observations and 5 ms logical processing age, each have 99% readiness and 100% frames within the region. Their actual inference wall times remain separate from logical accuracy time.

The 16-run visible loss-timeout sweep compares 60 and 80 ms across both sensitivities, constant fast motion, crossings, dropped detections and shake. Its 60 ms fast runs achieve 97–99% readiness and 80–82% frames within the region; 80 ms gives no consistent benefit. All sweep runs have zero attributed wrong-target and post-deadline movements, but crossings retain unknown identity samples. Increasing the timeout explicitly permits a longer prediction interval without trustworthy detections; these data are not approval of that tradeoff for a physical application. The simulator example uses 60 ms, while native defaults are unchanged.

Earlier stages are retained rather than replaced by the successful runs. The 44-run controlled development matrix at `calibrated-control-verified-20261019/` includes a higher-sensitivity shake regression (65% to 58% frames within the region), and 6000 px/s reversal cases remain difficult even when constant motion works. Earlier visible configurations also produced wrong-person crossing movements. Six successful fast repeats and a small exploratory asset set do not establish universal identity or shake robustness. The calibrated controller remains opt-in; quality gates remain uncalibrated.

## Visible startup correction and verification

The first constant-motion visible tests exposed a harness startup defect: capture initialization ran while world motion was already advancing. One first observation linked to rendered scene 17 at 207.8 ms, after the 6000 px/s target had left the capture region. Those runs were structurally linked but did not test initial fast-target acquisition fairly.

Capture is now warmed before the world motion clock starts; the scene window is also initialized before publishing that clock to capture/dispatch threads. The discarded startup desktop frame is neither an observation nor a saved scored image. Measured capture setup duration is retained separately. `check_visible_startup.ps1` requires the first constant-motion target to remain in the initial capture region and the warm-up provenance to be present. Paired hardware checks exclude measured setup duration while retaining hardware layout and protocol checks.

The native CUDA Release application builds. All four targeted suites pass: controller response, frontend HTTP/configuration, benchmark integration and checked dispatch. Original observation replay reproduces all 100 outcomes exactly in byte-identical repeated runs; alternative-policy replay remains an open-loop diagnostic. The camera check verifies actually applied counts and identical known-count response across modes. Browser checks pass the four-case final visible report and the earlier 22-case controlled comparison, including frame stepping, overlays, metrics and visible timing sections. No temporary debug logging remains in production code.

Before live adoption, measure the physical application's gain and response delay, calibrate loss/identity tolerances at its actual cadence, expand distinct target assets, and validate external-shake recovery. Further changes should address those measured limitations rather than promote the simulator profile as a universal sensitivity preset.

## Left-click downward movement with recoil toggles off

A subsequent live report identified a separate defect at the output stage, which tracking-only GPU simulations intentionally exclude. With the default legacy amount of 8, the live control loop enabled recoil whenever the amount was nonzero and a target was detected. Holding LMB or X1, or a triggerbot click, activated this direct output even though F7, Virtual Recoil Aim Offset and the fallback toggles were off. At the 120 Hz compatibility conversion, 8 means 960 counts/s. Calibrated position feedback can resist that constant disturbance and settle at an offset, making this look like a fixed downward aim shift.

The deterministic `delta_dispatch_tests` reproduction shares `LegacyRecoilOutput` with the live control loop: five idle ticks followed by fifteen 10 ms LMB ticks. It failed with 134 unwanted downward counts. Changing only the compensation amount to zero removed the output; changing only calibrated control to false retained all 134 counts. The defect therefore predates calibrated control and does not require virtual/advanced recoil or a changed detector aim point.

Direct legacy recoil now requires `legacy_recoil_output_enable`, default false. The frontend exposes **Recoil & Engage > Direct Legacy Recoil Output**, with runtime JSON and benchmark configuration support. Nonzero legacy values alone no longer enable movement. Enabling this switch intentionally retains the existing integrated rate and allows F7 fallback under its existing additional conditions. Virtual aim offset and advanced-profile recoil remain separately explicit features; this switch controls direct legacy output only. Existing users who intend direct legacy compensation must enable it explicitly after rebuilding.

The same regression now emits zero with the switch off and 134 with it explicitly enabled. It also checks disabling/re-enabling during a hold, the explicit-rate override, and suppression while virtual recoil owns output. A production tracking replay verifies that pressing LMB with recoil disabled does not alter the calibrated servo rate. The frontend HTTP test verifies both configuration APIs and recoil status. No physical mouse or serial device is opened by these tests. This does not establish firing/recoil performance or physical delivery latency.

After this output-stage fix, the native CUDA Release application builds and all four targeted suites pass again (1.70 seconds total). A fresh 100-observation controlled-time GPU smoke run at 6000 px/s, saved at `benchmark-results/calibrated-control-post-recoil-fix-20261019/`, audits RTX 3070 inference and no device output, passes evidence validity/correctness, and reports 99% readiness, 100% frames within the target region, zero attributed wrong-target movements and zero post-deadline movements. Its source fingerprint is separate from the historical paired tracking comparison above. The new direct-recoil switch deliberately changes the old implicit-on behavior; calibrated tracking still defaults off.
