# Visible-screen tracking diagnosis

This investigation keeps the production selection preferences and opt-in motion defaults. It separates simulator response, evaluator attribution, tracking decisions and scheduled delivery. Artifacts are local under `cpp_port/benchmark-results/`.

## Reproductions and confirmed causes

1. **Unmatched camera geometry.** A ten-count response check against the historical controlled and visible results failed: 3.999 px versus 12.796 px, a 3.2 ratio. Controlled mode forced 800 by 800 while visible mode used 2560 by 1440 with the same FOV and radians/count. Controlled mode now uses production static dimensions by default and accepts `-ScreenWidth` / `-ScreenHeight` for explicit experiments. Schema 5 records `scene.viewport`; comparison reticles use that viewport. Historical 800-square results are not comparable to the new geometry.
2. **Identity attribution used the wrong region.** The benchmark regression failed with `Head identity must use linked head regions rather than overlapping body unions`. Both render modes now retain class-specific linked truth regions. A head detection is attributed using heads, a body using bodies; same-class overlaps remain ambiguous. Rechecking the previous 25 wrong movements retained all 25 as target 2. Unknown/ambiguous movement attribution fell from 16 to 8 on that old trace. This correction clarifies evidence rather than erasing wrong-target output.
3. **Prediction dispatch had no delivery allowance.** A visible response sweep applied three movements at observation ages 40.40–40.56 ms despite a 40 ms expiry. The checked scheduler regression reproduced this using a servo whose deadline was two milliseconds away. Shared live/simulated scheduling now withholds explicit-deadline aim commands when less than one configured send interval remains, clears fractional carry and records the rejection. This applies to queued fresh commands and continuous servo output. Ordinary commands without an explicit prediction deadline retain their existing eligibility. Actual late applications still fail correctness: a reserve cannot guarantee Windows scheduling or physical-device timing.
4. **Crop fragments could confirm a false target.** The first response-adjusted visible suite still applied one movement from an incorrect head detection at 6000 px/s. Its source was a 12 by 99 pixel box touching the crop's left boundary; the annotated head was outside the crop. Two recorded observations suffice to reproduce the bug: a clipped body starts confirmation, then a clipped head completes it. The production replay regression failed with `Crop-clipped detector fragments must not confirm a new motion target`. The existing opt-in motion path now receives the actual capture rectangle and excludes boxes touching its boundary from acquisition/continuation evidence. Such observations cannot update velocity or refresh the deadline; a previously confirmed target can only use its existing bounded prediction. Legacy selection without the motion option is unchanged.
5. **A reversal gate created false identity certainty.** A later visible crossing trace contained a true continuity switch at observation 39, after observations 35–38 remained confirmed on target 1. Target 1's apparent screen motion reversed during camera response, so the matcher excluded it; a forward-moving target 2 then appeared uniquely plausible. A three-observation replay and a focused `noteMotionSelection` regression both reproduced the confident switch. The motion matcher now retains physically reachable same-class alternatives as uncertainty evidence even if the strict direction/prediction gate rejects them. It compares these alternatives using the existing reconfirmation displacement metric and margin, withholds acceptance and leaves the last accepted timestamp untouched. `ambiguous_motion_change` identifies this case in reports. Its logged alternate score uses displacement, rather than the strict predicted-position residual. A unique reversal still follows the existing reconfirmation policy.

Crop tests cover all four edges. The detector clips right/bottom coordinates to the last pixel (`width-1` / `height-1`), so checking only the exclusive crop extent missed those fragments. The regression for that convention failed before the final boundary correction. Separate checks confirm that a clipped observation preserves only already-valid bounded prediction and cannot extend its deadline.

## Freeze the input before changing the tracker

Schema 5 retains exact raw detections, confidence, crop, source IDs, integer nanosecond timestamps and relevant dispatch feedback for each inference observation. Ground truth stays outside the tracking input. `observation_replay.hpp` serializes this seam and `delta_closed_loop_gpu --replay` drives the production tracking loop without capture, neural inference or live device output. The original outcome includes selected boxes, prediction, command, readiness and motion decision; wall-clock processing costs are deliberately excluded from equality.

Two replays of `diagnosis-recorded-crossing-20261019/closed-loop.json` reproduced all 100 original outcomes exactly. This trace contains six wrong-identity output-ready observations with identity matching and ten with stability alone. Inspection places the wrong acquisitions after loss/reset and reconfirmation; the trace does not demonstrate a silent jump between two continuously confirmed identities. Retaining identity across such loss episodes would require a separately specified reacquisition policy; this change preserves the agreed production preferences.

That first trace did not exhaust the failure types. The later `diagnosis-clipped-visible-20261019/crossing-1500-motion-identity` trace supplied the distinct continuous-track reversal regression above. Exact replay made it possible to distinguish these causes rather than treat every wrong movement as the same bug.

Alternative policies replay the original feedback and observations. They are open-loop diagnostic comparisons: their commands do not alter the recording's camera, so they cannot establish counterfactual closed-loop following quality. Schema 4 files omitted raw confidence and complete input timing and must be recorded again for exact replay.

## Camera-response experiment

With a matched 2560 by 1440 viewport, clean 1500 px/s controlled runs using the motion candidate gave:

| Radians/count | Output ready | Frames within target | Mean normalized error |
| --- | ---: | ---: | ---: |
| 0.0007 | 54% | 73% | 0.378 |
| 0.00035 | 90% | 98% | 0.140 |
| 0.00021875 | 99% | 98% | 0.153 |

This isolates a major response/gain confounder. The smaller sensitivity restores the old scene's approximate horizontal pixels/count; it does not calibrate the model to an external application, nor establish the best controller configuration. The benchmark default remains 0.0007; the lower value is an explicit experiment via `-MouseRadiansPerCount`. No production runtime setting or frontend preference was added or silently enabled.

## Commands

```powershell
& cpp_port/benchmarks/run-closed-loop-suite.ps1 -Mode visible -MouseRadiansPerCount 0.00021875
& cpp_port/benchmarks/run-closed-loop-suite.ps1 -Mode controlled -MouseRadiansPerCount 0.00021875
& cpp_port/tests/check_scene_response.ps1 -Controlled CONTROLLED/closed-loop.json -Visible VISIBLE/closed-loop.json
& cpp_port/tests/check_observation_replay.ps1 -Recording VISIBLE/closed-loop.json
```

The camera check compares geometry and known-count response across modes and verifies actually delivered backend counts against recorded camera rotations. The replay check requires exact original outcomes and byte-identical repeats, then reports default and stability-only decisions on those frozen inputs. Neither test substitutes for RTX 3070 closed-loop runs or physical camera calibration.

## Final verification

Final visible artifacts are in `benchmark-results/diagnosis-edges-visible-20261019/`: ten baseline/candidate runs, 1,100 observations, including the added shake pair and a five-case `comparison.html`. All pass evidence validity and deadline correctness with zero attributed wrong-target movements. The final code also passes the matched controlled clean/crossing GPU runs at `diagnosis-edges-controlled-clean-20261019/` and `diagnosis-edges-controlled-crossing-20261019/` (another 200 observations).

| Visible case | Output ready: default / candidate | Frames within target: default / candidate |
| --- | ---: | ---: |
| Clean 1500 px/s | 86% / 98% | 79% / 93% |
| Fast 6000 px/s | 4% / 24.7% | 28.7% / 25.3% |
| Occlusion | 76% / 77% | 73% / 76% |
| Crossing | 88% / 93% | 83% / 88% |
| External shake | 63% / 80% | 61% / 68% |

The 6000 px/s case remains outside a demonstrated reliable range: readiness improves but frames within target regress slightly. Crossing includes nine ambiguous/unattributed candidate movements; zero attributed wrong movement does not establish their correctness. The motion candidate remains opt-in and quality gates remain uncalibrated. Real-time repetitions vary, so the earlier failing fixtures and intermediate suites are preserved rather than used as interchangeable paired measurements.

The clean candidate's DXGI-presentation-to-simulated-application latency is 13.45 ms median / 24.86 ms p95 across 182 movements. Inference wall p95 is 3.94 ms across 100 observations. Stage distributions are retained; the remaining time must not all be attributed to neural inference. These remain scene-specific measurements with simulated transport and application, excluding physical device delivery.

The final camera check verifies identical 2560 by 1440 viewports and a ten-count response of 3.9988 horizontal / 2.2493 vertical pixels across both modes, plus actual command-to-camera rotations. Exact observation replay passes byte-identical repeated outcomes and policy comparisons. Browser checks pass all five cases, frame stepping, metrics, mode labeling and full-report latency sections. The minimized crop-fragment and reversal fixtures now withhold the erroneous output. A negative controlled run with 50 ms transport still returns exit 2 and preserves valid evidence of expired applications at `diagnosis-expiry-negative-20261019/`.

The full CUDA Release build, including the native application, succeeds. Full rebuilt local CTest is 14/16; only the previously documented ignored startup-default assertions fail. No `[DEBUG-...]` probes remain in production sources.
