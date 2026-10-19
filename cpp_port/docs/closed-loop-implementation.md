# Closed-loop GPU benchmark: implemented slice

For the current continuity fixes and corrected interpretation of the crossing experiment, see [Tracking continuity diagnosis](tracking-diagnosis.md).
For matched camera geometry, class-specific attribution, exact observation replay and prediction dispatch allowance, see [Visible-screen tracking diagnosis](simulation-diagnosis.md). Schema 5 supersedes the geometry/attribution comparisons below; historical results remain diagnostic records.

The controlled-time closed-loop runner is `benchmarks/run-closed-loop.ps1`. From the repository root:

```powershell
& cpp_port/benchmarks/run-closed-loop.ps1 -Frames 150 -Speed 1500 -Scenario clean
& cpp_port/benchmarks/run-closed-loop.ps1 -Frames 150 -Speed 6000 -Scenario shake
& cpp_port/benchmarks/run-closed-loop-suite.ps1
& cpp_port/benchmarks/run-closed-loop-suite.ps1 -Broad
& cpp_port/benchmarks/run-closed-loop.ps1 -Mode visible -Frames 100 -Speed 1500
& cpp_port/benchmarks/run-closed-loop-suite.ps1 -Mode visible
```

The script builds the C++ runner, executes the benchmark integration test, then writes `closed-loop.json`, `report.html`, and rendered PNG frames under a new directory in `cpp_port/benchmark-results/`. No live mouse or serial device receives movement. The runner requires the local `train_backups/delta_train_backup`, model, CUDA runtime and RTX 3070 at CUDA ordinal 1. The scene manifest is `benchmarks/scenes/delta-backup-first.json`; its target image has linked body/head labels and a separate empty-label background from the validation split. The result fingerprints the assets and model, records the dataset version/license, and audits the inference provider.

Each run executes the production GPU detector and replay tracking/control path, the shared production aim/servo scheduler, and the selected output backend's numeric movement transform. Applied movement rotates a simulated camera around the screen-center aim point. Target world motion, camera shake, rendering, logical inference availability, scheduled dispatch and camera application use explicit logical timestamps. `clean`, `occlusion`, `dropout`, `shake` and `crossing` can be selected independently; injected dropout is labeled separately from natural misses. The report lets a human scrub captured frames against detector boxes, ground truth, commands and an aim-error plot. The JSON retains frame-level events for programmatic review.

This first scene is exploratory. Its source frame and labels were checked visually, but the foreground extraction is a feathered ellipse rather than a verified segmentation mask. The scorer projects the union of the linked head/body labels for identity region and error normalization; the aim point follows the selected head/body rule. Field of view and mouse sensitivity are uncalibrated. One target image, one background, a deterministic path and no pose change do not establish detector robustness or a supported tracking-speed range. In controlled-time mode actual GPU wall time is measured separately and does not advance logical time. Logical inference delay is currently limited to one render interval because observation production is synchronous. Presentation and capture timestamps are simulated, so this mode cannot establish physical capture-to-output latency.

The quick suite compares default settings against `motion-identity-config.json` on identical scene parameters. Broad mode adds sizes, cadences, shake and injected dropout. Every result retains its exact runtime snapshot. Quality thresholds remain under calibration; no candidate is promoted automatically. Schema 2 corrects region scoring to use the projected labels; schema 1's aim-centered region values are superseded.

The suite writes `suite.json` and `comparison.html`. The latter scrubs both runs together, with identity truth, detector overlays, reticle and per-case metrics. It verifies that paired runs used identical assets, model, scene parameters, hardware and executable source fingerprint. For an existing schema 2 suite, regenerate it with `cpp_port/tools/write-closed-loop-comparison.ps1 -SuiteRoot PATH`.

Schema 3 adds a separate correctness verdict: movement past the configured loss deadline fails the CLI and suite even when evidence validity passes. It also records shared production dispatch feedback on each observation and evaluates aim points using the configured production head/body ratios. The current broad verification is `benchmark-results/review-fixes-schema3/`; see `tracking-diagnosis.md` for the four regression reproductions. Earlier schema 2 feedback-dependent comparisons are provisional.

The first schema 2 paired run is saved locally at `benchmark-results/closed-loop-paired-backup-v2`. These historical results preceded the continuity fixes. The crossing row mixed initial selection with following and is superseded by the corrected crossing scenario described in the diagnosis. RTX 3070/TensorRT execution was audited in all eight runs. At 8 ms cadence and 5 ms logical inference delay:

| Case | Default output-ready | Candidate output-ready | Default normalized error | Candidate normalized error | Wrong-target moves, default / candidate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean 1500 px/s | 97% | 75% | 0.153 | 0.213 | 0 / 0 |
| Fast 6000 px/s | 2.7% | 34% | 1.134 | 0.684 | 0 / 0 |
| Occlusion 1500 px/s | 72% | 56% | 0.383 | 0.476 | 0 / 0 |
| Crossing 1500 px/s | 94% | 95% | 0.180 | 0.171 | 4 / 5 |

No historical run applied movement past its configured loss deadline. The old candidate improved fast-motion continuity but regressed clean motion and occlusion. The old crossing numbers alone cannot establish a following-identity regression. GPU inference p95 varied from 3.8 to 8.7 ms across those short runs. That is measured inference wall time, not capture-to-output latency. The browser smoke test verified case selection, synchronized stepping, loaded scene panels and metric rows, and saved `comparison.png` alongside the report.

The scene copies one sprite for both identities in the crossing case, so this measures geometry-based association under ambiguity. Exact overlap is labeled ambiguous rather than assigned an invented identity.

## Visible-screen mode

`-Mode visible` opens a borderless scene window on the configured capture output, with physical-pixel DPI coordinates and the display's actual resolution. It runs independent real-time render and dispatch threads, the production Desktop Duplication capture source, real GPU inference, the production replay tracking/control loop, shared servo scheduling, and backend numeric transforms. Camera output is simulated; live mouse, serial, firing and recoil adapters are never constructed. The console and saved report identify the no-device mode. Closing the scene window ends and invalidates the run.

The current hardware path is display rendering/composition and capture on RTX 3090 adapter 0/output 0, host-memory BGR capture, then RTX 3070 inference on CUDA ordinal 1. This deliberately measures the existing cross-GPU layout. The synchronous host capture path is recorded explicitly; this is not a claim about CUDA interop or the live application's unavailable cross-GPU zero-copy path.

A 96 by 8 pixel provenance marker at display center +140 px vertically links captured pixels to a rendered scene record. It must remain inside the adaptive capture crop; the standard centered minimum 320 px crop includes it. Its complementary rows reject corrupt/mixed markers. Ground truth and persistent identity remain in the evaluator. A separate observation ID links controller output back to the captured render record, and the raw DXGI source ID and presentation timestamp are preserved. Duplicate captures of the same rendered state are skipped rather than counted as new observations. Corrupt linkage, cached frames, backward IDs, scene closure, placement/DPI mismatch, or an overlapping visible window invalidate the run and retain diagnostic results. Presentation time is DXGI's reported source timestamp; DWM flush is recorded as a fence, not mislabeled as physical scanout.

Schema 4 separates real measured latency from controlled logical delays. Visible reports include render submission to DXGI presentation, presentation to capture completion, capture to inference completion, tracking processing, inference/tracking completion to dispatch, dispatch to simulated application, and world update to simulated application. Actual captured presentation intervals expose renderer/capture cadence limits. Inference does not pause world motion or scheduled control. The configured transport delay remains simulated; actual OS wake delay is included when camera movement is applied. No physical mouse/device or photon-to-output latency is established.

Captured frames are retained in memory and PNG encoding runs after the measured loop. A 1 GiB recording limit invalidates oversized runs; use shorter runs rather than silently dropping observations. Rendering is currently CPU/GDI through DWM, so its cost and achieved cadence are part of these real-time measurements. It does not promise that a requested render interval is achieved. Real-time baseline/candidate observations differ; paired playback aligns observation index and displays each run's own timestamps. Comparison generation rejects mixed modes/hardware.

The visible harness shares the live control loop's scoped 1 ms Windows timer-resolution request. An initial longer probe omitted it: nominal 1 ms transport waits took up to 16 ms and caused real loss-deadline correctness failures. Those failed artifacts are retained under `benchmark-results/visible-paired-20261019/`; they must not be treated as passing quality evidence. Background rendering now precomputes the scaled image and pans by row copies, preserving pixels while avoiding per-frame per-pixel coordinate division.

### First visible-screen evidence

The final quick suite is saved locally at `benchmark-results/visible-final-20261019/`, including `comparison.html`, captured images and per-run reports. All eight RTX 3070/TensorRT runs (900 observations) passed capture validity and loss-deadline correctness. This establishes that the real capture loop runs and exposes failures; it does not establish acceptable tracking quality.

| Case | Output ready, baseline / candidate | Frames within target, baseline / candidate | Attributed wrong moves, baseline / candidate |
| --- | ---: | ---: | ---: |
| Clean 1500 px/s | 40% / 27% | 47% / 18% | 0 / 0 |
| Fast 6000 px/s | 2% / 7.3% | 30.7% / 5.3% | 0 / 0 |
| Occlusion 1500 px/s | 10% / 16% | 4% / 11% | 0 / 0 |
| Crossing 1500 px/s | 3% / 47% | 0% / 30% | 0 / 25 |

All 25 attributed wrong moves in the candidate crossing run occurred after intended-target acquisition; another 16 movements had ambiguous/unattributed source identity. Neither variant is an acceptable quality baseline merely because correctness passes. The previous timer-matched suite also produced wrong-target following, and readiness varied substantially between repeated real-time runs. Current scene response is uncalibrated, physical display resolution differs from the 800 by 800 controlled scene, and actual render/capture cadence replaces the logical 8 ms cadence. These factors must be isolated through camera-response calibration, repeated runs and cadence/gain sweeps before attributing the difference to one tracking change or enabling the candidate.

For the final clean candidate run, DXGI presentation to simulated application was 14.5 ms median / 31.1 ms p95 across 48 applied movements. Inference wall time p95 was 9.1 ms; captured presentation intervals were 14.3 ms median / 24.9 ms p95. A requested 8 ms render interval therefore did not establish 125 Hz observation delivery. The report retains stage distributions and sample counts; these short scene-specific measurements are not a supported operating range or physical device latency claim.

Verification includes the BGR crop/marker roundtrip and corruption checks, a controlled-time GPU smoke run after the renderer change, and browser checks of all four paired cases, synchronized observation stepping, real-time mode text and measured-latency sections. `tests/check_visible_capture.ps1` deliberately creates an overlapping window: the tested run stopped after 12 observations, returned exit code 2, named the obstruction and preserved its report under `benchmark-results/visible-obstruction-20261019/run/`. This negative test opens a temporary scene and obstruction; invoke it only on the test desktop. Full local CTest remains 14/16 with the two previously documented stale startup-default assertions.

Calibrated camera response, hand-validated cutouts, additional target identities, broader paired sweeps, and physical output delivery remain separate work. The scene realism limitations above apply to both modes.

The CUDA Release build succeeds and the versioned benchmark integration checks pass, including scheduler cadence/freshness, backend rounding/splitting and label-projection scoring. The full local CTest suite has two failing legacy assertions: `delta_predictive_pid_tests` expects zero dead-zone ratios and `delta_control_tests` expects SendInput startup. The committed configuration already uses nonzero ratios and Serial startup; these ignored local tests are stale relative to that configuration. The implementation preserves those startup choices.
