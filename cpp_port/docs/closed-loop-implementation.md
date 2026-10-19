# Closed-loop GPU benchmark: implemented slice

For the current continuity fixes and corrected interpretation of the crossing experiment, see [Tracking continuity diagnosis](tracking-diagnosis.md).

The controlled-time closed-loop runner is `benchmarks/run-closed-loop.ps1`. From the repository root:

```powershell
& cpp_port/benchmarks/run-closed-loop.ps1 -Frames 150 -Speed 1500 -Scenario clean
& cpp_port/benchmarks/run-closed-loop.ps1 -Frames 150 -Speed 6000 -Scenario shake
& cpp_port/benchmarks/run-closed-loop-suite.ps1
& cpp_port/benchmarks/run-closed-loop-suite.ps1 -Broad
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

The visible-screen mode using production DXGI capture, calibrated camera response, hand-validated cutouts, additional target identities, broader paired sweeps, and physical output delivery remain separate work. The controlled-time runner does not claim those measurements.

The CUDA Release build succeeds and the versioned benchmark integration checks pass, including scheduler cadence/freshness, backend rounding/splitting and label-projection scoring. The full local CTest suite has two failing legacy assertions: `delta_predictive_pid_tests` expects zero dead-zone ratios and `delta_control_tests` expects SendInput startup. The committed configuration already uses nonzero ratios and Serial startup; these ignored local tests are stale relative to that configuration. The implementation preserves those startup choices.
