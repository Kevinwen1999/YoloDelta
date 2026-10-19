# Tracking continuity diagnosis

The diagnosis exercised the production C++ replay boundary with controlled source times, then checked each behavioral change using the RTX 3070/TensorRT closed-loop runner. Tracking changes remain behind the existing motion flags. No new runtime setting or live-device output was added.

## Reproductions and causes

The regression loop is `cpp_port/build-cuda-benchmark/Release/delta_benchmark_tests.exe`. Before each fix, the relevant assertion failed in roughly 0.1 seconds; the complete passing integration checks take about one second.

1. **Reversal recovery kept obsolete motion.** Four observations acquire a target, confirm forward motion, reverse, and continue in the new direction. The old matcher discarded the first new-direction observation while retaining the old position/velocity, delaying reconfirmation. The failing assertion was `Two consistent post-reversal observations must reconfirm without waiting for loss timeout`. A unique, physically reachable observation now starts a new confirmation sequence with movement withheld. The next distinct consistent observation can confirm it. Ambiguous evidence cannot refresh the prior accepted observation.
2. **Continuation underused the configured motion allowance.** A three-observation replay copied the displacement pattern in the GPU trace: 47 px forward, then 60 px forward and 22 px vertically over 8 ms. That final displacement is within the 8000 px/s allowance, but the old continuation gate used a quarter-speed contribution and rejected its prediction residual. Continuation now uses the full elapsed-time allowance, with separate checks for excessive displacement and meaningful direction reversal. Widening the residual gate alone made the reversal test fail, so that change was not kept alone. A separate excessive-shift regression also failed before the independent displacement check was added.
3. **Identical detector boxes invented identity competition.** The GPU trace contained identical same-class boxes, and the matching-margin check treated the copies as separate targets. A replay containing an identical repeated detection reproduced false ambiguity. Exact same-class boxes are now deduplicated, retaining the highest-confidence copy. Distinct boxes still compete and still require a clear matching advantage.
4. **The crossing fixture mixed initial selection with continuity.** In the old scene, target 2's head was initially closer to screen center. Every reported wrong-target movement occurred during initial acquisition, before target 1 was acquired. Selecting target 2 followed the existing preference; these results did not establish a crossing identity regression. The distractor now appears after 40 ms so the intended identity can be established first. Initial acquisition and subsequent wrong-target movement have separate scores. Old crossing results are retained locally but are not comparable to the corrected scenario as a continuity test.

## Evaluation corrections

Applied movement now takes its evaluator identity and aim class from the command's source-frame record, rather than whichever target is selected at dispatch time. This matters when a command waits in the scheduler while a later observation changes selection. Missing or inconsistent source attribution invalidates the run. Ground truth stays in the evaluator; none of these identity annotations reach the production tracker.

Each frame records the motion decision, ambiguity, matching costs and residual, and the result summarizes decision counts. The paired browser report shows the decision and separates wrong moves before/after intended acquisition. This makes future failures distinguishable from detector misses and confirmation waits.

## Validation boundary

The schema 2 results below preceded the dispatch-feedback correction described at the end of this document. They are historical; the schema 3 rerun now supplies the current evidence.

Final artifacts are saved locally under `cpp_port/benchmark-results/diagnosis-guarded-final/`, including `comparison.html` and `suite.json`. All 20 paired RTX 3070/TensorRT runs (2,400 frames) passed the current harness validity checks with the guarded continuation and duplicate fixes. The configured motion candidate stayed opt-in.

| Case | Default readiness | Fixed candidate readiness | Default normalized error | Fixed candidate normalized error |
| --- | ---: | ---: | ---: | ---: |
| Clean 1500 px/s | 97% | 98% | 0.153 | 0.154 |
| Fast 6000 px/s | 2.7% | 56.7% | 1.134 | 0.668 |
| Occlusion / injected dropout | 72% | 79% | 0.383 | 0.304 |
| Corrected crossing | 97% | 99% | 0.152 | 0.147 |
| Small target, 3000 px/s | 2% | 92% | 2.080 | 0.561 |
| Large target, 3000 px/s | 95.3% | 94% | 0.183 | 0.197 |
| 16 ms observation cadence | 95% | 99% | 0.226 | 0.202 |
| 4 ms observation cadence | 98% | 98.7% | 0.131 | 0.128 |
| External shake | 81% | 98% | 0.254 | 0.160 |

Compared with the original opt-in candidate on the unchanged clean/fast/occlusion scenes, readiness improved from 75% to 98%, 34% to 56.7%, and 56% to 79%, respectively. Readiness alone is insufficient: the fast case still spends only 41.3% of rendered frames inside the labeled target region, and the large-target case regresses slightly. These results do not justify enabling the candidate by default.

No attributed wrong-target movement or movement after the configured loss deadline was observed in the final suite. Both crossing runs have 28 movements with ambiguous source identity under the evaluator's conservative overlapping identity-box attribution. These are shown separately and cannot be counted as proven correct-target following. More precise class-specific attribution and distinct target assets remain useful before crossing quality gates are calibrated.

Five additional GPU runs disabled identity matching while retaining the fixed stability policy. Their artifacts are under `benchmark-results/diagnosis-stability-only/`. This isolates the matching contribution without changing motion limits:

| Case | Stability only: readiness / error | Stability + identity: readiness / error |
| --- | ---: | ---: |
| Clean | 99% / 0.149 | 98% / 0.154 |
| Fast | 51.3% / 0.723 | 56.7% / 0.668 |
| Occlusion | 82% / 0.235 | 79% / 0.304 |
| Crossing | 99% / 0.149 | 99% / 0.147 |
| Large target | 92% / 0.230 | 94% / 0.197 |

Identity matching helps the fast and large-target cases here, while stability alone does better on clean motion and occlusion. There is no uniformly best preset in this small scene set. All five additional runs were validity-clean with zero attributed wrong-target moves and zero moves after the loss deadline; the crossing run again had 28 ambiguous-identity movements.

The focused regressions cover reversal reconfirmation, modest camera-induced screen-motion changes, excessive shifts, duplicate boxes, distinct competing boxes, source freshness, prediction expiry and head/body identity scoring. The CUDA Release build, versioned benchmark checks and ten-case browser comparison smoke test pass. The full local CTest run remains 13/15: legacy assertions in `delta_predictive_pid_tests` and `delta_control_tests` expect different committed startup defaults. Those unrelated defaults were preserved. The browser check verified ten case choices, synchronized frame stepping and all eight quality/identity/expiry metric rows, and saved `comparison.png` in the final suite directory.

The GPU scene still uses one validation target and background with an approximate cutout and uncalibrated response. Logical observation age is simulated; GPU inference wall time is measured separately. This work does not establish Windows capture-to-applied-output latency or a supported 6000 px/s operating range. The candidate remains opt-in pending broader assets and calibrated quality thresholds.

## Code-review regression fixes

Four independent reproductions caught the review findings before their fixes:

1. A three-observation production replay received two successful delivery events. Its latency and sent-velocity feedback remained zero. Live dispatch and replay now share the feedback update: non-servo command latency uses adapter completion, sent velocity uses pre-backend command deltas, and idle ticks decay velocity. The regression now observes 3.6 ms latency EMA and (150, -75) px/s velocity EMA. Future completion events are rejected. Simulated transport/application remains a separate timing boundary; this feedback does not claim physical delivery measurement.
2. A GPU scene with nondefault head/body aim ratios disagreed with its projected annotations. Evaluator aim projection now reuses the production `detectionAimPoint` calculation. Both head and body projections pass explicit nondefault-ratio checks.
3. A ten-frame GPU run with 50 ms transport and a 40 ms candidate loss timeout applied two expired movements but returned success. Schema 3 now records a separate `correctness` verdict and returns exit code 2 for these violations. Evidence validity remains true when the measurement itself is sound. The same reproduction now fails correctly and preserves its report; suite and comparison scripts reject failed correctness.
4. A checked-STL scheduler test at 1, 10 and 19 Hz aborted on invalid `std::clamp` bounds. Servo integration now uses the intended 50 ms upper cap without an invalid lower bound. The checked regression passes at 1, 10, 19 and 20 Hz.

Current artifacts are under `benchmark-results/review-fixes-schema3/`. All 20 RTX 3070/TensorRT runs (2,400 frames) pass validity and correctness, and all contain nonzero dispatch feedback. Quality values match the table above for the tested presets; the optional latency/ego compensation paths are not enabled by those presets. The motion candidate remains opt-in, including the existing large-target regression and crossing-attribution limitations. A separate deliberately failing deadline run is saved at `benchmark-results/review-fix-deadline-green/`.

The benchmark and checked scheduler regressions pass, and the browser smoke test passes all ten cases and eight metric rows. Full local CTest is 14/16; the same two unrelated startup-default assertions remain failing. No runtime setting or frontend control was added by these four fixes.
