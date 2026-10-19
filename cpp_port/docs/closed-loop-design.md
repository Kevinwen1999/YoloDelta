# Closed-loop tracking evaluation design

Status: confirmed design. This document describes the agreed design choices; implementation is a separate phase.

## Objective and boundary

Measure whether the C++ system follows the intended moving target quickly and safely when its own movement changes subsequent camera images. Both test modes run production detection, tracking, control and scheduled output, with backend numeric transforms preserved and actual mouse/serial delivery replaced by a simulated camera receiver. The receiver rotates the view around a fixed screen-center aim point. Target motion and external shake remain independent.

Automatic firing, recoil, physical output delivery, pose animation and a full 3D world are outside the first milestone. The existing replay dispatcher alone does not execute the complete servo behavior and cannot support a full-loop claim.

## Modes

| Mode | Clock and source | What it establishes |
| --- | --- | --- |
| Controlled time | Independently scheduled world/render, observation availability, inference result and control dispatch. GPU inference supplies detections; configurable logical processing delays govern when results become available. | Repeatable behavior and paired configuration comparisons. |
| Visible screen | Dedicated unobscured 2.5D scene window on the configured display, production adaptive capture crop, RTX 3070 inference and wall-clock control scheduling. | Actual capture, inference and scheduling latency in the current hardware layout. |

The visible-screen renderer and diagnostics are kept in separate windows/regions so overlays cannot contaminate detector input. It validates display coordinates, DPI, occlusion and crop alignment. The current baseline uses CUDA ordinal 1 for RTX 3070 inference and the configured capture adapter; report renderer/capture/inference GPU placement and any cross-GPU host copy. Other placements are separate configurations.

## Scene and camera

Start with a small hand-checked library of photographic target cutouts and backgrounds at explicit depths, projected through configurable horizontal/vertical field of view. Scenes have simple occlusion and deliberately no pose animation. Curated assets record source, license, annotation version and train/validation split. Existing detection labels do not supply persistent identity or linked head/body truth; those annotations must be added. The simulation is uncalibrated until mouse-count-to-angle sensitivity and field of view are measured against the intended application. Sweep these parameters before claiming an operating range.

Model command processing in layers: selected backend gain/rounding/splitting, transport delay and quantization, camera sensitivity, and independent external shake. The chosen output adapter preserves the production control path through servo fractional carry, limits, freshness and dispatch, then applies movement to the simulated camera. No live mouse or serial device receives commands in scored simulation.

## Truth, validity and scenarios

Each scene stores time-stamped target world position, visible head/body aim points, occlusion, persistent identity and a selection contract covering intended initial target and allowed handoffs. Ground truth and future motion stay with the evaluator; the tracker sees ordinary capture metadata and detector output only. Changing head/body aim point is distinct from changing identity. World truth advances through missed detections and full occlusion. Bounded prediction stops at the configured loss deadline; movement resumes only after distinct observations reconfirm a permitted target.

Each rendered frame has a scene-state ID linking world update, render submission, presentation, capture, inference, dispatch and simulated camera application. A broken link, wrong model/GPU, unexpected cached frame, blocked scene window or capture alignment error invalidates the scored run but preserves diagnostic logs. Ordinary detector misses remain valid tracking outcomes.

The first scored suite spans 500–6000 px/s, small/medium/large target sizes and varied observation cadences. It tests crossings, reversals, crop exits, occlusion, external shake and naturally missed/noisy detections separately, followed by selected combinations. Deliberately dropped or perturbed detections are injected after inference in separately labeled fault-isolation runs. Provide a quick fixed-seed suite and a broader on-demand multi-seed sweep; set runtime budgets after measuring cost.

## Scoring and review

Report initial selection independently of following. Quality metrics include wrong-target movement time, identity switches, normalized aim error at applied movement, time within the intended target region, loss/recovery, overshoot and oscillation. Report world-to-presentation, presentation-to-capture, capture-to-inference, inference-to-dispatch and dispatch-to-applied-movement latency separately, with distributions and observation age. Do not collapse quality and speed into one score.

Freeze the default configuration as baseline and compare candidates on identical scenes and seeds. Fail harness correctness violations immediately. Calibrate quality thresholds before turning them into automatic gates; the current opt-in motion policy requires improvement in wrong-target movement before promotion. A setting change starts a new scored segment with configuration, model, asset and seed fingerprint. Save machine-readable events and a browser report that synchronizes scene, detections, identity, camera/aim traces, commands and plots. Pausing or scrubbing the report does not pause a live world.
