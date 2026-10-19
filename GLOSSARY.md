# Tracking evaluation

Evaluation of target detection and camera-based following, including the effect of movement commands on subsequent observations.

## Language

**Closed-loop tracking simulation**:
A test environment in which dispatched movement changes the simulated camera view used for subsequent target observations.

**Simulated camera**:
A viewpoint rotated by movement commands, with an aim point fixed at the screen center. Targets move independently of that viewpoint.

**Controlled-time run**:
A simulation run whose time progression is controlled by the test environment so that tracking behavior can be compared under repeatable conditions.

**Visible-screen run**:
A real-time simulation run that displays a scene for screen capture and measures the tracking system under actual processing and scheduling delays.

**Applied movement**:
A dispatched movement command that has taken effect on the simulated camera. Its application is distinct from command generation or dispatch.

**Target identity**:
The persistent identity of one simulated target, shared by its head and body observations even when one is temporarily hidden.

**Aim point**:
The selected location on a target that the fixed screen-center aim is intended to follow; head and body aim points belong to the same target identity.

**Natural detection failure**:
A missed or inaccurate detection produced by the detector while processing a rendered scene, including scenes with occlusion or camera shake.

**Injected detection fault**:
A deliberate change to detector output after inference, such as dropping or perturbing an observation, used to isolate tracking behavior.

**Intended target**:
The target identity a scenario specifies for following, including any explicitly allowed identity handoff.

**Selection contract**:
The scenario's rules for initial target identity, permitted identity handoffs and head/body aim-point changes.

**Loss deadline**:
The configured time after the last unambiguously accepted observation when predictive tracking movement must stop until reconfirmation.

**Scored segment**:
A continuous evaluation interval using one frozen configuration, model, scene asset set and random seed.

**Scene-state ID**:
An identifier that links one rendered view of the simulated world to its capture, inference, command dispatch and applied movement records.

**Rendered scene ID**:
The provenance marker identifying an image submitted by the visible scene renderer. A captured observation records this ID and its DXGI presentation source; skipped rendered images do not become controller observations.

**Invalid run**:
An evaluation whose capture, timing or scene provenance is incomplete or incorrect, so its tracking and latency scores cannot be trusted.
