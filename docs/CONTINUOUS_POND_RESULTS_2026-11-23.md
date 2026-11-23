# Food-free pond swimming — November 23, 2026

**Closed, unqualified, not adopted.** The creature cannot yet swim continuously
around this pond. The new candidate completed 64 native CUDA PPO updates and
632 Adam steps without a physical fault, but passed **0/16** sixty-second
starting cases. It learned short-horizon slowing/stalling, not dependable
moving turns. The qualified saved calm-water swimmer remains active.

The independent final evidence audit passed, with **qualified=false**. No new physical evaluation,
retry, extra training, viewer adoption or unused-budget reuse is authorized.

## What ran

The frozen [protocol](CONTINUOUS_POND_PROTOCOL_2026-11-23.md) trained sixteen
state-aware yaw coefficients around the frozen checkpoint336 backbone. Its
four actuator outputs, Model6 physics, torque limits and exact old swimming
and holding branches were preserved. No food, scripted navigation, body
rotation, bounce, rescue or live world reset supplied movement.

The fresh identical-command physics gate passed all 26 cases: 2,480 controls /
24,800 substeps, retaining the original tolerances. GPU zero-integration and
independent preparation/gate/calibration checks passed before the first Adam
step. Calibration collected 937,590 actual controls and 9,361,305 substeps.

Total actual physical spend was **75,555,675 controls / 755,025,046 substeps**.
Partial bank-terminal controls count their actual accepted substeps; inactive
padding is not physical work. Combined native process time was **232.991457 s**
of the 900 s cap. All physical phases exited normally. The eight reserved
30-minute withheld cases, constant-only ablation and 200-control Godot fidelity
capture did **not** run because selection never passed.

The RTX5070 ran 4,096 worlds with 64-thread physics blocks and a 64-block grid
covering its 48 SMs. Across 236 training-phase device telemetry samples,
activity averaged 71.84% and peaked at 98%; peak VRAM use was 3,181 MiB. These
are whole-device observations, not proof of perfect kernel efficiency.

## What was retained

The final native retained-skill suite passed all original requirements:

| Fixture | Result |
|---|---:|
| Original short swimming |4/4|
| Previously passing 60 s windows |28/28|
| Fixed-depth recovery |3/3|
| Previously passing 300 s windows |148/148|
| Original 64 s holding |6/6|
| Withheld 64 s holding |6/6|
| Uninterrupted 300 s holding |2/2|

These remain separate from Free swim. Old request modes still start from rest;
they do not establish braking or safe arbitrary mode changes. No holding,
swimming or original experiment checkpoint was overwritten.

## Why this did not solve turning

The optimization objective and desired behavior diverged:

| Saved measurement | Before | Final |
|---|---:|---:|
| Sixty-second qualifying starts |0/16|0/16|
| Sixty-second survivors |0/16|2/16|
| Mean measured duration across starts |13.04 s|27.60 s|
| Mean horizontal COM speed across starts |6.11 mm/s|3.70 mm/s|
| Deterministic first-16-second return |0.373802|0.544192|
| Sampled training return, first/last batch |0.352347|0.529831|

All final cases missed the 6 mm/s mean-speed requirement and exceeded the
10-degree body-tilt limit. Four also missed depth; fourteen missed the
registered horizontal coverage requirement. Better sixteen-second return and
longer survival did not imply continued swimming.

A read-only audit of the original 4,096 calibration trajectories found eight
45-degree heading events at 6.31–7.10 mm/s, but all drifted 4.31–5.28 mm in
depth and none demonstrated a repeated or 90-degree turn. No recorded sliding
four-second window after startup combined a 45-degree event, 6 mm/s travel,
1 mm depth and 10-degree tilt. This demonstrates some physical steering
authority, not an adequate clean turning skill or a proof of impossibility.

The strongest demonstrated problem is training credit ending after sixteen
seconds while acceptance requires sustained behavior. Limited yaw-only control
also leaves depth coordination during a turn unresolved. The coefficients
persist per episode, but their commands already depend on live state; they
are not constant yaw commands.

The next implementation decision is a longer motion-and-turn objective with
bounded depth/state correction on all four existing outputs. Preserve the old
learned branches and physical limits. Another yaw-only run or simply removing
the viewer timer would not address this evidence. This is a recommendation,
not an implemented correction or permission for another study.

## Numerical audit and provenance

The observer uses binary32 arithmetic. Near the exact pond center, a fraction
of a nanometre in the computed COM changes its normalized radial direction.
An initial double-precision saved-action audit therefore disagreed with the
recorded yaw correction. Reconstructing the frozen CUDA FMA and half-angle
polynomial arithmetic brought all calibration actions within the original
tolerance (maximum error 4.5122e-6). No native source, record, sensor threshold,
physics gate or acceptance requirement was changed. Failed offline audit code
snapshots were preserved; those retries used zero physical/optimizer steps.

Native evidence remains under
`build/ppo/continuous-pond-2026-11-23-v1`; causal inputs and initial state were
sealed in `frozen-inputs.json` (SHA256
`2df08f6343f0d13a846f2693b5bbd95e9cf9dbac9967184f03784caa7d33c15b`).
The calibration proof is `e1b252f4854b3a64e88c09944565ef9527ff12a18c733af69bd54c00428ae8e2`.
The final proof is `3c66c7864f3e6cabf93f3c4be46edc4d15f716d55d158d9b14c566c2699c0399`.
It checks every actual training/assessment row, all64 checkpoints and all old
retention traces. Numerical audit code and failed pure-audit attempts remain
separately versioned; no simulated trajectory or optimizer update was repeated.
After this closure, the GUI adds the portable verified rejection to Details.
The original frozen GUI/source copies remain in the causal-input snapshot.
All8,148 previously protected files passed their final byte-hash comparison.
The compact rejected-study archive is
`assets/ppo-four-axis-checkpoints/continuous-pond-2026-11-23-v1-rejected/evidence.zip`
(772 files,11,870,294bytes). Its manifest binds all1,692 local evidence files;
full raw control records remain local and were not deleted or put in Git.
The host observer's binary32 arithmetic was also independently reconstructed
for the original five-minute straight swims; no acceptance tolerance changed.
Read-only reward diagnostics are in
`build/continuous-pond-preparation-2026-11-23/saved-reward-diagnosis.json`.
