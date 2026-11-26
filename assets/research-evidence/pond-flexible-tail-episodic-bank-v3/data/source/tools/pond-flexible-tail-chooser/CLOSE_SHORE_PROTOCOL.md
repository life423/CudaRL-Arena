# Close-shore chooser continuation — November 25, 2026

New user authorization: check the two failed near-bank starts, train the chooser
with native CUDA PPO, protect the motor, no scripted live avoidance or mouse,
and update Godot only after qualification. Prior studies stay closed.

## Diagnostic before learning

Replay phase8/north/negative tangent and phase9/north/positive tangent from the
curved-bank-v1 baseline-shore fixture, using installed appended16 and motor88.
Require all archived COM samples/actions, fault and accepted counts to agree
(COM <=1e-6m, actions/counts exact). Capture the full World before decisions at
0.8,1.6,2.4 seconds. These include body, actuator/filters, motor/course histories,
request, clock and visitation memory. Preserve each snapshot and its hash.

At each of six captures compare the installed chooser and all15 speed/rate
actions held for either2.4 or8s, followed by8s of zero course rate at the same
speed, then the installed chooser. Each branch lasts at most40s. No state reset
inside a branch. Predetermined sequences are diagnostic probes only; they do
not become Godot code or supervised labels. Record all186 branches. A witness
needs40s without guard/fault/reset, positive interval clearance, tilt<=10deg,
depth error<=1mm, >=6mm/s mean whole-body speed, max stall<=8.14s and final
certified gap>=15mm. Safe but not qualifying outcomes remain reported.

Cap diagnostic work at6,000,000 attempted substeps and90s native/100s external.
No silent retries. Stop on replay mismatch or numerical/native faults; guard
stops are outcomes, not exceptions. If neither original state has a witness,
do not blindly run the proposed close-start curriculum: report the blocker.

## Conditional learning, frozen before execution

If a witness exists from each first capture, use one128-update session with
1,024 CUDA worlds,8 decisions/world/update,1,024-row minibatches,four passes.
Keep curved-bank-v1 inputs39, reward, action set, PPO parameters, motor and
physics unchanged. Start from curved-bank-v1 full checkpoint80 (original12+4
passes), preserving actor, critic and both Adam ages/moments exactly. Verify
the parent roundtrip before explicitly creating this study's new initial worlds.

The focused change is initial-state exposure: half the worlds get random bank
tangents at15–30mm certified gap, balanced both directions, using training
phases other than the held-out evaluation8/9; a quarter keep earlier bank
approaches and a quarter interior starts. Sample bank azimuths continuously,
not the fixed east/north evaluation starts. No prescribed direction follows
initialization. No training time-limit reset is added. Reset only after the
existing scored physical terminal; do not conceal live failure. New study
initialization is explicit and included in resume/provenance checks.

One3-update exact-resume diagnostic <=31,457,280 substeps. New placement must
pass finite geometry and CPU/CUDA parity checks before optimization. Main cap
1,048,576 macro transitions/1,342,177,280 substeps/1200s native1800s external.
All checkpoints saved; original12x120s+conditional4x300s suite every8,17 max,
2,112,000 substeps/30s each. New close suite also every8 (not only when original
tests pass),17max,1,152,000 substeps/30s native35s external. Numerical failure
stops; guard failures are recorded learning outcomes. No automatic extension.

Keep all prior qualification criteria:12+4 original tests,8/8 close starts,
at least4 with15s atpositivegap<=20mm,maxstall<=8.14s,mean near time>=10.96s.
Up to four qualifying checkpoints get unchanged4x1800s coverage tests; require
all4 safe,no stops/resets,mean>=43.91304348%,worst>=42.60869565%,default>=43.76811594%,
maximum slowdown<=8.14s. No union-coverage substitution or changed thresholds.

Only then stage an isolated39-input native Godot DLL with exact selected300s
and old-model300s parity (480,000 substeps max),package/headless/rendered checks
without mouse or keyboard input. Preserve the installed bundle if any gate
fails. Rest motor is frozen; distinguish reused retention from fresh tests.
Archive all work, failures and bindings; stop after this session's result.
