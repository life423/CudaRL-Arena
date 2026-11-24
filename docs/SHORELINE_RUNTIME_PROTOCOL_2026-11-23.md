# Shared shoreline product qualification v1 — November23,2026

Status: frozen before its single dynamic launch. The user's current request
authorizes fixing the full shoreline, removing the smaller legacy safety limit
and retiring obsolete code. This is a new product-change check, not a restart
of any closed interface, current or training study. No optimizer or training is
allowed. Preserve saved motor/checkpoint bytes, original reviews and all failed
evidence. Keep changes local; publication was not requested.

## Model and success boundary

Product geometry remains water radii0.52/0.34m, flat bed0.22/0.14m and floor−0.16m.
The shared native height field supplies both the rendered bed and contact.
Remove the0.20/0.12 ellipse, old six-plane geometric domain and duplicated
benchmark10mm exclusions from the PRODUCT execution path. Retain finite-state,
joint±1rad, filter12rad/s, reference200rad/s², torque±10µNm and solve/energy
checks. The1Hz target filter,3Hz servo and10×5ms integration stay unchanged.

The new versioned pond runtime composes model6 inertia/water forces with
compliant terrain loads. Per-link minimum vertical gap g=y−H has potential
0.5k min(g,0)², k=0.8N/m; force is k max(−g,0)(−Hx,1,−Hz) through the actual
point Jacobian. Approach-only link-mass-scaled damping adds positive semidefinite
resistance along grad(g). No tangential force, restitution, route, heading
choice, pose projection, reset or rescue is supplied. Report contact potential,
spring work and damping loss. The existing2%/2e−10J discrete kinetic-work check
remains; contact force work is included explicitly and damping loss is tracked.
Its resistance-loss field includes water AND contact damping, not two additive
copies. Also report spring work+potential change and maximum passive-energy
excess K+U−K0−U0−positive actuator work−noncontact external work. Before the run,
fix the passive allowance at2% of(initial kinetic+positive actuator work), plus
2e−10J and the initial/current certified potential uncertainty. That uncertainty
is the sum of0.5k min(lower gap,0)²−0.5k min(upper gap,0)² per link; it exposes
the geometry error instead of silently treating spring work as external energy.
Nonfinite diagnostic/balance values fail explicitly.

The head ellipsoid and rod/composite-fin elliptical cylinders are retained.
The tail additionally includes the actual thin fin box: the inherited ellipse
misses its extreme membrane corners by about3.53µm, so merely retaining it could
not certify the whole fin. This versioned union correction changes no body
mass, inertia, joint, drag or rendered dimensions. Complete side/end surfaces are covered
by analytically bounded adaptive patches. Sampled minima are upper bounds;
whole-body lower bounds include terrain gradient/Hessian and shape coverage.
Maximum permitted compliant overlap remains0.5mm; query uncertainty≤10µm.
Each whole-body query has a fixed131,072-visit computational cap, chosen during
zero-integration preparation; this does not change geometric/physical tolerances.
An exhausted/invalid certificate is a failure, never a pass or pose correction.

This is a fully submerged pond model. It checks the actual highest whole-body
envelope against water y=0; emergence is an explicit unsupported-surface stop.
Do not leave full-submerged buoyancy active above the water or call this bank
climbing/land physics. Neither contact deflection nor continued playback proves
learned turning. No bank sensors are added to the frozen41-input actor.

## Preparation gates (zero integration)

Compile native extension/core and pure geometry, whole-body certificate,
observation and package-provenance checks with warnings as errors. Validate
original package hashes and all41 input values against preserved references.
Preserve every failed build/pure-check attempt and distinguish it from physics.
Seal actual source, executable, protocol and input hashes before dynamics.
Use short evidence paths and verify inventory before the one dynamic launch.

## One finite dynamic launch

CPU only, one process, at most900s and2GiB evidence. No automatic relaunch,
tuning, tolerance relaxation or unused-budget reuse after a dynamic failure.
No CUDA dynamic parity/retention claim follows from this CPU product check.
CUDA core compilation/pure query checks, if performed, must be identified
separately; no CUDA physics or optimizer work is included implicitly.

| Phase | Cases / maximum attempted controls | Maximum actual substeps |
| --- | --- | --- |
| Original vs new no-contact retained skills |30,400 each =60,800|608,000 at5ms|
| Bank impact/contact |8×200=1,600|16,000 at5ms|
| Half-step references |4×200=800|16,000 at2.5ms|
| Actual pond saved-motor outward playback |4×1,200=4,800|48,000 at5ms|
| Total cap |68,000|688,000|

Count BOTH baseline/new controls and all attempted substeps, including rejected
intervals, preparation/contact diagnostics and half-step references. No extra
pilot/tuning run is authorized. Record every planned/attempted/skipped case and
first failure PRE/candidate, not just successful controls. A shared numerical,
certificate, physics or retained-skill defect stops affected remaining work.

Retention uses4×160 short swims,2×1,200 long swims,12×1,280 holds and2×6,000 holds.
The six original holds are ids0..5; six withheld holds are poses1/2 at offsets
0,−0.25,+0.25mm. Endurance holds use poses0/3 with zero depth offset. Keep the
original short/long-window/hold gates,28 exact previous long-window pass keys,
known startup misses and original retention tolerances. Independently closed-loop
old/new actors use the same saved weights. Compare body/filter/torque state at
each control (existing2e−6 state tolerance; report exact differences as well).

Long straight reference trajectories cannot fit the actual pond. Use an
explicitly labeled clear-water compatibility fixture: water radii0.9/0.7m,
flat bed0.65/0.45m, same floor, original comparison arena0.6/0.4m. It must have
zero terrain contact. This is not a larger product pond or pond-navigation
success. Report matched no-contact product prefixes separately as PURE endpoint
queries of both reference/new compatibility traces until the first uncertain or
possibly contacting endpoint. This does not establish substep-by-substep product
prefix equivalence and adds no dynamic controls to the cap.

Bank fixtures use headings0,45,90,135,180,225,270,315degrees. Initialize each
whole body100µm clear of the real bed at its−80mm starting depth, with5mm/s
outward rigid velocity and zero motor commands. Initial-state placement is
fixture construction only. Preserve physical state during each run. Require
finite dynamics, certified overlap≤0.5mm, actual compressive contact response
and nonnegative damping loss; no world escape or unexplained energy gain.
Half-step cases repeat the0/90/180/270 fixtures with physical2.5ms steps while
holding each original5ms filter reference across two steps. Compare at matched
times: COM position≤0.25mm, velocity≤0.25mm/s, orientation≤0.5degree. These
new contact convergence criteria are fixed before the run, not relaxed later.

Actual pond cases start the preserved saved swimmer at center, headings
0/90/180/270degrees, and run up to60s. Record COM speed/course, depth/tilt,
minimum certified body gap, contact events and stop reasons. An honest
surface-scope stop or stalled frozen swimmer is not autonomous navigation.
Excessive penetration, unknown clearance, nonfinite state or numerical failure
fails physical adoption. Do not require straight travel through the bank.

## Installation

Only a qualified runtime may replace the current installed DLL. Keep a backup,
verify saved-policy hashes after checks and inspect rendered phone/tablet/
desktop layouts with zero physics. Remove retired study imports before deleting
their modules so ordinary Project Manager Run remains usable. Verify the actual
correct Project Manager entry and launch; keep the old save namespace. If any
gate fails, preserve evidence and show the honest incomplete outcome without
installing an unqualified physical model or starting training.
