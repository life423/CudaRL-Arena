# Frozen requested-course capability probe

Status: preparation only; fresh explicit approval is required for execution.
This is a new finite diagnostic, not a retry of either closed continuous-pond
study. Continuous-pond V2 remains on design hold. No PPO, optimizer, policy
selection, actor replacement, food, route planner, or Godot playback is included.

## Question and frozen controller

Can the original competent swimmer respond to a persistent change in requested
course while retaining its existing world-depth and lane feedback? The earlier
free-swim observer replaced these inputs with the instantaneous head direction
and removed horizontal lane error. A larger new controller is not justified
until this smaller capability question is measured.

Use model 6, the original update-336 backbone, and its exact zero-column
migration from 40 to 41 inputs (2,532 parameters). The appended input is zero.
The original 41-input requested-motion observer, tanh output decoding, 1 Hz
reference filter, four physical servos, torque limits, 5 ms mechanics, energy
checks, contact checks, and fluid equations remain unchanged. The saved holding
parent is copied and checked for provenance; its gain is not used in swimming.
There are no learned residual coefficients, Gaussian action samples or rewards.
The exact parent file SHA-256 values are
`810dbd593ce39475cdd6e95b709bef3fdb34d50c9fcbbb2c345c7a8882a0bef0`
(swimmer), `fcaa74c5da6adc1e000282776cebfc51f007b00633e7a6a29ed9d8a34e485e1e`
(its protocol), and
`c54f73b57ab4280d99802a7253c13004f95c754e39e0acf8f6833377bbb760c3`
(holding parent). The native loader also checks their exact payload checksums.

## Cases and persistent requests

The 22 probe cases are poses 0 and 3, each at offsets in this fixed order:
0, +5, -5, +15, -15, +30, -30, +60, -60, +90, -90 degrees. Initialize through
the original `straight_motor::initial_world`, using the original long-swim
contact arena. The visible pond's active ellipse is 0.20 by 0.12 m in radius;
the ordinary bank guard uses radii 0.195 and 0.115 m. Every actual native body
corner participates in this guard.

Each case has at most 320 physical control attempts (16 s). Controls 1–80 are
real warmup under the original fixed request. Immediately before control 81,
rotate the **original requested horizontal course**, once, by the offset about
world +Y using the simulator's quaternion convention. This is not relative to
the gait's oscillating head orientation. Positive yaw about +Y rotates +X
toward -Z. Keep the resulting request fixed for controls 81–320.

For offset zero, change no `ForwardTask` field at the latch: original heading,
target, velocity and lane origin remain bit-identical. For nonzero offsets,
rebase only the lane origin's X/Z to the actual COM at control 80, retain its
original Y (depth), set the new persistent heading and 8 mm/s terminal velocity,
and put the request target 72 mm ahead of that origin. This intentional request
change does not reset task controls, startup features, world clocks, physical
pose, velocities, filter state, held targets, reference rates, energy ledgers or
any learned parameter. Do not call `begin_forward` or `request_speed` again.
Legacy straight-task outcome is recorded separately and is not a turn scorer.

## Required prerequisite and numerical interpretation

Before the 22 cases, run four original initial poses 0–3 for 40 controls each,
on cloned CPU and CUDA worlds, using **the same CPU-chosen command** for both.
This is 160 controls per backend, 320 total. Compare every accepted endpoint:
position 2e-5 model units; quaternion and joint angles 2e-5; linear velocity
2e-7 m/s; angular/joint velocities 2e-5 rad/s; held targets, reference rates and
normalized torque 2e-5. Six cumulative energy quantities use
1e-12 J + 1e-4 times their maximum absolute value. Same-observation CPU/GPU
backbone and decoded commands must differ by at most 2e-5. No tolerance changes
are allowed after this document is frozen.

This tests backend physics with matched commands, independently of pointwise
inference. Independently closed-loop CPU/CUDA trajectories are not a physical
agreement gate: small inference differences can amplify under feedback, as the
closed 960-control gait attempt demonstrated. The actual 22-case probe uses the
CUDA observer, frozen CUDA NN, tanh decoding and CUDA mechanics.

## Stops, accounting and evidence

Hard physical, nonfinite, solver, energy, contact, active-ellipse escape,
I/O, evidence-cap or deadline failure closes the entire attempt. An ordinary
bank, absolute COM-depth error over 5 mm, or head tilt over 20 degrees stops
only that case at the first accepted 5 ms crossing; priority is bank, depth,
then tilt. No rescue, state projection, continuation after terminal, or hidden
restart is permitted. The first rejected PRE/candidate state is retained for
hard faults. These guard limits are not successful swimming standards.

At most 7,040 probe controls + 320 prerequisite controls = **7,360 control
attempts and 73,600 attempted 5 ms substeps**. Partial terminal controls count;
failed attempts are not refunded. The one process has a **120 s** deadline and
at most **128 MiB including frozen inputs and run evidence**, with room reserved
for the close report. A conservative pre-step projection reserves 700 bytes per
substep CSV row, 2,200 per control CSV row, complete binary endpoints and 1 MiB
for closing artifacts; oversized frozen input copies reject before integration.
No automatic retry, budget extension, extra offsets, or new training follows.
Preparation performs zero physical controls and zero optimizer steps.

Save all initial/control endpoint Worlds and task requests in checksummed
binary evidence, plus round-trippable CSV with requests, observation, four
backbone latents, actual four commands, body state, filter state, clocks,
counts and six energy quantities. Save accepted 5 ms body/energy states in
substep CSV and retain the first rejected candidate separately. The result
reports actual attempted/accepted counters, completed cases, ordinary case
failures, hard fault and elapsed process time. Evidence may stop early under a
registered fault; missing cases are unrun, not passes.

## Descriptive measurements

Report head-heading error and gait oscillation separately from COM velocity
course and displacement course. For every completed turn phase report signed
requested-course progress, horizontal COM path length, mean and last-4-s COM
velocity vector/course error, last-4-s head circular-mean error and head error
RMS, lane/depth/tilt maxima, work, and the first 20 consecutive full-control
endpoint samples with head error at most 10 degrees (nominally 1 s, if any).
This sampled diagnostic does not establish the unsampled in-between head error.
For failed cases report the observed prefix and leave unavailable last-4-s
fields null. A bank failure can coexist with good requested-course tracking;
report both facts separately. Compare each offset against
its same-pose zero-offset baseline. These measurements establish the frozen
motor's finite response; they do not qualify a continuous-swimming policy or
guarantee turning indefinitely.

## Preparation, freeze and explicit execution guard

The standalone target is built only by the new recipe under
`tools/frozen-request-turn/`; the existing app, held V2, and old build files are
untouched. `frozen_requested_turn_probe --prepare RUN PARENTS PROTOCOL_MD`
copies parents and this document, stores a versioned protocol checksum,
performs pure same-observation inference and request-latch fixtures, and writes
a zero-control preparation report. Freeze the new document, source, guard,
recipe, executable, parent files, transitive compiled libraries and physical/
observation source closure before asking for the fresh grant.
Execution uses an external 120 s whole-process watchdog as well as the native
deadline, which starts at program initialization and includes evidence writes.

`frozen_requested_turn_probe RUN` must refuse before creating its one-shot
physical directory unless a separate authorization envelope contains the exact
new protocol checksum, exact frozen-manifest file checksum, and actual fresh
user approval text. The prepared executable and parent checksums must also
match. Existing or held-study grants are incompatible. Preparation is allowed;
physical execution and grant creation are not authorized by this document.
