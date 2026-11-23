# Untimed Godot Swim: bounded product verification

The user approved making Godot's interactive Swim untimed, preserving the saved
motor, physics and pond checks and updating the actual app. This is a viewer
runtime change, not approval to restart any research comparison or train either
policy. All prior research allocations remain closed.

Implementation: a separate reset_untimed_swim API and bridge-only endpoint
bookkeeping omit the finite episode terminal. Original reset(id,false), research
recorders, observations, physical equations, source contracts, torque/filter/
servo/timestep settings and saved policy files stay unchanged. Body, request,
world/task counters, servo targets/rates and previous torques persist through
eight seconds and Pause/resume. Existing pre/post whole-body pond containment
and solver/domain/finite checks remain. An integer-counter overflow guard is
explicit; it is not an arbitrary session timer. Rest remains64seconds. This
does not enable rejected autonomous policies or establish repeated turning.

Before physical verification, bind the actual product DLL, source, test script,
policy and historical comparison trace hashes in build/us26/verification-inputs.json.
Save pre-change DLL and bridge source snapshots under short, flat evidence names.
No nested historical tree or research launcher is used.

One new native product-verification trajectory only: pose0, original saved
learned-amplitude policy, untimed Swim. At most800 attempted50ms controls,
8000 attempted5ms substeps,40 simulated seconds,60seconds process time and
20MiB new functional-test output. All attempts, including the rejected boundary
candidate, count. No physical retry or allocation transfer; failure closes this
check. Compilation and zero-integration bookkeeping/UI checks are separate and
cannot call physics integration. No optimizer or CUDA training runs.

Read the original saved CPU trace rather than executing a second reference.
Compare the first160 accepted endpoint poses under the existing2e-6 model/
quaternion/joint fidelity limit. Continue through161/162 without resetting,
rebasing or changing requests. At170, Pause/get_state/start must leave every
body/counter/request/servo field unchanged apart from paused; then resume171.
Continue until the original supported pond-boundary guard stops the attempt or
the800-attempt cap, at which point pause. Record every endpoint, allattempt/
accepted counters, last safe and candidate states, speed/distance/depth/tilt
and exact stop reason. A bank stop is retained, not treated as autonomous
navigation success. Zero-time Rest and old finite API metadata checks preserve
their1280/160 horizons without spending additional physical controls.

Verify the UI at1280x800,390x844,768x1024 and844x390 with processing disabled,
including elapsed time, measurements, keyboard focus, no overflow and accurate
scope text. These checks take zero physical steps. Finally install the reviewed
DLL into the correct live project after gracefully closing its old instance,
reopen paused and inspect the actual Godot window. No further automated Play
actions, policy training, research rerun, push or merge are authorized here.
