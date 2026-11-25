# Whole-beat motor and early-bank chooser — conditional stage

November 24, 2026. This is the conditional pond stage authorized in the user's
request to fix noisy steering, train with GPU PPO, turn early enough to avoid
the bank and publish to Godot only after passing. The motor stage is CLOSED.
No sharper-motor experiment is authorized by this stage; the user has asked
to brainstorm that next protocol after the current work finishes.

## Entry and fixed inputs

Use whole-beat-steering-v2 checkpoint 56, the sole evaluated motor checkpoint
that retained all eight original gentle-turn complete pass keys and all 12
swimming/from-rest cases. Its fitted gait remains frozen. Bind its actor and
observer metadata, assessment, original model hashes, motor manifest and closed
source audit. Its lack of 60° mastery remains explicit. Earlier anticipation
with repeated learned 15° requests is the limited navigation hypothesis.

Keep actual body/water equations, integration, actuator caps/slew, habitat and
whole-body clearance identical. No food, prescribed path, scripted turns,
force, inner boundary, teleport or deployed recovery reset.

## First frozen navigation gate

Use the preserved sustained-chooser checkpoint 56 without optimizer changes,
held-heading requests, whole-beat motor feedback and original sensing ranges.
Run the unchanged 12 fixed 120-second navigation cases and >=20% pooled
coverage requirement on the existing 345-cell estimate. Only after all pass,
run the existing four 300-second endurance cases. The gate has at most
2,112,000 substeps and a 20-second external limit. Preserve every failure.

## Conditional sustained PPO

Only if that frozen gate fails with valid native physics, initialize one
separate chooser task from the preserved chooser actor. Its critic and Adam
states are initialized once and then carried with its full world/RNG state.
Do not label this a resume of the incompatible old motor/observation task.

Keep the motor frozen. Use 1,024 CUDA worlds, eight macro decisions/world,
8,192 samples/update, minibatches 1,024, four PPO passes, actor/critic learning
rates 0.0003/0.001, gamma 0.97, lambda 0.90, clip 0.20, entropy 0.01, value
coefficient 0.50 and the existing KL/gradient/numerical gates. Keep the bank/
interior curriculum mix and reward. Change terrain observation distances to
40/200/400 mm forward and 150/350 mm sideways. These provide information to
the learned chooser; they do not command avoidance or change collisions.

At most 64 updates / 524,288 sampled macro decisions, 1,200 cooperative
training-process wall seconds and a 1,800-second external cap. Finish and
checkpoint an update already started. Save complete actor/critic/Adam/world/
sampling state and immutable exports each update. Verify exact resume using
three actually executed updates: continuous updates one/two, then reload one
and repeat two. Every eight updates, run the same unchanged fixed navigation
gate while GPU training continues. No silent retry or criteria/budget tuning.

### Physics accounting correction before this stage executes

The motor-stage paper protocol's conditional upper-bound arithmetic omitted
training episode warmups. No conditional chooser physics or optimizer work
had run when this correction was written. A macro sample can include up to
300 warmup controls plus 300 collection controls, each with 16 substeps.
The exact finite upper bounds are therefore **5,033,164,800 main substeps**
and **235,929,600 resume-diagnostic substeps**. Both include warmup. The
64-update, three-diagnostic-update and wall budgets are unchanged. The original
motor-stage protocol remains preserved for provenance; this correction applies
only before the new conditional stage starts, not retrospectively to results.
Count and report actual attempted/accepted warmup and collection work separately.

## Admission

Passing workflow/checkpoint tests is not successful navigation. Require every
fixed motor-retention, 12-case navigation and four-case endurance gate before
publishing. The exact same model weights, feedback and sensing mode must reach
the registered Godot project. Preserve the working live bundle for rollback.
Inspect the actual Project Manager entry and rendered runtime only on admission.
A failed candidate leaves the working live motor unchanged.

Record compilation, GPU physics/inference/PPO, transfers/host waits, evidence,
CPU evaluation and total elapsed time. Report actual coverage, behavior choices,
stalls and all certified-clearance stops. A clearance guard is not evidence of
a resolved bank contact. Host waits overlap GPU execution and are not additive.
