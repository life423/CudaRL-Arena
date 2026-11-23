# Continuous swimming v2 — proposed numerical protocol

**Preparation only. No new physical or optimizer execution is approved.**
This is a proposed new, finite study following the closed failed November23
attempt. Compile and zero-integration fixtures may run. A fresh explicit user
grant, bound to this protocol and complete prepared source/binary closure, is
required before the new physics gate, calibration or PPO. All earlier data,
checkpoints, rejected actors and unused budgets remain preserved and closed.

## Correction supported by the saved data

The first study optimized sixteen-second return from0.373802 to0.544192 while
passing0/16 sixty-second starts. Sampled moving turns lost4–5mm of depth; a
yaw-only correction could not demonstrate a clean turn with the existing body
requirements. This attempt changes both the training credit and motor residual
authority. It does not claim the isolated contribution of either change.

Preserve Model6, all physical dimensions/masses/water/contact equations, four
actuator axes,1Hz filtering, ten nominal5ms substeps/control and all+/-10microN*m
limits. Use no current or food. Canonical336 and holding source7 are the exact
frozen parents documented in the first protocol. Initialize the new residual
at zero, preserving the competent inherited backbone rather than inheriting
the rejected stalling coefficients. Old requested Swim and Rest bypass the
new residual exactly, using the original network and saved holding gain.

## Four-output state-aware residual

Policy/task version2 uses eighty episode-Gaussian means:20 coefficients for
each of the same four outputs. Sample once per full training episode with
fixed sigma0.15. Its first eight derived features are identical to v1. Append:

- `tanh(depth_error/.001)` and `tanh(COM_vertical_velocity/.001)`.
- `tanh(roll_degrees/10)` and `tanh(pitch_degrees/10)`.
- Four `tanh(joint_angle/physical_limit)` values.
- Four `tanh(joint_rate/12)` values.

For each output `j`, use
`latent[j]=backbone[j]+.75*tanh(sum_k(tanh(u[j*20+k])*feature[k]))`.
The zero vector returns the original outputs exactly. All old branch routing,
latent-action probabilities and bounded tanh actuator commands remain explicit.
This is restricted parameter PPO in one four-output motor, not a new brain per
ability, a fully plastic backbone or a live learning worker.

Use the same sixteen training starts and eight withheld starts as v1. The
active ellipse remains0.20/0.12m and behavioral inset0.195/0.115m, checked with
the full articulated body. A missed inset remains a valid true task failure.
An actual hard escape/contact/numerical/joint failure still stops the study.

Additional registered behavioral failures prevent exploration from reaching
unsafe or stationary states: absolute depth error>5mm or body tilt>20degrees
at an accepted5ms substep, and a two-second horizontal-COM arc speed<2mm/s at
control boundaries from control80 onward. The rolling window is40 actual
controls, with actual accepted durations. Bank/depth/tilt terminals take
precedence. These are failed learning examples, never pose corrections or
deployed recoveries. Final qualification keeps1mm depth/10degree tilt unchanged.

## Sixty-second motion credit

Use2,048 GPU worlds,128 per initial case, with1,200 controls/60seconds per
training episode. Do not end training credit at16seconds. At each actual
accepted interval use this rate:

```
clamp(horizontal_COM_speed/.008,0,1)
 - 2*((body_forward_COM_speed-.008)/.008)^2
 - .25*(body_side_COM_speed/.008)^2
 - .05*(depth_error/.001)^2
 - .05*(roll_degrees/10)^2 - .01*(pitch_degrees/10)^2
```

Multiply by actual accepted duration; subtract6250 times the difference of
recorded float PRE/POST positive joint work, and subtract4 once for any
registered behavior failure. Divide the whole return by60, never survival
time. Stopping therefore loses reward and eventually ends the episode. There
is no waypoint, scripted desired turn, artificial propulsion or special food
reward. Every successful full episode must continue moving around real banks.

Score the original80-dimensional sampled Gaussian exactly once per episode.
Use no critic/GAE/bootstrap, gamma1, same-case leave-one-out baseline over the
other127 samples, and a positive finite global within-case RMS fit on one
independent calibration batch. Calibration data never enter an optimizer.

Fresh Adam:LR0.003,beta1=0.9,beta2=0.999,epsilon1e-5,global norm cap.5. PPO clip.2,
KL guard.01, four epochs/four512-row minibatches maximum. At most24 updates /
384 Adam steps. Seeds:calibration`0x8e25a643`,training`0x137be951`,permutation
`0x04a71369+update*4+epoch`. No seed search, retry or automatic extension.
Initialize/observe in128-thread blocks; physical advance uses32-thread blocks
and a64-block grid covering all48SMs at2,048worlds. Record actual resource use.

## Prerequisites and qualification

Complete all numerical/CUDA zero-step fixtures and an independent audit before
the physical gate. Freeze sources, proposed protocol, binaries, libraries,
parents, initial means and optimizer. Require the same26-case matched-command
physics gate and all original tolerances, including torque/reference states.
Then the original baseline, one independent calibration, and a separately
verified readiness marker before the first Adam step.

At each update, run the same sixteen60second selection cases. Keep every v1
motion/depth/tilt/coverage requirement. Stop at the first16/16, freeze once,
then run eight uninterrupted30minute withheld cases plus all original46,240
retained-skill controls. Final depth1mm,tilt10degrees,overall speed6mm/s,
eight-second motion windows3mm/s,COM spans0.20/0.12m and genuine accumulated
45degree-hysteresis turns totaling2pi are unchanged. If zero residual already
passes selection, validate it without invented PPO. No retry after reserved
validation failure. Only a fully qualified actor permits one diagnostic
constant-intercept ablation and a200-control native/render capture.

## Finite limits and evidence

| Phase | Maximum controls |
|---|---:|
|Matched-command gate |2,480|
|Old baseline |10,720|
|Sixteen zero-residual60s probes |19,200|
|Independent calibration |2,457,600|
|24 fresh PPO collections |58,982,400|
|24 sixteen-case60s selections |460,800|
|Eight30min withheld worlds |288,000|
|All old retained skills |46,240|
|Postqualification constant-intercept ablation |19,200|
|**Native study total** |**62,286,640**|
|Postqualification actual Godot capture |200|
|**Overall total** |**62,286,840**|

Maximum native substeps622,866,400; overall622,868,400. Partial terminals count
actual substeps; inactive padding is not simulation. Enforce900 combined
native/process seconds including the capture, at most10.5GiB new evidence and
8GiB remaining free reserve. Preserve all earlier data; stop for insufficient
space. No paid/cloud resource is authorized.

Keep initial/full final worlds/tasks, sampled80-dimensional vectors, all
probabilities/checkpoints/optimizer states, the v1 compact40-float trace and
actual substep/terminal flags. New physiology features are reconstructible
from these states and fixed original depth. Preserve failed preparations.
Independent saved-data verification uses faithful binary32/FMA CUDA observer
arithmetic and nonFMA CPU observer arithmetic where needed, without changing
acceptance tolerances. Bind actual approval, sources, code snapshots, process
receipts, all records and final qualification. Saved inference remains separate
from actual training; no unqualified package may replace the working pond.
