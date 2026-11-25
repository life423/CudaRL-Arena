# Signed course feedback through 90 degrees — session 3

November 24, 2026. The user authorized correcting the course-error input,
another native GPU PPO session on a separate candidate, and checking swimming
and turning in both directions. Earlier sessions are closed and remain intact.
This study changes course information; no chooser training, request-cadence,
physics, body, reward, episode-duration or actuator-limit change is included.

## Input and initial policy

Use the existing whole-beat COM course estimate. In opt-in `whole-beat-90`
mode, input zero is wrapped signed course error divided by pi/2 radians.
Thus 20/30/45/60/90 degrees map to 0.2222/0.3333/0.5/0.6667/1, with
opposite signs for mirrored requests. The wrapped full range is [-2,2];
there is no information cap at 20 degrees or before 90 degrees. A stopped
or stale measurement retains its existing explicit invalid behavior.

Initialize from the separate qualified-gentle whole-beat checkpoint 56:
`build/flexible-tail/whole-beat-steering-v2/training/actor-56.bin`, SHA-256
`888BA2A571CEB9F90091D147A4CC56C6CECD092B3B6F58D679FBFCC779E98891`.
Copy it and multiply coefficient zero by 18 for the new 90-degree scale.
That preserves the prior small-angle function analytically within the old
20-degree range, subject to measured float rounding. It allows larger errors
to produce distinct policy activations. Other coefficients, zero intercept,
fitted gait and decoder start unchanged. Only the existing four steering
coefficients learn. No original checkpoint is rewritten.

The transferred actor requires its explicit observer-mode metadata; it cannot
be loaded through the older observer. The original live gait, original
update-eight offset and native Godot library remain protected. Historical
half-second/whole-beat mappings remain available by explicit mode.

## Finite native training session

Keep the preceding motor PPO configuration: 1,024 CUDA worlds; 64 sampled
80 ms decisions/world; 65,536 samples/update; minibatches 4,096; four passes;
actor/critic rates 0.0003/0.001; fixed Gaussian log standard deviation -1.2;
gamma 0.995, lambda 0.97, clipping 0.20, entropy 0.01, value coefficient 0.50,
target/hard KL 0.015/0.05. Keep the four-coefficient actor and 36→32→1 critic,
original reward coefficients, 13-second turn plus two-second onward episode,
physical timestep/substeps, torque limits, trim cap and slew. Preserve full
body, servo, history and trim state through turn/onward and across rollouts.

Balance paired left/right requests at 15/30/45/60/90 degrees using the same
training-only archive and seed machinery. Adding 90-degree practice is the
explicit task-range extension requested by the user. Evaluation starts remain
withheld. Only scored episode terminals initialize another training body.

Maximum 128 completed updates / 8,388,608 sampled decisions and
536,870,912 attempted physical substeps; cooperative training wall budget
1,200 seconds, external process limit 1,800 seconds including startup/final
save. Finish and checkpoint an update already started. Save complete actor,
critic, both Adam states, native worlds, episode/RNG positions and counters
after every update, with immutable actor exports and observer metadata.
No silent restart, budget extension or criteria adjustment. Invalid native
physics, nonfinite data or serialization faults stop the study with rejected
evidence and the previous complete checkpoint. Expected training clearance/
posture terminals remain scored and counted.

## Prerequisites and assessment

Before physical work, run native zero-physics diagnostics for exact frozen
gait and inactive bypass, angle monotonicity/distinction through ±90 degrees,
mirrored sign, finite critic observations and small-angle policy-transfer
agreement within 1e-6. Emit the exact transferred pretraining actor separately.
Run one paired baseline with original whole-beat checkpoint 56 versus that
transferred actor/new mode. Do not accidentally evaluate the unscaled original
coefficients with the new input scale.

Verify exact full CUDA resume with three actually executed updates: one/two
continuously, then reload one and repeat two. Allocation: 196,608 samples,
12,582,912 maximum substeps, 600 external seconds. Compare CPU/CUDA observer
features at initial/restored states within 1e-6, with zero additional advances.
Count diagnostic optimizer calls and physical work separately.

At each multiple of eight, compare immutable candidates against the same
whole-beat checkpoint 56 on four distinct held-out moving phases and both
signs of 15/30/60/90 degrees: 32 turn cases per policy. Keep 13 seconds to
turn, two seconds onward, two seconds beat closure and the existing final
two-second turn/first two-second onward scoring windows. Retain actual
mass-weighted COM course error <=5 degrees, projected and rolling speed
>=6 mm/s, tilt <=10 degrees, depth error <=1 mm and positive certified
whole-body clearance. Retain >=80% matched complete root/passive frequency
and swing gates. Record response/completion times, actual signed course
change, effective radius, clearance, raw reversals and all failures. Separate
motion-only diagnostics from complete passes without waiving a criterion.

Also retain all 12 straight/from-rest cases per arm, 12 seconds each with
their original two final two-second windows and exact inactive bypass.
This does not establish a new holding/braking skill. One evaluator invocation
has 1,100,800 maximum substeps and a 30-second external limit. One baseline
plus 16 scheduled checks has at most 18,713,600 substeps and 1,496 trajectories.
Reuse already accepted unchanged physics rather than restarting old studies.

Qualification requires all 32 turns and 12 swimming cases. Rank experimental
checkpoints that retain every original ±15-degree key and all 12 swimming
cases by 90-degree passes, then 60,30,15 and lower mean90-degree course error.
Report left/right separately and every scheduled result. Failure to master
turning is progress data, not an automatic abort while retention/native gates
hold. A regression in required swimming requests a cooperative stop.

No Godot model adoption or autonomous-navigation claim follows from this motor
study alone. Godot continues loading the preserved working bundle. Any future
publication still needs repeated-turn and full pond/endurance qualification
with the exact new observer mode.

Report compilation, GPU collection, inference/PPO, readback/waits, evidence,
CPU evaluation and total elapsed time separately. Host waits overlap GPU work.
Preserve original files, source/configuration identities and all attempted-run
receipts. Do not begin another learning change after this session automatically.
