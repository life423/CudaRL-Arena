# First flexible-tail motor fit

The pilot fit one two-output motor to fixed native diagnostic actions, then
ran it in closed loop on new starts. The candidate is saved locally under
`build/flexible-tail/motor-pilot-v1/actor.bin`. It is ineligible for live
adoption. The supervised fit completed 384 Adam steps on the RTX 5070 and
zero PPO updates. See the [fixed protocol](protocol.json) and [fit receipt](fit-receipt.json).

The first collection accepted all 150,400 physics substeps. Two teacher
turns missed the course gate. The cue used the half-second course sample
instead of the two-second warmup average used in the qualifying diagnostic.
The one [recorded correction](collection-repair.json) restored the
two-second cue definition and kept the half-second turning feedback. All
12 corrected cases passed, yielding 9,400 observations and paired yaw and
pitch targets. [Original results](teacher-original.txt) · [Corrected results](teacher-corrected.txt)

The fitted action MSE fell from 0.23943 to 0.00106 rad². GPU optimization
took 85.9 ms and the process completed in 0.298 seconds.

| Held-out case | Physical result |
| --- | --- |
| Forward swim | 4.93 mm/s projected speed, 0.98° course error and 16.9° peak tilt |
| ±15° turns | +4.19° and −6.26° of actual course change |
| ±30° turns | +2.90° and +6.46° of actual course change |
| Calm hold | 7.41 mm peak anchor error |
| Gentle-current hold | 48.41 mm peak anchor error |

All 89,600 held-out physics substeps completed without a fault. The
controller passed 0/7 behavior cases under the original speed, course,
posture and hold gates. The evaluation took 0.564 seconds. [Held-out results](held-out.txt)

All teacher pitch targets were zero. The candidate predicted pitch errors of
0.00647 rad RMS and up to 0.0266 rad on those inputs. A separate diagnostic
suppressed the learned pitch target, keeping its learned yaw. It removed
tilt in the fixed planar fixtures and still passed 0/7 cases. Both posture
and closed-loop course control require work. The pitch intervention stayed
in the scratch evaluation. [Axis audit](axis-error.json) · [Pitch comparison](pitch-ablation.txt)

The 256-world CUDA preflight completed every one of 32,768 substeps in
24.3 ms of GPU integration time. Its process took 0.292 seconds, including
startup. This measures physical collection, with zero optimizer updates.
[GPU preflight receipt](cuda-batch-receipt.json) · [GPU preflight results](cuda-batch.json)

The main pilot used 390,400 CPU physics substeps across both collections
and the candidate evaluation. The matched pitch diagnostic used 89,600
additional CPU substeps. The Godot preview continues to present the
verified recordings. A live mode requires a qualified flexible-tail motor
and a learned goal chooser.
