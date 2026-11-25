# Full-angle course input — completed, sharp-turn candidate not qualified

The authorized input correction and GPU PPO session are complete. Signed
course errors now remain distinguishable through 90 degrees. The new candidate
completed **128 updates / 8,388,608 actuator decisions in 99.578 seconds** with
1,024 native CUDA worlds. It did **not** improve sharp turning. The working
Godot motor and the protected whole-beat checkpoint 56 are unchanged.

## What changed

Opt-in `whole-beat-90` uses the existing measured whole-beat COM course with
input zero equal to signed wrapped error /90 degrees. It supplies 0.3333,
0.6667 and 1 for 30,60,90 degrees, with opposite signs for the other direction.
The natural wrapped +/-180-degree range maps to [-2,2]; no 20-degree input
cap remains in this mode. Other inputs and historical observer modes remain.

The separate candidate began from a copy of whole-beat motor 56, with its
course coefficient multiplied by 18 to transfer from the former 5-degree
scale. Measured small-angle latent agreement was within 1.1921e-7. This keeps
the initial gentle-turn function, but it increases the original policy's
response above its former saturation point. Therefore the study changes
initial large-angle steering strength as well as available information.
It is not a pure information-only ablation. The pretraining physical baseline
explicitly measures that consequence.

The fitted stroke, water/body equations, torque limits, timestep, trim cap and
slew were frozen. PPO trained only the existing four steering coefficients.
Rewards and 13-second turn/two-second onward training episodes were unchanged.
Balanced +/-90-degree requests were added to the existing angle practice.

## Actual results

The expanded fixed suite has 32 turn cases (four phases, two directions,
15/30/60/90 degrees) per policy, plus 12 swimming/from-rest cases. All original
motion, posture, clearance and root/passive rhythm criteria were retained.

| Policy | +/-15 complete | +/-30 | +/-60 | +/-90 | Swimming/from-rest |
|---|---:|---:|---:|---:|---:|
| Protected whole-beat motor 56 | 8/8 | 0/8 | 0/8 | 0/8 | 12/12 |
| Transferred full-angle policy, before PPO | 8/8 | 0/8 | 0/8 | 0/8 | 12/12 |
| Candidate 72 | 8/8 | 0/8 | 0/8 | 0/8 | 12/12 |
| Candidate 128 | 0/8 | 0/8 | 0/8 | 0/8 | 12/12 |

Checkpoints 24,72,80 retained all eight complete gentle-turn keys. Checkpoint
72 ranks first among those under the preregistered mean90-error tiebreak,
but no checkpoint passed a 30,60 or90-degree case. No motor qualifies for
sharp-turn or autonomous adoption. Final15-degree motion and physical envelopes
still passed all eight cases; the passive-frequency criterion rejected their
complete passes. This distinction does not waive the frozen requirement.

The large-angle propulsion loss existed before optimization. At60 degrees,
the protected baseline achieved **34.05 degrees at 9.73 mm/s**; the transferred
untrained policy achieved **11.91 degrees at 1.93 mm/s**. At90 degrees,
projected speed dropped **6.16→0.54 mm/s** and course error grew55.78→79.10
degrees. PPO partially recovered60-degree performance: its best mean result
(update96) achieved28.15 degrees at6.32 mm/s, still below the protected motor.
Final60/90 speeds were2.67/0.61 mm/s with root swing0.289/0.117rad.

This establishes that removing the input cap is insufficient. Larger steady
steering commands can weaken propulsion with the current fitted stroke and
steering readout. It does not establish that the body is physically incapable
of sharper turns, nor that a larger neural network or changed physics is needed.
Any next experiment must address steering/stroke coordination or an explicitly
behavior-preserving information extension under a new finite protocol. No new
experiment followed this closed session.

## Checks and finite accounting

Native zero-physics checks passed distinct/monotonic/mirrored input values
through +/-90 degrees, 3,584 actual observer comparisons (maximum difference
1.7881e-7), 1,536 small-angle policy comparisons (maximum1.1921e-7), frozen
gait equivalence, and7,344 exact inactive-bypass values. All12 physical swimming
cases retained exact inactive commands and paired final states at every check.
No new braking/holding, current or depth-transition qualification ran.

All128 PPO updates used8,388,608 decisions and535,461,888 accepted/attempted
substeps, with8,192 Adam calls per optimizer. Resume verification used three
actual updates,196,608 decisions,12,582,912 substeps and192 Adam calls per
optimizer. Its complete actor/critic/Adam/world/history/RNG payloads matched
exactly. Initial/restored CPU/CUDA feature differences stayed below1e-6.

One pretraining baseline plus16 scheduled CPU assessments each used1,100,800
substeps, totaling18,713,600 substeps across1,496 trajectories. Full study spend
was **566,758,400 accepted/attempted substeps**, with zero native, numerical,
clearance or posture faults. Expected low-speed learning outcomes were retained,
not discarded as invalid physics. Low-course controls were5.69% of training.

GPU physics/observation took91.125s, inference0.397s, PPO/Adam2.712s and evidence
writing4.909s. CPU physics/scoring totaled10.829s and writing7.852s, overlapped
with training. The recorded workflow took106.920s including its prerequisite
checks and incremental build. Host waits overlap GPU execution. Main throughput
was84,241 actuator decisions/s. Full-state audits verified all129 checkpoints
and exports, initialization/mode metadata and protected live bundle hashes.

## Preserved evidence

- [Frozen protocol](data/protocol.md)
- [Signed input table](data/equivalence/data/course-input-table.csv)
- [Zero-physics checks](data/equivalence/data/report.json)
- [Learning curve](data/learning-curve.csv)
- [Movement/rhythm classification](data/analysis/groups.csv)
- [Final accounting](data/training/result.json)
- [Full-state audit](data/saved-data-audit/report.json)
- [CPU/CUDA and exact-resume proof](data/verification/data/verification.json)
- [Candidate72 fixed results](data/evaluations/update-072/assessment/result.json)
- [Final results](data/evaluations/update-128/assessment/result.json)

Originals remain at `build/flexible-tail/course-angle90-v3`. This archive keeps
all actor exports and score records, selected full/control traces, complete
source/configuration identities, verification and hashes of every original file.
New actors require their `whole-beat-90` metadata; do not load them with the
old observer or copy them over the protected live offset. No Godot publication,
chooser training, physics change, commit or push was performed by this study.
