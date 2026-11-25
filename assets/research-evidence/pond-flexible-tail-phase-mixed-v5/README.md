# Mixed-angle PPO — 30-degree turns pass, combined retention not qualified

The authorized continuation completed128 NEW native CUDA PPO updates with1,024
worlds:8,388,608 decisions in104.741seconds (104.838seconds external process).
Checkpoint120 passes all eight fixed30-degree cases, left4/4 and right4/4.
No checkpoint meets both8/8 gentle and8/8 at30. The working Godot motor stays
protected; autonomous pond exploration is not yet qualified.

## Change and continuity

The only learning change is balanced15/30-degree practice in place of30-only.
Rewards, features, nine trainable coefficients, fitted gait, physics, actuators,
trim cap/slew, PPO settings and evaluation criteria stay unchanged. This is
the same trainer/runtime and shared phase-aware policy module, not another
behavioral network or copied training path.

The explicit full-state fork starts from phase-v4 update112. Actor, critic,
both Adam states, all1024 native body/servo/course/beat-history states and RNG
positions match the parent bit-for-bit before collection. Existing live30-degree
episodes finish normally. Only study diagnostic counters restart. FTSSv5 binds
the parent payload/task, lesson and imported ages; FTYO2 policy format is unchanged.
Exact fork/resume checks pass. All originals and old binaries remain preserved.

## Results

| Checkpoint | Complete15 | Complete30 | Complete60 | Complete90 | Swimming/from-rest |
|---|---:|---:|---:|---:|---:|
| Parent112 |3/8|6/8|0/8|0/8|12/12|
| New56: best retaining all gentle keys |8/8|5/8|0/8|0/8|12/12|
| New88: best joint count |7/8|7/8|0/8|0/8|12/12|
| New120: all30-degree cases |4/8|8/8|0/8|0/8|12/12|
| New128: final |6/8|5/8|0/8|0/8|12/12|

All15/30-degree motion/posture cases pass throughout this continuation. Every
remaining full failure at those angles is the passive-tail frequency criterion;
no threshold is waived. At120, mean30-degree late/onward COM course error is
0.552/0.450degrees, projected speed12.852/13.278mm/s, initial response1.679s and
effective turning radius251.93mm. These are measured test means, not guaranteed
maximum radius or avoidance distance near a bank.

The two misses at88 are phase1-left15 and phase2-left30, both passive frequency;
the latter ratio is0.797518 against the fixed0.80 threshold. Keeping fitted
weights frozen does not guarantee identical passive motion under learned trim.
Mixed practice improved the joint pass count, but learning remains variable.
This continuation is not a paired causal isolation against30-only from the same
parent; the previous phase study started from a different policy state.

All12 swimming cases (including8 from rest) retain exact inactive commands and
paired final native states. Calm submerged fixtures measure zero tilt/depth error;
this does not establish disturbed recovery, new depths, station holding or currents.
No new repeated-turn/bank/endurance trajectory ran.60/90 degrees were evaluated
but not trained and remain unqualified. No chooser training or Godot adoption ran.

## Actual accounting and checks

New training:535,461,888 accepted/attempted substeps,8,192 new Adam calls per
optimizer. Global optimizer ages7168→15360; sampling progresses from parent112
to240. There are44,032 fresh episode starts:11,008 each at left15,right15,
left30,right30, independently derived from saved per-lane episode counters.
Imported in-progress episodes are not counted as new starts.

Fork/resume diagnostic:3 executed updates,196,608 samples,12,582,912 substeps,
192 Adam calls per optimizer; full resumed payloads identical. CPU/CUDA feature
maximum difference4.77e-7 under the1e-6 limit. Fork import itself uses zero steps.
Baseline plus16 evaluations:1,496 trajectories,18,713,600 substeps. Actual whole
study spend566,758,400 substeps, below568,167,424 maximum; zero native/numerical,
clearance or posture faults. All129 checkpoint/export pairs pass independent
audit: onlyweights4–12 changed, frozen/unused slots and optimizer moments exact,
and all protected input/physics/live-library hashes remain unchanged.

GPU physics/observation96.016s; inference0.444s; PPO/Adam2.789s; training evidence
5.044s. CPU assessments total11.040s physics/scoring and10.615s evidence writing,
overlapped with training. Host waits overlap GPU execution. Throughput80,089
actuator decisions/s. Configure/build8.391/11.278s, baseline1.338s, fork/resume
verification2.926s externally; complete runner130.725s. Analysis/audit/archive
are separate zero-physics work, outside runner timing.

Next: interpret the remaining rhythm failures using the saved traces, then
target retention consistency before moving the lesson to sharper angles.
No further experiment or budget extension started automatically.

## Evidence

- [Frozen protocol](data/protocol.md)
- [Learning curve and detailed interpretation](data/analysis/README.md)
- [Exact practice exposure](data/analysis/exposure.json)
- [Full-state fork/resume](data/verification/data/verification.json)
- [Zero-step import equivalence](data/training/fork-equivalence.json)
- [Independent saved-data audit](data/saved-data-audit/report.json)
- [Training accounting](data/training/result.json)
- [All30-degree cases passing](data/evaluations/update-120/assessment/result.json)

All originals remain at `build/flexible-tail/phase-mixed-v5`. The archive keeps
all actor exports, assessment scores/beats, selected full states/control traces,
source/input identities, receipts and hashes of every original file. No commit,
push, physics modification or live model replacement occurred.
