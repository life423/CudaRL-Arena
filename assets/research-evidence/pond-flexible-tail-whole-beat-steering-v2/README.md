# Whole-beat steering PPO — motor stage closed

The user authorized fixing noisy steering feedback, a new GPU PPO session and
conditional pond navigation with Godot adoption only after passing. The motor
stage completed **128 updates / 8,388,608 decisions in 99.456 seconds** with
1,024 CUDA worlds. The original live motor, fitted gait, physics and habitat
remain protected. Sharp turns are still unqualified.

## Feedback correction

The opt-in observer uses actual root-rate crossings to measure a complete
tail-beat duration. A causal sliding COM window over that duration supplies
continuous course feedback, with interpolated measured history. Acquisition is
explicitly labeled, and low-speed/stale periods do not fabricate course from
the requested heading. The actor course input and associated reward terms use
the same estimate. Pre/post progress uses one common period so changing the
window cannot manufacture reward. Physical equations, gait, trim cap and slew
are unchanged. Historical half-second mode remains available.

On the same candidate-56 traces, eight gentle-turn cases supplied 800 late-turn
controls with fresh measured periods. RMS residual against the independent
two-second net COM course fell **3.898° → 0.220° (94.4%)**. This measures
feedback consistency against that reference, not perfect instantaneous course
accuracy. The synthetic periodic-motion check also passed, independently of
physical swimming qualification.

## Fixed results

| Controller | ±15° complete | ±30° complete | ±60° complete | Straight/from-rest |
|---|---:|---:|---:|---:|
| Original motor / half-second feedback | 8/8 | 0/8 | 0/8 | 12/12 |
| New feedback, before training | 4/8 | 0/8 | 0/8 | 12/12 |
| Candidate 40 | 6/8 | 4/8 | 0/8 | 12/12 |
| Candidate 56 | **8/8** | 0/8 | 0/8 | **12/12** |
| Candidate 128 | 1/8 | 0/8 | 0/8 | 12/12 |

The fixed physical, speed, course and root/passive rhythm gates were retained.
Only checkpoint 56 preserved all original gentle-turn pass keys; it is the
declared candidate for the limited early-bank stage. Checkpoint 40's sharper
results do not authorize replacing the preserved motor.

Candidate 56's mean late-turn course errors were 2.06° at 15°, 5.71° at 30°
and 25.95° at 60°. For 60° requests it achieved about **34.05°** by the late
13-second window while maintaining **9.73 mm/s** mean projected speed. This is
real underturning, not a hidden pass rejected only by rhythm accounting.
The tested calm planar states retained zero measured tilt/depth error and
positive clearance. Disturbed posture, currents, depth transitions and new
holding behavior were not qualified.

## Traceability

- [Frozen protocol](data/protocol.md)
- [Training curve](data/training/updates.csv) and [final accounting](data/training/result.json)
- [All checkpoint evaluations and noise analysis](data/evaluation-readonly/report.json)
- [Matched feedback residuals](data/evaluation-readonly/feedback-noise-vs-two-second-course.csv)
- [Candidate-56 gate](data/evaluations/update-056/assessment/result.json)
- [Resume/parity proof](data/verification/data/verification.json)
- [Independent saved-state audit](data/saved-data-audit/report.json)

The CPU diagnostic passed 458 synthetic assertions, 2,048 request cases,
frozen-gait equivalence and 4,896 exact inactive-bypass values with zero physics
or optimizer work. CPU/CUDA observations agreed within 4.7684e-7 against the
predeclared 1e-6 feature tolerance at initial and restored states.
Three actually executed resume-check updates used 196,608 decisions,
12,582,912 substeps and 192 Adam calls per optimizer, with byte-identical full
actor/critic/Adam/world/history/RNG checkpoints after restore.

Main training used 535,461,888 accepted/attempted substeps and 8,192 Adam calls
per optimizer. All 17 CPU baseline/scheduled assessments used 883,200 substeps
each: 15,014,400 total. Motor-stage total physical spend was **563,059,200
substeps**, with no numerical, native, clearance or posture failures.
All 129 complete checkpoints and actor exports passed the independent audit.
Only four steering coefficients trained; inactive weights and moments stayed
zero. The selected actor's metadata binds whole-beat observation semantics.

GPU physics/observation took 90.849 s, neural inference 0.395 s, PPO/Adam
2.758 s and training evidence writing 5.008 s. CPU checks ran concurrently;
host wait times overlap GPU work. The full main workflow and process receipts
are preserved separately from compilation.

Original files remain at `build/flexible-tail/whole-beat-steering-v2`.
This compact archive keeps all score/beat records, selected control traces,
all actor exports, selected full checkpoints, source/configuration snapshots,
verification and complete file hashes. A late checker-only addition was
preserved without execution; the exact tested source was restored before
closing the study. No training code or criteria changed during the run.

## Conditional pond stage

The approved follow-up uses frozen candidate 56 with the learned chooser and
the same native feedback implementation. It must pass all 12 navigation and
four endurance cases before Godot adoption. See the separate pond-stage
evidence and current AI status. No additional sharp-turn motor training runs
belong to this closed session; the user requested a subsequent brainstorm.
