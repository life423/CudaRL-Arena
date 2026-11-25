# Individual exploration: coverage-v1

The deployed chooser was safe but mostly circled. Across four uninterrupted
30-minute starts it covered only 21.45–29.57% of the 345-cell fixed-posture
reachable-area estimate. Earlier 60.6% coverage was a union across starts.

One separate native CUDA PPO continuation reduced the steady safe-transport
reward in recently visited cells. New water retains 0.04/s; recently credited
water earns 0.002/s. The existing novelty, revisit aging, stalls, approaching
bank-risk and failure terms remain unchanged. Observations, actions, task mix,
PPO hyperparameters, native physics and frozen motor88 remain unchanged.

The session imported selected chooser88 exactly: actor, critic, Adam moments
and ages, worlds, histories and visitation memory round-tripped bit for bit.
New-study update/work/sampling counters were explicitly rebased. The three
actual-update full-resume check passed. Main training then completed 128 new
updates across 1,024 CUDA worlds, 1,048,576 macro transitions and 4,096 new
Adam calls per optimizer in 141.535 native / 141.744 external seconds. GPU
physics took 133.580s, network inference 0.270s, PPO 1.450s and evidence 3.835s.
Transfer/wait overlaps physics and is not additive. All 17 scheduled fixed
assessments and four post-training continuous assessments are preserved.

| Thirty-minute start | Previous chooser88 | Selected coverage80 |
| --- | ---: | ---: |
| Godot from rest | 23.48% | 44.93% |
| Northeast | 21.45% | 31.88% |
| West | 29.57% | 35.94% |
| Southwest | 22.90% | 38.84% |
| Mean individual coverage | 24.35% | 37.90% |

Both policies complete all four long cases with zero native faults, guard
stops or resets. Selected80 also passes the unchanged 12×120s and 4×300s
qualification. Mean long-run speeds are 12.07–12.27mm/s, with zero measured
depth/tilt error and minimum certified gap 20.95mm. The southwest case spends
11.10s below the 6mm/s rolling movement threshold (minimum about 3.06mm/s),
but stays clear of terrain and resumes normal swimming. This is a slowdown,
not grounding. It is a retained limitation, not a quiet hold or a learned stop.

Adoption required at least five percentage points improvement in both worst
and mean individual coverage, original 12+4 qualification and 4/4 long
physical passes. The rule was recorded before post-training long assessments.
Selected80 meets it. Candidate48 has a higher mean (43.19%) but only 23.48%
worst-case coverage and is excluded. Final128 scores 2/12 and is excluded.

**No candidate reaches the 80% per-run milestone, let alone the whole pond.**
Maps remain a fixed-posture estimate at the existing 80mm depth; no run visits
outside-reference cells. Full pond exploration, multiple depths, currents and
learned braking/holding remain open. The next information question is whether
the chooser's six local novelty probes adequately distinguish distant unvisited
regions after nearby water is familiar. Do not silently repeat this closed run.

Godot now loads the separately named `exploration-chooser-coverage-80.bin`
through its deployment manifest. Previous chooser88 and every motor file remain
intact. The selected five-minute replay matches all 3,750 sampled COM/request
states exactly, using the unchanged native DLL. All 215 rendered UI checks pass.
Details discloses 32–45% individual coverage and that full coverage is unfinished.
The viewer performs inference and physical simulation, not background training.

Total counted work through parity: 744,894,147 attempted substeps, including
baseline, resume diagnostic, main collection, every fixed/continuous assessment
and 240,000 Godot parity substeps. See data/work-ledger.json. No protected
checkpoint was overwritten. Every main/full checkpoint remains in the local
`build/flexible-tail/coverage-v1` root. The baseline executable path was rebuilt
before the training phase; its original binary hash was not retained. Its
command, complete traces and unchanged evaluator/physical criteria are retained.
Training has exact executable/source/input bindings and a frozen source snapshot.

The archived runner snapshot is the executed version. Its baseline presence
guard was corrected afterward (`-ne 4` replaces a negation-precedence mistake).
The actual baseline was independently inspected as 4/4 before training, and no
physics or optimizer work was rerun for that shell guard correction.
