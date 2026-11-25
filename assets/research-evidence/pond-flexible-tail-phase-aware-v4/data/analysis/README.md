# Phase-aware V4 measured result

Read-only aggregation of the completed session. This analysis ran zero physical controls and zero optimizer updates. `analyze.ps1` derives its CSV/JSON outputs from existing checkpoint evaluations and recorded decision telemetry.

| Checkpoint | All fixed turns /32 | 15 degrees /8 | 30 degrees /8 | 60 /8 | 90 /8 | Straight /12 |
|---|---:|---:|---:|---:|---:|---:|
| Protected baseline / initialized candidate | 8 | 8 | 0 | 0 | 0 | 12 |
| 8, latest retaining every gentle pass | 8 | 8 | 0 | 0 | 0 | 12 |
| 16, highest total full pass count | 12 | 7 | 5 | 0 | 0 | 12 |
| 112, highest full 30-degree pass count | 9 | 3 | 6 | 0 | 0 | 12 |
| 128, final | 9 | 4 | 5 | 0 | 0 | 12 |

No checkpoint qualifies: the declared lesson gate requires all eight 30-degree passes and all eight 15-degree passes, plus all twelve straight cases and zero global faults. 60/90-degree cases were untrained generalization tests; none passed. This experiment establishes no autonomous bank avoidance, repeated-turn competence, current holding, or indefinite pond navigation.

The motion-only 30-degree count reached 8/8 at update 64 and remained 8/8 through 128. That excludes the unchanged stroke-retention gate and must not be substituted for the full score. At update 112, full 30-degree passes split 3/4 left and 3/4 right. At final 128 they split 2/4 left and 3/4 right. Every final 15/30-degree failure was solely a passive-tail beat-frequency ratio below 0.8 of its same-state frozen baseline in at least one scoring window. Root frequency and swing, passive swing, movement, tilt, depth and clearance met their corresponding criteria.

For final 30-degree cases, the mean late-turn course error was 1.316 degrees and onward error 0.675 degrees; mean projected speed was 13.189 and 13.496 mm/s respectively. Left/right late errors were 1.235/1.397 degrees and onward errors 0.302/1.047 degrees. Mean initial response was 1.766 simulated seconds; the evaluator's complete-beat effective radius averaged 327.94 mm. The response measure is initial detected course response, not completion time. `target_1s_begin_s` was unavailable, so no settled-time claim is inferred. Peak recorded tilt and depth error were zero; minimum certified wet clearance was 67.999 mm. All twelve straight cases, including eight starts from rest, passed at every scheduled evaluation and had exact paired final states. Station holding was not physically tested in this suite; inactive Hold command equivalence was checked algebraically.

Initial equivalence is unusually strong here: 3,072 held request/state combinations through both signs of 90 degrees had bit-identical actor means and 27,648 decoded target/applied-offset comparisons. The new phase/speed/full-angle inputs were initialized with zero weights; the original four coefficients remained frozen. The added feature reflection discrepancy was zero, full-angle feature error was 5.96e-8, and the old loader rejected the new FTYO2 format. Thus the initial larger-angle behavior was not changed by an automatic gain rescaling.

The learned controller did use phase information. In final 30-degree late windows, mean per-case RMS frozen-base contribution was 0.13390 latent units and learned contribution 0.73750. The exact direct phase-position/rate terms alone contributed RMS 0.23286; the remaining learned aggregate contributed 0.50779. The latter contains phase interactions as well as course/speed/lateral terms. Final direct phase coefficients were -0.30267 and +0.08678; the error-phase harmonic coefficients were +0.47832 and -1.03649. These numbers demonstrate phase-dependent modulation, but one combined training arm cannot isolate its causal benefit from learned course-gain and lateral feedback. `phase-contribution.csv` records the actual base, learned, direct-phase and remaining components by case/window; components can cancel, so their RMS values do not add.

The finite run completed 128 PPO updates, 8,388,608 sampled actions, and 535,461,888 accepted physical substeps. Trainer wall time was 99.663 seconds; GPU physics/observations 91.000 seconds, network 0.407 seconds, and optimization 2.756 seconds. Sync/readback wall timing overlaps GPU work and is not an additional summable stage. Evidence writing was 5.035 seconds. Evaluations averaged about 1.3 seconds each.

Recommendation: retain the protected working motor in the app. For a separately authorized continuation, preserve this same versioned observer/runtime and fixed gates, restore mixed gentle and 30-degree practice, and teach retention of the measured passive rhythm rather than expanding to sharper targets. Do not label update112 qualified or weaken its passive frequency gate. The useful result is clear 30-degree physical movement with swimming preserved; reliable full-gate turning remains unfinished.

Files: `learning-curve.csv`, `case-failures.csv`, `turn-metrics.csv`, `phase-contribution.csv`, `selected-weights.csv`, and `turn-analysis.json` retain all scheduled checkpoint results and failure reasons.
