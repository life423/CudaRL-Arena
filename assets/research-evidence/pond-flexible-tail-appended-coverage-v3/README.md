# Appended distant feedback — selected16 adopted

The chooser now keeps every existing local observation and receives three
additional distant visitation fields. Inputs0–31 retain their meaning and
bits per backend;32–34 summarize reachable water farther than24cm in three
course-relative120° sectors. The same338-cell sensing cache and unchanged
345-cell evaluation estimate at80mm depth remain. This supplies information,
not a prescribed destination, route or turn. No food is introduced.

The actor and training critic expand32→35 inputs with exact-zero added
first-layer weights. All inherited weights, biases, Adam moments and ages are
preserved. New columns have zero moments and a separate fresh Adam age saved
in version2 full checkpoints. An unmodified coverage80 parent checkpoint
compacts back byte for byte; old and expanded GPU actor/critic outputs match
exactly across1,024 saved worlds before any update. New distant fields match
CPU/CUDA exactly, and all32 local fields stay identical to their old observer
on each backend. The inherited bank-input9 CPU/CUDA difference2.235e−6 remains
recorded; physical equations, arithmetic modes and tolerances do not change.
Three actual PPO updates verify exact full-state resume, including new ages.

The separate native C++/CUDA session completes128 updates on1,024 worlds,
8 decisions/world/update,1,024-row minibatches and four passes:1,048,576 fresh
macro transitions. GPU-resident physics, inference, PPO and Adam are retained.
The frozen motor88, gait, physics, action choices and all reward scales are
unchanged. Safe transport stays0.04/s×(0.05+0.95×novelty), first visits+0.6,
revisits up to+0.15 after90–300s, stalls−0.10/s, proximity cost up to−0.10/s,
and guard failures−5. There are215 scored training guard stops and no posture
stops. They remain failures, not deployed recoveries.

Native training takes161.552s,161.813s externally. CUDA events report152.704s
physics,0.692s observation/inference and1.667s PPO. Evidence writing takes4.000s.
Transfer/wait overlaps GPU execution; its153.513s must not be added to physics.
The17 fixed assessments sum34.836s internally; four long assessments sum37.300s.
Independent CPU evaluations can overlap training, so these sums are not a
measurement of total end-to-end session wall time. Main throughput is about
6,491 macro transitions/s; these are0.8s chooser transitions, not50ms motor
controls. The session is closed at its cap, with no trainer remaining.

## Measured coverage and safety

Selected update16 passes all12 two-minute pond cases, all4 five-minute cases
and all4 uninterrupted30-minute physical tests, with zero guard stops, native
faults or resets. All17 scheduled fixed assessments and all four attempted
continuous assessments, including failures, are retained. The final128 passes
the short suite but stops at the clearance guard in two30-minute cases, so
its higher mean coverage does not qualify it for adoption. Update8 has lower
coverage; update120 fails three long cases. Only16 meets the declared adoption
rule. The latest checkpoint is not automatically the best checkpoint.

| Thirty-minute start | Previous coverage80 | Appended16 | Mean speed | Longest interval below6mm/s |
| --- | ---: | ---: | ---: | ---: |
| Godot rest |44.93%|43.77%|12.18mm/s|4.56s|
| Northeast |31.88%|42.61%|12.32mm/s|0s|
| West |35.94%|42.90%|12.36mm/s|0s|
| Southwest |38.84%|46.38%|13.15mm/s|8.14s|
| Mean |37.90%|43.91%|12.50mm/s| |
| Worst coverage |31.88%|42.61%| | |

The default Godot start slightly regresses; the improvement is consistency
across starts. Worst-case coverage rises10.72 percentage points, mean6.01.
The longest slow interval improves11.1→8.14s. Minimum certified whole-body
bed clearance is60.00mm; measured tilt and depth error are zero on this calm
fixed-depth runtime. No case reaches80% individual coverage. Accordingly the
raw continuous evaluator's `pass:false` means the80% milestone remains missed;
`physical_passes:4` and the separate preregistered improvement rule authorize
this adoption. Union coverage is never substituted for one swimmer's coverage.

Safe circling to avoid the bank is a plausible explanation for the old limited
coverage. It is not causally established by this comparison. Distant feedback
improves results without removing local risk information, but does not prove
complete coverage, indefinite survival, currents or multi-depth exploration.
The denominator is a fixed-posture geometric estimate, not all reachable3D
water; there are no outside-map cells in the selected long cases.

## Godot and protection

The actual registered `flexible-tail-preview` now loads separately named
`exploration-chooser-appended-16.bin`, model5/schema4, with unchanged motor88.
Old chooser80/88 and all motor files remain available. New35-input native code
zero-pads old32-input models and selects their original observation mode.
Unsupported32-input consumers reject model5 instead of misinterpreting it.

In an isolated copy of the actual project, old coverage80 and selected16 each
replay300s and3,750 sampled COM/request states exactly with the new DLL. Each
uses240,000 substeps and zero optimizer calls. All11 staged package hashes
verify. The rendered UI suite passes215 checks at390×844,768×1024,1100×760 and
844×390, with zero physics/optimizer work. Phone and desktop screenshots show
the updated43–46% qualification text; existing left-column controls, scroll
reachability and zoom behavior are preserved. The real project was reopened,
but final desktop inspection was interrupted by the user's physical Escape.
No further Windows input followed. This manual step is not claimed complete.

The first native DLL compilation failed on a constant-condition warning under
`/WX`, spending zero physics/PPO. Replacing serialization's equivalent `if`
with `if constexpr` builds successfully; this changes no data or math. The
executed trainer/evaluator source snapshot and hashes remain archived, alongside
the warning and successful compile logs. No study was restarted for this fix.

All736,233,121 attempted /736,232,872 accepted registered physical substeps
are counted, including resume diagnostics, main training, fixed checks, failed
long assessments and480,000 Godot parity substeps. The previous baseline was
reused without new simulation. See `data/work-ledger.json`, the exact source
bindings, migration receipts, learning curve, all case results, selected full
checkpoint and raw selected/long traces. Full per-update checkpoints remain
in local `build/flexible-tail/appended-coverage-v3`. Nothing is pushed or merged
by this local publication.
