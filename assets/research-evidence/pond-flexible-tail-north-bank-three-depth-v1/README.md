# North-bank chooser continuation with frozen motor 88

**The six previously failing north-bank swims passed. The candidate failed on the ordinary Godot start, so Godot keeps its installed appended-16 chooser.** All 24 held-out shoreline swims completed 180 seconds with positive whole-body clearance, zero resets and zero stalled controls. The parent completed 18 of the 24. The candidate completed three of the four independent 30-minute swims.

## What PPO learned at the bank

At 80 mm, the parent requested 12 mm/s and the strongest available turn, −8°/s or +8°/s, at every decision before its two early stops. With the same body, the selected chooser requested 6 mm/s. On the phase-8 start it requested −4°/s, then straightened at 1.6 s. On the phase-9 start it requested +4°/s through the former failure time. These are the actors' decisions from independent native continuations of identical fixtures. Speed requests are targets, not measured body speeds.

| Depth | Parent finished | Candidate finished | Min clearance, phase 8 north − | Min clearance, phase 9 north + | Candidate cases with ≥15 s near bank | Mean seconds near bank |
|---|---:|---:|---:|---:|---:|---:|
| 72 mm | 6/8 | 8/8 | 9.563 mm | 0.283 mm | 1/8 | 10.81 |
| 75 mm | 6/8 | 8/8 | 9.577 mm | 0.303 mm | 2/8 | 11.22 |
| 80 mm | 6/8 | 8/8 | 9.617 mm | 0.377 mm | 1/8 | 11.23 |

Four swimmers at each depth must spend at least 15 seconds near the bed, and the per-depth mean must reach 10.96 seconds. The longest uninterrupted angular span at 80 mm was 12.72°, above the 11.81° reference. The minimum gap is the native whole-body clearance bound. [All individual results for the installed chooser, parent and candidate](bank-comparison.md)

## Ordinary swimming and exploration

| 30-minute start | Installed cells | Parent cells | Candidate cells | Candidate time | Mean moving speed | Longest native stall |
|---|---:|---:|---:|---:|---:|---:|
| Godot, 72 mm | 151 | 185 | 10 | 42.474 s, clearance stop | 13.73 mm/s | 0 s |
| Northeast, 80 mm | 147 | 195 | 234 | 1,800 s | 12.74 mm/s | 0 s |
| West, 75 mm | 148 | 218 | 186 | 1,800 s | 12.85 mm/s | 0 s |
| Southwest, 75 mm | 160 | 201 | 257 | 1,800 s | 12.98 mm/s | 0 s |

Mean individual coverage, including the stopped swimmer, was 49.78% of 345 scored cells, down from the parent's 57.90%. The installed reference scored 43.91%. The candidate failed the individual parent floors at Godot and West. It passed 11/12 ordinary starts. The original evaluator withheld its four five-minute tests after the ordinary Godot clearance stop. All four independent 30-minute tests ran; the ordinary and long Godot trials reproduced the same stop without a reset.

During the Godot swim, the chooser repeatedly requested 6 mm/s and +4°/s from at least 31.2 to 42.4 s. At 40.0 s the native clearance bound was 27.14 mm. It fell to 9.50 mm at 41.6 s and 0.73 mm at 42.4 s. Actual horizontal COM speed at the last decision was 10.45 mm/s. The next rejected substep reported a −0.0071 mm bound. This was a clearance guard, with zero stalled controls. [Individual long swims](long-swimmers.csv) · [Publication decision](publication.json)

## Actual training and selection

One separate CUDA PPO session continued the full parent checkpoint at update 128. The parent round-tripped byte for byte before the new study counters and starting placements were set. The chooser weights, critic and Adam state carried over. Half of 1,024 worlds practised offset north-bank approaches at randomized angles, the rest continued ordinary exploration practice. Each depth used separate training poses excluding rows 8 and 9. The held-out bank starts at all three depths reproduced the installed and parent start tables byte for byte. Motor 88 and the fitted gait were frozen.

| Depth | Decisions that entered PPO | Share | Optimizer sample presentations | Extra spawns | Decisions within 20 mm, all cohorts |
|---|---:|---:|---:|---:|---:|
| 72 mm | 348,160 | 33.203% | 1,392,640 | 8,880 | 74,617 |
| 75 mm | 350,208 | 33.398% | 1,400,832 | 8,780 | 74,489 |
| 80 mm | 350,208 | 33.398% | 1,400,832 | 8,847 | 73,628 |

The 128 new updates took 229.37 s process wall time. All 1,048,576 collected decisions entered PPO, with four presentations each. The evaluator independently reconstructed actual minibatch rows from the 129 checkpoints and reconciled them with 4,096 actor and critic steps each. No unexpected depth was detected. Clearing visitation history changed distant-water inputs on 174/340, 177/342 and 171/342 final live worlds respectively. Changing visitation history need not change the best distant direction on every pose. The first training placements were explicitly initialized at the three assigned depths and the counts include subsequent respawns. [Depth ledger](depth-ledger.csv) · [Native training counters](training/result.json)

A separate exact-fork check executed three diagnostic updates in 6.23 s. Training used 220.92 s of GPU physics, 0.68 s of inference and 1.47 s of GPU PPO. Evidence writing took 3.68 s. Host synchronization overlapped device execution. The 17 validation assessments used 43.50 s of cumulative native process time and overlapped training. The held-out suites used a further 13.03 s. Training recorded 11,036 clearance stops and 15,494 valid time-limit truncations. These outcomes were retained in the training records.

The parent passed 6/8 bank and 3/4 interior validation cases. Checkpoint +112 passed 8/8 and 4/4 with two extended near-bank cases. It was selected using the registered rule before any exact candidate bank or navigation fixtures ran. A PowerShell sort type error occurred after all 17 validation batches finished. Ranking resumed from their saved receipts, verifying actor hashes, native executable and fixtures. No PPO or validation batch was repeated. All exact tests used the single selected actor. [Full validation curve](selection/scores.csv) · [Selection receipt](selection/selected.json) · [Protocol](protocol.md)

The selected actor and resumable full session are saved under `training/`. Local full checkpoints for updates 0–128 remain under `build/flexible-tail/north-bank-three-depth-v1/training/checkpoints/` with their individual hashes archived here. The package check verified all 11 installed files, including motor 88 and the installed chooser. Godot staging and live adoption were skipped after native qualification failed. [Checkpoint hashes](training/checkpoint-hashes.csv) · [Selected actor](training/actor-112.bin) · [Full session](training/session-112.bin)

The remaining failure is an approach from open water. A next comparison could preserve the learned tangent exits and practise carrying full swimming speed into differently angled approaches. Keep the exact Godot start and north-bank fixtures held out so gains on either can be checked against the other.
