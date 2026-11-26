# Turning early on moving bank approaches

Motor 88 can swim safely through all twelve tested bank orientations when given enough room. The new chooser learned some of those approaches and lost sustained exploration on the four long swims. Godot keeps the installed appended-16 chooser and motor 88. All eleven installed file hashes passed after the candidate failed its publication check. [Package verification](results/package-check.json)

## Which starts allowed recovery?

The frozen diagnostic reproduced the original +104 clearance stop at the same physical substep on all twelve starts. Whole-body speed at the first chooser decision was 10.30–10.95 mm/s. The 87 continuations per start included three autonomous policies, brief changes to speed or turn rate, and two-part commands. All 1,044 reached the clearance stop within 7.82 seconds. The result bounds the tested commands. Other sequences could still work.

The upstream comparisons preserve starting orientation, depth and motion while translating each start back along the incoming velocity. A qualifying recovery needs 180 seconds of native swimming with positive whole-body clearance, adequate transport and acceptable depth, tilt and stall. The distances below are extra room relative to each original start.

| Depth | Original starts with a demonstrated recovery | Starts recovered 60 mm earlier | Starts recovered 120 mm earlier |
|---|---|---|---|
| 72 mm | None | B | A, B, C, D |
| 75 mm | None | A, B | A, B, C, D |
| 80 mm | None | A, B, D | A, B, C, D |

The three-depth parent at update 128 completed all twelve of the 120-mm-earlier cases autonomously. Its successor +112 completed seven. The two previous +104 and +112 actors, the three-depth parent and motor 88 are protected by recorded hashes. [Recovery comparisons](../pond-flexible-tail-moving-start-recovery-v1/README.md) · [Protected files](protected-inputs.json)

## What PPO learned

One native CUDA continuation forked the full +112 chooser, including its critic and optimizer state. Only the training starts changed. The motor, physics, reward, 35 chooser inputs and 15 available requests remained fixed. The 1,024 worlds collected 1,048,576 decisions across 128 updates in 234.13 seconds on an RTX 5070. GPU physics used 225.35 s, inference 0.63 s and PPO kernels 1.56 s. The measured 3.74 s of evidence writing is included in wall time.

| Depth | Decisions used by PPO | Share |
|---|---:|---:|
| 72 mm | 348,160 | 33.20% |
| 75 mm | 350,208 | 33.40% |
| 80 mm | 350,208 | 33.40% |

A checkpoint audit reconciled four optimization passes, 4,194,304 minibatch presentations and no unexpected depth. At each depth the history-sensitive distant-water inputs responded to clearing the visitation history. Preflight measured 1,536 CPU/CUDA moving placements with 10.91–11.28 mm/s initial speed and both approach directions. Half the worlds practiced moving approaches, one quarter near-parallel swimming, and one quarter continuous interior exploration. Across ended moving training episodes, safe 48-second finishes were 2.87% at short range, 26.35% at medium range and 85.73% at long range. Healthy time limits and physical stops were counted separately. [Run and timings](results/training/result.json) · [Actual depth, respawn and history audit](results/depth-ledger.csv) · [Practice by runway](results/training-bands.csv) · [Training curve](results/training/updates.csv)

We ranked 16 candidate checkpoints on 18 independent moving approaches, six tangent swims and six 15-minute swims. The new update 104 (hash prefix `16B09627`) won with 6/18, 3/6 and 5/6. Its starting chooser +112 scored 4/18, 3/6 and 6/6. All six new moving successes came at approximately 125 and 150 mm of straight-ahead room. The 45, 60, 75 and 95 mm fixtures all failed. The twelve original starts were excluded from training and selection. This new checkpoint is distinct from the previous study's update 104 (hash prefix `932CCF8E`). [Selection rule](selection-protocol.md) · [Learning checks](results/selection/scores.csv) · [Selected resumable checkpoint](results/training/session-104.bin)

| Extra room at matched approach | Any tested motor-88 continuation | Previous +104 | Starting +112 | Three-depth parent | New chooser |
|---|---:|---:|---:|---:|---:|
| Original | 0/12 | 0/12 | 0/12 | 0/12 | 0/12 |
| 60 mm earlier | 6/12 | 3/12 | 5/12 | 4/12 | 6/12 |
| 120 mm earlier | 12/12 | 6/12 | 7/12 | 12/12 | 6/12 |

The new chooser requested the maximum available course rate, ±8°/s, on its first decision at 0.8 seconds on every original approach. On the 80-mm D approach with 60 mm more room, the previous and new +104 both initially requested 12 mm/s at +8°/s. At 1.6 seconds the new chooser requested 6 mm/s and continued turning. It swam 180 seconds with 4.15 mm minimum clearance. The previous +104 stopped at 76.68 seconds. Subsequent decisions also differed. At 120 mm earlier the new chooser passed A and C and lost B and D at each depth. The upstream tests reproduced all cue speeds, clearances and initial reference actions; cross-build raw world hashes differed, so byte-level equivalence for that supplementary comparison remains unverified. [First differing actions on the original starts](results/original-moving-decisions.csv) · [Upstream physical outcomes and comparison limits](upstream/README.md)

## Publication decision

All 24 dedicated shoreline swims finished, with zero resets. The numbers reaching 15 seconds in the near-bank band were 1/8, 1/8 and 0/8 at 72, 75 and 80 mm. Four per depth are required. Mean near-bank times were 9.34, 9.41 and 8.56 seconds against a 10.96-second floor. The longest uninterrupted 80-mm shoreline arc was 12.72° (11.81° required). The original ordinary navigation gate passed 5/12 with seven clearance stops, leaving its four conditional five-minute trials unrun. The additional original moving approaches passed 0/12. [Every bank swimmer](results/bank-swimmers.csv) · [All gate failures](results/publication.json)

| 30-minute start | Elapsed before clearance stop | Cells visited / required | Longest stall |
|---|---:|---:|---:|
| Godot | 1,158.69 s | 245 / 185 | 5.38 s |
| Northeast | 1,403.27 s | 247 / 195 | 5.46 s |
| West | 82.97 s | 33 / 218 | 0 s |
| Southwest | 70.18 s | 26 / 201 | 0 s |

Each swimmer met the 407-control stall limit. All four lost certified clearance without a reset. These are native clearance guard stops. Mean individual coverage was 39.93% of the 345 scored cells, compared with 57.90% for the three-depth parent and 43.91% for the installed chooser. Native publication failed; candidate Godot playback was skipped. [Per-swimmer comparisons](results/long-swimmers.csv)

The strongest starting point for another separately authorized experiment is the three-depth parent. It already completes all twelve 120-mm-earlier approaches and retains the required four long-swim coverage floors. Practice closer approaches with a demonstrated safe exit, and measure improvement alongside retention of those twelve early approaches and the long swims.
