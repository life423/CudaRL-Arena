# Chooser PPO for unvisited water · November 25, 2026

Update 8 reached more distinct water from all four starts. It lost whole-body clearance on the west approach after 23 min 58 s, and two other swimmers exceeded the stall limit. None of the 19 trained checkpoints tested qualified for Godot. The installed pair is appended-16 with motor 88.

| Chooser | Ordinary passes | Held-out shore passes | Completed 30 min | Mean coverage across four starts | Long swims exceeding 8.14 s stall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Installed, update 0 of this fork | 12/12 | 6/8 | 4/4 | 43.91% | 0 |
| Update 8 | 12/12 | 6/8 | 3/4 | 50.00% | 2 |
| Update 32 | 5/12 | 4/8 | 1/4 | 34.93% | 0 |
| Update 128 | 1/12 | 4/8 | 2/4 | 27.25% | 0 |

Coverage uses each swimmer’s physically credited 50 mm cells divided by 345 reference cells. The mean includes every start and counts what the swimmer had earned when a guard stopped it. The denominator estimates navigable water at a fixed reference posture. All 12 new long evaluations ran without a reset. The guard stops recorded a loss of certified positive terrain clearance. The existing full 30-minute installed baseline was reused. [Per-swimmer results](data/individual-swims.csv) · [All checkpoint tests](data/checkpoints.csv)

| Start | Installed cells | Update 8 cells | Update 8 finish | Longest stall |
| --- | ---: | ---: | --- | ---: |
| Godot | 151 | 172 | 30 min | 41.42 s |
| Northeast | 147 | 163 | 30 min | 3.08 s |
| West | 148 | 162 | Clearance stop at 1,438.28 s | 6.68 s |
| Southwest | 160 | 193 | 30 min | 14.42 s |

At the common 15-minute mark, all four update-8 swimmers had already visited more distinct cells. On the west swim, the chooser kept requesting 12 mm/s and −8°/s for its final 43.08 s. At its last decision it still had 15.15 mm of whole-body clearance and was moving at 22.62 mm/s. It reached the clearance guard 0.68 s later. This identifies the decision sequence for a matched-state follow-up. [Coverage over time](data/coverage-over-time.csv) · [Final west approach](data/west-final-approach.csv)

The two north-bank approaches stayed in evaluation. At update 8 they ended at 4.33 s and 3.89 s. From update 96 through 128 the first survived the full 180 s. At update 128 the second stopped at 3.76 s. No trained checkpoint passed all eight shoreline starts. [Shore results](data/checkpoints.csv)

## What ran

The native RTX 5070 run forked the installed 35-input chooser and its full training state. Only the chooser and its training critic learned. Distinct 150 mm regions earned an extra 0.30 per offset tiling after eight visited cells. The main run completed 128 updates across 1,024 CUDA worlds, 1,048,576 transitions and 4,096 actor and critic optimizer steps apiece. The region bonus paid on 36,729 transitions for 56,930 new regions summed across tilings and episodes. The training collector recorded 439 clearance stops. A separate three-update verification used 24,576 transitions to check resumption. The initial chooser matched the installed weights byte for byte. The independent region-count check passed all 15,360 comparisons. [Protocol](protocol.md) · [Training result](data/training/result.json) · [Reward counts by update](data/training/regions.csv)

Training took 157.55 s measured outside the process, including its checkpoint writes. CUDA events attributed 149.05 s to physics, 0.65 s to inference and 1.45 s to PPO. Training and evaluation overlapped. From the start of verification through the completed runner receipt was approximately 191 s, excluding compilation. All 11 Godot bundle hashes verified after the run. All 129 full training checkpoints and raw trajectories remain under `build/flexible-tail/regional-exploration-v1`; the scored portable actors and per-case results are also archived here. [Run summary](data/result.json) · [Checkpoint hashes](data/full-checkpoint-hashes.csv)

## Finding for the next run

The actual 35-input observer returned three zeros for distant unvisited water at the Godot, west and southwest starts. Their request depths were 72, 75 and 75 mm. At the 80 mm northeast start all three sectors initially read 1. The current geometric map has one depth slice, and all 256 archived training starts used 80 mm. This blind spot is verified independently of PPO. Its contribution to the failed approaches remains unmeasured. The next controlled change should generate geometry-checked distant inputs at the three deployed depths, include distinct practice starts at those depths, and retain the same held-out approaches. [Measured inputs](data/reward-check/depth.csv) · [Training depths](data/reward-check/train-depth.txt)
