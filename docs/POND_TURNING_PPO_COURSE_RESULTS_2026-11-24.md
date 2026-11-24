# Turning with a course-sensitive reward

The RTX 5070 ran the existing native C++/CUDA trainer. One motor collected experience in 512 independent worlds. Each update used 64 controls per world and four optimizer passes with minibatches of 4,096. Its separate candidate started from the same saved motor 336, seed and request distribution as exercise 2. The registered Godot project plays motor 336.

After a turn request, the candidate earned more forward-progress reward when its horizontal center-of-mass velocity pointed toward the requested heading. At the 8 mm/s target, a 15° error earned 66% of the aligned movement reward, compared with 98% previously. Rest, straight swimming, pre-turn warmup and the deterministic request sampler kept their previous behavior. The physics, actuators and qualification thresholds also stayed fixed. [Sealed protocol](POND_TURNING_PPO_COURSE_PROTOCOL_2026-11-23.md)

## Learning result

All **32 updates** completed. The first and last checkpoints passed the same three turns, right 5°, left 5° and right 15°, from the first starting pose. Each of the other 13 missed by turning too little. Mean absolute course error improved by 0.14° from the inherited baseline. The final score was 0.15° better than exercise 2 at update 32. Neither exercise qualified a new turn or established autonomous pond navigation.

| Update | Turns | Mean course error | Mean speed toward request | Right 15° completion | Long windows | Short swims | Rests |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3/16 | 23.45° | 6.68 mm/s | 15.40 s | 28/28 | 4/4 | 14/14 |
| 8 | 3/16 | 23.63° | 6.66 mm/s | 15.35 s | 28/28 | 4/4 | 14/14 |
| 16 | 3/16 | 23.49° | 6.67 mm/s | 15.95 s | 28/28 | 4/4 | 14/14 |
| 24 | 3/16 | 23.50° | 6.68 mm/s | 16.00 s | 28/28 | 4/4 | 14/14 |
| 32 | 3/16 | 23.31° | 6.66 mm/s | 15.95 s | 28/28 | 4/4 | 14/14 |

Course and speed use horizontal whole-body center-of-mass velocity over the last available full four seconds for all 16 turns, including two early terminations. Completion time runs from the request change to the end of the first qualifying four-second window. Right 5° took 4.00 simulated seconds and left 5° took 4.15 seconds at every assessment. All 13 misses lacked a qualifying window. The published tables also preserve the registered final-interval measures and each termination. [Case measurements](../assets/research-evidence/pond-turning-v3-course/turn-cases.csv)

At update 32, the right 30° request from pose 0 fell 10.37° short. The left 30° request fell 15.36° short. Across all 16 trials, 12 exceeded the 1 mm depth requirement, eight exceeded 10° tilt and four averaged less than 6 mm/s toward the goal. These counts overlap. Straight swims averaged 7.83 to 8.28 mm/s in their post-startup windows at the 8 mm/s target. Peak lane error across the retained tests was 0.92 mm. The two five-minute rests passed. All 250 protected manifest entries, the three named original policy hashes and the research checkpoint from exercise 2 verified after training. [Checkpoint skill summary](../assets/research-evidence/pond-turning-v3-course/checkpoint-summary.csv)

## Measured time

| Equal training allocation | Exercise 2 | Course-sensitive exercise |
| --- | ---: | ---: |
| PPO updates / transitions | 32 / 1,048,576 | 32 / 1,048,576 |
| Training processes | 292.20 s | 282.51 s |
| Transitions per training second | 3,589 | 3,712 |
| First / last eight updates | 3.16 / 112.84 s | 3.35 / 110.23 s |
| GPU world-step kernel | 288.82 s | 279.09 s |
| GPU optimization interval | 0.50 s | 0.55 s |
| Host rollout transfer and reward reduction | 0.10 s | 0.10 s |
| Five CPU assessments, including trace output | 40.05 s | 39.60 s |
| Checksum and atomic rollout/checkpoint writes during training | Separate timer unavailable | 0.83 s |
| Elapsed span from initialization sequence through final check | ~626.9 s | 599.9 s |

Training throughput increased by 3.4% across these two exercises. The changed reward allowed trajectories to diverge, so the isolated layout replay below provides the matched test. Five-second training would require **209,715 transitions/s**, about 56.5 times the measured rate. The new full training run completed 10,485,760 physical substeps and 1,024 actor plus 1,024 critic Adam calls.

The two local runner invocations used 324.16 s altogether. Fixing the receipt reader between them accounted for an additional 275.70 s of elapsed time. After recovery, all remaining training chunks and checks ran automatically with 0.08 s of combined handoff overhead. The first eight updates plus their complete assessment took 11.24 s. Kernel, file-write and transfer intervals occur inside the process times. CPU evaluation includes producing its evidence. [Timing breakdown](../assets/research-evidence/pond-turning-v3-course/timing-summary.json)

## Why the late updates took longer

The comparison replayed the first recorded control from batches 1 and 32 of exercise 2 on an RTX 5070. Each launch started from the same 512 saved worlds and used the corresponding 512 saved commands. Every world advanced ten physical substeps. Resulting body, servo and filter states and all meaningful output values agreed bit for bit across layouts.

| Threads per block | Blocks | Early GPU physics | Late GPU physics |
| ---: | ---: | ---: | ---: |
| 32 | 16 | 11.59 ms | 205.38 ms |
| 8 | 64 | 2.21 ms | 182.16 ms |
| 1 | 512 | 12.47 ms | 227.54 ms |

Eight threads gave the shortest late calculation, an 11.3% reduction. That layout was selected for the new exercise. Each layout was timed once on each snapshot, in the order shown. The early result is particularly sensitive to first-launch effects. The entire diagnostic command took **0.87 s**, including transfers, comparisons, evidence writes and process startup. [Frozen replay and timings](../assets/research-evidence/pond-quick-bench-v1/result.json)

On the 512 late starting states, a separate probe timed one full clearance query at **16.28 ms** and the first substep's water-force and solver calculation at **0.54 ms**. Clearance certification required 63,056 terrain samples on the late states, up from 2,048 on the early states. The corresponding solver counts were 513 and 512. All 1,024 CPU/CUDA clearance checks agreed within tolerance. A control interval runs 11 clearance queries. At the measured late starting-state cost those would take about 179 ms, close to the 182 ms observed for the full control. Bodies move during a control, so that estimate is an attribution check. Hardware performance counters were unavailable. The isolated timings identify the expensive operation without modifying the simulator. [Component timings](../assets/research-evidence/pond-quick-bench-v1/component-result.json)

## How the run continued

The native baseline passed all 20 retention cases and retained all 28 required long-swim windows. On the first invocation, the PowerShell runner rejected the final eight bytes of the genuine receipt. Those bytes hold the native payload's second checksum. The runner stopped with zero training updates. Its synthetic fixtures had omitted the same checksum.

The corrected reader verifies both checksums. It verified the saved baseline, candidate, unchanged protocol and native work ledgers before continuing the same allocation. The original invocation and receipt are preserved. The baseline ran once. After recovery, one local command ran every eligible eight-update chunk, saved its checkpoints and started its full skill check. [Stage runner](../tools/run-pond-turning.md) · [Recovery record](../assets/research-evidence/pond-turning-v3-course/baseline-recovery.json)

## Godot and checkpoint protection

The Project Manager entry points to the worktree's `live/` project. Its ordinary main scene loaded the corrected shoreline runtime and the original saved motor. Details shows **Course learning · v3**, 32 completed updates, retained swimming and resting, and 3/16 turns. The registered-entry verification passed all 10 checks, the source-only viewer passed nine and the desktop/phone layouts passed 128 checks. Those viewer checks ran with the simulation paused. The behavioral results above come from the independent native skill assessments. [Published study receipt](../live/research_data/candidate-study.json)

## Next learning experiment

There were 651 new turn requests among 1,048,576 training transitions. The first two simulated seconds after each request contributed 25,873 transitions, or **2.47% of all experience**. Of the 444,947 transitions after a turn request, 94.2% occurred more than two seconds after its onset. Remaining on course and initiating a turn are both skills, but this distribution gives the learner relatively few new starts. [Recorded request counts](../assets/research-evidence/pond-turning-v3-course/request-exposure.json)

Test one change to the request schedule. Give already-swimming worlds additional left and right heading requests while carrying their body, momentum, joint targets and filters forward. Keep a mix of longer holds so the candidate has time to complete turns, and retain straight swimming and rest. Compare the first eight updates with this exercise at update eight using the same held-out evaluation cases. Measure turn starts per million controls, actual signed course change, speed, depth and time to a qualifying window. This would test whether more turning practice per GPU second improves learning. The cleared candidate remains research-only.
