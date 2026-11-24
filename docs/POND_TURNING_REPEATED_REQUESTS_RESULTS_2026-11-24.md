# Repeated turning practice

**The new schedule nearly doubled turn starts per minute. The candidate finished all 32 GPU PPO updates, preserved swimming and resting, and passed the same three of 16 independent turns.** None of the 1,508 training requests met the agreed one-second stabilization rule. Godot runs the original motor 336 and now accepts small player-requested turns.

## What the candidate practised

The native trainer on the RTX 5070 collected 64 controls from each of 512 independent worlds per update. One shared motor learned from 32,768 transitions per update, using four passes and minibatches of 4,096. Only the timing of heading requests changed. The original motor, signed-angle distribution, reward, rest and straight practice, actuators, water physics, 8 mm/s target and qualification thresholds carried over from the prior exercise. [Sealed training protocol](POND_TURNING_REPEATED_REQUESTS_PROTOCOL_2026-11-24.md)

Requests started after two simulated seconds of settling. Each lasted at least five seconds. Starting on tick 101, successful stabilization required all four conditions on 20 consecutive control ticks: horizontal whole-body course within 5°, speed toward the request at least 6 mm/s, depth error within 1 mm and tilt within 10°. The first possible success was tick 120. A request timed out at tick 400. The independent long-hold draw selected 167 of the 1,508 requests. Replacing a request preserved the moving body, joints, servos and filters.

| Practice, 32 updates each | Prior single request | Repeated requests |
| --- | ---: | ---: |
| Total request starts | 651 | 1,508 |
| Starts per minute of training-process wall time | 138.3 | 273.7 |
| Stabilized by tick 400 under the agreed rule | 0 | 0 |
| Stabilized completions per training minute | 0 | 0 |
| Actual timeouts | No timeout in that schedule | 861 |
| Interrupted at the 60-second episode cap | 0 | 0 |
| Interrupted by an earlier task ending | 458 | 459 |
| Still active at the end of training | 193 | 188 |

All 861 same-episode replacements followed timeouts. Of the 459 earlier task endings, 456 crossed the training depth limit and three crossed the tilt limit. The previous exercise's stabilization count was reconstructed from its saved trajectories using the same rule. The prior schedule continued holding its one request. [Prior request records](../assets/research-evidence/pond-turning-v4-repeated/v3-training-summary.json) · [Repeated-request records](../assets/research-evidence/pond-turning-v4-repeated/v4-training-summary.json)

**What prevented the easiest turns from counting?** Across 310 requests for ±5°, course missed tolerance on 40,339 of 68,772 eligible ticks (58.7%). Projected speed missed on 873 ticks (1.3%). In each direction the longest streak satisfying all four conditions was eight ticks, or 0.40 s, against the required one second. A useful next check is to plot the stored per-tick horizontal velocity against the displacement over a full tail beat. This would distinguish a side-to-side oscillation around the desired path from a sustained course error before choosing the next PPO lesson. [Individual training turns](../assets/research-evidence/pond-turning-v4-repeated/v4-training-requests.csv)

The final deterministic trials help isolate this effect. Looking at the last substep of each eligible control tick, the right 5° trial held all four conditions for at most four consecutive ticks (0.20 s). The left 5° trial reached seven (0.35 s). Both passed all 260 eligible samples for speed, depth and tilt. The course alone broke the streak, even with exploration noise switched off. Both paths passed the independent four-second mean-course test. [Fixed evaluation cases and trace hashes](../assets/research-evidence/pond-turning-v4-repeated/analysis.json)

## Independent movement and retention

| PPO update | Qualified turns | Mean course error | Mean speed toward target | Right 15° completion | Long windows | Short swims | Rests |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3/16 | 23.45° | 6.68 mm/s | 15.40 s | 28/28 | 4/4 | 14/14 |
| 8 | 3/16 | 23.63° | 6.64 mm/s | 15.35 s | 28/28 | 4/4 | 14/14 |
| 16 | 3/16 | 23.69° | 6.65 mm/s | 15.95 s | 28/28 | 4/4 | 14/14 |
| 24 | 3/16 | 23.42° | 6.67 mm/s | 15.95 s | 28/28 | 4/4 | 14/14 |
| 32 | 3/16 | 23.42° | 6.68 mm/s | 16.00 s | 28/28 | 4/4 | 14/14 |

The same initial pose passed right 5°, left 5° and right 15° throughout. Right 5° took 4.00 simulated seconds throughout. Left 5° took 4.15 seconds through update 24 and 4.10 seconds at update 32. In these fixed deterministic trials, course and projected speed use four seconds of whole-body center-of-mass motion. Course error in the table uses the last available full window for each case, including early endings. Completion timing uses the first qualifying window. Training uses stochastic actions and checks each control tick individually. The two measures answer different questions. [All 80 turn measurements](../assets/research-evidence/pond-turning-v4-repeated/turn-cases.csv)

At update 32 all 13 unsuccessful cases undershot. From the first pose, right 30° fell 10.50° short and left 30° fell 15.39° short. Across the 16 turns, 12 exceeded the depth limit, eight exceeded the tilt limit and four averaged less than 6 mm/s toward the request. These groups overlap. The 0.03° difference in mean course error since baseline added no qualifying turn. The prior single-request exercise ended at 23.31°. The latest straight-swim windows after startup averaged 7.87–8.32 mm/s at the 8 mm/s target. [Checkpoint and retention measurements](../assets/research-evidence/pond-turning-v4-repeated/checkpoint-summary.csv)

## Measured runtime

| Measurement | Single request | Repeated requests |
| --- | ---: | ---: |
| Transitions | 1,048,576 | 1,048,576 |
| Native training processes | 282.51 s | 330.59 s |
| Transitions per training second | 3,712 | 3,172 |
| CUDA world-step physics | 279.09 s | 323.16 s |
| CUDA optimization | 0.55 s | 0.62 s |
| Five native CPU assessments including trace output | 39.60 s | 40.55 s |
| Total active runner time | 324.16 s | 373.20 s |

One automatic invocation completed this exercise in **6 min 13 s**, including all five assessments. Each assessment took about eight seconds. Runner handoffs together took 0.12 seconds. Transfer and host reward reduction took 0.13 seconds. The checksum and atomic file writes for all 65 timed saves took 0.96 seconds. These measurements overlap the process totals. The current run executed 10,485,760 physical training substeps and 1,024 Adam calls each for actor and critic. A further 183,503 evaluation controls used the independent CPU implementation. GPU physics accounted for most of the elapsed time. The prior exercise's active time excludes the extra pause for its receipt-reader repair. [Timing breakdown](../assets/research-evidence/pond-turning-v4-repeated/timing-summary.json) · [Closed run and native-ledger accounting](../assets/research-evidence/pond-turning-v4-repeated/final-summary.json)

## What runs in Godot

The Windows Project Manager entry opens `live/project.godot`. The visible swimmer loads the preserved motor 336 with its holding gain. In Swim mode, the player can request left or right heading changes in 5° increments while the motor drives the joints. Independent native and Godot runs agreed across 6,080 combined controls, including four successful small-turn cases from two poses, an eight-second straight swim and a 64-second rest. The trained candidate is saved separately. Autonomous shore avoidance and whole-pond exploration remain unqualified. [Godot steering evidence](../live/research_data/live-steering-check.json) · [Saved research candidate](../assets/research-evidence/pond-turning-v4-repeated/candidate-32.bin)

The preview failure came from switching the extension descriptor before installing its matching library. The project now ships that verified library alongside the descriptor, motor and receipts. A quick Windows package check reproduces the failure when they differ. A clean Git export passed the package check and launched in Godot 4.4.1. [Package check](../tools/test-live-package.ps1)
