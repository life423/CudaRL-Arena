# Open-water chooser continuation

**Godot keeps its installed appended-16 chooser and motor 88.** Native playback reproduced the previous +112 stop and demonstrated that one earlier turn decision can change the outcome for the next 30 minutes. The new chooser made the ordinary Godot swim last 14 min 11 s and completed all 24 dedicated shoreline swims. It lost clearance in three of the four long swims.

## What made the difference

The native replay reproduced all 53 archived +112 decisions and the rejected substep at 42.47375 s. We forked every 0.8-second decision from 24.0 through 42.4 s. Each state retained body momentum, actuator and motor history, the chooser clock and visitation memory. The 768 matched branches compared +112, the three-depth parent, and each of 15 speed/turn combinations held for one or three decisions before returning control to +112.

| State on the original trajectory | Test | Outcome |
|---|---|---|
| 24.0 s, actual horizontal COM speed 13.66 mm/s | At the same 12 mm/s target, request −8°/s for 0.8 s in place of −4°/s | +112 then swam 1,800 s. Minimum whole-body clearance 18.83 mm. |
| 27.2 s, actual 12.76 mm/s | At the same 12 mm/s target, request −4°/s for 0.8 s in place of +8°/s | +112 requested −8°/s on release and swam another 1,800 s. Mean moving speed 12.90 mm/s, minimum clearance 18.13 mm, longest stall 0 s. |
| 28.8 s, actual 14.43 mm/s | Hand the exact state to the parent, which chose −8°/s and remained in control | Parent swam another 1,800 s. Minimum clearance 4.75 mm and longest stall 4.08 s. All 30 brief interventions returning to +112 failed the 120-second check here. |
| 29.6–42.4 s | Both autonomous policies and the same brief alternatives | Every tested branch failed the 120-second recovery check. |

The latest tested brief recovery came 15.27375 s before the original stop. Ten of 14 interventions at 27.2 s that passed the short test also passed 30 minutes. A 0.8-second request for −8°/s at 27.2 s passed two minutes and stopped after 1,003.12 seconds. The 60.0 mm whole-body terrain gap at that decision measures submerged clearance, not distance along the swimmer's path. Longer or different combinations outside the tested matrix may have different outcomes. The exact replay states and actions stayed out of PPO and the viewer. [All fork states, outcomes and receipts](../pond-flexible-tail-open-water-replay-v1/README.md)

## Actual GPU training

PPO ran once from a byte-verified copy of the full +112 session. The chooser actor, critic and Adam state carried over. The startup checks ran zero optimizer updates. Half of the 1,024 worlds practised moving open-water approaches, one quarter near-parallel shoreline swimming, and one quarter uninterrupted interior exploration. Training excluded the first 12 pose indices at every depth. Motor 88, body physics, reward, 35 inputs and 15 available requests stayed fixed.

| Depth | Actual decisions used in PPO | Presentations across minibatches |
|---|---:|---:|
| 72 mm | 348,160 (33.20%) | 1,392,640 |
| 75 mm | 350,208 (33.40%) | 1,400,832 |
| 80 mm | 350,208 (33.40%) | 1,400,832 |

The 128 updates took 245.14 s of process wall time, including 236.49 s of GPU physics, 0.69 s of GPU inference and 1.60 s of GPU optimization. Evidence writing took 3.71 s. Host waits overlap GPU execution. The 1,048,576 decisions reconcile to 4,096 Adam steps each for actor and critic. The 3,045 physical training stops and 16,542 healthy time limits were counted separately. The native result's base episode clock is 24 s. Effective limits were 48 s for moving approaches, 24 s for tangents and no periodic limit for interior exploration. All three interior cohorts had zero time-limit resets. [Actual training by depth and practice group](results/cohorts.csv) · [Training curve](results/training/updates.csv)

We independently measured 1,536 moving training placements. Initial speed was 10.91–11.28 mm/s, with both approach directions equally represented at each depth. On the same native straight-ahead measurement, swimmers started with 100.10–219.73 mm of horizontal room in training and 38.57–62.50 mm on the 12 independent moving validation fixtures. This freezes the current body shape and translates it along the measured course. It describes initial conditions, not the future path of a turning swimmer. The range difference limits conclusions about what can be learned and recovered on the shortest approaches. [Training ranges](results/placement-audit/training-horizontal-ranges.csv) · [Validation ranges](results/placement-audit/validation-horizontal-ranges.csv)

## Selection and final tests

We assessed the two preserved policies and 16 new candidates on identical independent fixtures, for 432 swims. Each set included 12 moving approaches, six near-parallel starts and six 15-minute exploration swims spread across the three depths. No candidate passed a moving validation start. The separate, predeclared ranking selected +104 with 0/12 moving, 6/6 near-parallel and 3/6 long passes. Its hash was frozen before final fixture binding. The official tests ran only on this checkpoint. [Selection rule](selection-protocol.md) · [Selection curve](results/selection/scores.csv) · [Order and actor verification](results/selection/order-check.json)

| Held-out bank depth | Finished | Cases reaching 15 s close (four required) | Mean near-bank time (10.96 s required) | Longest stall |
|---|---:|---:|---:|---:|
| 72 mm | 8/8 | 0/8 | 8.98 s | 0 s |
| 75 mm | 8/8 | 0/8 | 9.04 s | 2.86 s |
| 80 mm | 8/8 | 0/8 | 9.03 s | 0 s |

All bank swims completed 180 seconds with positive whole-body clearance, zero resets and zero clearance stops. The longest uninterrupted shore arc at 80 mm was 12.72°, above the required 11.81°. Ordinary navigation finished 3/12. The nine other swimmers reached a clearance guard. The original evaluator withheld its four conditional five-minute swims after that failure. All four independent 30-minute tests ran. [Results for each bank swimmer](results/bank-swimmers.csv)

| Long start | Required cells | +112 cells | Selected cells | Selected duration | Longest stall |
|---|---:|---:|---:|---:|---:|
| Godot | 185 | 10 | 189 | 850.776 s, clearance stop | 5.04 s |
| Northeast | 195 | 234 | 272 | 1,800 s, completed | 7.22 s |
| West | 218 | 186 | 29 | 82.606 s, clearance stop | 0 s |
| Southwest | 201 | 257 | 26 | 70.175 s, clearance stop | 0 s |

Mean individual coverage was 37.39% of 345 scored cells, compared with 57.90% for the three-depth parent, 49.78% for +112 and 43.91% for the installed chooser. Each swimmer met the inclusive 407-control stall limit. None reset. The three early stops, reduced individual coverage on West and Southwest, original navigation failures, and insufficient near-shore duration block publication. Since the native gates failed, candidate Godot parity was not run. The archived selected full checkpoint remains available for analysis; the package check accepted the 11 installed files and rejected this candidate. [Per-swimmer comparisons](results/long-swimmers.csv) · [Exact gate decision](results/publication.json) · [Selected full session](results/training/session-104.bin)
