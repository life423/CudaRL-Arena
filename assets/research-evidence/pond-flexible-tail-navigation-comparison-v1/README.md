# Flexible-tail heading versus destination PPO

Both arms completed eight native CUDA PPO updates on the RTX 5070. Neither
passed a held-out case at any measured checkpoint. Both moved more slowly
after training. The destination arm remained very close to the heading arm
throughout the study. No motor was nominated for autonomous exploration.

The paired run used one flexible-tail motor architecture per arm, the same
initial weights and critic initialization, one common set of 256 starting states
and scheduled commands, and the same PPO seed and optimizer settings. The
two arms had identical physical measurements in the update-zero evaluation.
The candidate adds body-relative XYZ goal displacement scaled by 0.10 m and
uses signed reduction in COM distance for the same 0.65 transport term. The
shared capture check matched +2.00048 µm, −2.00048 µm and 0 for approach,
retreat and no motion at 0.1 mm/s over 20 ms. Both arms use the same anchor
observation and scoring for Hold. [Protocol](../../../tools/pond-flexible-tail-motor/NAVIGATION_COMPARISON.md)

## Learning curves

Mean projected speed and average reduction in distance to the held target use
the same 20 scored moving legs per arm. Peak tilt and depth are over the whole
scored suite. Each checkpoint passed 0/12 full cases, 0/20 moving legs and
0/2 hold legs in each arm.

| Update | Mean projected speed, mm/s (heading / destination) | Mean goal progress, mm (heading / destination) | Worst tilt, ° (heading / destination) | Worst depth error, mm (heading / destination) |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1.897 / 1.897 | 29.432 / 29.432 | 69.54 / 69.54 | 12.17 / 12.17 |
| 2 | 1.855 / 1.850 | 28.802 / 28.710 | 71.48 / 71.82 | 12.47 / 12.38 |
| 4 | 1.509 / 1.505 | 25.184 / 25.095 | 44.91 / 44.24 | 2.01 / 2.06 |
| 6 | 1.529 / 1.506 | 25.653 / 25.097 | 45.12 / 44.75 | 2.10 / 2.19 |
| 8 | 1.518 / 1.525 | 25.357 / 25.505 | 45.68 / 45.54 | 2.09 / 2.15 |

## Held-out results after update eight

| Skill | Heading | Destination | Observed failure |
| --- | ---: | ---: | --- |
| Straight | 0/2 | 0/2 | Projected speed 2.56–3.08 / 2.56–3.18 mm/s |
| Individual left turns | 0/2 | 0/2 | Course advanced +10.21° to +18.21° / +10.22° to +18.06°, at roughly 3 mm/s |
| Individual right turns | 0/2 | 0/2 | Actual course went left +12.91° to +26.85° / +12.81° to +26.80° |
| Three-turn alternating sequences | 0/2 | 0/2 | No qualifying turn. Projected speed by the third turn fell below 0.35 mm/s in all four runs |
| Onward swimming | 0/6 | 0/6 | Best projected speed 2.58 mm/s in both arms |
| Brake and hold | 0/1 | 0/1 | Up to 108.61 / 108.81 mm from the anchor in the hold window |
| Hold from rest | 0/1 | 0/1 | Up to 19.83 / 20.02 mm from the anchor in the hold window |
| Recovery | 0/2 | 0/2 | Barely advanced from a stationary start and moved away from the goal from the 75° misoriented start |

The effective COM turning radius was 126–444 mm in each arm across the six
measurable curves at update eight. The estimate divides chord path length by
net course change. All six estimates came from failed turns, including curves
in the wrong direction. Four of ten turns had insufficient course change or entry
motion for a radius estimate. Every held-out trajectory completed its scheduled
controls without a physics fault. Minimum certified wet clearance across
checkpoints was 49.82 mm.

## Execution and interpretation

| Stage | Heading | Destination |
| --- | ---: | ---: |
| Updates | 8 | 8 |
| Transitions and accepted physical substeps | 131,072 / 2,097,152 | 131,072 / 2,097,152 |
| GPU rollout, CUDA events | 1.211 s | 1.106 s |
| PPO optimization | 0.034 s | 0.037 s |
| Checkpoint writing | 0.032 s | 0.033 s |
| Trainer process wall time | 1.499 s | 1.307 s |

The whole run took 4.205 seconds excluding compilation, with five CPU
assessments at 0.251–0.260 seconds each. All processes finished inside the
4-second individual cap. The destination process ran second. Its shorter
startup is not evidence that its training algorithm is faster.

Both models reduced the worst posture and depth excursions relative to the
initial fit but lost propulsion and made less progress. The final destination
gain over heading was only about 0.007 mm/s and 0.15 mm of average progress,
while both failed every skill. The base actor could not deliver forward,
stable swimming at the start. The new objective and inputs did not overcome
that failure in this small, shared-budget test. The goal offsets were placed
beyond the scored moving windows to measure continuous approach, so the study
does not establish controlled arrival at arbitrary coordinates. The next
learning experiment should test a stable learned gait from rest and from a
moving start before composing repeated turns and braking. An additional
identical PPO run would not address that deficit.

All prior saves were preserved and the shared inputs matched their original
hashes after the run. The physical integration and constraints were reused.
The raw checkpoint pairs, exact executed source and binary hashes, per-case
and per-leg CSVs and receipts are in
`build/flexible-tail/navigation-comparison-v1` in this worktree. No extended
training, chooser training or Godot adoption followed.
