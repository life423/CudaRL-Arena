# Flexible-tail yaw feedback pilot

The reference-trained flexible-tail actor passed **12/12** held-out
straight swims, including all eight rest starts. Eight CUDA PPO updates
increased speed but caused tilt and depth regressions. The fitted actor
is the qualifying straight-swim checkpoint. The PPO candidates stay in
research.

## Where propulsion was lost

The old learned actor and the working physical reference each ran from
identical saved body, joint and history states. The test included exact
rest, bent rest and two moving starts. Each start ran the learned actor at
80 ms and the reference at both 80 and 20 ms. All 115,200 physics substeps
completed in a 0.759-second diagnostic process.

The moving learned actor's yaw command settled near +0.62 rad within about
half a second. In the 2–4 second window, the two moving starts had
0.026–0.027 rad of commanded yaw swing and less than 0.001 rad of passive
tail-bend swing. The matched 80 ms reference retained about 1.096 rad and
0.672 rad, respectively. Positive root yaw work per two-second window fell
to 0.000014–0.000020 µJ versus about 2.32 µJ for the reference. The two
learned moving starts fell below 6 mm/s at 3.64 and 3.78 s after the stroke
had already largely stopped. The cadence-matched reference finished above
11 mm/s from the same rest starts. The 20 ms reference gained only about
0.17 mm/s in the final window.

The training change supplied the network with measured yaw angle and rate
relative to stroke amplitude. A learned startup weight can act when those
measurements are small. The small least-squares initialization used 808
samples from eight reference trajectories on training starts, with mirror
augmentation. It accepted all 38,400 substeps and fit the target to
0.000282 rad² MSE. The reference supplies labels only. The actor receives
physical observations and produces both joint targets without a phase clock.
Both outputs remain trainable. [Fixed protocol](../../../tools/pond-flexible-tail-motor/GAIT_PROTOCOL.md)

## Held-out qualification

| Checkpoint | Exact rest | Bent rest | Moving starts | Final projected speed from rest | Worst tilt | Worst depth error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fitted yaw feedback | 4/4 | 4/4 | 4/4 | 11.05–11.08 mm/s | 0° | 0 mm |
| PPO update 4 | 0/4 | 0/4 | 0/4 | 11.95–12.02 mm/s | 13.51° | 2.30 mm |
| PPO update 8 | 0/4 | 0/4 | 0/4 | 11.99–12.02 mm/s | 13.09° | 2.30 mm |

Qualification required two consecutive final two-second windows at
6 mm/s or faster and at most 5° from the requested course. Whole-case
tilt must stay within 10°, depth error within 1 mm, and wet clearance
positive. The fitted actor reached 8.42 mm/s in the 4–6 s window from
exact rest, then 9.86, 10.66 and 11.08 mm/s in subsequent windows.
All 36 held-out cases accepted their 345,600 substeps without faults.

The RTX 5070 trained all eight updates in **1.408 s** process wall time
(about 1.177 s GPU rollout and 0.030 s PPO optimization). It accepted
2,097,152 physics substeps and made 64 actor and 64 critic updates by
Adam. The fit, training and scheduled evaluations took **2.176 s**
combined, excluding compilation and the preceding 0.759 s diagnostic.
Every process finished within its four-second cap. The comparison used
the existing physics, integration interval, actuator constraints and
clearance test.

The signed yaw trace and matches are under
`build/flexible-tail/yaw-trace-v1`. The passing actor is
`build/flexible-tail/gait-v1/seed/actor.bin`. Its archived copy is
[passing-actor.bin](passing-actor.bin) with SHA-256
`2192D3A710EE3DE4D4BC24E32891EB6AF9D09D1F81600215927AA38A59D716C6`.
The protocol, executed
binary hashes, PPO checkpoints, per-case data and receipts are alongside
it. The evidence qualifies the new straight skill for the fixed suite.
Turning, shoreline recovery, and longer autonomous swimming require
their own qualification. The existing Godot replays remain in place.
