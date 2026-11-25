# Tail-beat timing at the ±15° turn exit

The four right-turn trajectories kept alternating their tail strokes. The
fourth root reversal fell 5.6–31.2 ms after the two-second onward window.
The motor also swam at a slower rhythm through both turning directions.
The right-turn mean frequency was 90–92% of the matched reference, and
the slowest complete beat retained 89.5%. Right-onward root and passive
swing retained about 101–102% and 86%, respectively. These met the
diagnostic 80% floor, so no additional PPO ran.

| Right turn start | Reference mean beat | Learned mean beat | Next reversal after the window |
| --- | ---: | ---: | ---: |
| A | 0.998 s | 1.089 s | 31.2 ms |
| B | 0.999 s | 1.092 s | 19.4 ms |
| C | 0.994 s | 1.104 s | 5.6 ms |
| D | 0.995 s | 1.098 s | 11.4 ms |

A postprocess count moved both endpoints of the two-second window by
40 ms on the same recorded trajectories. Every right-turn count went
from three to four. Every matched reference still counted four. This
illustrates why the predeclared integer count was sensitive to phase.
The original score remains 4/4 left and 0/4 right.

## Retention and execution

The diagnostic used the original whole-beat fixtures, fitted actor and
update-eight offset. The old `cases.csv` and `segments.csv` reproduced
byte for byte. The extra two seconds of every turn continued at the
same held heading. The lowest trailing half-second speed in the extra
time was 9.435 mm/s. Tilt and depth error stayed zero. Straight swimming
passed 12/12, including 8/8 starts from rest.

One CPU process accepted all 320,000 substeps in 0.226 s, including
25,600 follow-up substeps. The fitted and offset checkpoints, native
physics and original capture files matched their SHA-256 bindings.
[Fixed protocol](../../../tools/pond-flexible-tail-motor/BEAT_TIMING_PROTOCOL.md)

Raw reversal timestamps, measured periods, extension scores and
window-sensitivity comparison are under
`build/flexible-tail/beat-timing-v1/assessment`. The left and right
stroke periods overlap. The next qualification can count complete
beats alongside progress, clearance, and stability through consecutive
turns. If the goal becomes closer retention of the reference rhythm,
measure cycle frequency and swing in the training objective and keep
the original physical and retention targets.

The separate signed course-error gain at `569aaa9` verified opposite
first trims and exact zero-gain outputs offline. It is a candidate
comparison for directional asymmetry if consecutive turns expose it.
Use the same complete-beat cue fixtures and evaluator to make that
comparison. The scalar gain uses a different checkpoint schema, and its
offline command checks leave sustained moving turns unmeasured.
