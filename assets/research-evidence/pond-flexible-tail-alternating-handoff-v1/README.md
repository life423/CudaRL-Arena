# Six alternating turns with the frozen flexible-tail motor

Both three-turn sequences passed all 12 scored windows at an 80 mm depth request.
Each used one continuous native body. The four hold-heading controls forked
from the exact cue states. All four held their course and passed the movement
criteria in both windows.

| Saved start | Request | Signed course change, turn / onward | Course error, turn / onward | Projected speed, turn / onward |
| --- | ---: | ---: | ---: | ---: |
| Row 0, moving_a_left | +15° | +10.30° / +11.01° | 4.70° / 3.99° | 9.51 / 9.47 mm/s |
| Row 0, moving_a_left | −15° | −11.58° / −11.26° | 3.42° / 3.74° | 9.45 / 9.43 mm/s |
| Row 0, moving_a_left | +15° | +10.79° / +12.80° | 4.21° / 2.20° | 9.43 / 9.45 mm/s |
| Row 1, moving_a_right | −15° | −12.56° / −12.46° | 2.44° / 2.54° | 9.52 / 9.48 mm/s |
| Row 1, moving_a_right | +15° | +10.84° / +12.82° | 4.16° / 2.18° | 9.43 / 9.45 mm/s |
| Row 1, moving_a_right | −15° | −11.59° / −11.27° | 3.41° / 3.73° | 9.45 / 9.43 mm/s |

Turn scores use the last two seconds of each 12-second turn. Onward
scores use the following two seconds. Minimum rolling COM speed was
9.30 mm/s across the test runs, with zero measured tilt and depth error.
Minimum certified whole-body clearance was 58.62 mm. The second and
third turns retained at least 99.0% of matched complete root-beat
frequency, 99.6% of root yaw swing and 100.2% of passive yaw swing.
Against the archived zero-offset references, the first turns retained
at least 89.7% of complete root-beat frequency and 86.0% of passive
swing. The first right turn still recorded three root reversals versus
four in the fixed onward window. The complete-beat frequency in that
window retained 90.0%.

The subsequent cues were at 15.12 and 29.44 simulated seconds for the
left start, and 15.20 and 29.52 for the right start. The decision
records show continuous offset activation. The stage-two diagnostic
continuation closed beat windows while stage three used the saved cue
state from that same trajectory. One native CPU process accepted all
126,656 attempted substeps in 0.845 seconds wall time, including the
geometry replay. It used zero resets or optimizer updates. The
evaluator, fitted stroke, update-eight offset, starts, reference traces
and physics were hash-bound in the [run record](data/bindings.json).

## East-bank geometry replay

The native envelope checked 12,801 poses per handoff, from the cue
through 16 seconds. The replay placed each incoming course 7.5° outward
from the local east-bank tangent at 80 mm depth. Each 15° heading
request then pointed 7.5° inward relative to the local tangent.
Setbacks below are measured along the inland normal from the east
shoreline to the COM at the cue.

| Handoff | Last uncertified setback | First certified setback | Minimum certified clearance | Inward COM travel in first 2 s onward | Limiting pose after cue |
| --- | ---: | ---: | ---: | ---: | ---: |
| Left → right | 406 mm | 407 mm | 0.427 mm | 1.24 mm | 16.000 s |
| Right → left | 315 mm | 316 mm | 0.072 mm | 1.75 mm | 0.468 s |

At the smaller setbacks the conservative clearance bounds were
−0.169 mm and −0.690 mm. The limiting head, tail and COM positions
are in the [left-to-right](data/left_start/east-bank.json) and
[right-to-left](data/right_start/east-bank.json) records. These are
local conservative estimates using replayed clear-water geometry.
They cover one depth, bank location, approach angle and saved starting
phase. Live near-bank turning and longer autonomous navigation still
need measurement.

[Protocol](../../../tools/pond-flexible-tail-motor/ALTERNATING_HANDOFF_PROTOCOL.md) ·
[Window scores and reversal counts](data/alternating-gates.csv) ·
[Substep summary](data/result.json) · [Process receipt](data/assessment.receipt.json)

