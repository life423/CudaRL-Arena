# One matched right-to-left handoff

The saved update-eight offset executed a left turn after the shared
right turn and onward swim. Both branches passed the moving course,
speed, tilt, depth and clearance gates. The left-turn branch retained
at least 99.1% of the control's measured complete-beat frequency in
the scored windows.

## Shared start and fork

The paired evaluation used zero-based row 1, `moving_a_right`, from the
whole-beat fixture. The first right turn and onward scores reproduced
the saved update-eight case. After the first right turn, the three
positive-going root rate crossings occurred at 12.955960, 14.031232,
and 15.133713 simulated seconds. Their horizontal COM displacement
defined the approach direction across two complete beats. The program
forked at control 760, 15.20 s, and verified both state copies before
editing either request schedule. The control kept its existing Swim
leg. The test requested a 15° left turn from that course, then held
the new heading for onward swimming.
[Fixed protocol](../../../tools/pond-flexible-tail-motor/HANDOFF_PAIR_PROTOCOL.md)

## Paired results

| Branch and scored window | Course error | Projected speed | Root / passive swing | Root reversals |
| --- | ---: | ---: | ---: | ---: |
| Control, end of comparison | 0.16° | 9.41 mm/s | 1.012 / 0.540 rad | 3 |
| Left turn, end of comparison | 4.16° | 9.43 mm/s | 1.010 / 0.545 rad | 3 |
| Control, onward | 1.22° | 9.42 mm/s | 1.016 / 0.545 rad | 4 |
| Left turn, onward | 2.18° | 9.45 mm/s | 1.016 / 0.546 rad | 4 |

The test's COM changed course by **10.84° left** in the final two
seconds of its turn and **12.82° left** in the following window.
The two-second projected speed gate was 6 mm/s. Minimum rolling speed
was 9.40 mm/s in the control and 9.30 mm/s in the test, including the
extra time used to close beats. Both reported zero tilt and depth
error and certified at least 67.999 mm of wet clearance.

The mean complete-beat period was 1.091 s for the control and 1.094 s
for the test near the end of the comparison, then 1.107 and 1.101 s
onward. The slowest measured test beat in those windows retained
99.15% and 99.53% of the corresponding control frequency. Root swing
retained at least 99.8% and passive swing at least 100.1%. The
longest interval between consecutive root reversals was 0.560 s in
the control and 0.622 s in the test.

The applied offset was already negative at the exact fork sample
(`-0.0178`). Its mean across the preceding complete beat was
`+0.0108`. The first test decision moved it to `-0.0378` in latent
units. The beat spanning the cue had a mean of `-0.0476` and
closed 0.985 s after the cue. The first beat entirely after the
cue averaged `-0.0504`. This distinguishes the change in average
steering direction from the offset sign at one stroke phase.

All **37,760** substeps were accepted in **0.276 s** process wall
time. The shared prefix used 12,160 and each branch used 12,800.
There were zero optimizer updates. The fitted stroke, original
offset, cue fixture and physics matched their original SHA-256
bindings. This is one right-to-left pair in clear water. The separate
signed-gain controller remains outside this comparison.

Raw decisions, complete-beat means, reversal timings, substep
receipts and per-window measurements are in
`build/flexible-tail/handoff-pair-v1`.
