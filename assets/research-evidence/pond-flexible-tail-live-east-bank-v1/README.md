# Live east-bank turns and the flexible-tail Godot controls

Both native turns passed at the certified east-bank cue setbacks with
the frozen fitted gait and update-eight yaw offset. Each test started
from a rigid placement of its archived moving state. State checksums
matched the two clear-water handoffs before placement.

| Live turn | Cue setback | Course error, turn / onward | Minimum rolling speed | Minimum certified clearance | Onward inward COM travel |
| --- | ---: | ---: | ---: | ---: | ---: |
| Right −15° | 407 mm | 3.42° / 3.74° | 9.42 mm/s | 0.424 mm | 1.24 mm |
| Left +15° | 316 mm | 4.16° / 2.18° | 9.30 mm/s | 0.072 mm | 1.75 mm |

The final two seconds of each turn changed course by 11.58° right
and 10.84° left, respectively. Both maintained at least 9.43 mm/s
projected speed in every scored window, with zero measured tilt or
depth error. Complete root-beat frequency retained at least 99.1% of
the matched hold-heading controls. Root and passive yaw swing retained
at least 99.8% and 100.2%. Per-window course errors differed from the
corresponding clear-water runs by less than 0.002°. The small differences
reflect integrating the rigidly placed state again.

The live test used 25,600 accepted bank substeps and 24,256 accepted
substeps to reconstruct the two cue states. All 49,856 substeps
were accepted in 0.349 seconds wall time. There were zero resets
within a trial and zero optimizer updates. The original fluid and
body dynamics ran at the native 1.25 ms substep. The fully submerged
model certified positive clearance at every substep and the initial
poses. The 0.072 mm minimum leaves little margin for other starts.
The run covers one east-bank approach angle, one 80 mm depth and two
matched cue states. It leaves contact recovery and whole-pond
navigation unqualified.

[Fixed protocol](../../../tools/pond-flexible-tail-motor/LIVE_EAST_BANK_PROTOCOL.md) ·
[Scored windows](data/live-bank-gates.csv) · [Process receipt](data/assessment.receipt.json)

## Godot

Godot Project Manager's **Tadpole Pond | Flexible Tail** entry opens
directly to **Live pond**, paused at a qualified rest start. **Play**
starts native inference and physics. **Left 15°** and **Right 15°** queue
a turn at the next suitable whole-beat cue. Use **P** to play or pause
and **Q / E** to queue left or right. **Pond view** and **Zoom to
tadpole** switch camera views. **Recordings** keeps the three saved
clips and pauses the live swimmer where it is.

The same frozen weights load from the project's live data. A Godot
headless check drove left and then right on one body starting from
rest. It reached onward swimming after 16.82 and 31.14 simulated
seconds at 9.49 and 9.61 mm/s, with zero measured tilt or depth error.
The desktop launch from Project Manager, left-turn queue, live motion,
Recordings tab, native pause and camera swap were inspected in Godot
4.4.1. The live and recordings layouts also rendered at 390 × 844 and
768 × 1024. The native motor still takes a heading request from the user.
Autonomous navigation needs a compatible learned destination chooser.

[Godot scene](../../../flexible-tail-preview/flexible_tail_preview.tscn) ·
[Native Godot bridge](../../../src/godot/flexible_tail_live.cpp)

