# GPU PPO continuation on bank approaches

The resumed chooser began turning at its first eligible decision in
every held-out case. All 12 swam longer than before. Each eventually
lost native bank-clearance certification. The frozen motor maintained useful speed, depth and posture.

| Identical held-out test | First chooser | Bank-practice continuation |
| --- | ---: | ---: |
| 120-second passes | 0/12 | 0/12 |
| Native clearance stops | 12/12 | 12/12 |
| Median time before stop | 37.82 s | 46.12 s |
| Range before stop | 22.34–71.63 s | 24.53–89.19 s |
| First turn request | 16.0–64.8 s | 4.0–4.8 s |
| Keep / left / right requests | 99 / 12 / 6 | 0 / 15 / 126 |
| Pooled coverage | 119/345 | 115/345 |
| Mean speed range after launch | 11.15–11.45 mm/s | 9.48–9.76 mm/s |
| Peak tilt and depth error | 0° and 0 mm | 0° and 0 mm |

The exact Godot rest start improved from 52.38 to 53.88 simulated
seconds. It requested 13 consecutive right turns. The direction of
measured COM travel changed 58.0° across 44 seconds of four-second
segments, averaging **5.28° of actual course change per 15° request**.
Distance divided by angle gives an effective radius of about **0.41 m**
for that arc. A study of turning-room requirements should use measured
course change. Repeated requests still left the swimmer on a wide arc
toward the bank. The bank guard reports loss of certified clearance.
Contact was outside this run.

## Frozen inputs and work count

The RTX 5070 ran four PPO updates in **14.098 seconds**. Another full
update did not fit the original cooperative cutoff. The 96-world run
collected 1,536 chooser decisions and completed 48 Adam calls per
network. It accepted 5,735,281 of 5,735,609 physical substeps,
including 1,302,400 accepted warm-up steps. It began 407 training
approaches, split 206 and 201 by direction. It recorded 328 training
clearance stops with zero cue timeouts or posture stops. All 512
frozen-motor comparisons matched exactly. The new update-zero actor
file matched the first study's final actor byte for byte.

Training used the original rewards, three candidate actions and
4–6 s cue timing. Mirrored outward headings and positions made up
the only change to episode starts. They supplied no steering labels.
The geometry check for the first 96 placements certified at least
36.90 mm of fixed-pose clearance along the 72 mm warm-up projection.
The real warm-up and course transitions were simulated in native CUDA.

The identical held-out executable completed the 12 tests in
0.500 seconds and accepted 470,682 of 470,694 attempted substeps.
The four conditional endurance cases stayed closed. The checkpoint
remains a candidate for research. The registered Godot app keeps its
live manual controls.

[Protocol](data/protocol.md) · [Matched case comparisons](data/matched-cases.csv) ·
[Training receipt](data/training.receipt.json) ·
[Held-out results and trace](data/heldout/cases.csv) ·
[Saved chooser](data/training/actor-final.bin)

