# Flexible-tail 15° yaw-offset PPO pilot

The PPO offset brought the creature closer to the requested course in
both directions while retaining propulsion, stroke, posture and the
ability to start from rest. None of the eight turn cases reached the
5° course-error gate. The offset checkpoints remain research evidence.

| Checkpoint | Left turns | Right turns | Mean course error, end of turn | Mean course error, onward swim | Straight / rest |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen motor with zero offset | 0/4 | 0/4 | 15.00° | 15.00° | 12/12 / 8/8 archived |
| PPO update 4 | 0/4 | 0/4 | 13.41° | 13.15° | 12/12 / 8/8 |
| PPO update 8 | 0/4 | 0/4 | 8.77° | 8.09° | 12/12 / 8/8 |

Update eight reduced late-turn course error by about 6.1–6.4° across
the matched cases. Final errors were 8.29–9.18° in the turn and
7.65–8.47° in the onward window. In both directions, two of four cases
changed course in the requested direction relative to the observed
course at the cue. The learned offset averaged 0.020 latent units in
magnitude in the late-turn window, inside the ±0.25 bound.

The ±15° cues were relative to the held approach heading. The measured
half-second COM course at the cue varied with the stroke phase, so the
same cues represented 7.52–22.48° changes from that instantaneous course
estimate. The paired zero-offset and candidate runs received exactly
the same cue in each case. All candidates still missed the absolute
target tolerance. A future comparison should set the cue from an
approach course averaged over whole tail beats for a consistent measured
starting angle.

## Retained movement

- All eight final cases stayed above 10.21 mm/s in their slowest
  rolling half-second window from cue through onward swimming.
  Final projected speed was 11.07–11.10 mm/s during the turn and
  11.11–11.14 mm/s onward.
- Peak tilt and depth error were both zero in every case. Minimum
  wet clearance across all measured cases was approximately 60 mm.
- All eight cases met the stroke gate at both checkpoints. At update
  eight, the worst matched ratios across all scored windows were
  101% for root swing, 98% for passive swing, 96% for root reversals
  and 80% for passive reversals, rounded to whole percent.
- All eight starts from rest still passed. The straight-request
  bypass matched the frozen actuator command exactly on each policy
  decision. Preflight also verified 45 exact actuator comparisons,
  including zero-offset turns.

The original fitted actor stayed frozen. PPO learned one scalar yaw
readout. Its output adds to the fitted yaw latent before tanh. Its
command bias is kept at zero, and every offset leaves pitch with the
fitted policy. The native physics, actuator limits and cadence were
preserved. [Fixed protocol](../../../tools/pond-flexible-tail-motor/YAW_OFFSET_PROTOCOL.md)

The RTX 5070 completed all eight updates across 256 worlds, 131,072
physical control steps, 32,768 sampled decisions and 2,097,152 accepted
physics substeps. Trainer wall time was **1.608 s**. CUDA rollout took
1.315 s, PPO optimization 0.027 s and checkpoint writing 0.043 s.
The whole pilot took **2.169 s** excluding compilation. The two CPU
assessments each took about 0.185 s and completed 299,520 substeps
without faults. Every process finished within its 4-second cap.

The frozen and yaw-only PPO straight-swim checkpoints matched their
original SHA-256 hashes after execution. The separate offsets, stage
receipts and raw results are in `build/flexible-tail/yaw-offset-v1`.
The offset did not qualify for adoption.
