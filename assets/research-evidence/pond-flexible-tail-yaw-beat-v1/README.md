# Whole-tail-beat turn cue and yaw-offset PPO continuation

The fitted flexible-tail motor stayed frozen. With the corrected cue,
both directions met the moving course and speed gates after eight
new PPO updates. All four left turns passed the full suite. All four
right turns missed the onward root reversal gate. Rest-start,
straight swimming, tilt and depth gates passed.

## Calibration and training

The capture measured horizontal COM displacement between the first
and third rising root-yaw crossings, using 1.25 ms samples and linear
crossing interpolation. It preserved all physical, actuator and
course history until the next 80 ms decision. The two cues from each
capture were exactly +15° and −15° of that measured course within the
0.001° preflight tolerance. The observed cycles lasted 0.992–1.000 s
each. There were 64 preserved training approaches and four held-out
approaches, with 140,160 accepted approach substeps and zero faults.

The trainer started from the preceding experiment's update-eight
offset weights with a fresh critic and Adam state. The incoming
offset was also scored on the same corrected cues before this run.
The reward, fitted actor, all physical equations, actuator limits
and decision cadence were reused. The unchanged fitted motor and
the straight-without-offset bypass passed exact command checks.
[Fixed protocol](../../../tools/pond-flexible-tail-motor/YAW_BEAT_PROTOCOL.md)

## Matched held-out results

All course errors are measured with whole-body COM motion. Speed
and course must meet the existing gates in both the final two seconds
of the turn and the two-second onward interval.

| Offset version | Left passes | Right passes | Mean final turn error left / right | Mean onward error left / right |
| --- | ---: | ---: | ---: | ---: |
| Incoming offset | 0/4 | 0/4 | 9.11° / 8.82° | 8.38° / 8.15° |
| After four new updates | 0/4 | 0/4 | 5.72° / 6.80° | 5.00° / 5.79° |
| After eight new updates | 4/4 | 0/4 | 4.61° / 2.46° | 4.02° / 2.42° |

The zero-offset reference passed 0/8 turns. At update eight, the
left cases turned 10.30–10.54° and the right cases 12.43–12.60° toward
the request. The eight candidate runs stayed above 9.45 mm/s on the
trailing half-second speed measure. The worst late-turn course errors
were 4.70° left and 2.57° right. Both directions held 0° measured tilt
and 0 mm depth error. Every checkpoint passed 12/12 straight cases,
including 8/8 starts from rest.

Each right continuation recorded three root-yaw reversals in the
fixed two-second onward window against four for its matched zero-offset
reference. That is 75% against the fixed 80% requirement. Each
right turn retained four reversals in its final turn window. The
rightward course itself and the continuous propulsion both qualified.
Preserve the declared gate when interpreting this comparison. A
follow-up can measure complete-beat durations through the turn and
its exit to distinguish changes in rhythm from window-boundary phase.

## Runtime and preserved evidence

All eight CUDA PPO updates completed in **1.455 s** trainer wall
time, including 1.204 s GPU rollout and 0.025 s PPO optimization.
The 256 worlds accepted 2,097,152 substeps and supplied 32,768
sampled decisions. Capture, preflight, training and the three
assessments took **3.068 s** total excluding compilation. The
assessments accepted 883,200 substeps without faults. Every process
finished within its four-second cap.

Raw cue measurements, exact receipts, checkpoints and cases are
saved in `build/flexible-tail/yaw-beat-v1`. The trainer exported
update zero identically to the incoming offset file and verified
the frozen actor. The protected passing fitted and straight-swim
PPO motors still match their original hashes. The final offset is
saved as research evidence because the full eight-case gate was
unmet.
