# Tail-beat timing diagnostic at the right-turn exit

Replay the previous whole-beat assessment with the fitted Gait actor and
the saved update-eight offset. Keep the original starts, turn target, servo,
body, policy weights and physics. Use the original eight turn fixtures
with paired zero-offset controls and the original twelve straight swims.
Retain the twelve-second moving turn and its two-second onward scoring.
Continue each turn trajectory for two extra seconds at the same held
heading to observe reversals on both sides of the original end time.
The additional observations are separate from the original scores.

Record continuous root and passive yaw-rate reversals across the turn
exit, using the original 0.02 rad/s dead band. Measure a root beat
between consecutive reversals of the same direction, with an opposite
reversal between them. Use complete beats with midpoints in the original
onward window. Compare their durations to the matched zero-offset trace,
and look for a prolonged gap, loss of swing or loss of useful movement.
Keep the original reversal-count result alongside the measured periods.

The original 80% stroke retention level corresponds to a complete-beat
frequency at least 80% of the matched reference. If each right case
sustains alternating beats at that rate, retains swing and continues to
meet motion, posture and depth checks, close the diagnostic without
training. If it loses a beat or consistently falls under that level,
use one separately bound short native CUDA PPO attempt of at most eight
updates to address the measured fault. Keep the fitted actor frozen and
recheck both turn directions and all straight/rest cases. A physics or
numerical fault stops the affected diagnostic for investigation.

Use the 3.5-second internal deadline and four-second process cap.
Maximum 320,000 native evaluation substeps, including all extensions.
Save all attempted cases and raw timing. The earlier two-second score
remains part of the evidence.
