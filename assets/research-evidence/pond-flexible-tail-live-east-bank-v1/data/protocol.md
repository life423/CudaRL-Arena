# Live east-bank turn pair

Run two native flexible-tail trajectories from the same handoff states as
the alternating-turn comparison. Reconstruct fixture rows 0 and 1 to
their first confirmed cues, controls 756 and 760 respectively, and
compare their state checksums with the archived clear-water forks.
Start the bank test with those entire moving states placed at 407 mm
for the right turn and 316 mm for the left turn, as certified in the
east-bank replay. Set the incoming course 7.5° outward of the local
tangent. A ±15° relative heading request points 7.5° inward.

Initialize each live case once by a rigid horizontal placement.
Rotate the native body and world linear and angular velocity,
the held requests, COM course history and beat positions together.
Retain depth, relative joint states, trim, held targets and clock.
Apply the frozen fitted motor and the exact update-eight yaw offset
to the integrated motion. Keep the 80 ms decision cadence and
1.25 ms integration. The fully submerged model advances the original
fluid and body dynamics and checks the native clearance certificate
at the initial pose and every subsequent substep.

Hold the turn request for 12 s, score the final 2 s and first 2 s of
the following swim, and run a further 2 s to close beat windows.
Require signed COM travel in the requested direction, ≤5° error,
and ≥6 mm/s projected speed in both windows. Require ≥6 mm/s rolling
speed, ≤10° tilt, ≤1 mm depth error and positive certified whole-body
clearance throughout. Require inward COM travel over the first 2 s
onward. Compare complete beat frequency and root and passive yaw
swing with the archived same-state hold-heading controls, applying
the same 80% lower limit. Record raw reversals separately.

Limit the pair, including reconstructed cue states, to at most 51,200
physics substeps in one process, with a 3.5 s internal deadline and a
4 s external cap. Use zero optimizer updates and zero resets during
either live case. Preserve both results on an early stop.

