# Whole-tail-beat turn cue and short yaw-offset continuation

Continue the previous yaw offset from its saved update-eight weights.
Keep the fitted Gait actor frozen and use the same yaw-offset observation,
reward, 80 ms action cadence and 1.25 ms physics. The stroke actor receives
the measured body state. The offset alone learns. Adam and the critic
start fresh because the prior export stores actor coefficients.

## Calibrating the turn cue

From each archived moving start, run the fitted actor on its straight
request and measure whole-body COM displacement over two complete root
yaw cycles. Use crossing events of the same sign and a continuous
body trajectory. At the next aligned decision, preserve the body,
joint, last-command and COM history state. Define the approach course
from the horizontal displacement between cycle boundaries. Set the
target 15° left or right of that measured direction. Use the same
captured state for the two directions and the zero-offset comparison.
Fail the capture when two cycles cannot be measured within 250 controls,
or when dynamics or clearance fail. The frozen actor is the only policy
active during approach. The offset is still inactive in the independent
straight-from-rest cases.

## Bounded PPO and held-out checks

- Run one CUDA trainer process with at most eight updates, 256 worlds,
  and 64 unchanged physical controls per world per update. This is
  131,072 controls, 32,768 offset samples, and 2,097,152 substeps.
  Train from the captured cue state for eight simulated seconds of
  turn and 2.24 seconds of onward swimming. Use matched left and right
  requests. Keep the previous yaw-offset hyperparameters and likelihood.
- Score the incoming offset before the optimizer run and the new
  checkpoints at updates four and eight. Use eight held-out cases,
  both directions from each of four captured moving starts. Run 12
  seconds of turn and two seconds of onward motion with continuous
  native state. Compare each with its matched zero-offset run.
- Retain the existing acceptance rules. In the final two seconds of
  the turn and onward swim, projected speed is at least 6 mm/s and
  course error is at most 5°. Course must change in the requested
  direction relative to the measured approach. Require at least 6 mm/s
  rolling horizontal speed from cue, 10° maximum tilt, 1 mm maximum
  depth error, and positive wet clearance. Root and passive swing and
  reversals must retain at least 80% of the matched baseline. Score
  the twelve original straight swims, including eight rest starts.
- Use 4.000-second wall caps and the existing internal deadlines.
  Preserve fixture captures, cue calibration, checkpoints, failed
  attempts and scores. Stop on physical, numerical, KL or time-limit
  failures. Save the newest scheduled checkpoint passing all eight
  turns and all twelve straight swims. Leave earlier motor files intact.
