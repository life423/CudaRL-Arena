# Flexible-tail 15° yaw-offset PPO pilot

Freeze the fitted Gait actor that passed all twelve straight swims. It
continues reading joint state and producing both actuator latents. PPO
learns one scalar offset readout. The two terms compose before the
existing tanh action transform. A straight request from rest takes the
fitted actor path directly. After a turn cue the offset remains active
through the onward swim at the same requested heading.

The single stochastic action is the raw offset latent. The applied
offset is `0.25 * tanh(z)` latent units with a maximum change of 0.02
per 80 ms decision. The readout sees signed course and facing error
scaled by 5°, its previous applied offset, and the body's yaw rate.
Its intercept stays zero. Its weights start at zero. Check exact
zero-offset command equivalence, plus the straight-request bypass,
before running PPO. Physical integration remains 1.25 ms, with the same
body, water, servo, torque limits and clearance checks.

## Training allocation

Use one native CUDA trainer process and at most eight updates of
256 parallel worlds × 64 original 20 ms controls per update. That
is 131,072 controls, 32,768 sampled offset decisions and at most
2,097,152 physical substeps. Use preserved moving starts, schedule
balanced ±15° requests and keep the same body and course history
throughout. Train for 0.4 simulated seconds before the cue, hold the
turn for eight seconds, and leave the new heading active for the last
1.84 seconds. Use the existing moving-turn reward. Actor/critic learning
rates are 0.003/0.0003, Gaussian log standard deviation −0.8, minibatch
1,024, two passes, seed 20261124, base gamma 0.998, lambda 0.9797,
clip 0.2 and exact KL cap 0.05. GAE and rewards account for decisions
held four controls.

## Matched qualification

Evaluate checkpoints four and eight on eight held-out turn cases. Both
directions start from each of four saved moving phases. Compare to the
frozen actor with zero offset from exactly the same state and request.
After 0.4 seconds of approach evaluate a 12-second turn and two seconds
of onward swimming. Require all of the following per turn.

- Correct signed change in actual whole-body COM course and a final
  course within 5° of the target in both the last two seconds of the
  turn and the two-second onward swim.
- Projected COM speed at least 6 mm/s in both those windows. Trailing
  half-second horizontal COM speed stays at least 6 mm/s from the cue
  through onward swimming.
- Peak tilt at most 10°, depth error at most 1 mm and positive certified
  clearance throughout.
- Actual root and passive tail yaw swing and reversal counts each
  retain at least 80% of the matched zero-offset run in the full turn,
  final two seconds of the turn, and onward windows. Save yaw work,
  offset, swing, and motion results alongside the gate decisions.

Check the original twelve-case straight-swim suite, including all eight
starts from rest. Keep the existing speed, course, tilt, depth and
clearance gates. Save the latest offset that passes all eight turns and
all twelve straight cases as a candidate. Preserve the fitted actor and
the earlier yaw-only PPO motor independently.

Use one process per scheduled stage, a 4.000-second external cap and
3.5-second internal deadlines for training and assessment. Record
failed attempts. Stop on physical, numerical, KL or time-limit faults.
No additional training allocation follows this pilot.
