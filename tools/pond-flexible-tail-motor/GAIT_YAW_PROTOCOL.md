# Flexible-tail yaw-only PPO retention experiment

Start from the fitted Gait checkpoint that passed all 12 straight-swim
cases. Its pitch output is exactly neutral. The earlier two-axis PPO
trial preserved propulsion but exceeded the tilt and depth limits.
This experiment keeps the fitted pitch output at its existing value and
uses PPO to train yaw. The actor and physical joint still support pitch.
The experiment addresses upright straight swimming from the saved
calm-water starts.

The trainer compresses only the output layer to its single active yaw
coordinate. Rollouts sample the yaw Gaussian alone and PPO uses that same
scalar likelihood and KL. Export restores the ordinary two-output Gait
shape with neutral pitch. Before training, verify the source pitch
weights and bias are zero. Validate the exported update-zero checkpoint
against the source and verify pitch coefficients in every saved actor.

Retain all 256 original training starts, the twelve fixed assessment
starts, the shared Gait observation and reward, 80 ms policy decision
cadence, and all original dynamics and constraints. The frozen reference
checkpoint and prior trial are read-only. Count at most eight PPO updates
of 64 original physical controls per world, 131,072 controls,
32,768 policy samples and 2,097,152 substeps. Keep the Gait yaw exploration
log standard deviation −2.6 and the same yaw random draws, RNG seed
20261124, two passes, 1,024 minibatch, actor/critic learning rates
0.0001/0.0003, gamma 0.998, lambda 0.9797, PPO clip 0.2 and exact KL cap
0.05. The Gait rollout discount contracts to the unchanged 80 ms
decision interval.

Score updates four and eight with the fixed deterministic evaluator.
Use the archived update-zero assessment after checking identical
weights and evaluator. Each case requires two final two-second windows
at ≥6 mm/s projected COM speed and ≤5° course error, tilt ≤10° and
depth error ≤1 mm throughout, and strictly positive certified wet
clearance. Save the latest checkpoint that passes all twelve, while
keeping the archived fitted checkpoint separately. Report how it
compares on acceleration, speed, course and posture.

Launch training once. Use the existing 3.5-second internal and 4.000 s
external wall caps per process. Score each available scheduled
checkpoint once. Stop the affected process on physical, numerical,
KL or timeout failure and retain all evidence. This allocation covers
calm straight swimming only.
