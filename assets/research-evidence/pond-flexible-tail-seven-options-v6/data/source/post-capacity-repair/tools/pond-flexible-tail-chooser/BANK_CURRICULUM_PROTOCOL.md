# GPU PPO continuation with bank approaches

Continue from the first run's final chooser actor. The fitted gait and
update-eight yaw offset stay frozen. Save new actor checkpoints in their own
study directory. Start a fresh critic and Adam state because the previous
study saved the actor alone.

## Training change

Balance the initial training approach directions around the pond. Sample
archived moving poses from the training-only bundle. Place each complete
pose and its history on an oblique outward approach, 34–44° from the local
tangent of the conservative clearance contour. A single full 15° request
would leave this initial direction pointing outward. The exercise gives
the chooser enough room to learn a sequence of earlier requests.

The native stationary-pose clearance calculation bounds the straight ray.
Use a 170–205 mm straight-ray distance to certificate loss, and require
more than 20 mm of certified clearance at a point 72 mm along the same
ray. This is spawn geometry, measured without physical stepping. The
actual first steering cue and subsequent clearance use the ordinary
whole-beat and native dynamics checks. Record generated starts and warm-up
work separately. The chooser samples every action from its learned policy.
The placement calculation supplies no requested turns, route or action labels.

The reward, 32 inputs, three actions, physics, limits, 4–6 second decision
cadence, and optimizer settings are those of the first chooser study.
The first study's archived held-out actor output and reports provide the
before measurements. The evaluator executable and held-out fixture bytes
are identical.

## Allocation and gates

One RTX 5070 training process with 96 worlds × four decisions per rollout,
up to 16 PPO updates, minibatches of 128 and four passes. The original
cooperative 17.5 s cutoff and 20 s external cap apply. The maximum
training allocation including warm-ups is 58,982,400 attempted substeps.
A numeric or contract failure rejects the run. Score the last completed
actor only. Preserve all evidence and optimizer counts, with no restart
after an attempted training stage.

Apply the identical held-out suite with greedy policy inference.
Require all 12 continuous 120-second swims and the original 20% pooled
coverage floor, then all four conditional continuous 300-second endurance
swims, with the original per-case movement, tilt, depth and strictly
positive whole-body clearance gates. No reset occurs within any case.
All evaluation is covered by the original 2,112,000 substep and 20 s
external process caps. Enable the candidate in Godot only after a
full pass. Report issued requests, first-turn timing, native clearance
stops, travel, coverage and paired time survived for all starts.

[Original protocol](PROTOCOL.md)

