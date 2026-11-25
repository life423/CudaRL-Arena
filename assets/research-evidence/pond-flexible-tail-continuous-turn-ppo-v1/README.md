# Flexible-tail continuous-turn PPO

An eight-update CUDA PPO run started from the first fitted flexible-tail
motor. The 256 worlds began with physical swimming states. Each world
retained its body, momentum, joint state, previous actuator targets and
course history through 0.4 s of forward swimming, an 8 s signed turn
request, and 1.84 s on the new course. No world reset or action fallback
occurred. The PPO optimizer consumed the sampled raw actions and their
likelihoods. Physics received the bounded root targets.

The RTX 5070 completed all eight updates, 131,072 transitions, and
2,097,152 accepted physical substeps in 1.604 seconds of external wall time.
Collection totaled 1.320 seconds and PPO optimization totaled 34.7 ms.
The largest final per-update exact KL was 0.00556 against the 0.05 cap.
Capture used 41,856 accepted CPU substeps in 0.318 seconds. The matched
CPU assessment used 92,160 accepted substeps in 0.590 seconds. These are
separate process measurements and exclude compilation.
[Protocol](protocol.json) · [Update metrics](updates.csv)

| Request | Saved course change | PPO course change | Saved → PPO turn speed |
| ---: | ---: | ---: | ---: |
| +15° | +12.85° | +12.59° | 2.99 → 3.01 mm/s |
| −15° | +13.74° | +13.76° | 2.13 → 2.13 mm/s |
| +30° | +19.50° | +19.26° | 3.07 → 3.07 mm/s |
| −30° | +16.16° | +17.93° | 1.68 → 1.64 mm/s |

Both frozen actors scored **0/4** continuous cases and **0/4** onward
swims. The gates require at least 6 mm/s projected COM motion and at most
5° course error in each scored window, at most 10° tilt and 1 mm depth
error throughout, and positive wet clearance. Both right requests produced
leftward course changes. Left peak tilt increased from 19.77° to 23.95°
and from 18.67° to 22.35°. All runs completed each phase without a physics
fault. [Paired physical results](cases.csv)

A read-only command audit used 16 saved poses and gait phases. With the
physical state held constant, the two signed 15° requests shifted predicted
yaw targets in the diagnostic's intended relative direction at all 16
states. The mean left-minus-right target was −0.0390 rad before PPO and
−0.0398 rad after. This establishes initial input sensitivity. The physical
trajectory shows that sensitivity was insufficient to steer the body.
[Command audit](command-audit.csv)

The eight candidate checkpoints, captured starts, stdout and stderr remain
in `build/flexible-tail/continuous-turn-ppo-v1`. The original actor hash
matches the protected initial binding. This experiment trained the moving
motor only. Rest, braking, bank recovery and the chooser remain outside the
result. Headless Godot startup loaded the recorded swim. The PPO candidate
is ineligible for adoption.
