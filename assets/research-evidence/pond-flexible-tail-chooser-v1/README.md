# First GPU PPO chooser run

The final chooser failed all 12 held-out bank-clearance cases. It maintained
swimming speed, depth and upright posture, but waited until the last few
seconds before a clearance stop to request its first turn. The trained
checkpoint is preserved for analysis. The registered Godot project
still opens the existing live manual controls.

| Measurement | Result |
| --- | --- |
| CUDA PPO updates | 4 of 16 planned |
| RTX 5070 training process | 13.957 s wall, 13.178 s GPU collection |
| Chooser decisions sampled | 1,536 across 96 worlds |
| Native training substeps | 5,319,769 accepted of 5,319,854 attempted |
| Frozen motor comparison | 512 / 512 exact actuator and trim matches |
| Held-out cases reaching 120 s | 0 / 12 |
| Held-out end time | 22.34–71.63 simulated s, 1.304 s process wall |
| Certified clearance stops | 12 / 12 |
| Credited collective coverage | 119 / 345 estimated cells, 34.5% |
| Post-launch mean horizontal speed | 11.15–11.45 mm/s |
| Measured peak tilt and depth error | 0° and 0 mm on all 12 |

The trainer completed 48 Adam steps on the actor and 48 on the critic.
Its time forecast declined to start update five because the slowest recent
CUDA wave put another complete update outside the cooperative cutoff.
Training recorded 85 clearance terminations and zero cue, posture or
numerical faults. Evaluation ran one continuous body per case with
zero optimizer updates and zero resets. The four endurance trials stayed
closed because their prerequisite failed.

## Failure mechanism

The held-out network selected keep-course on 99 of 117 decisions
(84.6%), left on 12 and right on six. The first turn started only
3.24–7.51 s before clearance loss in every case. No case made more
than two turn requests. On the exact Godot start, the chooser held
course until 48.80 s, then requested a left turn at 36.98 mm
certified clearance. The certificate reached zero at 52.38 s.
These are native clearance-guard stops. They do not measure body
collision or contact recovery. The observed failure was late recurrent
steering while the motor continued producing useful propulsion.

The fixed cases retained their archived 72–89 mm depth targets.
The 345-cell denominator is a geometry estimate at the exact
Godot start's 72 mm depth, using eight tested upright orientations
per 50 mm cell center. The 119 cells are pooled across starts.
No qualifying cell revisits were observed before the stops.

A subsequent experiment should concentrate training starts on safe
approaches that need two or more early decisions to redirect the course.
It can test longer-range anticipatory clearance feedback while keeping
the same frozen motor and the existing held-out criteria.

[Bound protocol](data/protocol.md) · [Training result](data/training/result.json) ·
[Held-out cases](data/heldout/cases.csv) · [Exact Godot-start trace](data/heldout/traces/godot_rest.csv) ·
[Final candidate weights](data/training/actor-final.bin)

