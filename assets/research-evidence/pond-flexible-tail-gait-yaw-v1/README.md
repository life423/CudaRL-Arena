# Flexible-tail yaw-only PPO retention

Eight CUDA PPO updates preserved straight swimming from rest and from
moving starts. Checkpoints four and eight each passed all twelve calm-water
cases. The earlier two-axis run passed zero after training, with 13.09°
peak tilt and 2.30 mm depth error at update eight.

The new experiment held the fitted actor's neutral pitch channel fixed.
PPO sampled and scored yaw as one stochastic action, and learned the yaw
path in the same network. The native joint servo, water equations, 80 ms
action cadence, 1.25 ms substep, body and clearance test were reused.
The exported update-zero actor matched the original file byte for byte.
All checkpoint pitch weights and bias remained zero. The final actor had
1,665 changed coefficients. [Fixed protocol](../../../tools/pond-flexible-tail-motor/GAIT_YAW_PROTOCOL.md)

| Checkpoint | Exact rest | Bent rest | Moving starts | Final speed from rest | Final speed from moving starts | Worst tilt | Worst depth error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Archived fitted actor | 4/4 | 4/4 | 4/4 | 11.05–11.08 mm/s | 11.50–11.51 mm/s | 0° | 0 mm |
| PPO update 4 | 4/4 | 4/4 | 4/4 | 12.01–12.05 mm/s | 12.39–12.40 mm/s | 0° | 0 mm |
| PPO update 8 | 4/4 | 4/4 | 4/4 | 11.46–11.51 mm/s | 11.89–11.90 mm/s | 0° | 0 mm |

The motion gate was at least 6 mm/s and no more than 5° course error
in each of the final two two-second windows. Tilt and depth limits were
10° and 1 mm throughout. The minimum clearance bound at update eight
was 60.00 mm. The exact-rest cases averaged 6.30 mm/s during seconds
two to four, up from 5.96 mm/s for the fitted actor. The most difficult
bent start ended at 4.92° course error, close to the 5° gate. Both
held-out assessments accepted all 230,400 substeps.

The RTX 5070 ran the fixed eight updates with 256 worlds, 131,072 controls,
32,768 policy samples and 2,097,152 accepted physics substeps. Trainer
wall time was 1.591 s, including 1.298 s GPU rollout, 0.035 s PPO
optimization and 0.032 s checkpoint writing. The training and both
assessments completed in 1.904 s excluding compilation. Each process
finished inside its 4.000 s cap. No second training attempt ran.

Update eight is saved as [passing-ppo-actor.bin](passing-ppo-actor.bin).
Its SHA-256 is
`771CB7C1937568C08C62D6F4403C9EA2CA7C09728DAA98D735D9094D61E8158F`.
The [passing fitted actor](../pond-flexible-tail-gait-v1/passing-actor.bin)
is archived separately. Release binaries, receipts, seed bindings,
checkpoints and per-case measurements are in
`build/flexible-tail/gait-yaw-v1` and `build/flexible-tail/gait-yaw-build`.
The reused evaluator and physical model matched the earlier hashes.

The gate covers upright straight swimming from planar calm-water starts.
Correcting a pitch or depth disturbance and preserving the stroke during
a turn require their own checks. A turning comparison can freeze the fitted
stroke and learn a small yaw offset while measuring actual swing, reversals
and whole-body course. The offset changes the physical state seen by the
stroke, so an unchanged actor file alone cannot establish stroke retention.
