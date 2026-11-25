# Flexible-tail straight swimming pilot

The RTX 5070 ran all 12 planned PPO updates. All four held-out checkpoints
scored 0/12. The final actor stayed upright and near its requested depth,
but propagated less forward motion than its initialization. The existing pond
replays remain the available flexible-tail demonstration.

| Checkpoint | Exact rest mean speed, mm/s | Perturbed rest mean speed, mm/s | Moving start mean speed, mm/s | Peak tilt across suite | Peak depth error across suite |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.279 | 0.278 | 3.288 | 0.00° | 0.000 mm |
| 4 | 0.199 | 0.198 | 3.233 | 2.68° | 0.123 mm |
| 8 | 0.154 | 0.153 | 3.030 | 4.98° | 0.232 mm |
| 12 | 0.107 | 0.106 | 2.816 | 3.04° | 0.224 mm |

Speed is whole-body COM projection in the last two seconds. Qualification
required at least 6 mm/s and course within 5° in each of the final two
two-second windows, with peak tilt at most 10°, depth error at most 1 mm
and positive clearance throughout. The final actor passed 0/4 exact-rest,
0/4 perturbed-rest, and 0/4 moving starts. The fastest held-out moving start
ended at 2.86 mm/s and entered at about 10.29 mm/s. The fastest perturbed rest
ended at 0.12 mm/s, with too little displacement to assign a usable course.
The best final course error in the moving group was 1.60°. The conservative
clearance bound was at least 59.57 mm. Evaluation accepted all 460,800
substeps without faults.

The pilot initialized the pitch output weights and bias to zero and kept them
trainable. All learned targets were held across four 20 ms controls. PPO
scored one Gaussian sample per hold while the original 1.25 ms integrator
executed every substep. This bundled change tests a bounded standalone
propulsion exercise. The results do not isolate exploration cadence,
initialization, or reward as individual causes. [Protocol](../../../tools/pond-flexible-tail-motor/LAUNCH_PROTOCOL.md)

The trainer processed 256 worlds, 196,608 controls, 49,152 policy decisions
and 3,145,728 accepted physics substeps. Trainer wall time was **2.020 s**,
including **1.734 s** GPU rollout, **0.048 s** PPO optimization, and
**0.049 s** checkpoint writing. Capture and the four assessments brought
total wall time to **2.585 s**, excluding compilation. All processes
completed inside their 4-second caps. The exact protocol and file hashes,
learning curve, timing, per-case data and all 13 actor checkpoints are in
`build/flexible-tail/launch-v1`. Original inputs, prior executables and
the body physics hash were verified.

## Working preview.2 reference

The supplied Downloads path was unavailable, so this audit used the exact
repository tag `v0.1.0-preview.2`, verified at
`2524748ecebcd1816e9ed5d0c88921c133f0a11d`.

- The early native motor learned to move from zero-velocity states with
  varied yaw, joint bends and depth. It observed a target in the body frame
  and earned reward for physical reductions in distance. The preview's
  shipping learner started from PPO motor update 64,350 and retained the
  learned chooser. The old body had four active axes and a 0.4 mm buoyancy
  offset for passive righting. [Motor task](https://github.com/life423/CudaRL-Arena/blob/v0.1.0-preview.2/src/envs/swimmer/deep.h#L30-L60) · [Physics](https://github.com/life423/CudaRL-Arena/blob/v0.1.0-preview.2/src/envs/swimmer/deep_physics.h#L144-L148) · [Release learner](https://github.com/life423/CudaRL-Arena/blob/v0.1.0-preview.2/docs/live_feeding.md#L11-L17)
- A motor-only study held targets fixed and measured **4,608/4,608** arrivals
  with the learned motor versus **140/4,608** with zero torque. The separate
  continuous study ran **16 sessions of 30 simulated minutes** with no
  reset or confirmed immobility. Live CPU and CUDA playback kept one body
  advancing as the learned chooser changed its target. [Fixed-target results](https://github.com/life423/CudaRL-Arena/blob/v0.1.0-preview.2/docs/experiments/swimmer/gentle-currents/fixed-target/README.md#L3-L18) · [Endurance results](https://github.com/life423/CudaRL-Arena/blob/v0.1.0-preview.2/docs/creature_reliability.md#L9-L32) · [CUDA playback](https://github.com/life423/CudaRL-Arena/blob/v0.1.0-preview.2/src/demos/pond_playback_cuda.cu#L41-L56)

The transferable sequence is to qualify the motor under its own control,
then qualify persistent target pursuit, then learn to choose new targets.
This pilot stops at the first requirement. The observed regression motivates
measuring sustained yaw strokes and their propulsion from matched rest and
moving starts before another PPO experiment.
