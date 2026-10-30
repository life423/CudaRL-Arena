# One compact straight-stroke learner, task3/model6

User explicitly approved learning stroke strength/speed first; steering and
braking/arrival follow later. Mechanical prerequisite passed in
`build/movement-model/2026-10-30-fixed-v2/`. This is ONE separately frozen run
under `build/stroke-motor/2026-10-30-v1/`, not a retry of older learners.

Hard ceiling:32 CPU/CUDA readiness controls +32768 training (64*16*32) +640
before/after evaluation = **33440controls/334400native5ms attempts**. Native wall
stop20s checked before every physical batch and optimizer epoch. No automatic
retry/extension/tuning. Count attempted substeps and reserved failed batches.
Any physical/domain/energy/numerical failure stops before further GAE/Adam.

The intentional oscillator supplies a planar90degree-lag tail stroke. Two
Gaussian latent outputs map through tanh to amplitude0..0.7rad and frequency
.25..1.5Hz. Steering is zero; lag fixed. Critical-damping parameter filters at1Hz
smooth changes; state includes their values/rates and phase. Analytic joint
angle/rate/acceleration references feed the validated model6 internal capped
servo. The policy supplies no body force.42observations: original36physical/task
features plus oscillator sin/cos phase, two parameter values and two rates.
Four physical joint axes remain; pitch references zero in this planar lesson.

Initial actor42-32-32-2, critic42-32-32-1, tanh hidden layers. Orthogonal gains
sqrt2/sqrt2/.01actor or1critic. Actor output biases initialize near amplitude.6rad,
frequency1Hz (latent.895879735/.202732554) before its small random weights. Thus
the untrained baseline already uses an engineered stroke; no gait-discovery claim.
Seed0x53574D32, critic+1, Gaussian sample+2/counter update*32+tick; fixed logstd-2.
GAE gamma.99/lambda.95; PPO clip.2, entropy0, critic coefficient.5, two shuffled
epochs/two256-row minibatches. Adam lr.001, betas.9/.999, epsilon1e-5, clipnorm.5;
maximum256Adam steps per network. Same reset/terminal handling and reward as
forward task1; new wrapper task3 records its distinct action/observation contract.

Four fixed starts, goal20mm ahead at persistent initial heading, terminal10mm/s.
Frozen deterministic initial/final CPU evaluations each up to80controls/4s per
case. Success requires COM crossing20mm with lateral<=2mm, velocity-vector error
<=5mm/s and heading<=10degrees for three consecutive controls. Report success,
episode return, controls/time, work and trajectories before/after; a policy
improvement must be measured against the untrained oscillator policy. Reward
alone is not success. Forward movement may come from the built-in primitive.

Same model6 finite/SPD/torque/joint/10mm envelope domain and per-episode energy
budget checks run at every native step. Save generator state, worlds, task/request,
episode counters, actor/critic and Adam at update boundaries in isolated research
artifacts. Resume not automatically authorized. No current/food/chooser, steering
lesson, production-save/contact adoption, push/merge/deploy or extra run.
