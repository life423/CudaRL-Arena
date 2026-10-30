# Paired direct-torque / fast joint-target motor experiment

Experiment `paired-motor/2026-10-29-v1`; new versioned paired protocol,
independent of the old128-update command-motor format. The October29 user request
authorizes this experiment only. The production tadpole, pond, body, actuator
limits, old checkpoints and previous failed results remain unchanged. No merge,
deployment, growth, larger pond, chooser training or budget extension.

The [two-policy request contract](CREATURE_CONTROLLER_INTERFACE.md) is documented
separately. Fixtures issue persistent position/terminal-velocity/heading commands;
there is no food, reward motivation selector or learned chooser in this study.

## One experimental change

| Direct-torque arm | Joint-target arm |
|---|---|
| Four Gaussian neural outputs at50ms; clamp each to[-1,1] and multiply by10µN·m | Four Gaussian neural outputs at50ms; clamp to[-1,1] and multiply by existing joint limit1rad |
| Same commanded torques over ten unchanged5ms physical substeps | Hold targets50ms; recompute `tau=clip(Kp*(target-angle)-Kd*joint_rate,±10µN·m)` before each unchanged5ms substep |

Use a new wrapper around `deep::substep`; do not edit the shared physical
equations, masses, contacts, buoyancy, drag, soft stops or actuator limits.
Only internal joint torques are applied. No whole-body position/heading forces,
scripted gait/stop/rescue or position resets during a trial. A zero joint target
is active regulation, not zero torque/passive coasting.

There is no additional rate limiter in this version: bounded targets use a
zero-order50ms hold. Held targets, last actual torques, effort telemetry and
all physical/random/optimizer state must be saved/restored. CPU, CUDA and live
viewer use the same target clipping and servo calculation.

PPO stores the **raw sampled neural Gaussian actions** and evaluates their
likelihoods in that same latent action space. For the joint-target arm these are
normalized target-action latents; clipped physical targets are a deterministic
actuator transformation. Never compute likelihoods from the resulting torques or
clipped targets; no change-of-variable Jacobian is required when the policy is
explicitly defined/evaluated on the stored latent actions. Record both latent
commands and applied targets/torques to expose saturation.

Both arms retain59 physical observations and the meaning of columns29–32:
previous **actual normalized applied torque at the last5ms substep**. They never
become previous joint targets. Compatible transferred feature columns9–32 thus
retain their semantics. The two alternatives start with identical compatible
feature weights, small fresh output heads, critic/Adam/exploration initialization
and counter-RNG keys. Their physical exploration differs by design; record it.

Keep the existing position/velocity/heading/whole-body objective. Compute its
effort term from the mean of actual normalized torque-squared over all ten
substeps and four joints; do not charge joint target-squared as torque. The
unchanged substep also accumulates actual positive mechanical work.

## Preregistered gain derivation and regulation readiness

Exactly one gain proposal: for each joint use the minimum
`Ieffective=1/(M_inverse[j,j])` over the fixed eight initial poses, using the
coupled10×10 articulated inertia matrix. Fix natural frequency2Hz and damping
ratio1; derive `Kp=Ieffective*(2*pi*2)^2`, `Kd=2*Ieffective*(2*pi*2)`.
This is a local gain-design approximation, not proof of critical damping in
coupled water dynamics. Save the pose catalog, computed inertia/gains, exact
coefficients, source hashes and limits in CPU-only preparation before execution.
No gain change or sweep after inspecting results.

Eight fixed joint-regulation cases, ten seconds each on CPU and CUDA:
four individual+.2rad target steps from rest, two bent stationary holds, and
the two preserved actual-swimming captures (reference fixture4/5) requesting
their captured joint angles clipped to the unchanged target limits. Preserve
complete starting body/momentum state; no recapture or velocity injection.

**Hard readiness budget:8×200 controls×2 backends=3,200 control steps.**
All joint errors must reach≤.02rad with joint rates≤.05rad/s by5s, and every
joint must satisfy both continuously over the fixed final two seconds
(t8.00–10.00, including endpoints). All states/substeps must be finite, within
physical clearance and torque limits. Record whole-body reaction/overshoot/work
without mislabelling joint regulation as body stopping, swimming or holding.
CPU/CUDA parity uses the prior registered physical float comparison
`2e-4+2e-3*abs(reference)`; it does not relax the joint/safety gates.

Reuse these recorded states/actions for focused observation, actual-effort,
raw-action likelihood replay and full-state restoration checks. GPU inference
and fresh optimizer snapshot checks add no physical steps or PPO updates.
The completed independent numerical audit is preserved, not reopened.

## Conditional paired PPO and evaluation budgets

Only if regulation and focused data readiness pass:

- **64updates per arm×512worlds×64controls**,2,097,152 training steps each,
  **4,194,304 total**. No different seed, early best selection, gain sweep,
  optimizer/reward variant, automatic extension or extra update for testing.
- Use the registered initial head/training keys `0x4D072601/0x4D072602`, final
  fixture seed `0x4D072604`, same complete initial worlds in both arms, PPO
  gamma.99/GAE.95/ratio.2/entropy.002/value.5,4epochs×8minibatches, initial
  logstd−.5 and fresh Adam lr.001/beta.9,.999/eps1e-5/normclip.5.
- Each arm ends at update64, **2,048 actor/critic/std Adam steps**; total
  learning budget across arms is128updates/4,096 steps per optimizer type,
  not128updates per arm. New labels, schemas and hashes bind interface/gains,
  request/action/observation semantics, source identity and paired budgets.
- Register identical64 final complete fixtures before learning. No mid-run
  development evaluation/tuning. Evaluate the two fixed final policies and the
  preserved swimmer once: **≤115,200 control steps** including budgeted fixture
  captures/warmups. Reuse actual-motion capture starts across policies.
- Preserve separate `torque/` and `joint_target/` checkpoints, actions and
  histories. Refuse incompatible old128-update saves; do not overwrite them.

Reject invalid or physically unsafe collected training data before GAE/Adam;
retain first failing substep/state and the last good full save. Do not turn a
safety breach into a reward/reset shortcut. On an arm/readiness failure, stop
under the declared rules and report actual updates/steps, not the maximum cap.

Final movement standards remain unchanged: purposeful3D travel,90° turning,
braking from actual1–3cm/s to≤2mm/s for.5s within3s/72mm, position≤7.2mm,
heading≤10°, and10s quiet whole-body holding with the previously registered
head/COM/surface/joint/angular/drift/safety thresholds. Report each skill and
failure condition separately, not just total reward. Acceptance requires every
final case and zero safety/nonfinite failures. One paired seed is bounded
evidence, not a universal superiority claim.

## Honest live comparison and stopping

The existing motor bench must decode the new paired format, interface and gain
set. It may show actual trained final or explicitly failed/partial candidates
in live physics with frozen weights, accurate64-update labels and distinct
neural target/applied torque units. It must not label old frozen references or
update0 initializations as the trained replacement. If readiness fails, show
the failure and **not trained**, without substituting an old actor.

No candidate replaces the production swimmer in this experiment, including an
accepted candidate. On any gate failure or cap exhaustion, retain the full run,
finish a bounded report and state the narrow next decision. No further sweep.
