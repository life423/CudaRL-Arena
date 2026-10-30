# Bounded braking probe — completed October 29, 2026

**Research readiness failed. No learned-motor PPO pilot ran.** The registered
search spent61,440 control steps and its separate final replay spent1,600.
Neither budget was extended. No failed result was rewritten, final movement
criterion relaxed, or controller deployed. See [the preregistration](MOTOR_BRAKING_PROBE.md).

The conventional controller's0/16 result is historical evidence, not a research
admission requirement. The newly authorized two-capture test was the proposed
alternative; **that alternative did not pass either**. The old false reference
gate remains unchanged. No passed research gate was generated.

| Exact existing capture | Entry speed | Selected commands | Stop dwell completed | Path to stop | Safe for20s | Quiet10–20s |
|---|---:|---|---:|---:|---|---|
| Fixture4, slower | 11.03mm/s | All48 values exactly zero | 9.35s | 52.83mm | Yes | **No** |
| Fixture5, faster | 28.29mm/s | All48 values exactly zero | 11.35s | 77.98mm | Yes | **No** |

The selected sequences equal passive coasting, so neither improves it. Both
also fail the unchanged3s braking gate and original7.2mm position/heading hold.
The slower replay's minimum clearance is2.696mm; the faster's38.180mm. Safety
was not the failure in these replays. Whole-body quietness and pose were.

During the fixed10–20s window, maximum head speeds were1.245/2.263mm/s versus
the.5mm/s limit, maximum surface-point speeds2.028/2.357mm/s versus1mm/s,
anchor drifts7.23/12.09mm versus3.6mm, and original heading errors73.51°/19.48°
versus10°. Joint-rate limits alone passed; a quiet joint does not guarantee a
stationary, correctly oriented body with residual momentum.

## Useful partial evidence, without an extra replay

All512 candidates were finite and safe over their six-second search horizon.
There were496 nonzero candidates plus16 counted zero/incumbent copies. Under
the fixed physical cost, zero stayed best in every round:171.4811578 for the
slow capture and536.4141023 for the fast capture. Costs/torque scales were not
retuned after this result.

Offline analysis of the existing traces found **three nonzero slow-start
sequences** that reached the exact.5s head-and-COM low-speed dwell, completing
at5.10,5.75 and5.95s. None of the248 nonzero fast-start sequences did so within
six seconds. The earliest slow candidate completed its dwell with head.694mm/s,
COM1.555mm/s, tail39.09mm/s, joint rate2.37rad/s, body omega.489rad/s and
heading error30.7°. That is translational slowing accompanied by a moving tail,
not whole-body settling. These three were not selected by the registered cost
and **were not given an additional20s replay**. Whether one would later hold
after a different command schedule remains unknown; do not infer its success
or failure outside its recorded six seconds.

This is limited positive evidence that allowed torques can affect braking from
one start. It is not evidence of a reliable feedback controller, fast-start
braking, ten-second holding, final-envelope feasibility, or uncontrollability.
A coarse48-parameter/.5s-knot/four-round search can miss useful torque sequences.

## Stream and actual learner data path

The current production upload was already correctly ordered on its supplied
stream. We retained that explicit same-stream dependency and documented it;
there was no absent race to claim as a newly repaired numerical defect.

The focused stream test passes on an actual nonblocking CUDA stream with73
worlds, registered pinned host data, three delayed producer uploads and exact
uploaded-state/observation checks. A deliberately observe-before-upload local
control demonstrates that stale observations are rejected. Actual production
save→restore→step state, observations, outputs and checkpoint continuation are
bit-identical.

Two initial **test** assumptions failed and remain recorded. A GPU-concurrency
negative control did not work under this Windows scheduling environment; its
replacement tests deliberate misordering deterministically. A separately
compiled test integrator was not bit-identical to the runtime-parameter kernel:
maximum state difference5.72e-6, observation8.94e-7, reward3.35e-8. Independent
kernel parity uses the already registered physical float criterion
`2e-4+2e-3*abs(reference)`. Exact production continuation was never relaxed.
These were not evidence of stale production uploads or a changed physical model.

The final scratch readiness check passes:8×64 transitions, preserved swimming
captures, CPU/GPU observations and actor/critic inference, same-row latent log
probabilities, one-step physics/reward, finite-horizon terminal/reset flags,
handwritten double GAE, minibatch gathering/normalization and a complete512-world
fresh-optimizer save/restore. Maximum latent-logp discrepancy1.26e-6 and GAE
discrepancy1.02e-6. **No command-motor backward call, Adam step or PPO update ran.**

That check caught and repaired a local safety-accounting defect: checking only
the post-step body could lose an illegal starting pose when existing containment
physics moved it back inside. The batch now retains the exact first unsafe
pre-state as well as detecting post-step breaches. No force, contact equation,
action, reward, safety threshold or reset rule was altered to fix the accounting.
The final negative test verifies the exact step0 failing state and invokes the
production rejection before GAE/Adam, without consuming its null data buffers.

The new command-task protocol also resolves its finite-horizon mismatch:
600 ticks is terminal (`done=1,truncated=0`) because remaining time is observed.
Successful task snapshots persist/check safety counters and recomputed physical
observations. Command-only compile-time checks reject unsafe/invalid rollouts
before GAE/Adam, and nonfinite data/gradients before consumption. Rejected-rollout
diagnostics preserve the last good complete checkpoint rather than overwrite it.
The full128-update training/abort lifecycle remains **unexecuted**, since the
research probe failed. Four existing PPO resume/integrity regression tests pass.

## Preserved evidence

All new evidence is under `build/motor-braking/2026-10-29-v1/`:

- `probe/protocol.json`, copied complete starts and executable/source hashes;
- all eight candidate-sequence populations, costs, physical CSV/full-state trace
  and immutable selected sequences;
- `probe/research-readiness.json` with separate research and unchanged final gates;
- `probe/offline-analysis.json`, which performs no new simulator/policy steps;
- all failed and final stream/readiness logs, final scratch rollout/full save,
  four-test legacy regression log and preservation manifests.

Original numerical results, source policies, completed studies, personal saves,
working pond sources and unrelated overhang work remain preserved. Nothing was
committed, pushed, merged, installed or adopted by this probe. The final motor
acceptance suite and its withheld cases were not used for search or evaluation.

Final preservation classification: **221 protected artifacts are byte-identical;
one personal boulder save was legitimately updated when the user closed the old
pond** (user confirmed; write time19:06:33). Its current bytes were copied to
`personal-boulder-observed-after.bin` and left in place, without a restore or
overwrite. The static tool-path audit and project log support a separate live
boulder session; no exact changed training counters are inferred. The save write
preceded both the19:08 probe and19:12 legacy regression. Frozen research inputs,
numerical evidence, reference results and source policies are all unchanged.

## One recommendation to discuss

Compare **joint-target actions with measured-rate damping in a fast physical
joint servo** against direct torque, keeping the same four-output PPO motor,
articulated physics and torque caps. First validate that inner joint-regulation
loop at the existing5ms physical step; do not assume arbitrary PD gains work.
It should turn requested joint configurations into bounded torques, not choose
food, script swimming, rescue a stall or teleport the creature.

This targets the observed gap between slowing the head and settling the whole
tail/body. It is a controller-interface hypothesis, **not a proven cure or a
capacity verdict**. The failed conventional PD/gait did not establish a stable
inner loop, and this coarse search did not prove direct-torque learning cannot
work. No servo redesign, additional network, new sweep or stopping-standard
change has been implemented or run. Discuss one bounded interface comparison
before spending another research budget.
