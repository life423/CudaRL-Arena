# Frozen request-interface comparison: stopped at backend gate

The one authorized attempt is closed. **No interface winner was determined:
the CPU/CUDA prerequisite stopped before any of the 88 main trajectories.**
No motor weights, body/water equations, filter, torque limits or thresholds
were changed. PPO updates and optimizer steps are both zero. The saved motor
remains the existing viewer policy; no candidate or interface was adopted.

## Exact recorded stop

Gate case0 uses pose0, original lane-following observations and a zero-offset
request. It is straight swimming, not a turn and not Arm B. Both backends
received bit-identical observations and actual joint commands. At control107
(5.35 simulated seconds per backend), one endpoint field crossed the fixed
matched-command tolerance:

| Measurement | Recorded value |
| --- | ---: |
| CPU global-X head angular velocity | -0.24834023416042328 rad/s |
| CUDA global-X head angular velocity | -0.248317152261734 rad/s |
| Absolute difference | 0.00002308189868927002 rad/s |
| Registered absolute limit | 0.000020 rad/s |
| Relative difference to CPU magnitude | about0.009294 percent |
| Maximum position difference | about0.558 micrometres; within its own limit |

Endpoint106 is within every body-component limit. All other endpoint107 body
components remain within their limits. Neither backend reports a native step,
energy, bank, depth or tilt failure. The hard stop is the cross-backend
comparison guard. Offline binary32 analysis finds the first substep limit
crossing at control107/substep2; the native parity check is at control endpoints.
All substep states were retained. The previous shorter80-control gate would
not have exposed this later discrepancy.

This does not establish that either request interface fails, that the motor
requires retraining, or that the body cannot turn. Those questions remain open.

## Read-only arithmetic diagnosis

The executed binary contains a fused CUDA instruction for the buoyancy
expression density*volume minus stored mass, followed by multiplication by
gravity. Source model6 assigns mass as volume*1000. Separate CPU rounding can
cancel the neutral-buoyancy expression exactly; fused GPU arithmetic can leave
a tiny residual. Read-only cuobjdump inspection confirms the executed FFMA
and subsequent gravity FMUL at instruction offsets0xa0d0/0xa0e0.

The recorded CPU external-work ledger is exactly zero through107 controls.
CUDA external work is -2.9358566016550225e-15 J at control1 and
-9.794027754404255e-14 J at107. This is a concrete arithmetic-consistency
difference. Filter/reference and other floating-point paths also show small
differences; buoyancy FMA is **not proven the sole cause** of the angular-rate
threshold crossing. No equation, compiler flag or tolerance was changed to
make the gate pass. No physical replay or repeat was performed.

## Actual allocation and preservation

| Work | Actual |
| --- | ---: |
| CPU physical controls | 107 |
| CUDA physical controls | 107 |
| Total physical controls | 214 |
| Attempted and accepted5ms substeps | 2140 / 2140 |
| Gate cases completed | 0 of4 |
| Main interface trajectories attempted | 0 of88 |
| Native process seconds | 0.391203 |
| External whole-process receipt seconds | 0.4607581 |
| Watchdog termination | false |
| PPO / Adam steps | 0 / 0 |

The final protocol cap was35200 controls/352000 substeps,240 process seconds
and512MiB evidence. Remaining allocation is not reused. Source/build and
zero-integration preparation ran outside the physical-process timer; offline
audits ran no additional native steps, CUDA kernels or optimization.

Independent saved-data verification passed as **verified_hard_stop**, not
successful physics parity or a completed behavioral comparison. It audited
all214 controls/2140 accepted states, observations, actual actions, filter/
torque records, energy identities, domain guards, approval/source bindings
and actual accounting. Separate bound binary32 forensics verified the exact
first mismatch. No force integration is replayed by these readers.

Before/after SHA-256 verification passed for **10183 original protected files,
62957731755 bytes**. The saved update336 checkpoint remains
810DBD593CE39475CDD6E95B709BEF3FDB34D50C9FCBBB2C345C7A8882A0BEF0.
The holding source and all old experiments are unchanged. Pure preparation
verified88 initial/tilted observer fixtures,520 scheduled events per backend
and12 guard fixtures, with zero dynamics or optimizer work.

## Evidence and verification identities

Run root: build/ppo/frozen-interface-comparison-2026-11-23-v1.

- [Portable exact-byte evidence archive](../assets/research-evidence/frozen-interface-comparison-2026-11-23-v1/evidence.zip)
  contains 160 files, 5,346,731 compressed bytes; archive SHA-256
  2af9f92a0b9597e2e2c006499fa428d3eb10f70d5d7f09e5c7dd5650e7f08dc0.
  Its sibling manifest binds each archived file. Raw originals remain in place.
- [Preregistered protocol](FROZEN_INTERFACE_COMPARISON_PROTOCOL_2026-11-23.md).
- Native result SHA-256: 17920b9acca3d854bf93ee08ad36623a794b751ef212497d781c86af4e549bb8.
- Executed EXE SHA-256: ac0ace79c217fd521fe541fc9c85c57cb49dc3c96cca4e4d1b57555d1623f816.
- Frozen122-input manifest SHA-256: 19704311d38dfd630d7c86bb97fb82cbe795f570088d916b165533fbf4c08d61.
- Preparation proof SHA-256: 519d6724961128e8b8b083bea23791550aaca60f128b6fe85083a9c836162604.
- Full stopped-data proof SHA-256: b89f33803aff753d3f80386f7ffe94abb0e5ed78d56414ca3f3445ba00bcffe6.
- Frozen offline reader SHA-256: 22b8779e7b3d2fe0d19c080e6ef684085b16f18ab19dd116ba5ffc878ec1aab6.
- Binary32 forensic proof SHA-256: 8acceb266cddd26bfcc95f4d99f7e056a77f0c12a143466ffcbf3041ad508509.
- Executed-binary arithmetic note SHA-256: 84dfd46e81d68245fedc650d14751d8290804d6b2080a6ee1ec729080d764a58.
- Protected manifest SHA-256: 7b2f0ba64b2769fd6802bac4a117123f2213ab56f564a6e26769427d04e5ba5d.

The first offline forensic JSON attempt had a NumPy-boolean serialization
failure. Its source/partial output and note were preserved; only JSON boolean
conversion was corrected in a separate reader. The frozen native executable,
original result auditor, tolerances and actual physical attempt were unchanged.

## Product disclosure and next decision

Godot Details receives exact byte-bound copies of the native result and
independent proof. It must state that the CPU/CUDA prerequisite stopped,
0/88 interface tests ran and there is no winner. Do not show zero successes
out of44 as if each arm had actually been evaluated. Preserve the old actor,
natural pond, camera/zoom and truthful frozen-inference labels.
[UI verification](FROZEN_INTERFACE_COMPARISON_UI_REVIEW_2026-11-23.md) passed
34 parser and 60 rendered assertions across four sizes, all with zero
additional physical/optimizer work. Root inspected desktop/portrait artifacts
and the actual Godot 4.4.1 worktree app paused with the new Details. Godot
Manager still registers the correct live project; no Play action was sent.

The next engineering decision is backend arithmetic agreement on this
unchanged straight case. A numerical mismatch cannot be repaired by motor
PPO. Resolve its cause/required contract before proposing another bounded
comparison; do not silently relax tolerances or reuse the stopped attempt.
No further physical run, retraining, adoption, push or merge follows from
this result. The combined plan remains the intended route after this obstacle.
