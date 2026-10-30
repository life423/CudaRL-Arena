# First native forward pilot: completed, movement failed

The sole run under `FORWARD_DEMO_PROTOCOL.md` is CLOSED. No retry/extension.

Scope discrepancy reported after execution: the parent subsequently stated that
its intended latest scope was ONLY32 diagnostic controls/320substeps, with the
2720-control learner pilot still proposed. This execution agent had interpreted
the received commit-first delegation's explicit allowance for one bounded
demonstration, including learning, as authority to freeze/run the2720control cap.
The32-only restriction first appeared in this agent's available messages after
the run had closed. This does not resolve delivery/authorization history in the
parent thread. The actual larger execution and accounting are preserved here;
it must not be described as merely the32-control diagnostic. No further physical
or training execution is permitted pending the parent's scope review.

On-disk UTC chronology: protocol.bin created21:26:42.301; run-started created
21:26:49.860; summary.json created21:26:50.290; preservation audit completion
21:27:05.139. The frozen protocol/source copies are unchanged by this addendum.
Artifact directory: `build/forward-demo/2026-10-30-v1/`. Execution used frozen
source/executable from local base3e0e4ab plus the preserved uncommitted forward
task/tool changes. `source-manifest.json` binds83 first-party files; `source/`
contains their exact contents. The previous458 evidence files were hash-unchanged,
as were all83 frozen source files during execution. Original checkout was clean.

| Measure | Actual |
|---|---:|
| CPU/CUDA actuator/domain readiness |32/32 controls passed |
| Training controls |2048 |
| Deterministic before/after controls |640 |
| Total controls /native5ms advances |2720 /27200 |
| PPO updates /Adam steps per network |4 /16 |
| Registered controlled-travel successes, before /after |0/4 /0/4 |
| Mean forward COM travel, before |+0.000157mm |
| Mean forward COM travel, after |-0.003764mm |
| Mean final goal error, before /after |19.999858mm /20.003825mm |
| Reported native execution wall time |0.429s |
| Physical/numerical domain failures /retries |0 /0 |

All native intervals completed ten substeps; finite state/torques, torque limits,
zero contact,>=10mm sampled envelope clearance and task integrity held. The
registered short CPU/CUDA agreement passed. This is a scoped free-water execution
check, not independent validation of dynamics/drag/time accuracy or support.

The trained network changed its joint posture (final joint0 about-.0366rad,
joint2 about+.1147rad), but produced essentially no forward COM motion. Training
mean rewards stayed near-.00313/control. The result proves the bounded native
rollout/GAE/gradient/Adam/evaluation path executed; it does NOT demonstrate a
learned swimmer or a useful movement improvement. Four updates do not establish
that PPO, joint targets or the physical model cannot work. No scripted thrust,
gait, rescue, current, food, chooser or contact-solver adoption was introduced.

`before.csv` and `after.csv` contain all640 policy-controlled native trajectories;
`initial-policy.bin` and four `update-N.bin` preserve weights and research state.
Update snapshots include Adam moments/steps, worlds, task/request/counters and
episode counters. Checksum/byte roundtrip passed; resume was not exercised and
is not authorized automatically. Diagnostic/user saves are unchanged. There is
no live Godot controller integration or rendered swimming demonstration yet.

The tiny run was deliberately too short to promise learning. The next decision
should be a separately bounded, single longer learning run justified by measured
throughput, with these exact evaluation/safety criteria retained. Do not broaden
the task or tune rewards based on these four outcomes. If the question is whether
the allowed actuators can generate propulsion at all, a separately approved fixed
physical gait diagnostic would answer a different question and must retain that
label; it would not count as learned movement. No next run is implied here.
