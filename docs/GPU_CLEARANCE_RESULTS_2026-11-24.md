# GPU clearance and turning results — November 24, 2026

Original motor 336 and its holding gain remain protected. Two separate native
CUDA PPO residual arms completed 32 updates each. Each collected 1,048,576
transitions and finished at 2/16 fixed turns against the original baseline's
3/16. Neither reached the declared improvement trigger. The visible swimmer
uses the original saved policy.

Straight and Rest bypass both residual heads. The evaluator verified exact
original coefficients and inactive residual behavior, then reused the archived
four short swims, 28 required long-swim windows and 14 rests. These arms ran
zero new retention controls. Their 16 turns were physically evaluated at each
eight-update checkpoint. The two completed allocations remain closed.
[Yaw-arm evidence](../assets/research-evidence/pond-residual-yaw-v1/evidence-manifest.json) · [Four-axis evidence](../assets/research-evidence/pond-residual-four-v1/evidence-manifest.json)

## Conservative contact rejection

The new opt-in path proves separation for each substep using the full body
envelopes, terrain curvature and gradient bounds, surface/wet-domain checks,
and root/joint motion bounds. The proof covers both the previous velocity and
the free-solved velocity. Uncertain cases use the existing full geometry and
contact solve. Physical equations, torque/filter limits and energy gates are
preserved. The exact query retains its 10 µm interval requirement.

Threshold-only collection reports a conservative clearance lower bound.
Unmeasured numerical minima carry an explicit validity tag and NaN. Full
quantitative inspection remains available for qualification and measurements
that require it.

The matched benchmark passed in 2.44067 seconds, including setup and readback.
It executed 2,056 GPU controls and eight CPU controls, totaling 20,640
physical substeps, with zero optimizer calls.

| Matched starting states | Worlds | Exact GPU time | Swept GPU time | Speed ratio |
| --- | ---: | ---: | ---: | ---: |
| Earlier batch 8 | 512 | 11.8413 ms | 14.7287 ms | 0.80× |
| Late batch 12 | 512 | 149.9369 ms | 2.0219 ms | 74.16× |
| Bank contact | 4 | 89.7159 ms | 90.7172 ms | 0.99× |

The comparison checked actual body motion, actuator requests, contact forces,
energy and conservative clearance classifications on CPU/CUDA. The late
control's speedup does not establish full PPO training throughput.
[Benchmark record](../assets/research-evidence/pond-broadphase-dynamics-v1/result.json)

## Full collector

The first attempt to check the complete 512-world, 64-control collection path
hit its four-second wall-clock cap before issuing its first physical control.
The watchdog recorded zero issued controls, zero completed controls/substeps
and zero optimizer calls. That allocation ended at the timeout.
[Timeout record](../assets/research-evidence/pond-collection-replay-v1/timeout.json)

The follow-up reads the frozen archive once, loads saved weights directly into
512-row inference networks, and skips 38 GPU allocations for training buffers.
It finished startup and uploads in 192.539 ms. The single v2 replay still hit
its four-second deadline inside first-control preparation, with zero controls
issued and zero optimizer updates. The original and follow-up allocations are
closed. [Timed replay](../assets/research-evidence/pond-collection-replay-v2/timeout.json)

A separate Driver API probe created a cold CUDA context and four events in
120.155 ms, then hit four seconds within `cuModuleLoad` for the exact extracted
36,119,504-byte observation module. It launched zero kernels. The same module
contains the physical step, reset, and initialization kernels, with 10.01 MB,
8.98 MB, and 8.98 MB of native instructions respectively. Native `sm_120`
code is present. The successful dynamics benchmark's swept step sits in a
13.09 MB module. Its 4.45 MB module belongs to the older baseline. Inspecting
the replay image revealed three inlined copies of the per-link search, totaling
25.67 MB. The shared implementation gives the sampler, per-link search, and
whole-body inspection one device definition each. The qualified arithmetic
flags and the exact search bodies are preserved.
[Module probe](../assets/research-evidence/pond-collection-replay-v2/module-probe.json)

## Shared device inspection qualification

The linked CUDA image fell from 36.12 MB to 14.15 MB (61% smaller). The replay
and residual trainer use the same separately compiled terrain sampler, link
search, and whole-body inspection. The sampler retains its no-inline
annotation. Each gate used a separate CUDA process, in the order below.
The module probe ran first, so driver and filesystem caches may have been
warmer for the later runs.

| Gate | Process time | Result |
| --- | ---: | --- |
| Load the four trainer kernels | 2.101 s | Passed, zero kernel launches |
| Matched CPU/CUDA dynamics | 2.492 s | Passed, 2,064 controls and 37 force-positive GPU bank substeps |
| Full saved 512-world × 64-control rollout | 0.418 s | Passed, 32,768 controls and 327,680 substeps |

The dynamics comparison found zero failed CPU/CUDA comparisons. The exact
and accelerated GPU paths agreed on motion and contact force, including all
37 bank-contact substeps. The late 512-world control took 152.22 ms with full
clearance and 2.08 ms with the shared accelerated build.

In the full replay, GPU physics and task work took 146.60 ms versus 12,354.26
ms in the matching archived collection. The rollout phase delivered 148,175
transitions/s. All compared observations, actions, probabilities, values,
rewards, movement metrics and accounting matched the archived reference.
All full-replay substeps were certified clear. The separate dynamics gate
exercised real bank contact with the exact fallback. Neither test ran an
optimizer. These results qualify this collector path without establishing
complete PPO training throughput or turning performance.

[Module load](../assets/research-evidence/pond-collection-replay-v3-module/result.json) ·
[Physics parity](../assets/research-evidence/pond-broadphase-dynamics-v2/result.json) ·
[Full replay](../assets/research-evidence/pond-collection-replay-v3/result.json)

## Build and integration checks

The current source passed 75 CPU foundation tests, 132 evaluator pure checks
and 43,405 shared-geometry assertions. Each native check completed within the
four-second execution cap. Godot's packaged library, saved motor and evidence
bindings passed their hash check. Hosted CI must complete for the exact pushed
commit before integration is verified.

Optional historical comparison tools require their preserved local snapshots
and hash-bound libraries. Restore those inputs to reproduce the old studies.
Normal CPU CI builds independently of those research artifacts.
