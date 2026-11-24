# Fixed-input native collector-kernel timing — November23,2026

Preparation only; wait for explicit execution signal. This diagnostic does not
admit candidate initialization, PPO or autonomous pond navigation. The optimized
CPU/CUDA qualification stopped at15/128 controls per case and remains failed.
No successful prerequisite, candidate0 or optimizer update is manufactured here.

## Exactly one finite attempt

Protocol ID `pond-collector-replay-v1-2026-11-23`. Two disjoint cohorts:
512 recorded flat-world copies and512 recorded bank-world copies, one50ms
control per world. Maximum1024 attempted controls/10240 nominal5ms substeps,
zero optimizer calls, zero integration warmups and zero repeats. Native run wall
cap120seconds, evidence cap64MiB. No new launch with less than10seconds remaining.
An already launched kernel is allowed to complete so its real work can be read
back; actual in-flight wall overrun is reported. Never hard-kill a kernel merely
to claim the deadline was met. A hard physics/certificate fault stops later work.
The once-only benchmark ledger is separate from training/evaluation ledgers.

Bind the final trainer EXE, extracted sm120 CUBIN and function symbols, linked
native MLP static library, host benchmark EXE/source identity, this protocol,
original saved motor, recorded frames, recorded command CSV and ABI descriptor
before launch. Record the official `cuobjdump` extraction/listing evidence.
This host-only adapter never recompiles or duplicates the physical CUDA kernel.
It launches the completed trainer's actual `observe_kernel` and `step_kernel`.

## Recorded states and inputs

Use the CUDA-frame half of `build/pw29/gpu/frames/step-0001.bin`: cases0..3 are
the recorded straight/flat starts, cases4..7 are recorded bank-impact starts.
Each of the four states is copied128times in its cohort. Preserve the entire
recorded World, including body, servo, filters, contact/energy history and
accepted control/substep counters1/10. After one accepted replay control these
must be2/20; actual counter deltas determine the diagnostic work count.

Both cohorts use the same four exactly-zero commands recorded for bank cases4..7
at control2 in the existing command CSV. This isolates the physical geometry
cost. Zero latent maps bit-exactly to those zero commands in the trainer decoder.
Nonzero recorded CPU commands are not silently approximated through an inverse
decoder. Benchmark Task requests are constructed from the recorded first PRE
COM/heading with straight-swim intent and control count1. This does not move,
prepare or reset the recorded physical world.

Ordinary saved-weight `GpuMlp` actor and auxiliary-critic forwards are timed on
the generated observations. Their outputs do **not** choose this replay's fixed
commands. There is no Gaussian sampling, reset, final-value forward, GAE,
backward pass or Adam. This measures the actual observation/inference/control
entrypoints under fixed inputs; it is not a complete on-policy64-sample collector
or an estimate of32-update completion time.

## Timing and evidence

Run initialization is explicit: context, stream/events, device buffers, extracted
module and two native MLP constructors, plus original actor upload. No candidate
or optimizer object is constructed, and the training allocation stays locked.
CPU-only preflight verifies SHA/ABI/recorded data without requesting a GPU
context or advancing physics. Binding validation is outside the native run timer.

For H2D/accumulator clearing, observation, fault readback, ordinary forwards,
physical control and final D2H, record confirmed same-stream CUDA-event intervals
and host enqueue/wait/wall spans. CUDA-event intervals include any stream idle
caused by host enqueue gaps; they are not hardware utilization or isolated DMA
claims. Unconfirmed intervals are null. CPU fixture copying, payload creation
and checksum/atomic write have separate host timers. Event time and host wait
overlap and must never be added. The whole-run wall observation is made directly
before the small final JSON write; do not pretend every unclassified host cost
has been isolated by the stage buckets.

Save every launched cohort's returned worlds/tasks, per-world attempts/substeps,
task/physics failure reasons, rewards and traces. Preserve all actual work if
the deadline or a gate stops the diagnostic. If CUDA readback genuinely fails,
report accounting unknown with512-control/5120-substep upper bounds per launched
cohort rather than reporting zero work. A partial or slow benchmark is final for
this allocation. None of its outcomes establishes physics parity, turning skill,
retention or training success.
