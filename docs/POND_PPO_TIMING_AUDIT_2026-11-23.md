# Native collector timing boundaries — November23,2026

Read-only audit of `tools/pond-ppo/train.cu`, configured source
`be3aaaf642a733c5f7aad312f2256955f28c051228e04d018974fad3be2e8fdb`.
No training, optimizer or physical benchmark is run by this document. The actual
stopped optimized CPU/CUDA gate remains a failed prerequisite; performance
diagnostics cannot substitute for its missing128-control completion.

## What the existing trainer records

| Field | Exact boundary and contents | Limits |
| --- | --- | --- |
| `collection_ms` | CUDA events on the collection stream, after initial state/parameter readback and buffer clearing, until the full rollout's final observation/value/finite checks finish. Includes observation kernels, actor/critic inference, Gaussian sampling, physical controls, final-observation bootstrap, task-reset preparation and fault readbacks between controls. | A stream elapsed span includes GPU idle intervals caused by host launch/readback/orchestration gaps. It is not a sum of isolated kernel execution durations. |
| `physics_ms` | Sum of same-stream event pairs around each `step_kernel` launch. That kernel decodes actions, advances ten5ms substeps, records task/diagnostic state and computes final observations. | Nested inside collection. Includes possible host enqueue gaps between event records. Excludes separate inference, fault-readback and episode-reset kernels. Do not add it to `collection_ms`. |
| `transfer_ms` | Host steady-clock span around `batch()` downloads and the subsequent CPU mean-reward reduction. Downloads allocate pageable host vectors and perform a separate `cudaMemcpyAsync` plus stream synchronization for each array. | Includes allocation, CUDA API overhead, stream waiting and CPU reduction. Does not isolate DMA time and does not include every transfer elsewhere in the update. |
| `optimization_ms` | CUDA event span from before four shuffled PPO epochs until after the final whole-rollout KL audit/readback. Includes actor/critic forward/backward, clipped losses, Adam, finite checks, final KL and host gaps from permutation creation, checks and readbacks. | Host permutation construction and synchronous statistics/fault reads can leave the stream idle inside the span. This is not isolated optimizer-kernel time. The earlier GAE/advantage-statistic work is outside this span. |
| `aggregate_wall_s` | Host elapsed since the native `Resources` object was created, plus the preceding candidate's carried wall charge. Includes context/resource/network initialization, state transfers, collection, checks, GAE, optimization, serialization, evidence/checkpoint writing and reporting inside the run. | Excludes compilation, protocol/prerequisite file validation before `Resources`, and independently budgeted CPU evaluations. Checkpoints conservatively carry a five-second finalization reserve into the next process. |

There is **no separate evidence-write, checksum, checkpoint-transfer or isolated
DMA timer** in the current CSV. Batch serialization/checksum/atomic disk write
occurs after `transfer_ms` and before GAE. Candidate readback/serialization/write
occurs after `optimization_ms`. Their cost is included only in aggregate wall.
The existing fields therefore cannot establish that logging, transfer or device
physics caused a particular fraction of total time.

CUDA-event time and the host wait for that same event are overlapping views of
one interval. They must not be added. A difference between event span and host
wall is not automatically CPU work: host enqueue gaps, pageable copies, driver
overhead and synchronization scheduling also matter.

The CSV row is written only after a whole update/checkpoint completes. Missing
rows do not mean zero attempted physics or Adam. Rejected/partial work is counted
in the stopped result, batch/failure records and ledger.

## What the current executable can and cannot benchmark

The current public modes are describe, zero-integration preflight, gated
candidate initialization and gated training. There is no no-optimizer512-world
collector mode. Initialization prepares geometry and writes candidate0; it does
not advance a control, so it cannot be presented as a dynamics throughput test.
The failed GPU receipt must keep all candidate initialization/training locked.

A host-only adapter could reuse the completed trainer's exact extracted sm120
device module and its existing observation/control kernels, plus the ordinary
native `GpuMlp` inference implementation. This avoids recompiling the physical
CUDA translation unit. Such a test must bind the trainer/module bytes, symbols,
struct strides, recorded worlds/commands and source profile, and verify every
attempt/substep. Its scope is fixed-input collector-kernel replay, not an
on-policy64-sample rollout or a completed physics qualification.

Proposed finite replay: one512-world flat cohort and one512-world bank cohort,
at most one50ms control each, at most1024 attempted controls/10240 substeps,
zero optimizer, no integration warmup or hidden repeats. Device/host preparation
and ordinary inference can be warmed without integration only if every call is
reported. A hard physics/certificate failure stops the test; a later cohort is
then explicitly skipped. Freeze a native wall/evidence cap and all bindings
before any launch. This proposal has not been run.

Report host initialization/serialization/I/O separately. Use CUDA events for
explicit H2D, inference and physical-step stream intervals; report host launch
and wait spans as overlapping measurements. Use a host timer around the actual
disk write. Never add nested/overlapping durations into a manufactured total or
call pageable-copy wall time pure device transfer.
