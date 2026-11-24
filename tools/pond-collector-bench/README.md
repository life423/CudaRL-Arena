# Exact collector-kernel replay preparation

Host-only C++ adapter for an extracted, already compiled trainer CUBIN. It links
the existing native GPU MLP library and launches the actual compiled observation
and step kernels through CUDA Driver API. It constructs no optimizer or
candidate checkpoint and does not modify the frozen trainer source.

Build (no CUDA translation unit is compiled):

```powershell
cmake -S tools/pond-collector-bench -B build/pcb29/native
cmake --build build/pcb29/native --config Release --parallel 1
```

`--preflight --protocol <bound-machine-JSON> --out <new-folder>` verifies files,
recorded state layouts and original weights without requesting a GPU context.
Only `--run` launches the separately authorized finite replay. Do not run until
the parent confirms the frozen bindings and execution signal.

See `docs/POND_COLLECTOR_REPLAY_PROTOCOL_2026-11-23.md` for the exact2×512×1
allocation,120second cutoff/in-flight accounting and interpretation limits, and
`docs/POND_PPO_TIMING_AUDIT_2026-11-23.md` for the actual trainer timer boundaries.
