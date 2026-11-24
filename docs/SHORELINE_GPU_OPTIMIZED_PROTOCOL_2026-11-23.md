# Optimized GPU readiness gate — November23,2026

Preparation. CPU runtimev3 is QUALIFIED:36/36cases,68000controls/688000substeps,
all four outward swims and unchanged retention/timestep criteria. No further
CPU shoreline study is needed for an unchanged physics header.

The first v3 CUDA diagnostic is CLOSED at its180s wall cap:2controls per case,
16CPU+16CUDAcontrols,160CPU+160CUDAphysicalsubsteps,2dynamicCUDAkernels,
185.346119s including the in-flight kernel finishing. Readback is complete.
No arithmetic mismatch was found in those completed controls; angular-rate
difference≤1.188e−13. It ran the deliberately unoptimized `--Ofast-compile=max`
diagnostic, which traded a6.56s build for unusable GPU kernel latency. Preserve
all source/binaries/frames/results at `build/pw28/gpu-v3/` and the sealed bindings.
That gate did NOT pass and its unused allocation is not reused.

The user's latest instruction authorizes focused optimization of measured
bottlenecks to finish validation and start native PPO. This NEW performance
correction removes only `--Ofast-compile=max`, restoring default optimized CUDA
compilation. Keep `--fmad=false`, validated original CPU cbrtf results, all math
equations and exact original CPU/CUDA tolerances. There is no fast math, changed
force, suppressed physical term, lower accuracy or enlarged runtime limit.
No shared physics source changes; bind the qualified header SHA256
2b2445eea27e5a00007ecdc5b75f504d9d6fbd3d5f165e3a54e8bece4416b28b.

One fresh optimized fixed-input gate:8cases×128continuedcontrols EACHbackend,
at most2048controls/20480substeps total,180native seconds and2GiB evidence,
zero optimizer. Use the same four straight and four bank cases, every5ms trace,
commands, criteria and first-failure capture as v3. Static preflight first;
seal compiler/source/executable/input hashes before dynamics. Identity is
`shoreline-product-v3-optimized-2026-11-23`; its CPU prerequisite remains the
completed `shoreline-product-v3-2026-11-23` result. Do not silently relaunch a
failed gate or claim that the two earlier matching controls qualified CUDA.

Measure actual kernel/whole-gate throughput after compilation; do not confuse
compiler time, GPU busy percentage or launch count with useful simulation work.
The same precise optimized physics profile will be used in the native candidate
trainer; optimized native NN/PPO libraries remain unchanged. Training admission
still requires the completed source-matched CPU AND optimized CUDA receipts.
The user's32-update candidate budget/checkpoint/evaluation rules remain as in
`POND_TURNING_PPO_PROTOCOL_2026-11-23.md`.
