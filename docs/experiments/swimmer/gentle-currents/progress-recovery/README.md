# First progress-aware PPO pilot — October 27, 2026

The user approved working toward reliable swimming in 0–1 cm/s steady currents. This bounded first experiment adapts food selection/recovery while freezing the original swimming motor. It does not claim to solve arbitrary water or change the Godot pond automatically. The user also asked about GPU utilization; a short batch-throughput measurement precedes the pilot.

## Protocol fixed before candidate training

- Source: `../frozen-policy.bin`, SHA256 `D9730ED4FB2A05904E09ACAB45C43BE47FF7B80B768AA5379C463A11B8D9611D`. The original packaged policy and full training saves remain untouched.
- Selector features grow from 32 to 34: normalized time since food (saturating at 20 seconds), and recent distance progress toward the candidate food over one decision period. Raw critic observations grow from 36 to 40. Progress history resets for a respawned food, preventing a relocated food from pretending to be approach progress. No flow sensor, prescribed target switch or inference rescue is introduced.
- Transfer all existing selector weights and biases. Initialize the two added input columns to zero; CPU scores match the prior function within `1e-6` in the regression test. Use the exact frozen motor. The candidate critic, optimizer and worlds begin fresh for the new architecture; this does not reset the preserved original learner. Full candidate saves retain actor/critic, Adam, currents, distance history, RNG counters and worlds.
- Native categorical PPO uses the existing GAE/loss/backprop/Adam kernels: 32 decisions per rollout, four epochs, eight minibatches, LR 0.0003, gamma 0.99, lambda 0.95, clip 0.2, entropy 0.01, value coefficient 0.5, existing gradient cap. Decisions remain one second; physical control is 50 ms with 5-ms substeps.
- Training worlds: 25% calm; otherwise uniform current speed between 0 and 1 cm/s and a uniform horizontal direction, held constant within an episode. Half of starts use independently sampled joint poses, half retain normal resting starts. Food collection supplies the reward; a 20-second food drought with low ground speed gets the existing −5 terminal penalty and a training-only reset. Time limits are five minutes. Physical forces, contacts, food and motor equations are unchanged.
- Short throughput test: one warm-up update and three timed updates at each of 1,024, 4,096 and 8,192 worlds. These are discarded candidate copies. Measured rates are approximately 2.16M, 4.61M and 6.27M physical simulation steps/s, including rollout and PPO optimization but excluding validation/rendering/startup. Use 8,192 worlds for the pilot; it is the best tested batch, not a proven global optimum.
- Two candidate seeds, **71 and 109**, each **128 updates**, then stop. Save candidates at 0/32/64/96/128 and full state every evaluation and on exit. No automatic continuation or replacement of the deployed controller.
- Development selection suite: 64 one-minute worlds, seed `0xA7D13101`, in calm, +X 1 cm/s and −X/−Z 1 cm/s currents. Rank mean collection across these three conditions, requiring calm collection at least 98% of that seed's transferred update-zero score. Also require recovery within 20 seconds from the already-known 60-second stall fixture. The recorded stall is a **development/regression case**, not an untouched holdout, and never enters gradients. Test all saved checkpoints on this gate before freezing one winner; ties favor lower seed, then earlier update.
- Fresh paired evaluation after selection: two layout seeds `0xD311B101`/`0xD311B102`, 64 worlds each, all eight directions at 0.5/1.0 cm/s plus calm, one minute per world. Further five-minute tests use seed `0xD311B201`, 64 worlds each in calm, +X 1 cm/s and −X/−Z 1 cm/s. Compare both controllers through the original compiled current-evaluation physical kernel, without episode resets or rescue. Inspect paired food/minute, >20-second gaps and worst gaps; do not promote a candidate merely for fixing one fixture.

New tests cover score-preserving transfer, food-specific progress/respawn handling, current sampling coverage, finite PPO, frozen motor parameters, policy round trip and bit-identical full-state resume. Existing save formats and the original selector trainer are unchanged.

## Tools

```powershell
cmake --build --preset release --target pond_progress_train pond_current_eval cudarl_tests --parallel 6
.\build\Release\cudarl_tests.exe "progress selector"
.\build\Release\pond_progress_train.exe --baseline docs/experiments/swimmer/gentle-currents/frozen-policy.bin --output build/pond-progress-study/seed-71 --envs 8192 --seed 71 --updates 128
.\build\Release\pond_progress_train.exe --baseline docs/experiments/swimmer/gentle-currents/frozen-policy.bin --output build/pond-progress-study/seed-109 --envs 8192 --seed 109 --updates 128
```

Updates are additional on resume; existing candidate saves supply their original batch/seed. The new experimental policy format is separate from the deployed pond policy. No Godot distribution, graphics, limbs or time-varying current work is part of this step.

`--updates 0` in this experimental tool performs evaluation/save only, with zero optimizer updates; it does not mean unlimited training. `--benchmark` measures discarded copies at 1,024/4,096/8,192 worlds; `--benchmark-large` measures 16,384/32,768. These benchmark updates are not saved as candidate learning progress.

## Results and stopping point

Both 128-update pilots completed at 8,192 worlds and stopped, with resumable state saved separately under `build/pond-progress-study/seed-71` and `seed-109`. Each pilot used 671,088,640 physical environment steps. The selected candidate is seed **71**, update **128**, SHA256 `5AB73B6AF596B2A5832DCB307B11E5538F12B144172339C6F9B600201D489A7E`. Its policy is archived as [selected-policy.bin](selected-policy.bin), and its full actor/critic/Adam/world save remains local. See [selection across all checkpoints](selection.csv) and [selection record](selected.json).

The original gets no food in the 60-second recorded-stall continuation. The selected candidate gets **eight foods**, with the first collected within **15 seconds** (sampled at one-second boundaries). Only this one checkpoint across the two seeds passed the predeclared recovery/calm gate. This is evidence of a learned recovery for this development case, not a seed-robust or universal recovery solution.

Fresh paired one-minute evaluation, averaging the eight directions at each nonzero strength:

| Condition | Original food/min | Candidate food/min | Paired difference, approximate 95% interval |
|---|---:|---:|---:|
| Calm | 17.54688 | 17.57813 | +0.03125 [−0.14926, +0.21176] |
| 0.5 cm/s | 16.83789 | 16.90137 | +0.06348 [−0.00929, +0.13624] |
| 1.0 cm/s | 14.21289 | 14.35449 | +0.14160 [+0.04144, +0.24176] |

For strength-level intervals, average the direction-wise difference within each seed/environment first, then estimate uncertainty across 128 matched layouts. Repeated directions on one layout are not treated as 1,024 independent observations. These intervals remain conditional on the selected training run, not variation over training seeds; no multiple-testing correction is claimed. The stronger-current improvement is about **1.0%**, rather than the 18% current penalty being eliminated.

Robustness is **not established**: among the 1,024 fresh one-minute 1 cm/s trials, the original has two >20-second food gaps and the candidate three. In new five-minute tests, +X current has two such worlds for each controller, with maximum gaps 21.25 s (original) and 27.15 s (candidate). Calm and −X/−Z groups have no >20-second gaps for either policy. Five-minute means are calm 17.76875→17.81875, +X 14.675→14.79063, −X/−Z 14.07188→14.26875. Fixing the known fixture and a small average collection gain are insufficient to replace the original or claim fewer stalls.

The original remains deployed. This pilot changes observations and the training distribution together; it does not isolate the causal benefit of progress inputs against a separately current-trained 32-feature selector. The used holdouts are now recorded evidence, not untouched data for subsequent candidate selection.

Raw results: [fresh one-minute cases](fresh-minute/cases.csv), [per-condition summary](fresh-minute/summary.csv), [layout-clustered aggregate](fresh-aggregate.csv), [fresh five-minute cases](fresh-long/cases.csv), [long summary](fresh-long/summary.csv), [verification](verification.json), and both training logs. No UI or pond model replacement was made. The original packaged policy, frozen baseline and both original full training saves are hash-verified unchanged. A **copy** of the real seed-71 save was restored at update 128 with zero additional optimizer updates; its complete checkpoint hash stayed identical. All **159 native tests passed**, GPU required, no skips.

## GPU measurements and optimization notes

| Training worlds | Short measured physical steps/s |
|---|---:|
| 1,024 | 2.162M |
| 4,096 | 4.606M |
| 8,192 | 6.271M |
| 16,384 | 5.571M |
| 32,768 | 6.376M |

The larger test was requested after the initial three-size comparison. It adds only about **1.7%** over 8,192, while 16,384 is slower in this sample. These are short local measurements, not proof of a global optimum or a confidence interval for the difference. Bigger batches alter PPO data per update; simulation throughput alone does not establish faster learning to a quality threshold. Preserve the pilot at its predeclared 8,192 worlds instead of changing the training budget after looking at results.

A recorded training-time NVIDIA sample shows 99% GPU activity, about 3,003 MiB allocated and 167 W. NVIDIA's local `nvidia-smi --help-query-gpu` defines utilization.gpu as the fraction of sampled time with one or more kernels running. It is **not 99% of peak arithmetic throughput or proof of optimal occupancy**. See [timing samples](batch-benchmark.csv), [larger batches](batch-benchmark-large.csv) and [hardware sample](gpu-active-sample.csv).

The user supplied optimization notes at this checkpoint. Preserve this order as a proposed next performance pass, not work already implemented:

1. Profile CPU submission, GPU execution, synchronization and copies on a unified timeline. [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems) supports this system-wide view; Nsight Compute is available locally for subsequent kernel-level investigation. Do not infer the bottleneck from utilization alone.
2. Keep neural calculations and the efficient batched physical step on the GPU. Reduce transfers or synchronize less where profiling identifies a cost. NVIDIA's [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) recommends measuring bottlenecks and overlapping independent work where useful.
3. Consider CPU workers for frozen-checkpoint screening across scenarios, report generation and safe snapshot file writing. The original pond has a tested CPU simulation/inference path. The new progress-aware candidate's long-horizon CPU evaluation has not been validated; the prior stall was execution-path-sensitive. Validate that path before using it for ranking, keep CPU scores separate, and retain authoritative GPU comparisons for promotion.
4. Prepare scenario descriptions/evaluation schedules on CPU where worthwhile; do not transfer simple randomization off the GPU without measuring a benefit. Preserve separate training, development and final evaluation data.
5. Only after profiling choose a measured optimization: fewer launches/CUDA graphs if CPU-bound, or physics/MLP memory and execution changes if kernel-bound. More host threads, larger networks, filling VRAM or a higher percentage meter are not success criteria.

No CPU evaluation pool, background checkpoint writer, profiler installation, CUDA-graph optimization or open-ended training was started. The user requested reaching a stopping point before discussing further hardware use; both candidate runs are saved and stopped.
