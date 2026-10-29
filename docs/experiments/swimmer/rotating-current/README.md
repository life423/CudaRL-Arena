# Frozen swimmer in slowly changing flow — October 28, 2026

The latest saved agent passes the [predeclared gentle-current suite](PROTOCOL.md) without additional PPO training: **60 sessions / 420 simulated minutes**, zero confirmed five-second immobility, invalid states, invalid food or skipped steps. Both networks remain frozen. This is evidence of response to an unfamiliar flow schedule using existing learned skills, not new learning or competence in arbitrary water.

## What was tested

Three uniform horizontal flows at the same **0.5 cm/s** speed: steady +X, rotating from +X toward +Z, and rotating from +X toward −Z. Each rotating vector completes one turn every **120 simulated seconds**, updated at the shared 50-ms control step. The same vector drives existing water-relative swimmer drag and suspended-food advection. There is no scripted swimming, target override, rescue, reset or changed physical equation.

Eight fresh seeds `0xC0DECC01`–`0xC0DECC08` run five minutes per condition/backend. The first two also run fifteen minutes, covering 7.5 current turns. Strict CPU and strict CUDA playback use the actual app playback path; silent fallback is forbidden. Initial layouts match by seed, while subsequent collection and respawns can diverge. CPU and CUDA may also diverge over long trajectories.

The controller is `gentle-currents/bank-recovery/selected-policy.bin`: selector update **320**, retained swimming actor **64,350**, SHA-256 `4186F47471EDE2D0034EB566EFA1EC959B03D0A015E686FFA5F4262FCEB4A563`. The motor learned calm-water swimming; the selector already received steady-current and bank-recovery practice. Neither is updated during this experiment. Eleven protected source-policy/full-save references retain their expected hashes in [preserved-inputs.json](preserved-inputs.json).

## Collection and reliability

| Backend | Minutes/case | Cases/condition | Steady food/min | Toward +Z food/min | Toward −Z food/min | Largest food gap across conditions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU | 5 | 8 | 17.43 | 17.68 | 17.60 | 12.50 s |
| CPU | 15 | 2 | 17.27 | 17.73 | 17.70 | 17.95 s |
| CUDA | 5 | 8 | 17.50 | 17.38 | 17.73 | 13.75 s |
| CUDA | 15 | 2 | 17.57 | 17.40 | 17.50 | 16.05 s |

Every rotating-condition mean exceeds the declared 90% of its matched steady mean; observed ratios span **99.05%–102.70%**. All food gaps stay below twenty seconds. CUDA paired mean differences are −0.125/+0.225 food/min for the short positive/negative runs and −0.167/−0.067 for long runs. Individual short-run paired differences range from −1.8 to +1.2 food/min on CUDA, so a close overall average does not mean each layout is equivalent. [Paired results](paired.csv), [all cases](cases.csv), [aggregates](aggregates.csv) and [machine summary](summary.json) retain the details.

These descriptive samples do not establish statistical superiority, training-seed variance or universal reliability. Higher collection in a particular moving-food condition can reflect easier food trajectories rather than a better strategy. This suite does not reproduce or resolve the earlier retained hard bank-start failures.

## Bank observations

The passive sampled bank proxy finds **366 visits with at least one cumulative second of contact**. Of those, **364 clear** the proxy for a full second before trial end. The two remaining visits begin at 295.40 s (CPU positive short, first seed) and 296.65 s (CUDA steady short, first seed); both remain open at the five-minute cutoff. They are censored observations, not counted as cleared or hidden by resets.

All 366 sustained visits have a subsequent food collection within twenty seconds of entry; the largest such delay is 17.50 s. Food collection and bank clearance are separate events: eating near the edge does not itself prove physical clearance. Maximum cumulative contact per visit is 2.85 s; maximum observed time to the one-second clearance criterion is 10.00 s. Across all sampled contacts there are 1.10 s below 0.5 cm/s ground speed, without a confirmed immobility event. [All bank events](bank-events.csv) include follow-up and censor flags.

This is an observational contact proxy, not a contact-force measurement or a deliberately difficult bank-recovery benchmark. Brief ordinary contacts passing here do not settle previously failing difficult starting states.

## Engineering checks and reproducibility

The three new focused tests pass with GPU required and zero skips: constant-speed/mirrored simulation-time scheduling, bank clearance versus feeding/censoring, and live CPU/CUDA agreement without resetting when the vector changes. Six existing current/reliability/food-drift regression tests also pass. The conflicting rotation/delayed-switch CLI combination is rejected before creating output. No full native-suite rerun is claimed for this tools-only change.

Both updated runners reproduce the prior steady drifting-food reference (`0xC0DECA01`, five minutes, **89 foods**) exactly. This checks that passive diagnostics preserve that reference trajectory. All **6,408 sampled flow vectors** agree with the declared speed and schedule within `1e-8 m/s`; measured maximum vector error is `3.52e-9 m/s`. The maximum speed error is `3.47e-10 m/s`.

Build `pond_reliability` with the release preset. Example positive-rotation CUDA trial, using a new output directory:

```powershell
& ./build/Release/pond_reliability.exe `
  --policy docs/experiments/swimmer/gentle-currents/bank-recovery/selected-policy.bin `
  --progress --backend cuda `
  --module godot-project/addons/cudarl/bin/cudarl_pond_cuda.dll `
  --drifting-food --current-cm-s 0.5 --rotation-seconds 120 `
  --minutes 5 --seed 0xC0DECC01 --output build/my-rotating-trial
```

Use period `0` for steady and `-120` for the opposite direction. Repeat all eight declared seeds (repeat `--seed`), both backends and the first two seeds at fifteen minutes to reproduce the suite. `--backend cpu` needs no optional module. If a running Godot app locks its DLL, build only the runner; the module source is unchanged by this experiment. A module with the existing current/food-control interfaces is required for strict CUDA.

Each backend/condition folder retains raw cases, immobility events, bank events, sampled state/current traces and mode metadata. The runner's existing `passes_provisional_gate`/case `passes` fields cover structural validity and immobility only; the root summary additionally applies this experiment's food-gap and matched collection rules. No diagnostic stall snapshots were triggered. Traces are diagnostic samples, not exact restartable CUDA challenge states.

## What changes in the app

**No pond gameplay or GUI changes.** The open pond still offers its existing steady gentle flow, latest saved agent and drifting food. Slowly turning flow is implemented in the research runner only. No learner is running, no policy is replaced and no save is upgraded.

The result does not justify spending more PPO updates on this gentle schedule by itself. The next proposed AI step is a separately declared, bounded **0.75 cm/s** frozen-agent current test, retaining calm/gentle references, to locate a meaningful weakness before choosing selector or motor practice. That experiment and any live rotating-current control remain proposals. The broader goal is still dependable interaction, measured adaptation with skill retention, then anatomy growth.
