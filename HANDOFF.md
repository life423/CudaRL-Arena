# PPO handoff — 2026-11-01

Training is stopped. Last completed update: **352**. Nothing was adopted into
Godot. Native implementation commits: `c49e5a0` (resume/shared duration) and
`8f336db` (larger 32s training arena). Unrelated Godot editor changes remain separate.

## Preserved, reloadable checkpoints

Both directories contain the complete actor, critic, fixed stochastic settings,
both Adam states/counters, matching protocol, provenance and compact evaluations.
These are research checkpoints; neither passes full motor acceptance.

| Role | Checkpoint | SHA256 | Result |
|---|---|---|---|
| Best overall | `assets/ppo-four-axis-checkpoints/update-288/ppo-update-288.bin` | `CEAB95BF500788F8F005EADC28F556CF0AB08069AB67625622199E3D7C07CB0C` | 4/4 short; 8/30 long windows; 0/2 complete long passes |
| Best new 32s result | `assets/ppo-four-axis-checkpoints/update-352/ppo-update-352.bin` | `A93637E9469DE591203C4465FCF7E630C0236BEC3EEDB6D4CC47D0DD0E14F933` | 2/4 short; no long test triggered |

The `protocol.bin` beside each checkpoint is required. Adam counters are 11,460
at 288 and 19,540 at 352. Resume restores all training state at a completed-update
boundary and starts fresh episodes; mid-rollout state is not checkpointed.

## Last experiment

The first 32s attempt reached the original +X safety margin at 19.575s before any
optimizer update. Training was then explicitly moved to the existing 1.2m × 0.8m
long-evaluation arena. Body/fluid equations, limits, reward and acceptance tests stayed fixed.

The larger-arena campaign completed 64 updates (288→352), 1,316,096 controls and
13,160,960 physical substeps, with no safety stops. Short passes at 304/320/336/352:
**0/4, 0/4, 0/4, 2/4**. At 352 only poses 0/2 fail, solely for scored speed:
7.08565/7.08566mm/s versus the unchanged 7.2 minimum. All other short gates pass.
The source 288 remains better. No new checkpoint reached 4/4, so no new long swims ran.

The first completed GPU update proved post-8s reward flow: 15,360 late reward rows,
GAE error ≤5.51e-6, and 195 late minibatch rows with nonzero policy gradients before
optimization. Full local evidence: `build/ppo/continuation-243-288to352-32s-wide-v1/`
and `build/ppo/episode32-wide-243-u304/duration-reward-verification.json`.
Ignored `build/` evidence is not stored by Git.

Baseline 288 long swims: pose 0 first leaves ±1mm at 16.15s, final depth −8.946mm;
pose 3 exits at 18.45s, final −7.044mm. Neither returns. Maximum/final horizontal
offset magnitudes: 4.187/4.187mm and 3.785/3.603mm. Both swims finish safely.

## Tomorrow

First inspect saved 352 startup/steady-speed traces and 32s reward components
against 288; choose the next single change before another campaign. Do not blindly
extend training or replace 288. No further training budget is currently approved.

If a new continuation is approved, from the worktree root this prepares one new
16-update, 32s block from the best checkpoint (it does not train):

```powershell
.\tools\run-ppo.ps1 -Name resume-288-next16 `
  -ResumeFrom .\assets\ppo-four-axis-checkpoints\update-288\ppo-update-288.bin `
  -ResumeProtocol .\assets\ppo-four-axis-checkpoints\update-288\protocol.bin `
  -EpisodeControls 640 -AdditionalUpdates 16
# Only after a new bounded run is authorized:
.\tools\run-ppo.ps1 -Name resume-288-next16 -Run
```

Focused tests and zero-step CPU/CUDA reload checks are recorded in each asset's
provenance. No training runs overnight; publication status is reported separately.
