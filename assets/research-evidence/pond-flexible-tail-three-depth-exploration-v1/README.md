# Three-depth chooser PPO with frozen motor 88

**Godot retains the installed appended-16 chooser.** The separate update-128 chooser covered more water on all four 30-minute swims. It passed the full ordinary navigation suite and all four long-swim stall checks. Both choosers hit the clearance guard on two north-bank approaches at each depth. [All 24 paired shoreline results, with minimum clearance and longest stall for every swimmer](bank-comparison.md)

## Actual GPU training

The native CUDA trainer ran 128 updates on 1,024 parallel worlds in 135.78 seconds of process wall time. Each update collected eight chooser decisions per world. A separate three-update exact-resume check accounted for another 24,576 verification decisions and is excluded below. Motor 88 and the fitted stroke were frozen throughout.

| Training depth | Initial worlds | Later respawns | Unique PPO decisions | Share | Actor/critic minibatch presentations |
|---|---:|---:|---:|---:|---:|
| 72 mm | 340 | 22 | 348,160 | 33.203125% | 1,388,205 |
| 75 mm | 342 | 34 | 350,208 | 33.3984375% | 1,396,370 |
| 80 mm | 342 | 27 | 350,208 | 33.3984375% | 1,396,417 |
| **Total** | **1,024** | **83** | **1,048,576** | **100%** | **4,180,992** |

The fork restored the installed chooser's full parent session and marked every inherited lane for a new episode before its first training decision. The parents' in-flight episodes contributed zero decisions. All 1,024 first episodes and all 83 later training respawns are included. A native guard checked the actual request depth in every row before it could enter the rollout. Zero accepted decisions had an unexpected request depth.

The offline audit checked the native checksum of all 129 full checkpoints. It reconciled each lane's saved action counters and reconstructed the exact minibatch rows using successive actor and critic Adam-step differences and the trainer's own shuffle. Every collected decision entered at least one optimizer minibatch. The four-pass maximum would have been 4,096 minibatches. The actual 4,083 each presented 1,024 rows to both actor and critic. Across all 131,072 post-update lane snapshots the largest difference between whole-body COM depth and the assigned depth was 0.000014 mm. COM was sampled at update boundaries. Request depth was checked on every decision. [Depth ledger](qualification/depth-ledger.csv) · [Trainer result](data/training/result.json) · [All checkpoint identities](data/full-checkpoint-hashes.csv)

## Distant unvisited water

The native depth maps contain 352, 344 and 338 reachable cell centers respectively. The 80 mm slice matches the installed slice exactly. The new maps feed the last three inputs of the 35-input chooser. All 32 local inputs retain their original definitions. The coverage denominator for scoring is still the original 345-cell reference so both policies are measured identically.

At each depth, empty visit history produced left/forward/right values of 1/1/1, fully visited water gave 0/0/0, and reversing course moved a single unvisited patch out of the forward sector. CPU and CUDA agreed exactly on the three distant inputs across 1,024 parity probes. As a separate test on real final checkpoint states, clearing visitation history alone changed the distant inputs for 340/340, 342/342 and 342/342 swimmers. Largest absolute changes were 0.9845, 0.9892 and 0.8723 respectively. [Input checks](data/verification/data/distant-senses.json) · [Depth map](map.json)

## Four uninterrupted 30-minute swims

Coverage is the number of credited cells per swimmer out of 345. Both choosers completed each swim with zero clearance stops and zero resets. The two depth maps with different cell counts at 72 and 75 mm change the chooser's sensing. The held-out score uses the same denominator and credit rules for both actors.

| Start | Depth | Installed cells | Candidate cells | Installed min gap, mm | Candidate min gap, mm | Installed longest stall, s | Candidate longest stall, s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Godot | 72 mm | 151 | 185 | 60.00 | 60.00 | 4.56 | 0.00 |
| Northeast | 80 mm | 147 | 195 | 68.00 | 64.85 | 0.00 | 2.58 |
| West | 75 mm | 148 | 218 | 63.00 | 63.00 | 0.00 | 0.00 |
| Southwest | 75 mm | 160 | 201 | 63.00 | 63.00 | 8.14 | 4.76 |

Mean coverage rose from 43.91% to 57.90%, a 13.99 percentage point gain. The update-zero parent with the corrected map averaged 42.25%. Updates 8 and 32 completed 1/4 and 0/4 long swims, respectively. The final candidate passed 12/12 ordinary starts and 4/4 five-minute swims. No final swimmer reached the separate 80% exploration milestone. [Learning checkpoints](data/checkpoints.csv) · [Coverage comparison](data/comparison.csv)

## Publication decision and stall regression

The package check reads the native per-swimmer stall counters. It allows 407 consecutive 20 ms controls and rejects 408. The previous regional update-eight Godot swim finished all 1,440,000 substeps with positive clearance. Replaying its archived checkpoint reproduced all original case columns and measured 2,071 consecutive stalled controls (41.42 s). The same gate that accepted the installed southwest swimmer at exactly 407 controls rejected that Godot swim.

The final candidate met the long-swim coverage and stall gates. It completed **6/8 shoreline cases at 72 mm, 6/8 at 75 mm and 6/8 at 80 mm**. Each set requires 8/8. Zero cases per depth met the sustained-near-bank requirement of four, and all three depth means fell below 10.96 seconds. The gate rejected the candidate before Godot staging. The 11 existing live package hashes verified. [Gate verdict](qualification/decision.json) · [Native regression replay](qualification/regression/continuous/assessment/cases.csv)

The two new fixture sets use held-out poses 8 and 9 of the 12-pose evaluation archive. Neither evaluation archive was used as training input. Training used the separate 256-pose archive and randomized interior or outward-angled placements. All 24 evaluated fixtures are tangent starts. Within each depth, the two actors' recorded start files matched byte for byte. The 80 mm start file also matched the original eight-case archive. The supplement ran frozen native evaluations only.

To reproduce the publication decision in this worktree, run the following from its root. The second command is expected to reject update 128.

```powershell
node --test tools/pond-flexible-tail-qualification/publication.test.mjs
$study = 'assets/research-evidence/pond-flexible-tail-three-depth-exploration-v1'
& tools/test-flexible-tail-package.ps1 -QualificationPath "$study/qualification" -CandidateActor "$study/data/training/actor-128.bin"
```
