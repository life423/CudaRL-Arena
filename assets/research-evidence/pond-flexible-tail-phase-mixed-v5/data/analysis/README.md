# Mixed 15/30-degree continuation: completed evidence

This read-only analysis executed zero physics controls and zero optimizer updates. It aggregates the existing 16 scheduled assessments and pretraining baseline; no additional evaluations were requested.

| Checkpoint | All /32 | Gentle 15 /8 | Trained 30 /8 | 60 /8 | 90 /8 | Straight /12 |
|---|---:|---:|---:|---:|---:|---:|
| Imported parent112 | 9 | 3 | 6 | 0 | 0 | 12 |
| Mixed24 | 12 | 8 | 4 | 0 | 0 | 12 |
| Mixed56, best all-gentle checkpoint | 13 | 8 | 5 | 0 | 0 | 12 |
| Mixed88, best total | 14 | 7 | 7 | 0 | 0 | 12 |
| Mixed120, every trained 30-degree case | 12 | 4 | 8 | 0 | 0 | 12 |
| Mixed128, final | 11 | 6 | 5 | 0 | 0 | 12 |

The full lesson gate remains unmet: no checkpoint simultaneously passed all eight 15-degree cases, all eight 30-degree cases, and all twelve straight cases. Every 15/30 case passed the movement, depth, tilt, clearance and root/passive swing criteria at every scheduled checkpoint. Full-pass fluctuations were exclusively passive-tail complete-beat frequency below 80% of the same-state protected reference in at least one original scoring window. The complete gate was kept fixed. No 60/90-degree generalization case passed; these angles were not trained.

Mixed120 establishes the clearest new capability: all four left and all four right 30-degree cases met every existing gate. Mean late-turn course error was 0.552 degrees, onward error 0.450 degrees, projected speed 12.852 then 13.278 mm/s, mean initial response 1.679 simulated seconds, and mean effective complete-beat radius 251.93 mm. The smallest passive frequency ratios were 0.834 left and 0.840 right, above the unchanged 0.8 gate. Four gentle cases failed solely passive frequency. Response is initial measured course response, not settled completion time.

Mixed88 retained 3/4 left and 4/4 right passes at both 15 and 30 degrees. Its only misses were `phase_1_left_15` and `phase_2_left_30`, both passive frequency. The 30-degree left onward minimum ratio was 0.797518, narrowly below 0.8; it remains a failure. Mixed56 retained every gentle pass and 3/4 left plus 2/4 right 30-degree passes. Final128 retained 4/4 left plus 2/4 right gentle cases and 2/4 left plus 3/4 right 30-degree cases.

All twelve straight/start cases passed at all checkpoints, including all eight starts from rest. Exact inactive commands and paired final native states were retained. Every recorded turn had zero tilt and depth error and at least 67.999 mm certified wet clearance. The suite did not physically test stationary holding, currents, repeated navigation, or live bank avoidance; those capabilities are not implied.

The study completed 128 new updates, 8,388,608 sampled actions, and 535,461,888 accepted physical substeps in 104.741 trainer seconds. GPU physics/observations took 96.016 seconds, network 0.444 seconds, optimization 2.789 seconds, and evidence writing 5.044 seconds. Sync/readback wall timing overlaps GPU execution and must not be summed as a separate compute stage. Three diagnostic updates were separately counted by the fork/resume receipt.

Exact starts were recovered from saved terminal Counts arrays and the fixed pair/episode parity rule: 11,008 new left15, 11,008 right15, 11,008 left30 and 11,008 right30, totaling 44,032. The imported active 30-degree episodes were continued and excluded from new-start counts. `exposure.ps1`, `exposure.csv` and `exposure.json` retain this derivation. The parent session SHA256 was 07D5841E9EA028F4BC087574ED2F78802A5D75E4CFAC7A37976C7B164D2001AB, matching the protocol.

Read-only source review found that the fork retained full native Lane bytes, actor, critic, both Adam states and counter-based sampling/episode positions. Adam age7168 and sampling origin112 continued while new study evidence counters were rebased. The zero-step import receipt and deterministic three-update fork/resume check passed, with identical full payload checksums and maximum CPU/CUDA feature discrepancy 4.77e-7. Physics, rewards, observer and evaluation gates were unchanged. The runner did not itself hard-code the protocol's parent112 hashes, but its bound manifest and actual import establish the correct parent for this completed run. Angle exposure was not a direct training JSON field and was recovered offline rather than invented from aggregate sign counts.

Recommendation: keep the current working app motor protected and retain mixed120 and mixed88 as separate research checkpoints. The next learning variable, if authorized, should target the demonstrated passive-rhythm retention while keeping mixed practice and all movement gates unchanged; increasing angle range or repeating the same schedule alone has weaker evidence. Do not deploy a checkpoint that has not jointly passed the agreed 15/30 gate, or label a clear-water turn as autonomous indefinite pond exploration.

The accompanying learning curve, per-case failure flags and angle/direction/window metrics preserve every scheduled result. `phase-contribution.csv` is limited to its declared sampled checkpoints and distinguishes actual learned latent contribution from the frozen prefix; it is not a causal ablation.
