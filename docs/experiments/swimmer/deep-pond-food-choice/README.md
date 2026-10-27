# Food selection with a retained swimming controller

The selected controller combines a learned PPO food scorer with the **unchanged** update-64,350 swimming network. The scorer chooses a physical food slot every simulated second; the motor network still produces four tail torques every 50 ms. Both inference and training use the existing deep-pond rigid-body/drag equations. No scripted swimming, nearest-food rule, rescue, teleport or live episode reset is part of deployment.

## Fresh evaluation of the selected controller

| Target choice | 512 new one-minute trials, food/min | 128 new five-minute trials, food/min |
|---|---:|---:|
| Original food ordering | 15.363281 | 15.690625 |
| Programmed nearest, every 50 ms | 16.888672 | 17.128125 |
| Programmed nearest, every one second | 15.707031 | 15.543750 |
| **Learned shared-score PPO selector** | **17.562500** | **17.834375** |

The learned selector improves the short-trial mean by **14.3%** over the original controller and **4.0%** over per-step nearest-food targeting. Paired improvement over the original is +2.1992 food/min (approximate 95% interval +2.0540 to +2.3444); over nearest it is +0.6738 (+0.5026 to +0.8450). Long-trial improvement over the original is +2.1438 (+1.9612 to +2.3263), and over nearest +0.7063 (+0.5788 to +0.8337).

In these fresh trials the learned controller collected from all three physical slots. Its longest observed food gap was 14.8 seconds in the short suites and 12.7 seconds in the long suites; its longest continuous speed below 0.5 cm/s was 0.25 and 0.65 seconds respectively. These are bounded observations, not proof that every future trajectory is stall-free or globally optimal. The learned selector changes targets more often than the per-step nearest baseline (about 9.5 uncollected-target switches per minute versus 1.7); fewer switches alone is not the objective. Some nearby foods can still be passed.

Raw per-world data, seeds, totals and paired comparisons are in [holdout A](holdout-a/summary.csv), [holdout B](holdout-b/summary.csv), [long A](long-holdout-a/summary.csv), [long B](long-holdout-b/summary.csv), and [aggregate calculations](aggregate.json). Cases use identical initial seeds/layouts across modes, but later respawns depend on collection history and body/other-food avoidance. The intervals summarize sampled layouts, not variability across all training seeds.

## Why a recovery phase was necessary

An initial three-output selector learned a fixed-slot shortcut. A shared per-food scoring network removed slot identity and improved collection, but its first five-minute holdout exposed a folded-tail stall in environment 8, seed 3481762059. The pre-recovery actor stayed below 0.5 cm/s for roughly 136 accumulated seconds in that trial. Those first holdouts are retained under `pre-recovery-*` and are discovery/regression evidence for the revised controller.

Branching that physical state showed that another target let the **same frozen motor** recover. Both the original learned selector and nearest-food baseline failed to collect over the 30-second branch; choosing another food allowed collection within about five seconds. See [the original branching probe](pre-recovery-probe.log). Forced choices appear only in this diagnostic tool, never in deployment.

The recovery phase trains on mixed independently sampled resting joint configurations and five-minute episodes. A food drought of at least 20 seconds plus current speed below 0.5 cm/s ends a training episode with a -5 penalty. These training resets do not occur in inference. The recorded stall gates checkpoint selection, but does not enter gradients. The selected actor independently recovers in the [CPU replay](recovery-probe.log), collecting eight pieces over 30 seconds, with first collection at about 5.3 seconds. Its [full prior long-suite regression](long-regression/summary.csv) also improves; it is not counted among the 128 fresh long trials.

## Selection and provenance

- Source pond checkpoint update: **80,525**; frozen motor best update: **64,350**. Source SHA-256 is in [aggregate.json](aggregate.json).
- Initial shared-score training: seeds 11 and 29, 800 updates each, highest fixed-suite checkpoint selected before opening the initial holdouts; the equal best scores were tied in favour of seed 11 at update 675.
- Recovery phase: seed 47, initialized from that learned selector, with a fresh selector optimizer/worlds and the same frozen motor. It ran 800 updates. **Phase update 250** was selected by fixed-suite score (17.8125 food/min) subject to collecting within 20 seconds from the recorded stall. The later learner remains saved for exact continuation.
- The [protocol saved before the new holdouts](protocol-before-final-holdout.md) and [selection record](selection.json) specify the fresh seeds and selection rule. These holdouts did not receive gradient updates.
- [Packaged selected controller](../../../../godot-project/pond/policies/deep_selector.bin): SHA-256 `D9730ED4FB2A05904E09ACAB45C43BE47FF7B80B768AA5379C463A11B8D9611D`. It includes both network parameter vectors and decision cadence. The frozen source checkpoint and full training checkpoints remain local; Git contains the deployed controller, its regression fixture and experiment evidence.

The [implementation and commands](../../../pond_food_selection.md) describe the architecture, PPO settings, full resume, remaining limitations and frozen Godot playback. Instrumented evaluation includes CPU state copies/reporting and is not a GPU throughput benchmark. Physical modelling remains a reduced articulated-body/drag simulation, not CFD or calibrated animal biomechanics.

The packaged-only reproduction command in that guide does not require the local full source checkpoint. It recreated `holdout-a/cases.csv` byte for byte. Final integration checks passed: 154 native tests and 374 Godot assertions, plus a CLI resume from update 800 to 801 on an isolated save copy.
