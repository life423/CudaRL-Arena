# Bank-recovery selector training protocol

Fixed October 27, 2026 before collecting the new data/training. The user approved keeping two controllers and training food selection to recover near the bank, with the swimming controller and physical equations fixed. They clarified that food must remain underwater and reachable from the water. No arms, land locomotion, graphics work or new flow model is included.

## Preserve and isolate

Keep the deployed frozen bundle (SHA-256 D9730ED4FB2A05904E09ACAB45C43BE47FF7B80B768AA5379C463A11B8D9611D), original Godot saves, earlier pilot full saves and selected pilot unchanged. Run native CUDA PPO on separate copies. Retain the existing progress-aware selector's 34 features, motor's 33 inputs/four torque outputs and all PPO hyperparameters. No new network or sensor. The learned actor and training-only critic may update; the swimming actor stays frozen.

## Recovery-state collection

Collect authentic physical states from frozen original-selector/motor trajectories with normal pond initialization. A frame qualifies at a one-second decision boundary after at least 20 seconds without food, ground speed below 0.5 cm/s, and a link near the sloped pond bed away from its center. Record at most one frame per source world, along with its actual prior target, per-food progress history and current. All three foods must satisfy the unchanged underwater `deep_pond::water` constraints (including bottom clearance). Do not move food to land or alter it to force a solution.

Use 8,192 source worlds per seed, five simulated minutes, 25% calm and otherwise random horizontal direction at either 0.5 or 1 cm/s. Training source seeds 0xBAA71001/0xBAA71002; development seed 0xBAA7D001; fresh recovery holdout seeds 0xBAA7E001/0xBAA7E002. Preserve original-state identities. The already-recorded stall fixture is excluded from gradients and remains a labeled development/regression case. If too few qualifying states are collected, document a revision before changing this protocol or training.

## Bounded training

**Collection revision 1, before any new training:** the initial 8,192-world/five-minute pass yielded only 12 training states, three development states and 12 holdout states. Increase to **32,768 source worlds per seed and ten simulated minutes**, with the same five seeds, qualification rules and separation. Preserve the first-pass manifests as an audit record. Pool-size counts alone were inspected; no candidate was selected or trained on holdout outcomes. Do not relax underwater validity or fabricate bank contacts to increase sample counts.

Start two copies from the earlier seed-71 and seed-109 update-128 full saves (actor, critic and Adam retained). Change only episode-start curriculum: 50% replayed training recovery states, 25% normal central starts, 25% independently folded central starts. Recovery frames retain their physical state/food/current/prior target/history; remap RNG identity to the new world and reset episode time/food-count. Cap historical time-since-food at ten seconds at a replay start so the inherited training-only 20-second failure rule leaves time to attempt recovery. This is reset-time sampling, never a deployed teleport or rescue.

Run **256 additional PPO updates per seed**, 8,192 worlds; stop with full resumable saves. Existing food reward, failure penalty, time limits, optimizer, action cadence and 5-ms physical substeps stay fixed. Evaluate/save at phase start and each 64 additional updates. The new curriculum and recovery pool must be included in resumable state; legacy saves remain readable.

## Selection and untouched evaluation

Before final selection, compare original, earlier selected pilot and each saved candidate on the development bank pool: food within 20/60 seconds, first-food time and foods in 60 seconds, no resets or rescue. Evaluate normal development layouts in calm and two 1 cm/s directions. A selectable checkpoint must retain at least 98% of original normal calm/current mean collection, recover the known fixture within 20 seconds, and improve development 20-second bank recovery over the original by at least ten percentage points. Rank recovery first, mean collection second; ties favor earlier phase update, then lower training seed.

Freeze at most one winner before using fresh data. Compare it with the original and earlier pilot on the fresh bank pools, plus fresh normal one-minute trials across eight directions at 0.5/1 cm/s and calm, and five-minute swims. New normal layout seeds 0xBC171101/0xBC171102; long seed 0xBC171201. Report paired recovery, food collection, >20-second food gaps and worst gaps. Do not promote merely for fixing one fixture or raising average reward. If no checkpoint passes development gates, report that result and retain all source policies.

No model is automatically installed into Godot by training. Fresh evidence should determine whether promotion is justified. Success here concerns steady imposed currents up to 1 cm/s and swimming recovery; broader/changing water conditions remain later curriculum/physics work. Keep the saved GPU optimization backlog for a future performance pass.
