# Appended distant sensing — November 25, 2026

The user authorizes retaining every local input, appending zero-weight distant
feedback, another separate GPU PPO chooser session, and conditional Godot update.
The existing coverage80/motor88 pair and all historical evidence stay protected.

One change from coverage-v1 selected80: expand actor/critic from32 to35 inputs.
Inputs0–31 stay bit-identical per backend. The existing three distant sector
means occupy32–34. Both networks' added first-layer columns start at exact zero.
Keep all inherited weights, biases, Adam moments and ages; added columns have
zero moments and their own fresh age. World, servo, course, visit and episode
state survive import. A compacted parent roundtrip must be byte-identical,
and original versus expanded GPU actor/critic outputs must be exactly equal
over1,024 saved worlds before any optimizer update. New35-input checkpoints
explicitly serialize both fresh-column ages. Model version5/schema4 binds
the same native80mm geometric cache. Version3/4 models load with zero padding
and their original observation meaning. No second behavioral network is added.

Require new sensory CPU/CUDA difference≤1e−6 and all32 old fields bit-identical
on each backend. Record the already identified unchanged bank-input discrepancy
separately; no physical tolerance or equation changes. Use the prior338-cell
sensing cache and unchanged345-cell evaluation estimate at80mm depth.
Keep reward scales, actions, task mix, PPO parameters, physics and motor fixed.
No food, routes, scripted steering, action replacement or automatic live resets.

One new session:1,024 CUDA worlds,128 PPO updates,8 decisions/world/update,
1,024-row minibatches and four passes (1,048,576 fresh macro transitions max).
One three-actual-update exact full-resume check. Main native/external wall caps
1,200/1,800s and1,342,177,280 physical-substep ceiling. Preserve every checkpoint.
Stop numerical/native failures; bank guard stops are scored training failures.
No silent failed-study restart or extra optimizer budget.

Unchanged fixed12×120s+conditional4×300s assessments every8 updates (17 max,
2,112,000 substeps/30s each). At most four qualified post-training candidates
get four uninterrupted1,800s starts (5,760,000 substeps,60/65s per assessment).
Reuse coverage80's baseline; never substitute union coverage for one swimmer.
Adoption requires12+4,4/4 long physical passes,zero faults/resets, minimum
individual coverage≥36.88405797%, mean≥42.89855072%, and maximum below6mm/s
interval≤11.1s. Rank worst individual coverage then mean. Report80% milestone
separately; it is not proof of whole-pond coverage or indefinite survival.

Conditional Godot adoption requires old-model compatibility plus selected300s
native replay, at most480,000 additional parity substeps, and UI/package checks.
Keep current Godot weights and DLL unless the candidate qualifies.
