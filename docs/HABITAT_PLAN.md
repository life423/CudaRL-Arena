# Obstacle habitat plan

Prepared October 29, 2026 at the user's request. Stage 1 is now implemented locally and open for live review; later stages remain proposals. The immediate scope is a believable, navigable underwater habitat. Shelter-seeking follows the obstacle work; limbs and land movement remain later.

## Starting point

The local scene already has a larger basin, one physical rock arch, native C++/CUDA PPO, inherited swimming and food-selection controllers, real learning/save/resume, and bounded 0/0.5/0.75 cm/s current controls. The pre-current checkpoint is commit `fc65b72`; the current follow-up is still uncommitted. See [the current report](experiments/swimmer/arch-current/README.md). Preserve the working scene, full learner and approved actors before geometry changes.

**Stage 1 delivered:** `pond/cove_pond.tscn` adds an outward asymmetric cove with shared native bed/contact, current, food and rendered geometry. The main central depth profile is retained and no old water volume is cut away. Automatic food covers the main pond and cove. The separate child imports the entire parent arch learner at update **4,539**, including the approved actor **4,352**, optimizer, practice worlds and visible physical state. The parent save remains unchanged. Native geometry/parity, full-save migration and rendered checks pass; the inherited actor visits and collects food in the cove in brief calm/0.5/0.75 cm/s probes. This establishes a working local stage, not broad reliability or a measured learned improvement. See [the cove report](experiments/swimmer/cove-navigation/README.md). Public feeding/download builds remain separate.

## Proposed sequence

| Stage | Change | Purpose | Evidence before proceeding |
| --- | --- | --- | --- |
| 1. Natural outline | Keep the main swimming volume and add one smooth asymmetric cove. Preserve existing depths initially. | Give the pond a natural outline and an additional reachable area. | Visible shoreline matches native water/bed boundaries; food and saved creatures remain valid; CPU/CUDA geometry agrees; old controller is tested before deciding what needs training. |
| 2. One boulder | Add a solid rock interrupting a direct route, with two generous alternatives. Keep the existing arch. | Learn when to go around a solid obstacle instead of pushing toward a blocked target. | Whole-body clearance, reachable food, recovery and measured routes; calm first, then existing currents. |
| 3. Rock cluster | Arrange a small number of solid rocks to create one broad channel/gap. | Choose a passage and control approach/body orientation. | Gap admits the complete articulated body with margin; no initial overlaps or inaccessible food; unseen gap placements tested. |
| 4. Short tunnel | Add a hollow log or short broad rock tunnel. | Enter, traverse and leave an enclosed passage. | Contact/perception match the hollow visual; head entry alone earns no completed-passage credit; no long confinement or false rescue. |
| 5. Depth and layout variation | Vary safe obstacle positions, opening heights, food locations and routes through the habitat; introduce a depth-defined route separately from the initial shoreline change. | Navigate unfamiliar arrangements in three dimensions and use the broader pond. | Separate practice, candidate-gate and withheld final layouts; retained old swimming and current tests; longer actual live sessions. |
| 6. Shelter behavior | Identify or build a physically useful protected area, then teach when to reach/use it. | Add a purpose beyond food collection using the completed habitat. | The refuge actually reduces the declared disturbance; learned response is measured separately from reactive inference. |

Stage 1 is implemented and shown live. Review this stage and its remaining reliability checks before placing the boulder. Each later stage introduces one challenge rather than adding the entire habitat at once.

## First stage in concrete terms

Use an outward extension of the existing pool for the first cove. Avoid cutting away the water that contains the current creature, food or saved practice worlds. Keep the present swimming depth profile in the main pool; do not combine shoreline expansion with a new shallow-crawling or land task.

The implemented `cove-basin-v1` descriptor supplies native bed height/gradient, water containment, food clearance and current boundaries; Godot samples that same terrain and shoreline. Layout metadata is outside the unchanged raw world-state structure, with tested ARCH v1/v2-to-v3 migration. The legacy arch scene keeps its original basin. Future obstacles still require extending torus-only sensing/contact deliberately; changing only a visible mesh would leave hidden old physics.

Make the new area genuinely usable. Include valid cove/return placements in later coverage tests and automatic practice, with whole-body clearance and food that can be reached. Keep automatic food generation; the user should not have to provide thousands of training examples. Existing many-pellet feeding support should remain available in its working scene while the bounded habitat task develops.

An indentation or bay is only a potential refuge. Its shape does not establish calmer water, and the current prescribed field does not simulate a wake. Shelter behavior waits until the physical environment offers a measured reason to use it.

## Shared engineering and learning rules

- Preserve PPO, native C++/CUDA training and the two decision controllers. Carry learned weights forward. Do not start a fresh unrelated network for each obstacle or skill.
- Test the inherited controller briefly on each changed environment before choosing additional training or observations. Keep ordinary swimming and prior obstacle/current practice in the training mix.
- The 18 obstacle rays currently see the torus. Introduce a shared obstacle-distance query for boulders, clusters and tunnels; decide explicitly whether bank sensing is needed. Preserve useful weights when extending observations. Identical weights alone do not guarantee retained behavior when sensor meanings or physics change.
- `deep::Habitat` is embedded in the raw saved arch state. Adding shoreline/obstacle fields directly can invalidate full saves. Use a versioned descriptor or tested migration and verify actor, critic, optimizer, worlds and visible state, as in the v1-to-v2 current upgrade.
- The same native geometry and local flow must drive CPU reference, CUDA training, visible body, food and tracers. Preserve all original reference policies/full saves.
- Do not hide navigation failures with teleports, forced steering, invisible obstacle removal or live-body resets. Background episode resets are training machinery, not a deployed recovery skill.
- Keep each new stage observable live in Godot once its basic physics works. A new layout can be a development scene until its learned controller passes validation; the known working scene stays available.

## Useful ideas recovered from the older plans

Reviewed October 29 at the user's request before implementation. These findings refine the proposed plan; they do not authorize implementing every old idea. Later explicit decisions and current architecture override historical status statements.

| Recovered idea | Older source and status | Where it fits |
| --- | --- | --- |
| Save and practice difficult encounters | [Architecture audit](CODEBASE_AUDIT.md), lines 98–103; authentic difficult-state practice also improved the old [bank recovery experiment](experiments/swimmer/gentle-currents/bank-recovery/README.md). Generic practice/promotion exists now, but automatic collection of the visible creature's failed obstacle encounters is still unfinished. | Version the habitat so failures can be reproduced; add a bounded challenge bank as actual boulder/tunnel failures appear. Capture physical state, food, goal/decision history, flow, layout and policy identity. Practice nearby restored starts using **fresh on-policy PPO rollouts**, not stale rollout replay, mixed with ordinary old tasks. Never reset the live creature as a rescue. |
| Raise difficulty after demonstrated competence | [Architecture audit](CODEBASE_AUDIT.md), line 101. Current groups randomize speeds, but are not a competence-driven obstacle curriculum. | Start with broad gaps and short tunnels. Increase offset, length, orientation and current variation only after the easier declared suite passes, retaining successful earlier tasks. This is a training thread through the existing stages, not an extra obstacle stage. |
| Make food exercise the whole habitat | [Three-food learning](three_food_learning.md), line 11, and [deep pond](deep_pond.md), line 69. Region-changing, separated placements existed in older tasks; the current two-food arch task is intentionally narrower. | Include reachable cove/return targets during the shoreline stage, then targets behind/through each new obstacle. Avoid retaining only the same two arch-side locations until the end. Preserve useful region/spacing principles, not the old three-pellet UI limit. |
| Diagnose target choice separately from movement | [Food selection](pond_food_selection.md), lines 13–15, and [saved stall diagnosis](experiments/swimmer/gentle-currents/diagnosis/README.md), lines 13–25. An unchanged motor escaped when a diagnostic branch chose another goal; nearest-food and removing current did not resolve that case. | For a recorded obstacle stall, test reachability, selector choice and motor capability before deciding which controller needs practice/features. The current arch selector is frozen; more motor PPO is not the assumed solution to every failure. A learned temporary open-water escape goal is a contingency if evidence requires it, not a scripted target override or a third network by default. |
| Preserve invariance to object ordering | [Food-order bias](experiments/swimmer/food-order-bias/README.md), lines 35–45. Shared food scoring already addressed the old slot problem; repeating the shuffled-food experiment is unnecessary. | New shared obstacle queries should return the same sensed geometry when the rock list is reordered. Add a concrete permutation test rather than giving arbitrary object indices a new neural meaning. |
| Measure graceful movement as well as collection | [Physical quality plan](pond_physics.md), line 68, and [swimmer ideas](next_demo_ideas.md), line 67. Speed variation, turning, joint smoothness and mechanical work/travel were proposed diagnostics. | Track route/arrival time, sustained stalls, turning and command smoothness alongside food/minute. The old [autonomous comparison](experiments/swimmer/autonomous-heldout/README.md), line 18, collected about 96% of supplied food, so pellet arrival rate can hide useful navigation gains. Add measurements before changing rewards; body mechanical effort is separate from GPU electricity and does not imply a new stamina restriction. |
| Keep navigation observable | [Pond presentation](pond_presentation.md), lines 19–23, describes the stable camera and render-only occlusion aid already implemented for the pond. | Preserve or extend the existing aid only where new rocks/logs hide the creature. Contacts remain solid while presentation makes passage/recovery readable. This is functional visibility, not a return to the deferred graphics backlog. |

The strongest near-term additions are encounter-specific recovery practice, competence-based difficulty and whole-habitat food coverage, with selector-versus-motor diagnosis when failures appear. They keep the shoreline → boulder → gap → tunnel sequence intact.

Keep these older ideas later: bounded player repositioning of obstacles after fixed layouts work ([next demo ideas](next_demo_ideas.md), line 75), then limbs with an actual underwater bracing purpose after the obstacle/shelter work ([architecture audit](CODEBASE_AUDIT.md), line 102). Preserve the [approved fictional creature identity](concepts/creature-growth-v1.md); articulated arms/hands were not settled by that concept. Slowly changing current direction is a useful later obstacle test, but the older open-water result is not proof for the new habitat.

Older README/audit statements about racing being the current focus, missing live learning/flow control, a permanently frozen motor or only three selectable foods are historical. The attention-policy roadmap, further GPU optimization, distribution and cosmetic upgrades remain separate/deferred; do not revive them from old wording during this habitat pass.

## Common validation gate

For every stage, verify finite physical state, native/render geometry agreement, CPU/CUDA agreement, food reachability, body/tail clearance and full save continuation. Measure completion, collection rate, route time, contact duration, food gaps and immobility. Report around-obstacle routes separately from through-opening passages.

Use fresh candidate-gate layouts and a separate withheld final set; do not train on the final evaluation cases. Check calm water and the existing 0.5/0.75 cm/s controls. Preserve the current provisional product criterion of no five-second confirmed immobility event in the declared finite normal-viewer suite, and investigate actual playback failures. A finite pass is evidence for the tested conditions, not global robustness or optimality.

Before the combined habitat is treated as dependable, run longer actual live-path sessions across multiple safe layouts and seeds, compare retained old tasks and inspect the rendered result. Do not equate higher training reward, more updates or completed meshes with better navigation.

## Experience and scope

The creature stays easy to see, with the current stable overview/closer views and restrained interface. Preserve visible creature identity, learning/current feedback and straightforward closing/saving. The habitat changes should create useful route choices; unrelated decoration, broad graphics redesign, distribution work, stronger/turbulent flow and anatomy expansion are outside this pass.

The broader journey remains: dependable swimmer → challenging interactions → measured adaptation → physical growth and new movement skills. This plan completes the obstacle foundation before shelter, then later underwater bracing, crawling and land movement can build on retained swimming.
