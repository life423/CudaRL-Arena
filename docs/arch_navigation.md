# First obstacle-learning pond

This local development scene adds a larger habitat and one submerged rock arch. Food appears automatically on opposite sides. The initial calm-water stage is now followed by bounded current interaction, with suspended food drifting through the water.

The original feeding pond, its trained policies and full saves remain separate. This prototype is not part of the public installer or website.

**Latest habitat stage:** the [boulder pond](experiments/swimmer/boulder-navigation/README.md) runs in `godot-project/pond/boulder_pond.tscn`, importing the complete cove learner into its own save and adding one solid obstacle plus a bounded physical downstream wake. The preserved [natural cove pond](experiments/swimmer/cove-navigation/README.md) runs in `godot-project/pond/cove_pond.tscn` and imports the full arch learner. This guide's original `arch_pond.tscn` remains the legacy basin/source. All retain online PPO/current controls; none replaces the public feeding build.

## How to watch

Open `godot-project/pond/arch_pond.tscn` in Godot and run that scene. The native extension must include the arch environment. This first local training prototype requires a healthy NVIDIA CUDA device; ordinary feeding still supports CPU learning.

The visible creature starts with the newest validated swimming actor from the existing full CUDA feeding save. A copy learns obstacle navigation through native PPO in 1,024 unseen practice worlds. The inherited food selector chooses between active foods. The swimmer's four outputs still apply joint torques through the shared physical solver; there is no scripted route or movement.

The graph measures the visible swimmer's actual session collection rate. AI details show the background learner, matched evaluation and adopted actor updates separately. An increase in training updates does not establish improvement. Candidates only reach the live swimmer after passing the measured improvement and calm-skill checks, and adoption preserves its physical pose and session clock.

Use Current on/off and the 0.5/0.75 cm/s selector to change the water without resetting the creature. Food, physical drag and native visual tracers use the same local field. Close the window normally to save and stop. Reopening resumes the separate arch checkpoint and saved current. Manual food controls remain hidden in this automatic-food task.

## Physics and perception

The basin's horizontal radii increase from 4.6/3.5 to 6.9/5.25 model units; creature size and physical units are unchanged. One model unit represents four centimetres. Water depth remains the existing analytic basin profile.

The rock is an analytic torus with its bottom buried in the bed. Its center is `(0, -2.25, 0)`, major radius 1.35, tube radius 0.5 and opening radius 0.85. Native geometry supplies the Godot mesh dimensions. Torques, distributed water drag, gravity, buoyancy and compliant articulated contacts determine motion. Local prescribed flow deflects near boundaries and vanishes inside solids; it is not an incompressible fluid, wake or turbulence solver. Contact samples cover the head and both tail links, including fin clearance.

Eighteen body-relative distance rays supplement the existing 37 motor observations. The actor expands to 55 inputs by copying inherited weights and setting all new input weights to zero. Before fine-tuning, the expanded actor produces the same actions even with arbitrary obstacle readings. The two decision controllers remain food selection and body control; PPO's critic is a training helper.

A passage is counted only after the whole articulated body changes sides through the opening. Collecting food by going around the rock is valid food collection but is not an arch passage. These counters observe behavior and never reposition the creature.

## Training and preservation

One quarter of background worlds retain the old calm-water task and another quarter retains the calm arch task. With current enabled, the remaining worlds practice random steady and smoothly varying speeds bounded by the selected strength, with genuine calm intervals. Safe initial foods have clearance from the creature and rock; collection replenishes food across the arch rather than at the same spot.

The initial reference is eight ten-second trials. Every 16 updates, the inherited actor, adopted incumbent, pre-current calm reference and candidate run matched fresh 16-world, 30-second comparisons. Promotion requires finite behavior without a recorded five-second immobility span, separately retained old-calm and calm-arch food collection, no fewer foods and no more contact ticks than the incumbent, plus progress or clean-passage improvement. Current-context changes during evaluation prevent stale publication. These are development gates, not proof of reliable navigation or optimality. Longer unseen-layout and actual-playback evaluation remains necessary before replacing the ordinary pond.

The source is the outer **validated** 37-input actor from `user://pond-policies/feeding-learning-v1/training.bin`, not its unvalidated latest learner. The separate `user://pond-policies/arch-navigation-v1/training.bin` stores source provenance, visible world, applied and pending validated actors, current context and full PPO actor/critic/optimizer/practice state. The folder name stays stable; ARCH v3 reads v1/v2 without resetting training and records basin layout outside the unchanged raw world state. The original feeding file is read-only. Resuming the arch stage uses its embedded source even if the personal feeding learner changes later. The cove stage separately imports this complete arch save; it does not restart from the old 37-input feeding actor.

## Verification record

The [initial arch verification](experiments/swimmer/arch-navigation/README.md) records inherited-weight learning and the contact bug found during verification. The [current follow-up](experiments/swimmer/arch-current/README.md) reports 202/202 native tests, 2/2 portable suites, 21 byte-level migration checks and 52 rendered checks with actual 1,024-world PPO, current context and full resume. These are bounded development checks, not proof of obstacle or current mastery.
