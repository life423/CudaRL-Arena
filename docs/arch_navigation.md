# First obstacle-learning pond

This local development scene adds a larger habitat and one submerged rock arch. Food appears automatically on opposite sides. The initial task uses calm water and suspended food so obstacle navigation can be studied separately from sinking food and flow around rocks.

The original feeding pond, its trained policies and full saves remain separate. This prototype is not part of the public installer or website.

## How to watch

Open `godot-project/pond/arch_pond.tscn` in Godot and run that scene. The native extension must include the arch environment. This first local training prototype requires a healthy NVIDIA CUDA device; ordinary feeding still supports CPU learning.

The visible creature starts with the newest validated swimming actor from the existing full CUDA feeding save. A copy learns obstacle navigation through native PPO in 1,024 unseen practice worlds. The inherited food selector chooses between active foods. The swimmer's four outputs still apply joint torques through the shared physical solver; there is no scripted route or movement.

The graph measures the visible swimmer's actual session collection rate. AI details show the background learner, matched evaluation and adopted actor updates separately. An increase in training updates does not establish improvement. Candidates only reach the live swimmer after passing the measured improvement and calm-skill checks, and adoption preserves its physical pose and session clock.

Close the window normally to save and stop. Reopening resumes the separate arch checkpoint. Current and manual food controls are hidden in this controlled first task.

## Physics and perception

The basin's horizontal radii increase from 4.6/3.5 to 6.9/5.25 model units; creature size and physical units are unchanged. One model unit represents four centimetres. Water depth remains the existing analytic basin profile.

The rock is an analytic torus with its bottom buried in the bed. Its center is `(0, -2.25, 0)`, major radius 1.35, tube radius 0.5 and opening radius 0.85. Native geometry supplies the Godot mesh dimensions. Torques, distributed water drag, gravity, buoyancy and compliant articulated contacts determine motion. This is a reduced physical model, not computational fluid dynamics or an exact continuous mesh collision solver. Contact samples cover the head and both tail links, including fin clearance.

Eighteen body-relative distance rays supplement the existing 37 motor observations. The actor expands to 55 inputs by copying inherited weights and setting all new input weights to zero. Before fine-tuning, the expanded actor produces the same actions even with arbitrary obstacle readings. The two decision controllers remain food selection and body control; PPO's critic is a training helper.

A passage is counted only after the whole articulated body changes sides through the opening. Collecting food by going around the rock is valid food collection but is not an arch passage. These counters observe behavior and never reposition the creature.

## Training and preservation

One quarter of background worlds retain the old calm-water task. The remaining worlds use arch navigation, with modest variations in arch position/yaw and food placement. Safe initial foods have clearance from the creature and rock; collection replenishes food across the arch rather than at the same spot.

The initial reference is eight ten-second trials. Every 16 updates, the inherited actor, adopted incumbent and candidate run matched fresh 16-world, 30-second comparisons. Promotion requires finite behavior without a recorded five-second immobility span, retained calm food collection, no fewer arch foods and no more contact ticks than the incumbent, plus progress or clean-passage improvement. These are development gates, not proof of reliable navigation or optimality. Longer unseen-layout and actual-playback evaluation remains necessary before replacing the ordinary pond.

The source is the outer **validated** 37-input actor from `user://pond-policies/feeding-learning-v1/training.bin`, not its unvalidated latest learner. The separate `user://pond-policies/arch-navigation-v1/training.bin` stores source provenance, visible world, applied and pending validated actors, and full PPO actor/critic/optimizer/practice state. The original feeding file is read-only. Resuming the arch stage uses its embedded source even if the personal feeding learner changes later.

## Verification record

The [verification record](experiments/swimmer/arch-navigation/README.md) reports 196/196 native tests, 2/2 portable suites and 62 rendered checks with actual 1,024-world PPO, validated adoption and full resume. It also records the contact bug found and fixed during verification, protected-source hashes and remaining reliability limits. These are bounded development checks, not proof of obstacle mastery.
