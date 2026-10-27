# Stopping point — October 27, 2026

The user requested committing the food-selection work to its branch, merging into main, and a stopping point for sleep. Do not start another experiment, leave training running, or schedule background work on their behalf.

## Delivered

- Deep water plays the packaged learned food selector paired with the retained update-64,350 swimming network. Four tail torques, shared physical dynamics and the existing scene are preserved.
- The selector chooses among all three foods using a shared scoring network trained with native categorical PPO. Recovery training uses independently sampled resting joints and longer episodes. The original training save is intact; deployment has no rescue, teleport or training reset.
- Fresh results: 17.5625 versus 15.363281 food/minute across 512 one-minute worlds; 17.834375 versus 15.690625 across 128 five-minute worlds. This is the strongest tested controller, not proof of global optimality or a guarantee against every future stall.
- [Raw results, failed-candidate evidence and provenance](experiments/swimmer/deep-pond-food-choice/README.md); [architecture and usage](pond_food_selection.md).

## Saved progress and reopening

- Open **Tadpole Pond → Deep water** for saved-controller playback. It is explicitly labeled as trained inference, with no background learning.
- The packaged controller and recorded-stall fixture are versioned under `godot-project/pond/policies/deep_selector*.bin`.
- Full resumable selector training is at `user://pond-policies/deep-pond-selector-v1/training.bin`: current phase update **800**, best phase update **250**, seed **47**, frozen motor update **64,350**. This local full save includes optimizer/world state and is outside Git. The menu Training guide resumes it headlessly; further candidates do not replace the packaged model automatically.
- Original motor training remains at `user://pond-policies/deep-pond-v1/training.bin`. Both full saves were copied into a verified `pond-policies/sleep-checkpoint-2026-10-27-*` backup folder before committing. Preserve these local files.
- No Godot game or native headless trainer was running at the stopping-point process check; only the Godot project manager remained. No ongoing training is required to preserve the saved progress.

## Verification

Release build passed. **154/154 native tests**, GPU required, no skips. **374 Godot checks** passed: saved controller 27, original deep-pond training/resume 67, presentation 138, menu 103, original deep water 39. Packaged-only evaluation reproduced holdout A's cases byte for byte. A separate copy resumed from selector update 800 to 801; the installed save remained at 800. Rendering/layout checks covered desktop, tablet, portrait phone and landscape phone; physical touch/assistive-device checks remain pending.

## Next discussion

Keep cosmetic work deferred. The larger goal remains a consistent fictional creature that retains swimming while developing two weight-bearing legs, then learns crawling and eventually upright locomotion with PPO and native neural networks. Legs, shoreline crossing and walking have not been implemented or authorized as the next coding task. Review the current controller briefly, then agree on one bounded anatomy/physics/retention experiment before coding the next stage. Avoid restarting existing skills from random weights or claiming biological evolution from designed growth stages.
