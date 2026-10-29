# Approved creature experience and working roadmap

Saved October 27, 2026 at the user's explicit request. This is the guiding product plan, superseding an experience centered on watching prolonged untrained flailing or treating research metrics as sufficient release quality. Existing experiment reports remain historical evidence.

> A dependable young swimmer → increasingly challenging interactions → genuine measured adaptation → physical growth and new movement skills.

The app should be interesting immediately, work well when only watched, and support meaningful player interaction. The creature begins with competent learned swimming and grows more capable over time. An optional untrained/from-scratch demonstration can explain learning without making the ordinary experience wait through failure. The portfolio contribution is believable learned physical control, sound AI experiments, native C++/CUDA engineering and an engaging Godot presentation.

## 1. A dependable young swimmer — current milestone

Start with a trained controller that swims purposefully, collects reachable underwater food and recovers promptly. Keep physical equations, creature identity and source saves intact. The food-selector bank experiment is progress, not a finished reliability claim: it recovers 83/86 fresh bank starts within 20 seconds, while occasional ordinary long-run gaps remain.

First implementation step, now delivered: support evaluating the current and progress-aware food selectors through the **actual single-creature playback used by the app**, on CPU and CUDA. The original path is preserved. The [reliability report](creature_reliability.md) records immobility events, diagnostic snapshots, food timing, finite-state/underwater-food checks and actual backend identity. Sixteen thirty-minute calm-water sessions pass the provisional gate; this does not settle the retained difficult current cases. The latest bank-trained candidate can now run live in `pond/latest_agent.tscn`, separate from the original default. A recorded recovery comparison is also available. Neither viewer performs learning updates.

Initial engineering gate (provisional targets, not user-measured quality): no five-second sustained visible immobility in the declared normal-viewer test suite. A suspected immobility window is three seconds during which the head stays within 0.1 model units (0.4 cm) of its current location. Report qualifying events rather than moving/resetting the animal to hide them. Record food gaps separately; swimming slowly toward distant food and being stationary are different observations. Test normal thirty-minute sessions and retained difficult-state regressions. Passing a finite suite is bounded evidence, not a guarantee for all interactions or water.

Do not replace the visible controller merely because it has a higher reward or bank-recovery percentage. Use actual-playback evidence. If remaining failures require swimming-controller recovery training, diagnose them before selecting the next change. The user previously froze that controller for the selector experiment; physical recovery adaptation is a later evidence-driven decision.

## 2. Increasingly challenging interactions

First bounded interaction is implemented on `codex/pond-gentle-current`: the latest saved agent offers Calm/steady 0.5 cm/s flow in the live pond, preserving its physical state. [Measured response and limitations](experiments/swimmer/live-gentle-current/README.md) cover 24 five-minute CPU/CUDA sessions with no confirmed immobility; no new learning phase was needed for this sample. This is fixed-policy response to a real force change. Practice, challenge export/restore and candidate promotion remain to build, and harder bank cases remain separate.

Add one bounded interaction after the reliable baseline: for example, changing a gentle current or placing food at a valid underwater location. Keep the initial supported range physically attainable, with immediate understandable creature response. Later broaden to slowly changing currents and spatial variation, extending the fluid model where necessary. Do not promise arbitrary-water competence from a uniform-flow simulator.

The [frozen-agent slowly changing flow experiment](experiments/swimmer/rotating-current/README.md) now tests that next variation behind the scenes: 60 CPU/CUDA sessions / seven simulated hours with drifting food, 0.5 cm/s flow turning once per two minutes in both directions. It passes the declared collection/food-gap/immobility suite without training. Bank contacts are observed with explicit censoring; retained difficult bank starts remain a separate limitation. No live rotating-current control or new practice phase is delivered by this research result. Find a bounded meaningful weakness before adding training simply to produce weight updates.

The [0.75 cm/s follow-up](experiments/swimmer/current-075/README.md) also passes its declared suite: 100 sessions / 700 simulated minutes including calm and 0.5 references, both networks frozen, all protected saves intact. Stronger-flow collection retains 96.07%–101.13% of matched reference means, with no confirmed immobility. The visible pond still has the steady 0.5 switch. Turning uniform flow is not a spatial whirlpool. A future rock interaction needs a validated local disturbance/contact model, varied training and held-out testing; visual ripples alone do not supply physics or generalization. Do not promise competence under arbitrary unseen or physically overwhelming conditions.

The subsequent [live strength control](pond_current_controls.md) exposes the two evaluated steady strengths, 0.5 and 0.75 cm/s, alongside Current on/off. The same native vector drives body drag, food and visuals; switching preserves the session. Both networks remain frozen. This completes one bounded visible interaction, not actual practice/promotion or rock-impact physics. Turning flow remains in the research tools.

This stage is real reactive inference from learned skills. A changed action is not proof that network weights changed. Validate both the physical environment and agent observations before making the challenge harder. No new interaction UI or flow-field implementation is delivered merely by saving this plan.

## 3. Genuine measured adaptation

**Core requirement clarified October 28:** the creature must genuinely learn while the user interacts with its world. Frozen-policy interaction is an interim demonstration, not the completed product. Background native C++/CUDA PPO should practice conditions created by supported user interactions while the visible creature keeps running. Candidate improvements must periodically reach that same visible creature after measured improvement/skill-retention checks, preserving its current physical/session state. This is not a requirement to restart the app or manually leave play for a separate practice-only mode. Original policies remain frozen references; learner copies receive actual weight updates. Learning can plateau, and adoption timing/improvement must be shown honestly rather than promising an immediate gain after every interaction.

Keep a dependable controller in charge while a background copy practices unfamiliar conditions using PPO. Evaluate candidate updates on separate scenarios and retain existing abilities before promoting them. Preserve the full actor/critic/optimizer/world state so learning resumes rather than starting over. Show actual measured improvement and distinguish it from fixed-policy reactions. Practice copies remain copies of the same functional controllers, not a new brain per skill.

The user subsequently authorized the first [live sinking-food learning loop](live_feeding.md), requesting a brief starting test and rejecting a three-pellet UI limit. The new CUDA feeding scene adds native particles and a 128-active resource guard without replacement; the existing shared selector scores all active foods. Swimming inputs expand 33→37 with initially zero added weights, and native PPO adapts the inherited motor concurrently in user-influenced practice worlds. The selector stays frozen in this stage. The 1,024-world check demonstrates real weight updates, a validated update-16 actor reaching the same live body without resetting, and full optimizer/world/physical-pellet resume. The source policies and old saves remain separate. This is the first bounded implementation of the loop, not comprehensive robustness, guaranteed improvement after every interaction, spatial fluid simulation or morphology growth. The initial reference remains eight ten-second simulated trials; no pre-trained sinking actor is substituted for learning during play.

Define rewards, observations and supported tasks deliberately; the agent will not invent arbitrary new abilities merely because the player interacts. Never label a demonstration as learning if it only replays behavior or runs frozen weights.

## 4. Physical growth and new movement skills

Retain the approved creature identity across aquatic baby, new-limb stage and eventual upright adult. Growth changes the real articulated body, mass, inertia, drag and contacts. Extend/fine-tune a copy of the body controller with new joint observations and torque outputs; test how much learned swimming transfers. Train swimming alongside new capabilities to retain it. A controller operating multiple body stages needs appropriate body/limb-presence information and training coverage.

Food collection can participate in a designed progression trigger, while the ability to operate new limbs must be learned and verified. Crawling, shore transition and upright walking remain substantial later milestones. **No arms or legs are implemented in the current reliability step.**

## Shared architecture and priorities

- Two decision controllers: food/goal choice and swimming/body movement. PPO's value estimator is a training helper, not a separate new creature ability.
- Preserve the original capable creature, best tested actor, full learner state and honest fallback/error reporting. Neural actions obey the shared physical model.
- Optimize for the visible experience and verified adaptability. Higher GPU utilization, more training steps or a nicer graph does not establish a better creature.
- Keep [GPU optimization findings](gpu_optimization.md) for a later pass. Distribution, installers, broad platform expansion and additional cosmetics remain paused. The user-facing product is now Tadpole Pond: hide CartPole/Racing from its UI while preserving their code, scenes, policies, developer routes and training. Keep the underlying Godot application name/save paths stable.
- Work one milestone at a time. The broad roadmap is approved direction; a stage is only implemented when code, actual behavior and relevant verification support it. Record outcomes and residual failures in [the session checkpoint](SESSION_CHECKPOINT.md).
