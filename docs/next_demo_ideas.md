# Next demonstration — discussion notes

Recorded October 26, 2026 at the user's request. These are proposals, not approval to implement a new game. Preserve CartPole and Racing. Navigation Lab remains deferred.

The objective is a memorable employer-facing demonstration of Godot presentation, native C++ integration, and efficient hand-written CUDA reinforcement learning, without Python or PyTorch.

## Current discussion: physical creatures and water-to-land development

Current implementation decision: the user approved three simultaneous food pieces and explicitly requested learning to collect more food per minute by choosing efficient routes. This is implemented as a fresh 13-input policy trained by 1,024 CUDA swimmers, with consumed-only food replacement and continuous learning. The graph measures actual food per minute over fixed one-minute trials. The underlying torque-driven articulated physics is unchanged. See [three-food learning, results and limits](three_food_learning.md) and [physics](pond_physics.md). The prior one-food phase remains a regression fixture. Limbs and the full lifecycle are still deferred; keep future steps manageable.

Latest user feedback and implementation: food must cover the pond rather than cluster centrally, and an eaten piece must respawn well away from its previous location so one corner cannot be farmed. The current sampler uses a 3.9×2.8 ellipse, 2.2 minimum piece spacing, and at least 2.6 relocation into a different quarter, preferring an empty quarter. Other food stays fixed. The user also reported stopping at a folded tail pose while training continued. Full-range resting starts, an outward-at-stop reward penalty and three-food entropy regularization now train recovery; 64 folded starts and ten five-minute probes provide bounded regression evidence. Do not replace that with scripted swimming, teleports or periodic fresh starts. Tadpole Pond is now in the startup menu via Watch it learn, with Back to demos and Escape returning cleanly.

The approved racing-style statistics sidebar and bottom graph are implemented. Visible-animal metrics are distinct from background training counters. Pause/resume, paced speed and Maximum speed retain the current learner; Start over clears it and the graph. The original 10-second one-food success curve is superseded by mean food collected in 64 separate 60-second three-food trials, at update zero and every 25 updates. No update cap or periodic optimizer reset is applied. Food count does not prove graceful movement or optimal routing; observed slot preference and long-run stalls remain documented follow-ups.

The user selected the physics-creature idea for further exploration, not implementation. The latest proposal begins with a single-cell-like organism, adds a tail, then arms/hands and legs, and eventually has the agent swim, feed, crawl out of water, and walk on land. The user wants a compelling way for a person to interact while the app remains impressive when only watched.

Subsequent authorization: the user explicitly approved one small initially-untrained swimmer experiment and asked to watch it. That isolated C++/CUDA/Godot experiment is implemented; see [results and viewer](experiments/swimmer/README.md). This does not authorize the full lifecycle game. The user requested that each implemented morphology have a visible way to inspect it, prioritized a classic tadpole (tapered head, broad membrane, two actuated tail joints), and proposed baby eel and proto-amphibian alternatives. Those latter forms remain queued, not implemented. They asked for controlled morphology comparisons with success, median food time, travel per tail stroke, simulated work per distance and turning ability. Changes must affect physical forces, not only drawing. They also encouraged independent judgment rather than treating suggested changes as automatically correct. Simulated mechanical effort must be clearly distinguished from GPU usage/electricity; no creature stamina restriction is approved.

Preserve the prior creature concept: physical, learned limb control rather than canned walking animation; expressive creatures in a polished 3D world; optional environmental interaction and high-level direction control; possible later object retrieval/competition; robust evaluation on unseen terrain and disturbances; shared authoritative training/display dynamics and measured native CUDA performance. Earlier arena and four-legged-body suggestions are alternatives, not settled requirements.

Historically the user considered pretrained agents for the employer-facing demo; their later explicit decision is live training in the current pond. Preserve the distinction between real weight updates and behavior produced by a fixed policy. The eventual portfolio entry experience and lifecycle progression remain separate design questions.

Distinguish designed growth stages within a creature from biological evolution across generations and from learned control. Automatically discovering new bodies and their controllers is a separate research scope. Morphology changes, swimming/buoyancy, joint/contact stability, simple grasping, and the water-to-land transition are substantial unresolved engineering work. No open-ended intelligence, emergent strategy, simulation speed, or training time is established.

Proposed interaction direction: the visitor shapes a small habitat, places reachable food, moves physical objects and creates routes/challenges; agents choose their own physical actions. Preserve optional spectating, persistent creature identity, and an immediate satisfying result. Keep hand control simple before considering individual finger joints. A bounded feeding/swimming task and a low-bank water-to-land transition would provide concrete feasibility gates before expanding the lifecycle. These are discussion proposals only.

## User's RTS proposal

Two different races battle and learn against each other while the viewer watches. The user raised asymmetrical balance as a concern and has not decided whether human gameplay is needed.

Proposed evaluation approach: establish a symmetric combat/training baseline, then introduce controlled race differences. Distinguish rules balance from training strength. Evaluate saved policies against fixed opponents and older checkpoints, across map sides and seeds; current opponents splitting wins evenly is not evidence that either improved or that the races are balanced. A small opponent pool may help limit forgetting and strategy cycling. Start with a bounded combat task before committing to a complete economy, construction tree, or long match horizon. All of these choices remain open for discussion.

## Alternative to remember: team arena / capture the flag

Assistant proposal retained at the user's explicit request: a compact 3D arena inspired by Unreal Tournament's capture-the-flag mode. Two teams try to take the other team's flag back to their own base while defending their own. Spectating would work immediately, with optional following of an agent, changes to arena conditions, and later taking control of a participant.

Potential learned behaviors include pursuit, evasion, interception, and cooperation; none is established or guaranteed. The classic games already had AI: the proposed contribution is learned policies backed by measured evaluation, not a claim that older games lacked AI. A bounded movement/pursuit task would precede full combat and team objectives.

## Shared constraints

- Prefer a convincing, focused 3D scene and measurable intelligent behavior over unbounded engine feature scope.
- Share authoritative gameplay rules between CUDA training and the displayed simulation. Godot rendering/physics does not automatically become a GPU-batched training environment.
- Separate policy inference, which responds to observations with fixed weights, from actual learning updates.
- Make spectator use valuable without requiring a reviewer to learn game controls. Human participation remains optional and undecided.
- Choose an experiment to prove learnability and training cost before promising large populations or complex emergent strategies.
