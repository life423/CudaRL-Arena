# Approved creature experience and working roadmap

## Master direction — restored October 30, 2026

Build a physically simulated creature that learns increasingly capable behavior
through our own C++/CUDA PPO system. Godot owns presentation, interaction and
inspection, not hidden propulsion or a replacement learning runtime. Python and
PyTorch are not being introduced as training/runtime replacements.

The pond tadpole is the first body stage. The product journey is dependable
swimming → meaningful player interaction and measured real-time adaptation →
physical growth and underwater limbs → shallow crawling and leaving the pond →
walking and eventual manipulation. Those later stages remain unimplemented.
Changing anatomy requires real mass/drag/joints/contact and explicit compatible
policy/schema transfer, followed by tests of retained skills.

Keep exactly two learned behavioral policies: a goal chooser for feeding,
safety, exploration and holding/rest, and one reusable movement motor. PPO
critics support training; they are not behavioral controllers. The chooser owns
persistent intentions; the motor executes destination, terminal velocity and
heading requests. See [the ownership/request contract](CREATURE_CONTROLLER_INTERFACE.md).
Supply requests directly in the current motor study; do not train the chooser.

Movement acceptance covers purposeful travel, turning, braking, controlled
arrival and sustained position/heading holding. Report unsupported calm holding,
holding relative to the pond against a specified validated current, and
supported bottom rest separately. Active joint motion is allowed. Food rate,
servo regulation, imitation error and training counters do not establish these
skills. Preserve the registered tolerances, actuator limits and study budgets.

Cleanup serves this direction: simplify scene/session ownership, remove proven
obsolete paths and duplicated orchestration in buildable groups, and preserve
the simulator/PPO foundation, personal saves and reproducibility. Organization
problems are not proof of the cause of a learning failure. The current contact
gate failure below remains a separate physical issue; no new learning is running.

**Newest physical prerequisite — October 30:** contact physics v2 is implemented
as an isolated research path, but its mandatory dynamic gate failed before any
new motor learning. [MOTOR_V2_RESULTS.md](MOTOR_V2_RESULTS.md) separates the
133 geometry and 2,400 CPU/CUDA comparison passes from the rolled-head recontact
failure at all three timesteps. Bottom contact is now allowed within the fixed
0.5 mm allowance; excessive overlap and escape remain failures. Only 4,440
contact substeps ran. Support-window/free-water/final timestep acceptance are
uncompleted; readiness, PPO and policy evaluations are zero. No new interface
winner or dependable movement claim exists. The next research decision is a
bounded contact-only normal-response correction, requiring new authorization.
Keep the two-policy design, old swimmer, historical servo gains and all saves.
The later whole-Godot audit/cleanup is separately authorized presentation work;
it does not reopen the stopped contact or learning experiment.

**Newest motor comparison — October29:** keep exactly two learned behavioral
policies: goal chooser and one reusable motor, with PPO critics explained
separately. [The request contract](CREATURE_CONTROLLER_INTERFACE.md) is now
documented; chooser training remains excluded. The new isolated torque/fast-servo
experiment ran under [its preregistration](PAIRED_MOTOR_PROTOCOL.md).
Joint regulation passed8/8 on CPU andCUDA and data readiness passed. Torque PPO
stopped after6completed updates when rollout7 crossed the strict distal-fin/bed
clearance while native contact proxies remained clear. Joint-target PPO has
0updates; no final behavioral comparisons ran and no candidate is accepted.
Read [PAIRED_MOTOR_RESULTS.md](PAIRED_MOTOR_RESULTS.md) for actual budgets and
the exact geometry mismatch. Preserveglobal-stop, successfulservo gains, the
working pond/body/saves and previous results. No additional run or contact
change is authorized automatically; discuss the narrow geometry-consistency
decision before another study. Feeding, safety choice, exploration/rest and
growth remain behind dependable move→brake→stop→hold.

**Latest presentation decision — October29:** retain the current pond and old
tadpole while improving the new motor. The user deferred removal/replacement and
a larger water environment until the new creature works better. No production
UI change or replacement was made. Keep research comparisons separate and do
not turn this timing decision into permission for a new training sweep.

**Newest bounded motor result — October29:** numerical and actual stream/data
checks pass, but the newly authorized two-capture braking probe fails readiness.
The full conventional16/16 prerequisite was removed for research; its old failure
remains historical. Exact61,440+1,600 search/replay steps found no safe braking
plus10s quiet hold; no PPO launched. Read [motor braking results](MOTOR_BRAKING_RESULTS.md).
Discuss one bounded joint-target/fast-servo interface comparison; do not add
another brain, extend the search or weaken final movement gates. No redesign
is implemented by that recommendation. Goal selection remains separate.

**Latest motor replan — October 29:** dependable physical movement is the next
gate, not food rate, imitation accuracy or counters. The user permits replacing
the motor/training setup while preserving all old work for rollback. Keep goal
choice separate from one reusable body controller; no network per skill.
[The motor plan](MOTOR_CONTROL_PLAN.md) recommends a persistent-command locomotion
task and calm open-water move→brake→stop→hold before shelter and predator choice.
The supervised pilot held5/16 unseen cases with cover-only fitting (feeding
regressed),1/16 with joint fitting (feeding retained), and0/4 familiar nominal
starts for both. None passes the full gate. Small residual commands amplify
after creating joint/rate/previous-action states absent from the stationary
labels. This is not a capacity proof or a diagnosis of a broken numerical core.
The tiny independent real-actor check now passes. The32-trial physical preflight
does not: fixed conventional0/16 full passes, passive2/16. The isolated task,
bounded trainer and live motor bench are implemented, but **zero PPO pilot
updates ran** because the physical gate remains false. Read
[motor results](MOTOR_CONTROL_RESULTS.md). Resolve braking/reference adequacy
before expanding trials; no new motor is adopted and no app rewrite has started.

**Latest shelter-object order — October 29:** add an open-sided rock ledge
leaning out from the back bank **before lily pads**. Native collision and
predator sight obstruction must match the displayed solid, with an opening
facing the viewer. Keep this opt-in habitat separate from the working boulder
learner and its saves. The user asked to open the predator demonstration only
after its hiding test passes. Adding cover geometry is not learned hiding;
the control/action-path diagnosis remains pending before further PPO variants.

**Latest shelter decision — October 29:** the user chose a **passing predator
shadow**, rather than a current surge, as the first event for prioritizing
cover over food. The [bounded pilot](experiments/swimmer/predator-shelter/README.md)
has native whole-body occlusion, separate full saves and gated two-controller
PPO, but has **not passed holding readiness or replaced the boulder swimmer**.
Both brief body-training trials retain feeding but achieve 0/24 five-second
holds. Diagnosis shows strongly saturated raw joint commands in this new task;
review action parameterization/transfer before more training. Preserve the
strict tests and source actor. Lily pads, an open-sided stone ledge, a hollow
log, exposed roots and sparse leaf cover are discussed options for visible
shelter, not authorization to add every object. Keep the creature visible to
the camera while native predator sight obstruction remains real.

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

**Autonomous practice approved next:** the user does not want to feed thousands of training worlds manually and explicitly requested random current speeds, fluctuations and no-current conditions. The local follow-up replaces immediate food refill with timed seeded arrivals, mixes random pond-wide and recent player placements, and adds calm/steady/smoothly varying 0–0.75 cm/s practice groups without changing the swimmer/particle force equations, PPO or save layout. An optional visible Auto feed toggle supports unattended watching; its current controls remain manual while background conditions vary independently. [Protocol](experiments/swimmer/autonomous-practice/PROTOCOL.md). Validate actual learning, retention and copied-save continuation before claiming useful generalization. Do not infer mastery of arbitrary currents from the implementation checks.

## 4. Physical growth and new movement skills

**First obstacle environment implemented locally October 29:** after the autonomous-practice comparison showed no clear food-rate gain in mild open water, the user approved a larger pond, one real rock arch and automatic food on both sides, initially in calm water. The [arch stage](arch_navigation.md) preserves the inherited selector and expands the validated swimming actor from 37 to 55 inputs with function-preserving transfer. Native CPU/CUDA contacts and sensing, actual CUDA PPO, matched adoption without a visible reset and separate full resume pass the bounded development checks; see the [verification record](experiments/swimmer/arch-navigation/README.md) and latest [session checkpoint](SESSION_CHECKPOINT.md). The user wants this live Godot scene opened when ready. Default feeding/public release remain separate until longer reliability and unseen-layout checks justify promotion. Later explore varied routes, tunnels/logs, local flow and growth. A visual rock alone does not teach navigation. Do not expand beyond this first milestone automatically.

**Priority clarified October 29:** the user likes shelter-seeking as a future learned behavior, but wants to finish the pond's obstacle habitat and navigation work first. The existing arch plus bounded current is the foundation, not completion of the broader habitat idea. Prioritize real route choices through/around obstacles, body-clearance and recovery, and tested navigation in calm/current conditions before shelter, holding-position tasks or limbs. A solid boulder creating an alternate route is the next suggested manageable addition; its exact layout and further obstacle types remain proposals, not implemented features or a previously approved fixed obstacle list. Shelter must later correspond to a genuinely useful protected region in the physical model; the present prescribed field does not simulate a rock wake.

**Shoreline stage implemented locally October 29:** the user approved the first step of the habitat plan. `pond/cove_pond.tscn` now adds an outward asymmetric cove while retaining the main central depth profile and all previous water volume. Native terrain/contact, food clearance, current boundaries and Godot geometry share the versioned basin definition. Automatic food draws the inherited swimmer through the main pond and cove. The new separate full save imports the parent arch learner at update **4,539** and its approved actor **4,352**, retaining actor/critic/optimizer, practice worlds and visible state; the parent save is unchanged. [Verification](experiments/swimmer/cove-navigation/README.md) includes native CPU/CUDA agreement, old-save migration, rendered/live learning checks and brief frozen-actor collection in the cove at calm/0.5/0.75 cm/s. Longer and withheld-layout reliability remains open; no new-policy improvement or physically calmer shelter is claimed. No boulder, new depth route or limbs are added in this step.

The requested [obstacle habitat plan](HABITAT_PLAN.md) records the manageable sequence: outward irregular shoreline/cove with existing main depths → one boulder → small rock cluster with a safe gap → short hollow log/tunnel → depth/layout variation and reliability → physically useful shelter behavior. Each stage has geometry, save-preservation and learning gates, and a live Godot review. The shoreline/cove and one boulder are implemented locally; the cluster, tunnel and shelter remain proposed. Review this working stage before choosing the next addition.

**Next stage authorized October 29:** commit the verified cove, then add one solid boulder while carrying forward its entire current learner. The suggested later shelter event is a brief signaled current surge. The user specifically requires weaker downstream flow to affect actual physics. The boulder work therefore includes a bounded shared wake model and checks its physical effect; this does not implement the surge event or teach shelter-seeking yet. Keep navigation, physical shelter and learning when to shelter as distinct verified steps.

**Boulder implemented locally:** `pond/boulder_pond.tscn` imports the full cove learner at update 5,491/applied actor 5,136 into its own save. One bed-rooted sphere has shared native contact/rays/food/current and exact rendered dimensions. Bounded directional wakes affect body-relative drag, food and tracers; a tested point pair reads 0.50→0.248 cm/s. Actual PPO, retained cove checks, widened food coverage and honest per-visit metrics are included. Verification caught and fixed a nonblocking CUDA initialization/restore race without resetting the visible creature or relaxing save validation. [The report](experiments/swimmer/boulder-navigation/README.md) records geometry, byte preservation, actual weight changes and residual stronger-current gaps. Physical low-flow shelter is now approximated; the surge event and the decision to seek shelter are still later work.

Retain the approved creature identity across aquatic baby, new-limb stage and eventual upright adult. Growth changes the real articulated body, mass, inertia, drag and contacts. Extend/fine-tune a copy of the body controller with new joint observations and torque outputs; test how much learned swimming transfers. Train swimming alongside new capabilities to retain it. A controller operating multiple body stages needs appropriate body/limb-presence information and training coverage.

Food collection can participate in a designed progression trigger, while the ability to operate new limbs must be learned and verified. Crawling, shore transition and upright walking remain substantial later milestones. **No arms or legs are implemented in the current reliability step.**

## Shared architecture and priorities

Later hosting option saved October 28: the user is willing to consider **at most $15/month of Azure GPU spending** for browser play, with a genuinely enforced spending cap. This is a deferred idea, not authorization to provision a GPU now. Finish the Windows CPU/CUDA release and public download site first. Investigate enforceability and actual economics before presenting a future GPU service as capped or safe from overage; do not treat budget alerts alone as a guaranteed cap.

- Two decision controllers: food/goal choice and swimming/body movement. PPO's value estimator is a training helper, not a separate new creature ability.
- Preserve the original capable creature, best tested actor, full learner state and honest fallback/error reporting. Neural actions obey the shared physical model.
- Optimize for the visible experience and verified adaptability. Higher GPU utilization, more training steps or a nicer graph does not establish a better creature.
- Keep [GPU optimization findings](gpu_optimization.md) for a later pass. Distribution, installers, broad platform expansion and additional cosmetics remain paused. The user-facing product is now Tadpole Pond: hide CartPole/Racing from its UI while preserving their code, scenes, policies, developer routes and training. Keep the underlying Godot application name/save paths stable.
- Work one milestone at a time. The broad roadmap is approved direction; a stage is only implemented when code, actual behavior and relevant verification support it. Record outcomes and residual failures in [the session checkpoint](SESSION_CHECKPOINT.md).
