# Natural cove habitat — October 29, 2026

The user approved the first stage of the [habitat plan](../../../HABITAT_PLAN.md): replace the regular outline with an outward asymmetric cove, preserve the main swimming volume and learned creature, then show it live. This stage adds no boulder, cluster, tunnel, shelter behavior or limbs. Work remains local on `codex/autonomous-pond-practice`, following commit `fc65b72`; no push, main integration, installer or website change was requested.

## How to watch and continue

Run `godot-project/pond/cove_pond.tscn`. The scene starts the actual inherited agent and native CUDA PPO with 1,024 background swimmers, automatic food and the existing calm/0.5/0.75 cm/s controls. Its own full save is `user://pond-policies/cove-navigation-v1/training.bin`. Close normally to save and stop; reopening resumes that child. The existing `arch_pond.tscn`, its save and the ordinary CPU/CUDA feeding scene remain available separately.

The user's visible window was opened with Godot **4.4.1**, imported the full parent at update **4,539** and restored the saved **0.75 cm/s** current. It is responsive, has an empty error log and continues real PPO. The approved live actor is initially **4,352**. Training updates and visible actor updates remain distinct; early gates reject candidates rather than concealing old-skill/contact regressions. No newly improved policy is claimed from this implementation check.

## Shared physical habitat

The fixed `cove-basin-v1` descriptor (layout 1) combines the main analytic bowl with a smoothly joined outward bowl. At the existing habitat scale, the cove center is `(5.8, 0, -1.5)`, horizontal radii are `(3.2, 2.8)`, depth is 3 model units and blend width is 0.35. One model unit represents four centimetres. The main center's deeper profile remains exactly unchanged; the new floor never removes previous water volume. The cove is shallower than the main pool, with no added crawling/land task.

Native bed height and analytic gradient determine contact, water containment, food clearance and boundary-aware current. Godot obtains the 192-point shoreline and samples native terrain height/gradient for its bed mesh. Water, terrain, shore decoration placement, camera bounds and native visual tracers follow that descriptor. No separate hand-drawn boundary changes physics. The legacy layout remains layout 0.

Automatic food cycles main-front, main-back and cove regions, including reachable points outside the old shoreline. Imported foods and visible body are preserved initially; subsequent genuine collections replenish coverage. This bounded habitat task still has two automatic foods; it does not remove or replace the many-pellet player feeding task.

The same prescribed local field drives articulated drag samples, food drift and visual tracers. It is a bounded boundary-aware approximation, **not** an incompressible fluid solver, rock wake or turbulence model. A cove is not automatically demonstrated shelter. Swimming remains driven by learned joint torques under the same mass, gravity, buoyancy, drag and compliant contacts. No live teleport, forced route or body-position projection hides failures.

## Full-state transfer

The actual parent `arch-navigation-v1/training.bin` was saved normally and preserved under ignored `build/cove-navigation/preserved-20261029113418/`. Its SHA-256 is `5f3ac35d996ce4500875031cc26c1d1e34ab7b83c4df9bc090b14dbbe9985665`.

It contains learner **4,539**, approved/applied actor **4,352**, 749 collected foods and physical time **2,833.800049 s**. Import retains its actor, critic, deviation, Adam moments/counters, practice worlds, visible position/orientation, foods/IDs, actor versions and current. Original source-motor provenance 10,000 remains historical provenance; it is not the actor from which this stage restarted.

ARCH v3 reads v1 and v2. Raw `deep::Habitat` and `arch::State` layouts are unchanged. Versioned basin metadata lives outside that raw world state. A v2 zero-update import preserves all nested trainer bytes except the required appended four-byte legacy layout field: **6,067,096 → 6,067,100 bytes**. Desired context changes to the cove while the saved practiced context stays legacy until the next rollout boundary. Older v1 migration adds the complete new context, retaining its old bytes as well.

`arch_resume_verify <protected-save> <fresh-child> cove` passes **22 checks** against the actual v2 parent. The older v1-to-v3 copy migration passes **21 checks**. Sources are byte-for-byte unchanged. No full private checkpoint is copied into this report or committed.

PPO keeps the same 55 motor inputs and inherited food selector. One quarter of practice retains old calm swimming, another quarter retains calm arch navigation, and the rest uses the cove with bounded steady/varying current. Candidate checks retain old swimming and arch skill separately. Saved layout identity prepares reproducible future experiments; automatic capture of difficult visible encounters remains future work.

## Verification

| Check | Result |
| --- | --- |
| Complete native suite, real GPU required | **208/208**, zero skips |
| New cove cases | **6/6**, included in the full suite |
| Existing portable feeding/playback suites | **2/2** |
| Full actual-parent import / older v1 migration | **22/22 / 21/21** |
| Rendered cove on Godot 4.4.1, actual user engine | **52/52**, empty stderr |
| Rendered cove on Godot 4.7.2 | **52/52**, empty stderr |
| Legacy arch/current rendered regression, Godot 4.7.2 | **52/52**, empty stderr |
| Ordinary feeding CPU / CUDA regression, Godot 4.7.2 | **32/32 / 32/32**, empty stderr |
| Historical protected policy/full-save references | **11/11** unchanged |

The six cove cases cover unchanged raw layout/version invariants, outward water-volume preservation and exact deep-center profile, contour/gradient, food coverage, particle/current behavior, and CPU/CUDA/layout-save parity. A substring-filtered run reports 13 passes because it also matches existing recovery/coverage names; it is not 13 new cove tests.

Rendered checks compare actual old-build pose/foods/counters before stepping, sample native/render geometry, exercise currents and keyboard controls, observe real PPO and context gating, and save/reload the separate full learner. Godot 4.4.1 independently resumes child update **4,548 → 4,549**, from imported parent 4,539. The earlier 4.7.2 run resumes **4,564 → 4,565**. These are isolated QA child saves, never installed as the user's agent.

Representative rendered sizes are 1280×720, 768×1024, 390×844 and 844×390, plus expanded AI details. The user's own 4.4.1 window continues independently. Physics tests and screenshots are distinct evidence; screenshots alone do not establish swimming reliability.

### Brief inherited-controller probe

With PPO paused, the same inherited actor 4,352 runs from the actual saved physical state for one simulated minute per condition. There are no resets or rescues during a trial.

| Current | Foods in 60 s | Cove foods collected | Maximum measured immobility | Maximum food gap | Whole-body arch passages |
| --- | --- | --- | --- | --- | --- |
| Calm | 13 | 2 | 0.25 s | 7.9 s | 5 |
| 0.5 cm/s | 12 | 3 | 0.55 s | 7.6 s | 4 |
| 0.75 cm/s | 11 | 2 | 0.50 s | 10.4 s | 1 |

All three stay finite, visit the cove and collect food there. This shows that the new volume is genuinely usable by inherited swimming. It is one saved start and three short trials, **not** a held-out long-run reliability study, proof of optimal routes or evidence that cove training improved the controller. Passage counts distinguish going through the arch from collecting food around it.

## Remaining work and scope

Review the live cove first. Longer actual-path sessions and independent safe layouts/seeds remain necessary before calling the combined habitat dependable. Continue to diagnose goal choice versus movement when failures appear. Neither more updates nor a rejected/accepted development gate settles generalization.

Internal UI review finds hierarchy, consistency and feedback usable; remaining portrait framing and accessibility verification merit specific follow-up: overview makes the swimmer small, so the existing Closer view remains important, and physical touch/assistive-device testing has not been performed. Do not turn these caveats into an unrelated graphics redesign.

The next proposed habitat change is one solid boulder with two broad routes, followed later by a small cluster/gap and short tunnel. Those features require matching native contacts/perception and their own checks; they are not silently included in this stage. Shelter and anatomy growth remain later. The public release is unchanged, and the local cove prototype requires a working CUDA backend; CPU learning remains supported in ordinary feeding.

Raw logs, summaries, preservation checks and source fingerprints are stored beside this report. Rendered screenshots and all private checkpoint binaries remain under ignored `build/cove-navigation/`. The source scripts are `tests/godot/cove_pond_smoke.gd`, `tests/godot/arch_current_frozen_probe.gd` and `tools/arch_resume_verify.cpp`.
