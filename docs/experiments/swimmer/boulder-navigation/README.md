# One boulder and a physical downstream wake — October 29, 2026

The user requested committing the verified cove, then adding the next manageable habitat stage. Commit **`8fee3b4`** saves the current/cove work locally on `codex/autonomous-pond-practice`. This follow-up adds one solid boulder to the cove pond. The user also explicitly required weaker physical flow behind obstacles, so the new stage includes a bounded downstream wake shared by swimming, food and visual tracers.

This is navigation and a physical prerequisite for future shelter. There is no new surge event, shelter goal/reward, learned shelter-seeking, rock cluster, tunnel or limb. The ordinary feeding pond, public installer and website remain separate.

## Watching and saving

Run `godot-project/pond/boulder_pond.tscn`. Native CUDA PPO continues with **1,024** unseen practice swimmers while the visible creature runs its last approved actor. Current off/0.5/0.75 cm/s, Closer view, pause, manual save and ordinary close/autosave remain available. Close normally to save and stop; reopen this scene to resume.

The separate child is `user://pond-policies/boulder-navigation-v1/training.bin`. Its initial source is the complete `cove-navigation-v1/training.bin`, saved normally before rebuilding. It imports learner **5,491**, applied/approved actor **5,136**, 860 collected foods, physical time **3,492.75 s**, the original live pose/foods/IDs and saved **0.75 cm/s** current. It does not restart from the old feeding actor or use a trained QA checkpoint. Parent and historical policy/full-save references are preserved.

Parent SHA-256: `7a6185f0c730be97cf15ac31dbe7c56eef00a7ec139fc6060c02a7f624e0941d`. A protected copy and the old working binaries remain under ignored `build/boulder-navigation/preserved-*/`. No private checkpoint binary is included in this report.

## Geometry and learned control

Layout **2**, `cove-boulder-v1`, retains the existing cove/bed and arch. One exact sphere is rooted in the bed at `(3.8, -2.65, -1.25)`, radius **1.3** model units, fully submerged. One model unit is four centimetres. Godot reads these dimensions from native code; stone color/normal variation does not alter its silhouette.

The shared solid-distance query combines the torus and sphere. It supplies articulated head/link/fin contacts, the existing 18 body-relative rays, whole-body clearance diagnostics, food placement, particle contact and current boundaries. The motor stays at **55 inputs / four torque outputs**, with the inherited food selector. No extra network, scripted swimming, runtime waypoints, live-body reset or teleport supplies navigation.

Two sampled whole-body detours have **0.1644** model units minimum solid clearance and **0.7828** minimum water clearance across 329 poses. This is geometric feasibility, not proof that the policy executes those particular paths. The test's example routes never enter the app's controller. Automatic food continues to exercise the main pond and cove, with both solids excluded from valid placements.

### Food coverage correction

A five-minute calm probe exposed a task-design problem: one uneaten cove pellet crowded later cove arrivals into a narrow envelope, so fallback placements kept renewing food near the arch. Only one cove pellet was collected despite 70 total foods. Stage 2 now uses a wider physically checked cove envelope and bounded extra attempts, allowing two separated cove foods. All existing active foods remain unchanged until eaten; the controller still chooses its goals. Layout 1 retains its old generator. A 512-case regression checks the second cove arrival, physical clearance, separation and preservation of the first pellet.

The corrected calm probe collects 15 cove foods. Total collection changes because the task now requires wider travel; that is not a neural-policy improvement or a fair same-task rate comparison. The new graph separately measures this visit's collection, excluding inherited lifetime counters while preserving those counters in the full save.

## What the wake does

The new layout multiplies the existing boundary-aware velocity by a dimensionless downstream deficit. The boulder uses an analytic rounded-solid source. The arch uses twelve tube-centerline sources; its aperture is not replaced with an opaque bounding sphere. Source deficits combine with a maximum, avoiding artificial compounded stagnation and arbitrary source-order dependence.

The deficit has a smooth onset/lateral fade, an expanding footprint and downstream recovery. It follows the imposed current vector, including reversal/rotation, and leaves upstream water unchanged. Model constants are a maximum deficit of 0.72, initial lateral width 1.25 radii, spread 0.16 per trail unit and squared recovery `r / (r + 0.25 * trail)`. These are explicit reduced-order parameters, **not calibrated hydrodynamic measurements**. The wake factor stays between 0.28 and 1; boundary layers may separately reduce velocity to zero at solids, bed and surface. Original layouts 0/1 keep their old field.

Body drag samples, passive food and visible tracer advection use that same final native field. Tests measure reduced force on a stationary body and reduced food displacement in the wake, not only shader appearance. Drag still depends on body-relative water velocity. The rendered native probe at equal offsets upstream/downstream reads **0.50 → 0.248 cm/s**, ratio **0.496**. This is one point pair, not a promise that every downstream point has half the speed.

This remains a prescribed mean-flow approximation. It does not solve pressure, incompressibility, vortices, turbulent shedding or propagation of an actual wave/surge. A later current-surge event and shelter curriculum need their own physical limits, observations, goals and evaluation. We have made a lower-flow region; the creature has not yet learned when to seek it.

## Learning and full-state preservation

ARCH v3 and the raw state/context layouts are unchanged. The new child imports every old actor/critic/deviation/Adam/practice-world byte; desired layout changes from 1 to 2, with the saved practiced context retained until a rollout boundary. **22 byte-level checks pass**; the nested trainer payload stays **6,067,100 bytes**. Original references and visible state remain unchanged. New visible-body/food overlap checks reject an invalid insertion rather than moving the creature or deleting food.

Practice is split evenly between old calm swimming, calm arch, retained cove and new boulder worlds. Cove/boulder groups each include random steady and smoothly varying flow, with genuine calm intervals. Candidate gates check old swimming, calm arch and cove food collection separately. Cove retention compares against the matched incumbent and existing pre-current reference; it is not a separately frozen copy of the exact imported cove actor. Contact counts remain pooled over non-retention evaluation worlds, not an individual-obstacle guarantee.

Imported practice worlds are kept intact. Some may initially intersect newly introduced solid space; native compliant contact and ordinary training episode timeouts handle those copies. The visible import is separately checked and never reset as a rescue. Real PPO continues through the first changed-context rollout and full save/reload without non-finite weights or worker errors in the declared checks.

Candidate comparisons remain development gates. Actual weight updates are verified directly against the inherited learner, rather than inferred solely from an update counter or comparison to the older feeding actor. A newly improved boulder policy is not claimed from implementation checks.

In the actual user session, matched gates accept updates including 5,600 and 5,680 while preserving the same running body. The later real autosave at 6,003 has all 1,024 practice clocks consistent and all 2,980 learner actor parameters changed from the inherited 5,491 learner. These are genuine learning/adoption/persistence results. Independent broad reliability or statistically established performance improvement still needs separate evaluation.

### CUDA restore race found and fixed

The final rendered resume check exposed a startup race: initial world generation was queued on a nonblocking training stream, then saved worlds were copied on the default stream. Late initialization could overwrite part of the restored population. A failed QA save at update 5,508 had mixed episode clocks of 1,088 and 912, while the protected source had all clocks at 1,024. This is the exact fresh-versus-restored split after seventeen 64-step updates; it was not a physical stall.

`Batch::load` now fences pending device work before replacing states. This is a one-time restore cost, not a synchronization added to the training loop. A deterministic delayed-producer regression verifies all 1,024 restored worlds, continuation through the 1,200-step timeout, CPU clock agreement, repeated full resume and continued strict rejection of genuinely mixed-clock data. The failed private QA file is retained for diagnosis, not silently repaired or installed as the user's agent. The original good cove save remains the source.

## Verification

| Check | Result |
| --- | --- |
| Focused native boulder/route/contact/wake/parity/restore cases | **8/8**, real GPU required, zero skips |
| Complete native regression suite | **216/216**, real GPU required, zero skips |
| Ordinary portable CPU learning/playback suites | **2/2** |
| Full cove-to-boulder byte-level import | **22/22** |
| Rendered Godot 4.4.1 geometry, physics, controls, PPO and resume | **62/62**, empty stderr |

The initial contact fixture omitted its base translation and therefore sampled above the intended solid; it was corrected. An abruptly rotated example route put its straight tail outside the bank; the test now turns inside the available water and preserves its original clearance thresholds. These were test/route-fixture corrections. Runtime physical geometry, contact stiffness and agent movements were not changed to hide the failures.

The same inherited actor **5,136** also runs the actual live path with PPO paused for five minutes per condition, after the wake and food-coverage changes:

| Current | Food in 300 s | Cove foods collected | Maximum measured immobility | Maximum food gap | Minimum whole-body boulder clearance |
| --- | --- | --- | --- | --- | --- |
| Calm | 59 | 15 | 0.60 s | 8.35 s | −0.0428 model units |
| 0.5 cm/s | 46 | 16 | 0.80 s | 10.55 s | −0.0480 model units |
| 0.75 cm/s | 46 | 17 | 3.85 s | 31.70 s | −0.0540 model units |

All remain finite and avoid a five-second immobility event in this bounded suite. The 31.7-second food gap and 3.85-second near-stall in stronger current are real remaining weaknesses, not mastery. Negative clearance is small compliant penetration, up to about **2.16 mm** in this sample; the contact model is not an exact rigid mesh constraint. These are three trials from one saved start. They do not establish long-run reliability, optimal routes, arbitrary-current competence or an improvement from further training.

Actual rendered 1280×720, 768×1024, 390×844 and 844×390 views plus AI details were inspected. No critical clipped/overlapping controls were found. Internal review retains hierarchy/consistency/feedback at 9/10 and portrait framing/accessibility verification at 8/10: portrait overview makes the swimmer small, so Closer view remains available, and touch/assistive-device testing is not claimed. The foreground bank/vegetation substantially obscures the swimmer in the final overview captures; test closer framing or extend the existing render-only occlusion aid in a future visibility pass. Static screenshots do not establish visibility in every position. No broad graphics redesign is part of this stage.

Preservation checks confirm eleven historical references and both immediate arch/cove parent saves unchanged. The separate mutable feeding save has a different hash from the prior cove audit (timestamp 12:43:33); its writer was not attributed. Its current contents were retained without overwrite and copied only into ignored private build storage. It is not reported as unchanged. The boulder stage writes its own child path.

Raw logs, compact summaries, preservation results and source fingerprints are beside this report. Screenshots and all private saves stay under ignored `build/boulder-navigation/`. Source checks are `tests/test_boulder_geometry.cpp`, `tests/test_boulder_resume.cpp`, `tests/godot/boulder_pond_smoke.gd`, `tests/godot/arch_current_frozen_probe.gd --boulder --probe-seconds=300`, and `arch_resume_verify ... boulder`. The tool's `compare-learning` mode directly compares saved learner actor parameters and prints practice-clock integrity diagnostics.

Review the live stage and collect longer independent-layout evidence before treating the habitat as dependable. The next proposed obstacle is a small rock cluster with a broad gap, then a short tunnel. A surge/shelter event is a separate later behavior; do not add it or further anatomy merely because the physical wake now exists.
