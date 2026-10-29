# First rock-arch navigation stage — October 29, 2026

The larger pond, physical rock arch, inherited-weight obstacle sensing and live PPO loop are implemented locally on `codex/autonomous-pond-practice`. The default feeding pond and public release are preserved. [Usage and experimental design](../../../arch_navigation.md).

## What was verified

| Check | Final result |
| --- | --- |
| Native arch geometry, contact, sensing, transfer, passage counting and continuation | 11/11, actual GPU required, zero skips |
| Complete native suite, including retained swimmer/CartPole/racing behavior | 196/196, actual GPU required, zero skips |
| Existing CPU-only feeding learning and playback | 2/2 portable ctest suites |
| Rendered arch, 1,024-world actual PPO, controls, promotion, full resume | 62/62, empty Godot stderr |
| Default feeding scene after shared presentation changes | CPU 32/32 and CUDA 32/32, empty stderr, both restore update 33 |
| Protected historical policies/full saves | 11/11 unchanged |
| Current personal feeding full save | SHA-256 unchanged from the pre-build copy |

The inherited motor is the **validated feeding actor at update 10,000**, loaded from the outer snapshot of the existing full save. It is expanded from 37 to 55 inputs with zero weights for the 18 added proximity readings. The inherited food-selection actor remains frozen. The source save SHA-256 is `25a147d693edd673abcb5dff417849dbeac81d71be5e2735caf91b069cc5388e`. The new arch stage has its own update counter, optimizer and full checkpoint; it does not overwrite that source.

The final live check uses a protected source copy and a unique QA checkpoint. Its starting reference is eight ten-second worlds: 13.0 arch foods/minute, five passages, 22 contact steps. This brief reference is intentionally not a mastery estimate.

At the first matched development gate (16 worlds, 30 simulated seconds each), update 16 collects 15.833 vs 15.167 foods/minute for the inherited/incumbent actor, retains 38/38 calm foods, and has 94 vs 97 contact steps. It is accepted. At update 32 the candidate and incumbent both collect 16.167/minute; the candidate passes the declared progress/clean-passage and contact/retention gates, with 86 vs 98 contact steps and 36 vs 35 reference calm foods. It is also accepted. These are fresh matched development checks, not a final unseen-layout superiority claim. Food-rate ties must not be described as faster collection.

The approved actor reaches the same live swimmer on its next physical step, preserving the session clock. A full checkpoint at learner update 36 restores the exact visible position, orientation, food and elapsed time, then continues to update 37. During the 12-second live physical path check, the swimmer collects three foods and makes one whole-body arch passage, with two contact steps and 5.526 model units of displacement. The passage observer separately rejects head-only crossings and paths around the opening.

## Physics correction found during verification

The initial contact expression calculated the collision normal in one subtraction operand and read it in the other. C++ does not guarantee their evaluation order; CPU support could be evaluated with an unset normal, while CUDA used a populated normal. The new head/tail/fin force tests and CPU/CUDA trajectory comparison exposed this failure. Computing distance/normal in a separate statement fixes it. All final evidence here uses that corrected solver.

Contacts are now reported when any of the ten 5 ms substeps applies a rock-contact load, including overlaps resolved before the 50 ms step ends. Counters count **contacted simulation steps**, not distinct collision impacts. The solver uses compliant sampled body envelopes, not continuous exact collision against every rendered triangle. Body movement remains force-driven; no geometric body projection or rescue is used.

Earlier rendered evidence predating this physics fix is isolated in ignored `build/arch-navigation/rendered/pre-contact-fix/` and is not used for the final results above.

## Presentation and remaining limits

Desktop 1280×720, tablet 768×1024, portrait 390×844 and landscape 844×390 were rendered and inspected, along with expanded AI details. The overview shows the hole and swimmer with a stable camera; Closer view provides another distance. Keyboard activation of camera, pause and details works. The graph explicitly shows live session collection; candidate evaluation and approved/applied versions are separate. Startup failure, unsupported CPU arch training and learning-error feedback are explicit.

Internal UI critique: hierarchy 9/10, consistency 9/10, feedback 9/10, responsiveness 8/10, accessibility verification 8/10. These are review heuristics, not measured user scores. The portrait overview makes the swimmer small; improve portrait framing after the local obstacle milestone if that platform becomes a target. Physical touch and assistive-device testing remain unperformed. This Windows local prototype is not a completed mobile release.

The new arch learner requires CUDA in this first stage; the CPU reference shares geometry/physical integration/sensing and passes parity checks, but no new CPU arch PPO session is claimed. Existing CPU feeding remains supported. Two automatically supplied suspended foods isolate the navigation task; the ordinary pond still supports many user-dropped sinking foods. There is no local flow model around the rock, leg growth or land movement. Longer actual-playback and unseen-layout reliability testing is still required before replacing the default pond or claiming robust obstacle navigation.

Raw native, portable, rendered, preservation and source-fingerprint records are beside this file. Full QA checkpoints, source backups and screenshots stay under ignored `build/arch-navigation/`; personal full learner saves are not added to Git. No commit, push, merge, website deployment or installer update is part of this stage.

The user's actual arch scene was then opened in a normal window with a fresh separate personal arch save and the preserved feeding source. Native logs confirm 1,024 background worlds, source validated update 10,000 and accepted arch update 16; startup stderr is empty and the window is responsive. Continued learning/autosave is intentional while the user watches. Closing saves/stops; reopening the arch scene resumes.
