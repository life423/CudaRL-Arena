# Currents in the rock-arch pond — October 29, 2026

The user requested committing the working arch stage, then restoring the water current. Commit `fc65b72` saves the prior verified stage locally on `codex/autonomous-pond-practice`. The subsequent current implementation is local work on that branch; no push, merge, public installer or website update was requested.

## Behavior

The arch pond now offers Current on/off and 0.5/0.75 cm/s strengths. Changes preserve the creature's pose, food, physical clock and learned weights. Suspended food drifts through the same local field used by body drag and native visual tracers. Surface motion reflects the imposed current. Manual food remains outside this controlled arch task; the ordinary many-pellet feeding pond is preserved.

The native prescribed field smoothly removes near-wall normal velocity and fades to zero in a thin layer at the rock, bed and water surface. It is zero inside solids and never exceeds the imposed speed. Every articulated drag sample uses the local field, and food/tracers share midpoint advection and small-sphere contacts. This is a bounded boundary-aware approximation, **not** a divergence-free fluid solver, wake model or turbulence simulation. No creature teleport, scripted route or body-position projection is added.

PPO still uses 55 motor inputs and the same two decision controllers. The food selector remains inherited. Current context reaches background training at rollout boundaries without restarting worlds. Practice includes 25% original calm-water retention, 25% calm arch, 25% random steady strength and 25% smoothly varying strength with calm intervals, bounded by the selected live speed. Turning current off makes all practice groups calm.

Candidate gates protect original calm swimming and calm arch food collection separately; pooled current gains cannot conceal loss of either. A pre-current approved actor is embedded as a calm reference. A current change during candidate evaluation prevents publication of that stale-context result. The UI distinguishes the selected, practiced and evaluated context, and keeps session collection separate from matched gate scores.

## Full-save preservation

The user window was closed normally and its v1 save/binaries backed up to ignored `build/arch-current/preserved-20261029104613/`. The original snapshot has learner update **1,468**, approved/applied arch actor **1,376**, 235 collected foods and elapsed physical time **831.900024 s**. Its SHA-256 is `19557cf97ff870acdefb4e2a20b8686e4918d07aff35c6ba07e3f400ebba0211`.

ARCH v2 reads v1 without changing its raw world-state layout. It adds desired/practiced current contexts and the calm-reference actor outside the old state. A zero-update upgrade on a copy passes **21 checks**: all original actor, critic, deviation, Adam and practice-world bytes are exactly preserved. The nested trainer payload changes from 6,067,080 to 6,067,096 bytes solely by appending the 16-byte calm context. Live state, actor versions and embedded source also match; the original file is unchanged. Tool: `arch_resume_verify`, with protected v1 input and a fresh output path.

The rendered migration check independently matches the old build's captured pose, orientation, foods, pellet IDs, learner and applied actor before stepping. V2 restores saved 0.75 cm/s current/context and continues PPO. The user's personal session then resumes its own save, not a QA actor. Its previous v1 file remains preserved in the backup.

## Verification

| Check | Result |
| --- | --- |
| Native arch/current cases | 17/17, real GPU required, zero skips |
| Complete native suite | 202/202, real GPU required, zero skips |
| Existing CPU feeding/playback suites | 2/2 |
| Byte-level zero-update v1 migration | 21/21 |
| Rendered current controls, migration, food/flow, PPO/context and resume | 52/52; empty stderr |
| Default feeding scene | CPU 32/32 and CUDA 32/32; empty stderr |
| Historical protected policy/full-save inputs | 11/11 unchanged |

Four rendered sizes (1280×720, 768×1024, 390×844, 844×390) and calm/current captures were inspected. Keyboard controls work; current controls stay usable, native tracers remain outside rock, water appearance changes and food drift is measured before collection. An initial food-drift assertion lost both tracked IDs to collection; the final test observes consecutive steps before collection instead. This was a test-censoring issue, not a physical rescue or food replacement change.

The current-context gate at update 1,472 practices 0.75 cm/s maximum. The candidate improves current food rate 14.5→15.75/minute but loses original calm (34 vs 35 foods) and calm arch (31 vs 33); it is correctly rejected. This verifies protection against forgetting, not a successful new current policy or guaranteed future improvement.

The frozen approved actor 1,376 also runs through the actual native live path from the same saved physical state for one minute per condition, with PPO paused:

| Current | Foods in 60 s | Maximum measured immobility | Maximum food gap | Arch passages |
| --- | --- | --- | --- | --- |
| Calm | 20 | 0.25 s | 6.60 s | 7 |
| 0.5 cm/s | 16 | 0.60 s | 12.35 s | 3 |
| 0.75 cm/s | 16 | 1.20 s | 6.85 s | 3 |

All three remain finite, keep the same frozen actor and avoid a five-second immobility event. This single-start, one-minute-per-condition probe shows an attainable challenge and collection-rate loss in current; it does not establish broad reliability, long-run safety or improvement from current training. Script: `tests/godot/arch_current_frozen_probe.gd`.

## Current local state and limits

The user scene is reopened in a normal visible Godot window with 0.5 cm/s enabled and the actual resumed 1,024-world learner. The window is responsive and its error log is empty. Ongoing training/autosave is intentional; closing normally saves, and reopening the scene resumes current and learning state.

No additional network or input expansion was necessary for this first current task. More detailed flow, stronger currents, broader direction changes, long-run reliability and user-controlled geometry remain future work. Internal UI review remains 9/10 hierarchy/consistency/feedback, 8/10 portrait framing/accessibility verification: portrait overview makes the swimmer small, so Closer view is available. Physical touch and assistive-device tests are not claimed.

Raw logs, summaries, source fingerprints and preservation hashes are beside this report. Full private saves and screenshots stay under ignored `build/arch-current/`; no personal checkpoints are committed.
