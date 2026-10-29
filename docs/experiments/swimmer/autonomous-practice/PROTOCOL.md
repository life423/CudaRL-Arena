# Autonomous feeding and varied-current practice — October 28, 2026

The user approved unattended pellet feeding, random current speeds, smooth speed fluctuations and no-current conditions. This is a local AI development step on `codex/autonomous-pond-practice`, after the Windows installer/site release. Do not publish or overwrite that release as part of this experiment. Microsoft Store signing remains deferred.

## Change and bounds

- Keep the two decision controllers, inherited weights, existing 37 motor inputs, native PPO/GAE/Adam settings and body/particle force equations. Only the swimming motor learns in this stage.
- Non-retention practice receives seeded surface arrivals rather than instant replacement after eating: one arrival opportunity per six-second window at an offset of 1–5 seconds, sometimes a pair. Adjacent opportunities are 2–10 seconds apart. Initial worlds retain 1–8 varied-age foods; new arrivals are born at the surface and sink physically.
- Sample half of locations from recent player feeding spots when available and half across the valid pond ellipse (radius fraction at most .86). Without player history all locations are random. Keep at most 16 active practice pellets; suppress additional arrival attempts at that bound without replacing existing food. The visible pond's existing 128-active guard remains separate.
- Keep four equal world groups: original calm/static-food retention; calm sinking-food practice; random steady speeds; smoothly fluctuating speeds. Speeds remain within 0–0.75 cm/s. Fluctuating keyframes are 4–10 simulated seconds apart with smooth interpolation and exact calm intervals. Direction follows the existing imposed vector, defaulting to +X; no vortices, spatial fluid field or abrupt random directional changes are introduced.
- Derive schedules from seed, world, episode and physical step count, independently of action/eating history. Matched policies receive the same nominal arrival attempts and water conditions; occupancy can suppress an attempt. No new raw State/Context fields, save layout, network dimensions or optimizer reset.
- Apply updated player context between rollouts without resetting practice-world clocks. Candidate evaluations explicitly run the same schedules without episode resets; live manual playback retains its existing current controls. Accepted actors still reach the same physical body only after the existing matched improvement/calm-retention gate.

## Verification plan

1. Seeded timing/coverage/bounds/zero-flow tests, including no consumption-triggered replacement and preservation of stationary-food retention.
2. Actual CPU/CUDA schedule/physical-state agreement and exact uninterrupted-versus-restored continuation. Preserve terminal-observation bootstrap ordering.
3. Native GPU-required regression suite and the C++-only CPU learning suite. Verify protected reference/full-save hashes.
4. Brief rendered CPU/CUDA learning and full resume, plus copied historical feeding-save compatibility. Use isolated QA saves, never the user's original learner for tests.
5. Optional visible Auto feed control: simulation-time arrivals, continued manual feeding, finite capacity, keyboard operation and representative viewport inspection. Background current variation remains independent of the visible manual controls.

The initial reference remains eight ten-second simulated trials. Candidate checks are small provisional gates; food rates from this new arrival/current protocol are not comparable to old fixed-stock or three-food scores. Passing implementation checks does not establish useful generalization, optimal behavior or robustness to arbitrary water. Do not force a candidate promotion or run lengthy offline pretraining before showing the new mode.
