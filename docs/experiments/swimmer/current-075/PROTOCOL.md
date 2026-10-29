# Frozen-agent 0.75 cm/s test — October 28, 2026

Declared before trials. User authorized testing 0.75 cm/s with both decision controllers frozen and the original trained policy preserved before further training. No motor/selector update, reward change, rescue, reset or new physical equation. Preserve candidate SHA-256 `4186F47471EDE2D0034EB566EFA1EC959B03D0A015E686FFA5F4262FCEB4A563`, original policies and full saves.

## Conditions

Use the unchanged actual CPU/CUDA app playback runner and suspended-food advection:

1. Calm, zero imposed flow (food-drift mode enabled, zero transport).
2. Steady +X 0.5 cm/s reference.
3. Steady +X 0.75 cm/s.
4. 0.75 cm/s turning from +X toward +Z once every 120 simulated seconds.
5. 0.75 cm/s turning from +X toward −Z once every 120 simulated seconds.

Flow remains spatially uniform at any instant. Turning its direction is not a spatial whirlpool. Food uses the same imposed vector as body drag, with unchanged underwater no-flux feeding-clearance boundary.

Eight fresh seeds `0xC0DECD01` through `0xC0DECD08`, five uninterrupted minutes per condition on each strict backend. Repeat the first two seeds at fifteen minutes per condition/backend: **100 sessions / 700 simulated minutes**. Match layouts and controller weights by seed; subsequent collection and respawns may diverge. Compare each backend/duration separately. No fallback in strict CUDA trials.

## Measurements and decisions

Retain the existing structural/immobility monitor and passive bank proxy from the 0.5 cm/s rotating-current protocol. Record food/minute, maximum food gap, confirmed five-second immobility, invalid state/food, sequential-step integrity, bank contact/clearance/food follow-up with censoring, and sampled diagnostic trajectories.

Provisional gate: zero confirmed immobility/invalid states/invalid food/nonsequential steps; no food gap above twenty seconds; every 0.75 condition's mean food/minute at least 90% of its matched 0.5 steady reference. Calm is a reference, not a stronger-current denominator. Preserve all raw failures and directional/layout differences. A failed gate triggers diagnosis; it does not authorize training, overrides or an automatic replacement of the working agent.

Preserve source hashes before and after. The unchanged weights make calm/gentle references controls, not a claim about retention after fine-tuning. No training occurs. Earlier difficult bank starts and arbitrary turbulence, obstacle wakes, rock impacts or spatially varying flow are outside this suite. Menu simplification is a separate UI task and cannot alter the physics or trial configuration.
