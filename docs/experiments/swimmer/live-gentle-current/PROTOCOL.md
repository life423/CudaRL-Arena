# Live gentle-current baseline — October 28, 2026

Declared before the response trials. This bounded step adds the first live water interaction; it does not train or promote any controller.

- Frozen bank-trained selector update 320, SHA256 `4186F47471EDE2D0034EB566EFA1EC959B03D0A015E686FFA5F4262FCEB4A563`; retained motor update 64,350. Original policy/full saves stay intact.
- Use actual `PondPlayback`, not a separate evaluation-only environment. Run CPU and CUDA separately, strict backend identity, unchanged 50 ms steps with ten 5 ms physical substeps.
- Four fresh seeds: `0xC0DEC801`, `0xC0DEC802`, `0xC0DEC803`, `0xC0DEC804`; environment zero, five continuous simulated minutes per seed/condition/backend.
- Conditions: calm throughout; +X flow at 0.5 cm/s from the start; calm for sixty seconds then the same flow for the remainder. Same initial body/food per seed; no resets, scripted recovery, policy changes or learning. Later food samples can diverge as trajectories/collection differ.
- Report foods and food/minute, longest food gap, suspected/confirmed immobility, invalid body/food state, sequential steps, source integrity and actual backend. Preserve all cases, including failures. Keep backend results separate; long CPU/CUDA trajectories need not match exactly.
- Existing provisional immobility definition: all head positions in the prior three seconds within 0.1 model units of current position; five continuous seconds confirms an event. This gate alone does not establish efficient foraging; explicitly inspect >20-second food gaps and rate changes.
- Native/UI checks must establish state continuity at the switch, truthful mode/backend labels, no inherited training curve, preservation of flow across the random-weight comparison, and zero-current regression. Validate layout/input at representative desktop/tablet/phone sizes.
- The flow is the existing imposed uniform velocity through water-relative drag, not CFD. Food stays at existing sampled underwater locations. No new observations, reward, optimizer or physical equations are introduced.
- No candidate selection occurs from this suite. If the agent handles the gentle challenge, record it and move on; if it struggles, retain identities and diagnose goal selection/motor/observations before choosing further PPO training. Existing difficult bank/current cases remain unresolved regardless of a pass here.
