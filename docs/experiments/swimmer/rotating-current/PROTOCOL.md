# Frozen-agent rotating-current test — October 28, 2026

Fixed before trials. The user approved comparing the existing saved agent in steady and slowly changing 0.5 cm/s current, with drifting food. Both networks remain frozen: selected bank-trained selector update 320 and retained swimming actor update 64,350. Preserve candidate SHA-256 `4186F47471EDE2D0034EB566EFA1EC959B03D0A015E686FFA5F4262FCEB4A563`, original policies and full learner saves. No new PPO training, reward change, observation change, rescue, autoreset or automatic replacement of the working pond.

## Matched conditions and samples

- Steady: uniform +X flow at 0.005 m/s.
- Positive rotation: the same magnitude, starting +X, turning toward +Z, one full turn every 120 simulated seconds.
- Negative rotation: the same magnitude, starting +X, turning toward −Z, one full turn every 120 simulated seconds.
- Update the imposed vector at each 50-ms control step using completed simulation steps, not wall-clock time. The same vector drives existing swimmer drag and native suspended-food advection. Food/body physics, depth constraints and respawn rules remain unchanged.
- Eight fresh layout seeds `0xC0DECC01` through `0xC0DECC08`, five uninterrupted simulated minutes per condition on each of strict CPU and strict CUDA playback. Repeat the first two seeds for fifteen minutes per condition/backend: **60 sessions, 420 simulated minutes** in total.
- Match initial layouts and weights by seed. Later consumption/respawns can diverge. Report each backend and duration separately; this is a bounded sample of the selected controller, not training-seed variance or proof of arbitrary-water competence.

## Measurements and decision rule

Record per case: food/minute, maximum food gap, confirmed five-second immobility, invalid states/food, sequential-step integrity and actual backend. Preserve all failures. Sample diagnostic state/current/target information every five seconds and once per second around bank contact or food gaps of at least ten seconds. Preserve stall snapshots as diagnostics; they are not exact resumable GPU exports.

Add passive bank observations using the prior bank-study proxy: three points on each link, normalized bank radius squared greater than 0.35, sampled surface gap below 1 mm, head below −0.2 model units. A visit starts at first observed contact and clears after one full second without contact. Record contact duration, low-speed contact time below 0.5 cm/s, first subsequent food and time to clear. Report visits with at least one cumulative second of contact separately. Unfinished visits and insufficient follow-up at trial end are censored, not counted as successful recoveries or automatic failures. Ordinary layouts may produce no meaningful bank-recovery opportunities; say so when applicable.

Provisional acceptance for this declared suite: zero confirmed immobility/invalid states/invalid food/nonsequential steps; no food gap above twenty seconds; each rotating direction's mean collection at least 90% of matched steady collection within the same backend and duration. Report paired differences and raw results, including directional asymmetry. Do not tune the policy or the protocol after inspecting outcomes. A failure triggers diagnosis and discussion before any training; passing does not itself justify more training.
