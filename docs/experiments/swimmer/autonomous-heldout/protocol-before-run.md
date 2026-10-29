# Frozen before/after autonomous-feeding comparison — October 28, 2026

The user approved the next bounded comparison after automatic feeding/current practice was implemented. Freeze both snapshots before running the new layouts. Keep the actual local learner running; never instantiate a training session, acquire its lease, write its save, alter its physics or install a policy during evaluation.

## Policies and provenance

- **Before:** untouched `build/autonomous-practice/original-feeding-training.bin`, saved before the autonomous-practice change. This is the existing feeding learner, not the older packaged swimming network. Its learner age is 4,751; report the separate outer validated-actor update after parsing.
- **After:** one immutable copy of the running save, obtained through a shared-delete read handle so atomic autosaves continue normally. Evaluate the outer saved validated actor. The inner latest learner can be unvalidated and is not the candidate for this test.
- Validate checkpoint framing/checksum, format, parameter counts, finite weights and identical inherited selector/motor provenance. Record file SHA-256, actor fingerprints, both validated-actor versions and learner ages. Keep full personal snapshots under ignored build output; do not version them.

## Predeclared cases

Eight fresh seeds: `631A4B01`, `7D92E653`, `438BC7A9`, `95C1234D`, `2B6F84D7`, `C17A93B5`, `5E209F6B`, `A83D7C19` (hexadecimal). These differ from the live training seed and the FEED-prefixed candidate-gate seed family. Do not select a newer actor or tune the model in response to these results.

For each seed, run **16 newly initialized worlds for 120 simulated seconds per actor**, deterministic mean actions, no gradients, no resets. This gives **32 paired cases per subgroup**, 128 per actor, 512 total simulated minutes. Use one fresh common context: no saved player locations; +X current reference 0.5 cm/s. Initialize both actors identically with the same frozen selector and new world seeds.

Report the four groups separately: original calm/static retention; calm sinking-food; seeded steady speeds; smooth fluctuating speeds including true calm intervals. The latter two remain within 0–0.75 cm/s. Invoke scheduled environment events explicitly with autoreset disabled, so food/current evolution is the same task as the new practice beyond the small promotion gate. Nominal arrival requests are matched; actual accepted arrivals may differ at the 16-pellet practice occupancy bound.

## Measurements and interpretation

- Per-world food/minute and paired differences. Report means, percentage change, paired wins/ties/losses and deterministic 10,000-draw paired **seed-cluster** bootstrap 95% intervals. Eight seed clusters are a small sample; do not claim universal confidence or statistical significance from overlapping aggregate/subgroup summaries.
- First-food latency, noting no-food observations are right-censored; longest inter-food/start/final gap; longest food-available/no-collection interval; empty-pond time; final waiting interval. Waiting for a timed arrival is not a feeding failure by itself.
- An immobility event requires five seconds with head positions within 0.1 model units of the current position, food continuously present and no food collected. Track continuous qualified windows as one event and identify ongoing events at trial end. This is an explicit held-out observation criterion, not identical to the older anchored State.max_still gate.
- Finite body state and physically valid food. New sinking pellets legitimately start near the surface and contact the bed: do not use the old suspended-food envelope. Check finite positions/velocities, surface and .94 radial containment, and the actual sloped-bed sphere gap with 0.1-mm numerical tolerance. Report invalid tick/case counts rather than hiding/resetting them.
- Active food at the end, total food created, and time at the practice cap. These explain resource-bound divergence in actual arrivals.

Run an identical-policy numerical self-check before the comparison and verify snapshot/policy hashes afterward. GPU inference and the passive metric monitor run separately from the live Godot process. This is a correctness/progress comparison, not an isolated throughput benchmark.

The comparison measures whether the saved creature improved during additional training. It does **not** prove this curriculum outperforms an equally long continuation of the old curriculum. Do not automatically replace the visible actor or change either network from the test results. If subgroup weaknesses remain, use them to choose a later development task; retain these layouts as evaluation evidence rather than feeding their results back into this test.

Provisional next-stage interpretation, declared before results: any after-policy confirmed food-present immobility or invalid state/food leaves a reliability issue to investigate before anatomy expansion. A subgroup mean loss over 5% is a material regression in this bounded test; supporting retention more strongly requires its paired interval lower bound to exceed minus5% of that subgroup's before mean. A general food-rate improvement claim requires a positive overall paired interval lower bound and no material subgroup loss. These are development decision aids, not universal safety/optimality guarantees or multiplicity-adjusted statistical tests.
