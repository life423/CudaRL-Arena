# Distant visitation feedback — November 25, 2026

New user authorization: improve distant unvisited-area feedback, run a separate
GPU PPO chooser session, preserve the working motor, update Godot only after
better individual coverage with bank safety, and commit the work.

The only learning change replaces inputs21–23 (three 24cm point-novelty probes)
with mean novelty in three 120° sectors centered at −120°,0°,+120° relative to
measured COM course, or body facing when course is unreliable. Only geometrically
reachable 50mm cells farther than24cm contribute. The static native whole-body
geometry cache has338 cells at80mm target depth; it is generated without physics
or optimizer work. Its floor-aligned sensing grid differs from the unchanged
345-cell centered evaluation estimate. No policy test trajectory builds this
cache. No destination, route, action mask or physical force is imposed.

Each mean remains in[0,1], using existing first-visit and aged-revisit novelty.
Inputs0–20 and24–31, the32→32→15 chooser, reward scales, PPO parameters and
task mix remain unchanged. Version4/schema3 model files include the map checksum;
version3 preserved models retain their original observations. The same observer
and file-mode decoder serve CPU, CUDA and Godot. Current80mm support is explicit;
other depths require their geometric cache and qualification.

Fork full state from coverage-v1 selected80. Require exact checkpoint roundtrip,
CPU/CUDA agreement within1e-6 for the three new inputs over1,024 saved worlds,
bit-identical other inputs on each backend, and fresh/visited/body-relative sector tests before a
three-actual-update exact-resume diagnostic. New input meanings may change
untrained behavior; record checkpoint0 qualification honestly. Never equate
unchanged weights with unchanged behavior.

One new session:1,024 CUDA worlds,128 new updates,8 decisions/world/update,
1,024-row minibatches,four passes;1,048,576 fresh macro transitions maximum.
Native/external wall caps1,200/1,800s; training substep ceiling1,342,177,280.
Keep complete immutable actor/critic/Adam/world/history/RNG checkpoints every
update. Evaluate unchanged12×120s+conditional4×300s every8 updates,17 evaluations
maximum with2,112,000 substeps and30s per assessment. At most four qualified
post-training candidates receive four uninterrupted1,800s starts each, with
5,760,000 substeps and60/65s caps per assessment. Reuse selected80's existing
30-minute baseline; do not recompute a union and call it individual coverage.

Adoption requires original12+4 passes,4/4 long physical passes,zero faults/resets,
minimum individual coverage≥36.88405797% and mean≥42.89855072% (five percentage
points above selected80), and maximum below6mm/s slow interval≤11.1s. Rank by
worst individual coverage, then mean. Report80% milestone separately; it is not
whole-pond mastery. Keep working policies unless qualification succeeds.
Godot adoption also requires old-model compatibility and new selected300s replay,
at most480,000 extra parity substeps. All failed assessments remain recorded.
Stop native/numerical faults; bank guard stops are scored training failures.
No automatic restart, budget extension, new motor training or physics change.

Pre-training audit correction: the first uniform32-input check failed before
physics/PPO. Two read-only inspections isolate the worst difference to unchanged
input9, lane976: CPU−0.366296798 vs CUDA−0.3662990332, a normalized difference
2.235174179e−6 (about0.112µm clearance). The legacy observer has exactly the same
maximum difference; the new three inputs have ZERO difference, and all other
GPU inputs are bit-identical to legacy. No physical tolerance or arithmetic is
changed. The corrected affected-input gate checks the new three at1e−6 and
requires bit-identical legacy fields on both backends, while retaining the full
matrix maximum in evidence. Original failed receipts, snapshots and executables
remain under distant-coverage-v2. That root spent zero physical/optimizer updates.
Use distant-coverage-v2-corrected for the one three-update resume diagnostic and
main session; this does not add training allowance or hide a failed learning run.
