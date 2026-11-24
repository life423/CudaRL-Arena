# Corrected shoreline certificate v4 — November23,2026

Preregistered before one new CPU dynamic launch. Preserve all v1/v2/v3 sources,
binaries, receipts and failed evidence. Prior v3 physical completion does not
validate its incorrectly signed sloped-gap certificate. This is a fresh scoped
correction, not a retry under the old identity and not training authorization.

## Narrow correction and identities

For gap g=y−H(x,z), the terrain normal n is proportional to
(−Hx,1,−Hz), so the gap gradient is (nx/ny,1,nz/ny). The previous search warm-start,
tangent bound and downhill order reversed those horizontal signs. Correct those
query calculations. Returned `gradient_x/z` remain height derivatives Hx/Hz;
the physical contact force continues using (−Hx,1,−Hz). Preserve the intended
force, spring/damping constants, integration, servo/filter, torque/joint limits,
body dimensions, fin union, terrain and acceptance tolerances.

The runtime also reuses an accepted POST geometry query as the next PRE only
inside the same50ms control, where body and terrain are identical and only the
servo reference changes. Control entry always obtains a fresh query. This avoids
duplicate pure queries; it does not skip physical steps or change contact state.
Device-only noinline boundaries do not alter CPU arithmetic or enable fast math.

Runtime version is4, query arithmetic version2 and collision envelope version2.
The unchanged saved amplitude package remains version2, the frozen motor remains
update336 plus its seven learned holding updates, and observation schema4 stays
41 inputs. CPU result protocol ID is `shoreline-product-v4-2026-11-23`.

The native pure geometry/independent-reference checks must pass against the exact
corrected header before physical execution. Then complete one zero-integration
checker preflight, compile/seal source, executable, protocol and immutable input
bindings, and execute once. Record every preparation/build attempt separately;
pure queries are not physical steps. No CUDA dynamics or optimizer is part of
this CPU allocation. No additional physical pilot, silent restart or budget reuse.

## Unchanged finite CPU qualification

Use the exact starts, phases, score windows, original28 long-window pass keys,
state tolerance and physical criteria from `SHORELINE_RUNTIME_PROTOCOL_2026-11-23.md`,
with the v3 duration-integrated contact law documented in
`SHORELINE_RUNTIME_V3_PROTOCOL_2026-11-23.md`. Do not alter criteria after results.

| Cases | Maximum attempted controls | Maximum attempted substeps |
| --- | ---: | ---: |
| 20 original/new matched retention cases |30,400 each =60,800|608,000|
| 8 compass bank contacts,200 controls each |1,600|16,000|
| 4 matched half-step bank checks,200 controls each |800|16,000|
| 4 center-start outward saved swims,up to1,200 controls each |4,800|48,000|
| Total |68,000|688,000|

One CPU process, at most900 seconds and2GiB evidence; zero PPO/optimizer work.
Count rejected controls/substeps. Stop on the first shared physical, numerical,
certificate, retained-skill or convergence failure. Record all attempted and
skipped cases plus the first failure's before/candidate state. No automatic rerun.

Retention contains4×160 short swims; poses0/3×1,200 long swims; six original plus
six withheld×1,280 holds; poses0/3×6,000 endurance holds. Long straight comparison
uses explicitly labeled compatibility terrain with water radii0.9/0.7m and flat
bed0.65/0.45m, original comparison arena0.6/0.4m and the same−0.16m floor. It is
not product pond navigation. Keep the2e−6 body/filter/torque state tolerance and
the original swimming/holding criteria. Match all28 original passing long keys
exactly for this unchanged actor, including its two known startup misses.

Bank fixtures remain at headings0/45/90/135/180/225/270/315 degrees,100µm certified
clear at the original depth,5mm/s outward rigid translation and zero motor
commands. Half-step cases are headings0/90/180/270 with2.5ms physics while each
5ms filter reference is held through two physical steps. At matched times retain
COM position≤0.25mm, velocity≤0.25mm/s and orientation≤0.5 degrees. Query uncertainty
stays10µm, compliant overlap≤0.5mm and runtime query visit cap131,072; unknown
certificates fail. Keep finite energy/passivity checks and explicit water/contact
loss accounting.

All four outward cases use the actual approved0.52/0.34m pond and fixed original
actor. Retain the existing honest unsupported-water-surface stop semantics and
record its actual shortened horizon, if encountered. A physical scope stop,
deflection or slow swimmer is not learned navigation. Do not require straight
travel through the bank or conceal a failure with a reset.

## Outputs, Godot and adoption

Checker: `build/pr30/native/Release/pond_runtime_check.exe`.
Pure preflight: `build/pw30/preflight-v4`.
The single physical result is `build/pw30/run-v4/result.json` with raw CSV/failure
evidence alongside it. Freeze input/source copies and executable hashes before
that launch. Preserve original checkpoint/save hashes before and after execution.

Build the corrected Godot bridge separately into `build/pw30/bin`; compile and
pure UI inspection are not physical trajectories. Install/rebind only after the
required CPU qualification passes. Bind the actual installed DLL SHA and exact
CPU proof SHA, runtime4/query2/envelope2, and expose honest counts/scope limits.
If any gate fails, keep the existing DLL and show qualification pending/failed
instead of claiming a corrected physical runtime is verified.

Keep the saved motor untouched, the user's save namespace, aesthetics/zoom and
current-state Details. Verify the actual registered `live` entry path headlessly
without a new physical trajectory; inspect representative rendered sizes in a
private paused project. Avoid interrupting the user's running app where an atomic
file replacement can preserve it. A new DLL takes effect on its next normal Run.
GPU parity and any candidate PPO remain separate, independently bound allocations.
