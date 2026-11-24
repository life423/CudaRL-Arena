# Shoreline runtime v1 — closed at timestep convergence

The single CPU product qualification is CLOSED. It preserved the saved motor's
tested swimming/holding behavior and passed eight physical bank-contact cases,
then failed its fixed half-step convergence gate. The candidate was NOT installed
and no outward-swimming or learning case followed. User subsequently requested
the next test and training; a corrected, separately bounded v2 is being prepared.
Do not relaunch v1 or reuse its unused budget.

| Measurement | Observed result |
| --- | --- |
| Retention |20paired cases;30,400 controls each old/new; exact0 body/filter/command/COM differences |
| Short swimming / long windows |4/4; same28/30 exact original keys, both known startup misses retained |
| Holding |6original+6withheld64s cases and2five-minute cases passed |
| Bank contact |8/8 prescribed10s impacts passed; actual contact/damping occurred |
| Whole-body minimum gap |−89.870657µm; within fixed−500µm limit |
| Half-step |2completed passes; third failed at control42/2.1s; fourth skipped |
| Failure criterion |COM position difference0.252716709mm > fixed0.25mm |
| Other convergence measures |velocity0.157905102mm/s <0.25; orientation0.358985° <0.5 |
| Outward saved-motor cases |0/4 ran; skipped after the failure |
| Controls / substeps |62,842 attempted and accepted controls;632,840 attempted substeps |
| Wall / optimizer / CUDA dynamics |209.640465s /0 /0 |
| Planned cases |36:31attempted,30completed passes,1failed,5skipped |

Raw files are at `build/pw26/run-v1/`; `result.json`, `controls.csv`, `cases.csv`,
`physics-cases.csv`, `windows.csv`, `matched-product-endpoints.csv` and
`planned-cases.csv` preserve all attempts and skips. Product-prefix queries are
endpoint-only geometry checks, not independent substep product trajectories.
Sealed source/input bindings are at `build/pw26/sealed-inputs.json` and `causal/`.
The native process exited2. All250protected asset `.bin` hashes matched afterward;
the live policy/checkpoint originals also remain unchanged.

Independent raw CSV aggregation confirms the control/substep/case counts, exact
retention comparisons and28long-window keys. The failure was NOT a penetration,
nonfinite state or retained-skill regression. It remains a failed qualification:
do not relax its threshold because the excess is only2.716709µm.

## Numerical diagnosis without another trajectory

At heading180 the bank contact ends around0.15s. The5ms versus2.5ms impact damping
losses are9.164932e−9 versus1.06749755e−8J, differing16.48%; the resulting residual
COM velocity offset accumulates to the failed position difference by2.1s.
This identifies impact/contact-impulse discretization as the next engineering
target; it does not prove every possible source of error has been isolated.

An additional explicitly nonintegrating native C++ algebra audit examined16
frozen boundary snapshots, using16body queries/125,576terrain visits,
16assemblies and32velocity algebra solves. It advanced no coordinates or world
steps. Four contacts were active and two damped; no attractive solved contact
force was found in those snapshots (minimum multiplier+1.01425e−5N). Internal
unsaved substeps were not thereby cleared. Evidence is under
`build/pw26/contact-algebra-audit/`.

The current explicit spring/old-velocity damping branch is not generally a
fully implicit unilateral contact law in coupled dynamics. The proposed v2
correction evaluates contact spring/damping at the new normal velocity with a
bounded active-set solve, preserving the same physical constants and original
no-contact motor/water path. Keep the failed v1 source and evidence intact.
Prepare fresh CPU/GPU arithmetic/parity and unchanged retention/contact/timestep
criteria before its one separately identified launch. No optimizer may run until
the relevant physics and motor-interface gates pass.

## App recovery and current product boundary

The cleanup briefly removed five old study GUI modules before main.gd imports
were removed, producing the user's gray window. The log confirmed missing
preload parse errors. Those dependencies and obsolete UI paths are now removed;
Godot4.4.1 startup/render check passed with0controls/substeps and no startup error.
Godot Manager points at the correct `minimal-foundation/live` project. The old
qualified installed DLL remains in use while the new physical model is held.
It still has the smaller compiled guard; do not claim full shoreline adoption
or indefinite navigation from source cleanup or a passing private screenshot.

Private zero-step v1 UI checks passed37assertions at each of1280×800,390×844,
768×1024 and844×390, but require new artifact binding if the corrected DLL changes.
The selected next product goal is WHOLE-POND EXPLORATION, not merely a safe
perimeter loop. See `POND_NAVIGATION_PREPARATION_2026-11-23.md`; it does not run
training. Simulation, inference, native checks and future PPO remain C++/CUDA,
with no Python runtime or training dependency.
