# Motor-v2 tasks and fixed behavioral acceptance

Registered October 30, 2026, before contact-regression execution or new PPO.
The experiment compares two alternatives for one reusable motor; the learned
goal chooser is excluded. All requests persist after arrival. Four outputs
actuate two universal joints with two controlled axes each, not four joints.
PPO critics are training auxiliaries, not additional behavioral controllers.

These are new task/physics semantics. Previous regulation success and the old
six-update torque actor are historical evidence, not matched new-physics results.
No candidate is required to imitate that actor or preserve its food rate.

## Fixed withheld catalog

Each arm receives the same complete 64 initial states, in the same order,
with fixed final key `0x4D082604`. Every episode lasts at most 600 control
intervals of 50 ms (30 s), with ten physical substeps of 5 ms per interval.
There are no adaptive retries, enlarged budgets, best-checkpoint selection,
goal changes, extra forces or rescues during evaluation.

| IDs | Count | Task | Entry and motion requirement |
|---|---:|---|---|
| 0–7 | 8 | Purposeful movement | A 36–72 mm 3D displacement; halve target error within 4 s and enter the position band within 12 s |
| 8–15 | 8 | Turn | Alternate +90°/−90° level-heading requests; reach heading band within 5 s |
| 16–23 | 8 | Brake | Real captured swimming at 10–30 mm/s; immediate zero-terminal-velocity request at the entry position; head and COM speed at most 2 mm/s continuously for 0.5 s within 3 s and at most 72 mm head path |
| 24–31 | 8 | Slow controlled arrival | An 18–36 mm displacement; enter the position band with head and COM speed at most 2 mm/s within 15 s |
| 32–43 | 12 | Unsupported calm hold | Pond-relative stationary destination, calm water, no floor support |
| 44–55 | 12 | Unsupported current hold | Pond-relative stationary destination in steady 5 mm/s horizontal water flow, no floor support |
| 56–63 | 8 | Bottom-contact rest | Initially placed at the geometrical floor with a declared 1 mm/s downward approach; zero terminal velocity, legitimate contact permitted |

Braking uses the eight archived, qualified `reference-v1` actual-swimming
captures 4–11. Preserve their orientations, all linear/angular/joint velocities,
joint angles and previous actual torques. Only translate the initial head to
the tank center `(0, −2, 0)` and issue the new persistent stop request, with
level forward heading. This is fixture preparation before the episode, not a
deployed teleport. No velocity is invented or rescaled, and no new capture
simulation is needed. Their exact source bytes are hashed in the new registry.

Other starting positions, headings and joint poses use the fixed final seed;
these states are not training or tuning data. Current holds cross four body
headings with three fixed world-current directions (+X, +Z, −X), with fixed
small heading perturbations. Current speed remains 0.005 m/s. These tests do
not establish arbitrary-current, obstacle or shelter competence.

## Common arrival and sustained holding gates

The requested position is the head/root position in pond coordinates. Desired
terminal velocity is zero in this first slice; terminal heading is a level
world unit vector. Future nonzero-velocity and moving requests are represented
in the interface but are not established by this evaluation.

After any task's initial motion, the controller must begin a qualifying hold
by 15 s and maintain it for a continuous ten seconds. A transient arrival and
a successful ten-second hold are recorded separately. During every 5 ms
sample of the hold, require:

- Head/root position error at most **7.2 mm**, one tenth of the 72 mm body length.
- Heading error at most **10°**.
- Pond-relative head and COM speed at most **2 mm/s**, one tenth of the 20 mm/s
  motor velocity scale, rather than velocity relative to the water.
- Head/body angular speed at most **0.05 rad/s** (about 2.9°/s).
- Finite physical state and no excessive overlap, escape or actuator violation
  under the new contact version and preregistered overlap tolerance.

Tail motion, joint motion, joint-angle excursion and maximum fin-point speed
are **not** stationary-holding failure criteria. Actuation may be required to
oppose current. Record those motions, applied torque and positive actuator work
so vigorous or inefficient station keeping remains visible.

For **unsupported** holding, require the full body's floor clearance to remain
at least **2 mm** and the applied floor contact force to remain zero throughout
the hold. Reaching the bottom cannot earn hovering credit. A body drifting with
the current fails the pond-relative position/speed requirements.

For **bottom-contact rest**, require floor gap to stay between the negative
overlap limit and **+5 µm** throughout the hold. This positive allowance is a
floating-point contact band, not the 0.5 mm penetration allowance or an allowed
hover height: at the 0.16 m tank depth it is about 300 single-precision ULPs
and about 0.007% of body length. Require at least one actual positive floor
force above **1e−8 N** within the same ten-second window. Continuous nonzero
load is not required for a neutrally buoyant body touching at equilibrium.
Record the fraction of control intervals with positive floor force, maximum
overlap, and geometric contact separately. A geometry-only hold without the
required force evidence is not reported as a supported hold.

The ten-second gates use maxima/minima recorded over every physical substep,
not just endpoints at the slower neural-control rate. Actuator efforts use
actual applied torques, including servo reactions, not requested target angles.
Mechanical work and kinetic energy are not hunger reserves.

Acceptance requires **64/64 task passes and no physical/numerical failures**.
Report the success count for every stratum as well as initial arrival,
continuous hold duration, position/heading error, head/COM/angular motion,
tail/joint motion, applied torque/work, contact and excessive penetration.
One fixed-seed catalog supports only these tested conditions; it is not a
universal optimum or broad environmental-robustness claim.

## Existing current model and observations

The current path already exists in `deep::assemble`: distributed drag depends
on point velocity relative to water, and the semi-implicit equation includes
the matching fluid-velocity force. Uniform current does not directly translate
the body, increase motor torque or provide a whole-body stabilizer. Historical
checks in `tests/test_pond_current.cpp` cover passive advection, velocity-boost
invariance, explicit calm equivalence and CPU/CUDA agreement. Those are earlier
results; the new contact/current combination needs its focused new checks
before training. This remains prescribed flow with reduced-order drag, not CFD.

The new motor observations add body-relative water velocity / 0.02 m/s in
columns 4–6, full-body floor clearance / 0.072 m in 7, and the contact flag in 8.
These new columns start with zero transferred first-layer weights. Columns
9–32 retain their physical meanings, including 29–32 as previous actual
normalized applied torques. Relative desired-terminal-velocity observations
remain 33–35; desired heading remains 56–58. Both arms use the same observations.

Obstacle-ray columns 36–53 remain zero and column 54 explicitly says obstacle
geometry is unavailable. Later shelter integration needs body-relative distances
and directions from the same solid queries used for contact/validation, with an
explicit validity mask. This experiment does not implement that perception
system or claim the motor can navigate an overhang.

The goal chooser will later select Feed/Safety/Explore/Hold intentions and issue
persistent destination, terminal-velocity and heading requests. Goal identity,
validity, commitment and completion belong above the motor. Actuator channel
identity, units and morphology require versioned schema changes when limbs are
actually added; no extra per-skill motor is assumed. Growth, limbs, shore
movement, walking and manipulation are explicitly outside this experiment.
