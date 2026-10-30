# Motor v2: focused contact correction and bounded comparison

Registered October 30, 2026, before execution. Study:
`build/motor-v2/2026-10-30-v1/`. Baseline is clean
`codex/autonomous-pond-practice` at
`dbbcf7b0caf2dadbfd14f08523415dea8d524f01`.
The complete user authorization is archived in `sources/user-request.txt`.
The old paired run, original controllers, personal saves and previous results
are preserved. Neither new arm inherits the old six-update partial actor.

## Scope and physical version

Exactly two learned behavioral policies remain the design: goal chooser and one
reusable motor. Critics are PPO training helpers. The chooser is excluded;
fixtures supply persistent destination, terminal velocity and heading requests.
Four actuator outputs control two axes at each of TWO universal joints.
The target interface requests angles every 50 ms; the deterministic servo
computes internal torque every 5 ms using the historical validated gains and
unchanged ±10 µN·m limits. No angle overwrite, propulsion, rescue or whole-body
stabilization. Preserve dimensions, mass, drag, buoyancy and joint soft stops.

Contact physics v2 is opt-in for this rectangular motor laboratory. Contact and
geometric validation represent the directional head ellipsoid, tail and fin,
including both endpoints. Tail endpoints share each link's coefficients.
The 0.8 N/m spring remains; contact-only approaching damping explicitly changes
to `2*sqrt(0.8*link_mass)`. The legacy non-contact equations/integration are
preserved. This changes physical contact forces and trajectories: it is not
literally unchanged physics. The production pond path remains untouched.

## Contact allowance and mandatory gate

Fix overlap tolerance BEFORE tests at **0.0005 m**: the largest link's static
1% negative-buoyancy deflection at 0.8 N/m, plus two 5 ms travels at a declared
20 mm/s normal approach speed, plus 2 µm numerical margin, rounded upward.
This is about 0.69% of the 72 mm body length. Density 990 is a regression-load
stimulus only; motor training retains density 1000 and existing body mass.

Signed gaps from −0.5 mm through zero are acceptable compliant contact. Greater
overlap is excessive and rejected; entire-body escape beyond a plane is
separately detected. Invalid/nonfinite states stop affected work. The overlap
allowance is NOT a positive hovering band. No threshold changes after results.

One immutable preparation records numeric states and hashes before one run:
an independent CPU-double geometry oracle, old fin/proxy reproduction,
endpoints/orientations and contact/excess/escape/invalid classifications;
eight dynamic cases ×5 s×three timesteps (5/2.5/1.25 ms)×CPU/CUDA, capped at
112,000 substep calls; four free-water old/new comparisons add at most 1,600.
**Total cap:113,600 calls**, equivalent to 4,960 controls of 50 ms duration.
No PPO, sweep or automatic repeat. Require upward response, loaded support,
no excessive overlap/escape at any substep, registered timestep sensitivity,
CPU/CUDA parity and exact unchanged free-water old/new arithmetic. Numeric
criteria and all fixtures are frozen in `contact/protocol.json` before execution.
On gate failure do not train or expand the contact solver automatically.

## Movement tasks, observations and reward

[MOTOR_V2_TASKS.md](MOTOR_V2_TASKS.md) registers the identical 64 withheld
episodes per arm:8 move,8 turn,8 brake,8 slow arrival,12 unsupported calm holds,
12 unsupported current holds,8 bottom rests. Each is at most600 controls/30 s.
Eight braking starts reuse complete archived actual-swimming captures, with no
new warmup. No held-out data is training or tuning data.

Ten-second holding gates require pond-relative position≤7.2 mm, heading≤10°,
head AND COM speed≤2 mm/s, body angular speed≤0.05 rad/s. Joint/tail motion is
allowed and reported. Unsupported credit requires floor clearance≥2 mm and
zero floor load at every 5 ms sample. Supported rest requires gap from the
negative tolerance through +5 µm and positive floor spring force>1e−8 N at
least once in that SAME ten-second window; report load fraction separately.
All64 task gates and zero physical/numerical failures are required for full
acceptance; partial actors cannot pass full acceptance.

Use the existing validated uniform-current drag model at5 mm/s in fixed
horizontal directions. It is prescribed reduced-order flow, not CFD or all
water conditions. Historical tests remain historical; the new contact/current
combination receives focused checks. New observations4–6 give body-local water
velocity/.02;7 gives signed floor clearance/.072;8 contact. Previous-action
29–32 remains last actual normalized torque. Reserved local-obstacle channels
36–53 remain zero with availability54 zero. No new perception or shelter claim.

Both arms use matched fresh output heads, critics, exploration and Adam, with
compatible original55 body features9–32. New feature weights start zero.
Seeds:head `0x4D082601`,training `0x4D082602`,withheld `0x4D082604`.
Seven matched lessons supply calm/current movement and bottom-rest requests.
Fixed cost is position error/72 mm squared +.25 terminal velocity error/20 mm/s
squared +.1(1−heading dot)+.02 body angular velocity/.5 squared +.002 mean40
actual normalized torque squared; reward=−.05cost. No joint/surface stillness
penalty defines active station keeping. Positive actuator work is reported;
mechanical work and kinetic energy are not hunger reserves.

PPO stores/evaluates raw sampled Gaussian latent actions, including values
outside clipped targets/actuator bounds. Never use resulting torque or clipped
targets for the likelihood. Preserve actual-torque effort, previous-action
meaning, held-target state and complete random/physical/optimizer checkpoints.

## Readiness, execution caps and failure handling

After contact passes, focused data readiness spends at most32 motor controls:
eight fixed starts×two interfaces×CPU/CUDA×one50 ms step. Check action transforms,
actual effort/work, observations/current and parity. Gaussian replay/full-save
checks add no physical integration or optimizer updates.

Maximum training:64 updates EACH×512worlds×64 controls,2,097,152 rollout slots
per arm,4,194,304 total. PPO gamma.99,GAE.95,ratio.2,entropy.002,value.5,
4epochs×8minibatches,fresh Adam lr.001,beta.9/.999,epsilon1e−5,normclip.5,
initial logstd−.5. Record actual substeps separately from frozen rollout slots.

A finite physical safety failure stops its arm and preserves failed data/last
completed checkpoint. The other arm may independently proceed from its matched
verified-safe initialization under its own cap. Shared physics, numerical,
data-integrity or evaluation defects stop all affected work. Validate finite
data BEFORE treating a physical failure as arm-local. No silent retry, tuning,
best-of-heldout selection or automatic extension.

Each terminal actor receives ONE evaluation on the same64 fixtures, at most
38,400 controls per arm/76,800 total. Label partial actors and unproved skills.
Keep evaluation separate from training reward. New schemas/hashes/viewer must
bind task/physics v2. Every presented run states its scene and whether weights
update; frozen live inference is not learning. No production replacement,
chooser training, limbs/growth, app overhaul, merge, deployment or automatic
follow-up is included.
