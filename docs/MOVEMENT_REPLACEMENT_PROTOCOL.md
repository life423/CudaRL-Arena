# Movement model6: fixed short mechanical acceptance

User authorized replacement and quick validation21:49:06; no obligation to reuse
broken tools. Existing runtime/save physics2 and all earlier evidence remain
unchanged. New opt-in native functions in `motor/movement_model.h` operate on
caller-owned bodies, with explicit model6 registration and no Godot/session owner.

## Design

Remove the old implicit joint damper torque_limit/12 (.833333microNm*s/rad).
It is5.75times the tail servo's own derivative damping. The predicted dry
overdamped.5Hz response (.318/.142rad,50.45/73.49degrees) closely matches the
measured wet stroke (.323/.162rad,49.35/70.46degrees). This identifies a concrete
mechanical limitation without declaring fluid effects absent.

Use internal torque-limited floating-base inverse dynamics with a3Hz,zeta.9
joint reference tracker and angle/rate/acceleration feedforward. Solve the six
unactuated root momentum equations, request ONLY four joint torques, clamp at
unchanged10microNm, then solve the complete actual dynamics. No body force,
target projection, unlimited actuator or scripted rescue. Ideal universal joints
have zero passive hinge friction; actuator damping is explicit in the controller.
Invalid/reference bounds and joint travel reject rather than project the body.

Retain20/26/26mm lengths and masses (total2.537446925g). Represent the distal
26mm rod(radius1.5mm) plus two.1mm-thick membrane wings of total added height9.7mm:
outer span12.7mm. Compute composite inertia; enclosing geometry is a conservative
collision envelope, not solid-cylinder volume. Pressure areas derive from this
same description, with explicit Cd (.3/.8/.8 head,.15/1/1 other links). Local
point velocity includes rotational Jacobians. Three-point Gauss weights sum to
one and integrate the axial second moment. Viscosity uses a declared equal-volume
sphere approximation. Uniform density means coincident mass/buoyancy centres.
All drag assembles a positive-semidefinite resistance matrix. No lift/added-mass
or wake term is fabricated; this remains an approximate submerged drag model.

## Sole fixed run

Output `build/movement-model/2026-10-30-v1/`. Freeze numeric/source/executable
before execution. One centered straight stationary initial body, zero current.
8s per case,0.6rad1Hz yaw targets,90degree tail lag, smooth2s ramps at both ends;
other axes zero. New tracker sees analytic position/velocity/acceleration every
native step; legacy retains its50ms held-position interface. This cadence change
is part of the actuator replacement, explicitly not an unchanged-interface claim.

Cases CPU/CUDA: legacy; new servo with old water/geometry; complete new wet model;
complete new dry model; new wet resting baseline; new wet model at2.5ms. First
five have1600native5ms steps each per backend; finer case3200steps/backend.
**22400 attempted physical substeps TOTAL**, failures/launch reservations included;
maximum1920control advances, no retries/sweep/extension. Native wall cap10s checked
before each batch. Compilation/setup time is separate. Static independent mass,
span, initial clearance and rotating-point dissipation checks precede integration.

Every step requires finite state/reference/torques, actual10microNm cap, full
envelope gap>=10mm, joint angle<=1rad, valid SPD solve without pivot flooring,
normalized solve residual<=1e-8, nonnegative fluid dissipation. State/backend
agreement uses existing2e-4+2e-3abs(CPU), counts exact. Record signed/positive
actuator work, fluid dissipation and backward-Euler quadratic velocity-increment
loss separately. Energy-budget residual must stay within2%of accumulated positive
work+fluid loss, plus2e-10J; no hidden energy source is allowed as free thrust.

Acceptance before learning: complete new wet central2..6s speed>=.0072m/s
(0.1bodylength/s), actual yaw amplitude>=.48rad, tracking RMS<=.12rad for both
wet/dry, and central-speed refinement<=5%. These are new honest mechanical gates,
not replacements for earlier learned20mm travel/terminal-speed/heading criteria.
Failure closes this run and blocks learning; do not tune/retry under this budget.
Any numerical/physical failure stops the comparison. Diagnostic motion remains
prescribed; there is no learned controller or runtime adoption in this stage.
