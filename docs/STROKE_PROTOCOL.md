# One prescribed stroke versus rest, not learned movement

Explicit user approval21:37:33. Output `build/stroke/2026-10-30-v1/`.
Default centered straight stationary BodyState, position(0,-2,0)model units,
identity orientation, zero joint angles/velocities. Existing joint-target mode,
fixed gains, fluid, geometry/masses, physics2 and10microNm cap unchanged.

Two matched conditions (stroke and all-zero targets), CPU and CUDA,8s each:
160controls*2conditions*2backends = **640controls/6400native5ms slots TOTAL**.
Charge batches before integration; failures included. No retry, sweep, PPO,
extra fixture, extension, adoption or migration. Freeze source/executable/binding
before sole run. Abort all affected work at any existing physical/domain,
finite/torque,10mm native-sampled clearance, zero-contact or parity failure.

Control-start t=n*.05s, targets held50ms. Yaw joint0=.5E(t)sin(pi*t)rad,
yaw joint2=.5E(t)sin(pi*t-pi/2)rad; pitch axes1/3zero. E=sin²(pi*t/4)for0..2s,
1for2..6s, sin²(pi*(8-t)/4)for6..8s. Four0.5Hz cycles, quarter-cycle lag;
two interior cycles avoid relying on startup bending. All-zero matched targets
are a resting servo baseline, not disabled physics. No extra body thrust.

Compare whole-COM stroke-minus-rest forwardX, lateralZ and verticalY displacement;
report individual2s cycle progress, actual joint ranges/phase, control-end target
tracking RMS, requested/applied peak torque and accumulated positive actuator
work. Existing defined-field parity tolerance2e-4+2e-3abs(CPU); gaps2e-6+2e-3abs(CPU);
counts/time/flags exact. Clearly measurable forward propulsion requires total
netX>.1mm AND>5times CPU/CUDA netX disagreement, and positive net progress in
BOTH2..4s and4..6s cycles, on both backends. Report measured negative/small
motion regardless. This diagnostic threshold does not replace20mm learned-task
acceptance. One failed waveform cannot prove every body/gait incapable.
No integrated environmental force was previously logged; do not call actuator
torque a measured forward force or equate positive work with useful propulsion.
