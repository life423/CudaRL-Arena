# Model6 first validation: stopped, not accepted

The frozen comparison in `build/movement-model/2026-10-30-v1/` executed once
and CLOSED on its first outer interval.140substep slots charged,136physical
attempts (CPU70/CUDA66), native time.254s. CUDA's2.5ms case rejected attempt16;
the other cases completed their first50ms interval. No tracking/speed acceptance
was reached; no learning or runtime adoption ran.22260unused slots are unavailable
automatically under this closed no-retry protocol.91frozen source files and1017
previous evidence files stayed unchanged during execution; original checkout clean.

Initial recorded geometry remained safe (73.6mm gap), and recorded energy
errors were small. The original generic failure message did not identify which
substep gate rejected; its exact cause is NOT established. Do not infer a body,
tracking, propulsion or safety failure from the printed zero speed: measurement
windows were never reached.

Subsequent source inspection found an operator-consistency defect: stored float
mass/resistance triangles can differ by rounding, while Cholesky uses the lower
triangle and the audit/controller previously used both. Current source now forms
one explicitly symmetric double operator and uses symmetric mass consistently.
The1e-8residual limit is unchanged. Added stage-specific failure codes/residual
logging, and stopped failed candidate work from entering committed-work totals.
These fixes have NOT undergone a new physical run.

Strict CUDA rebuild passed.32/32CPU/CUDA static synthetic matrix cases passed
with ZEROphysical advances. Worst raw-operator residual4.28648e-9, corrected
2.89504e-16, matrix asymmetry4.54747e-13. This verifies the numerical correction,
but does not prove it caused the earlier uninstrumented rejection. Log:
`build/takeover-checks/movement-static-numerics.log`.

Implemented replacement remains experimental: explicit composite fin geometry,
geometry-driven dissipative drag, corrected inertia, and floating-base inverse
dynamics with four capped internal torques/reference feedforward. Physical
performance, acceptance and any compact stroke policy remain unverified.

## Renewed fixed-source registration: mechanical acceptance passed

After renewed user instruction22:03:13, a separately frozen registration in
`build/movement-model/2026-10-30-fixed-v2/` completed22400/22400attempts in1.014s,
with no retry inside that registration. Original140charged/136attempted run and
unused budget remain closed. Static32-case zero-advance check preceded this run.

Matched.6rad1Hz case speeds: legacy.275224mm/s; new actuator/old water7.968169mm/s;
complete model8.227876mm/s (.1143bodylength/s). Full model travelled56.1714mm/8s,
lateral-.063568mm, vertical0CPU. Tracking RMS.003706/.003705rad versus legacy
.442382/.442098rad. Peak actual torque1.268464microNm; positive work13.304515microJ.
Finer2.5ms speed8.269222mm/s, relative difference.5000%. Maximum model solve
residual below4.7e-16. Resting CPU body stays still. Energy-budget, native clearance
(minimum73.6mm), torque/finite and matched-backend checks passed.

Dry COM drift.200083mm/8s is a remaining integration/conservation limitation,
about.36%of wet travel; do not label the model exact or real-world calibrated.
The dominant speed improvement is actuator/reference tracking, with roughly3.3%
additional speed from the coherent geometry/drag/inertia change. Acceptance
permits the separately bounded compact learner stage; it is not learned swimming
or runtime adoption. New stroke helper/tool preparation follows this run.
