# Fixed position and depth lesson: 2026-11-04

Frozen before physical execution. User explicitly requested continued PPO teaching
to slow down and hold position. Preserve checkpoint336 and all straight-swimming
tests; all four joint targets remain network outputs. No posture clamp, joint
freeze, external propulsor, scripted rescue, physics change, push or merge.
One new study only; no retries, extensions, extra seeds or automatic adoption.

## Evidence and design

Prior reward6 stopping trial336->352 reduced final speed to2.80/2.87mm/s but
never achieved the0.5mm/s stopping gate and regressed short swimming4/4->0/4.
Only zero terminal-velocity inputs were exposed. The path features explicitly
removed displacement along the heading, so longitudinal holding error was absent.
At2.8mm/s, old horizontal stopping cost was0.1225 (8mm/s denominator).
Old return/GAE verification established that post32s rewards reached optimization.
This is a task-observation/objective change, not a claim that previous PPO was off.

Physics inspection: model6 sets every link mass=volume*1000kg/m^3; default water
density1000 gives neutral net static buoyancy when fully submerged. The head's
0.4mm buoyancy-center offset supplies physical righting torque. Gravity, water,
distributed drag, body geometry and all torque/filter limits remain unchanged.
Tail/joint motion is allowed while holding; there is no zero-joint-motion reward.

New explicit observation schema3 retains the40inputs and2mm body-frame path
scale. For swimming it is exactly schema2. For zero-speed holding only, path
inputs0..2 become full target-minus-COM position error. At the32s stop request,
capture horizontal COM once as the fixed hold anchor; retain original world-depth
target(-80mm) and original heading/lane reference. No body/filter/time/AR reset.
The anchor never follows subsequent motion. Checkpoint schema4 pairs schema3
with reward8 and rejects mismatched readers; prior artifacts remain unchanged.

Reward8 preserves swimming reward5. During zero request, score horizontal velocity
against a target-error/2s return reference capped at2mm/s. This reference only
scores outcomes; it never supplies actions or body forces. Normalize horizontal
velocity error by1mm/s, vertical error from existing depth-settling reference by
1mm/s, retaining original depth/lane/posture/effort penalties. Add horizontal
anchor error cost0.05*||error/2mm||^2. At2.8mm/s and zero anchor error, horizontal
cost7.84 is64times prior stopping cost. Keep the acceptance tilt limit10degrees.

Restore a copy of336, both actor/critic, both Adam histories/counters and fixed
sigma/rho. Actor rate0.000125, critic0.001. Same40->32->32->4/1 networks, native
CUDA PPO and Gaussian action-probability/AR accounting. To retain swimming,
even-index worlds perform original8mm/s for64s; odd worlds perform8mm/s32s then
zero32s. All use original pose0/3 and fixed-target0,+/-0.5mm depth-start mixture.
Half the worlds keep original swimming throughout. No separate skill network.

## One frozen budget

Identity:build/ppo/position-hold-r8-336to368-v1.
Source336 SHA256:810DBD593CE39475CDD6E95B709BEF3FDB34D50C9FCBBB2C345C7A8882A0BEF0.
Source protocol SHA256:FCAA74C5DA6ADC1E000282776CEBFC51F007B00633E7A6A29ED9D8A34E485E1E.
Exactly one attempt, at most32additional updates336->368; intermediate
assessments344/352 and final368.32worlds x1280controls,64s true episodes,
0.05s controls/0.005s physical substeps. Up to1310720training controls,
13107200training substeps,10240additional Adam steps per network. Seed1247037243,
rho0.8,log_sigma[-1,-1,-1,-1],two epochs,minibatch256,gamma0.99,lambda1.
Training and60s arena1.2x0.8m, unchanged vertical bounds/clearances.

Combined training/CPU/CUDA budget1351264controls and13512640substeps includes
64axis/parity controls; baseline4x8s+6x64s; each of three candidates4x8s+6x64s+
2x60s. This counts all backend/internal attempts including failures; no physical
replays outside this single run. Native cap180s, own process cap180s.
On first safety/nonfinite/numerical fault stop the whole attempt, preserve partial
evidence, do not retry. No new training using unused budget after closure.

Before launch: native build, focused CPU/CUDA synthetic and zero-integration
tests; verify mixed production observation kernel, fixed anchor/depth, no reset,
all40forward observations identical to schema2, finite rewards and paired schema
rejection. Freeze executable/source/protocol/initialization/parent hashes and
record launch-attempted before launching. Native run-started guards execution.
Report actual training start/first update and intermediate results.

## Frozen comparisons and acceptance

Baseline uses exact336 weights with the new explicit holding inputs, identifying
the input-interface change; historical reward6 stopping evidence remains separate.
Six64s stop cases:poses0/3 crossed with centered,+/-0.5mm initial depth offsets.
All request8mm/s0-32s thenzero32-64s, same fixed hold anchor and depth references.
Report first one-second low-speed streak, final uninterrupted stopping time,
stopping distance, final speed, backward travel, max depth/lane/tilt, signed roll
and pitch, hold drift/anchor error, and all four commands/joint rates/torques.

Retain old stopping gates: actual3D COM speed<=0.5mm/s at every50ms sample48-64s;
final56-64s longitudinal excursion<=2mm,depth<=1mm,lane<=2mm,tilt<=10degrees.
Depth/lane/tilt gates apply across the whole64s. Pre-stop2-32s swimming must pass
7.2-8.8mm/s,3D RMSE<=2.5mm/s,path<=5degrees,work<=0.001J/m. Additionally require
3D distance to fixed stop anchor<=2mm throughout final8s for actual position hold.
Joint/tail motion is allowed. No discounted or relaxed tilt criterion.

Each candidate also receives four original8s swimming tests and two original60s
swims, even when stopping fails, to expose regression. Keep all original gates
and startup windows. Report full formal long pass counts separately from retaining
336's28/30 passing windows4-60s;336's two startup misses remain failures.
A usable holding milestone requires4/4short,6/6stop/hold, and every previously
passing4-60s window retained. No automatic viewer adoption. No300s cases, new
32/64s pure-swimming recovery study or indefinite-performance claim is authorized
by this study. Preserve all candidates, failures, protocol and actual resource use.

## Completed result: rejected

Training started2026-11-04T17:11:06.755Z; first update337 completed17:11:08.615Z.
Process launched17:11:05.053Z and ended17:12:02.688Z(exit0,57.628s).
Completed32updates336->368,1310720training+40544evaluation=1351264controls,
13512640substeps,10240additional Adam steps per network(final31940each).
No physical/numerical safety stop, retry or extension. Training is stopped.

| Stage | Original short swims | Stop+hold | Worst depth/lane/tilt across6stop cases | Worst final8s anchor error |60s windows |
|---|---|---|---|---|---|
|336,new hold inputs,no updates |4/4 |0/6 |7.53mm /6.83mm /123.80deg |70.11mm |Historical28/30 |
|344 |0/4 |0/6 |8.23mm /2.54mm /120.55deg |66.57mm |28/30 |
|352 |0/4 |0/6 |8.92mm /1.35mm /117.48deg |65.53mm |28/30 |
|368 |0/4 |0/6 |9.00mm /5.96mm /119.53deg |64.45mm |0/30 |

Final speeds0.1060-0.1065mm/s are low, but permanent low speed occurs18.7s
after request, missing the16s deadline. Every stage fails stopping deadline,
holding and whole-body state gates. Low late drift is not target holding: the
creature remains~64mm away,~9mm below target and greatly overtilted. Final short
mean speed6.3556mm/s is below7.2minimum; pure60s swimming also regresses. Earlier
344/352 retain28previous4-60s windows but fail short startup speed. No adoption.

The new input interface itself destabilizes inherited336 before learning:
historical zero-goal baseline keeps tilt8.83/9.36degrees but continues8.37/8.41mm/s;
same weights with longitudinal hold error slow down while tilting~124degrees.
Across1280centered precommand controls, ten common motion fields match exactly.
This isolates a compatibility problem rather than evidence that336 could hold.
The stronger reward also changes return scale substantially(first-update mean
-92.43,std141.29), so simply extending this run is not justified.

Both networks and both Adam moments/counters restore exactly; only planned actor
LR differs. All4joint outputs and actual rates/last torques are recorded, along
with signed head roll/pitch. No joint/posture clamp, scripted body action or
physics change was introduced. Static buoyancy/righting remain as inspected.

Build and31focused CPU/CUDA checks pass,0skips. Actual8holding synthetic
CPU/CUDA cases agree within3.82e-6; production mixed observation kernel passes.
Post32s20480reward rows and126minibatch rows with nonzero policy gradients
reach PPO; GAE absolute discrepancy2.13e-4 is within existing scale-aware tolerance.
Three saved candidate checkpoints reload at344/352/368 with both Adam counters
24260/26820/31940 and zero dynamics/optimizer updates. A stale legacy test that
treated reward7 as unknown was fixed before launch; unknown9still rejects.

Two generic readiness labels were corrected after the completed run: holding's
actual wrapper cap is180s and its regression evaluation is unconditional. Frozen
original evidence remains unchanged; a zero-step metadata preparation confirms
the correction. No extra training or physical evaluation was run.

Preserve all candidates as rejected, original336 unchanged, and unrelated Godot
edits/evidence untouched. Durable record:
assets/ppo-four-axis-checkpoints/position-hold-r8-336to368-rejected.
Full per-control traces and executable/source snapshot remain in the ignored run
directory. No push/merge/viewer installation. The one study is closed.

Next proposal: retain existing lane/depth feature semantics and add longitudinal
holding information through a migration initialized to preserve336outputs;
use a gradual bounded curriculum and balanced reward scales. This is a proposal,
not a demonstrated cure or permission for an automatic new trial. No successful
position/depth hold, disturbance recovery or indefinite performance is claimed.