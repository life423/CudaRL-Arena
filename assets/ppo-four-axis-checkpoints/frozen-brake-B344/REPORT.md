# Frozen braking-response diagnostic: completed

The exact approved6240control diagnostic completed. No pulse met the predeclared
useful-braking threshold.24/48 pulses produced small reductions relative to their
matched zero-goal branch without exceeding branch depth/lane/tilt limits. The
strongest response reduced last.5s speed by0.121290mm/s and2s forward displacement
by0.200241mm; both required reductions were>=0.5. It still traveled8.150128mm/s.
This is weak relative slowing, not stopping or demonstrated full braking ability.
Only9/48pulse branches had a last.5s mean speed below their own32s initial speed;
the largest such absolute reduction was0.055976mm/s.

## Results and physical constraints

Six original pose0/3 x depth-offset0,-.5,+.5mm cases. FrozenB344, deterministic
CPU inference and original model6 physics. Each32s8mm/s prefix was captured once;
ten2s branches from identical state: continue8, requestzero, and +/-0.2latent
single-axis pulses on four learned outputs for.5s then unmodified learned output
for1.5s. Every other output stayed learned. No posture clamp, external propulsion,
physics/filter/torque change, optimizer, training, runtime installation or adoption.

| Metric | Measured |
|---|---:|
| Completed branches |60/60 |
| Pulse branches meeting both braking thresholds |0/48 |
| Pulse branches improving speed AND displacement relative tozero |24/48 |
| Pulse branches within depth/lane/tilt envelope |48/48 |
| All branches within depth/lane/tilt envelope |60/60 |
| Worst branch depth error |0.546179mm (limit1mm) |
| Worst branch lane error |0.620530mm (limit2mm) |
| Worst branch combined tilt |6.027208degrees (limit10) |
| Peak applied branch torque |2.082918microNm (limit10) |
| Minimum speed anywhere in the branches |7.853515mm/s |

The best response is case3(pose3,-.5mm), branch7(axis2,+.2latent). Depth0.504112mm,
lane0.545774mm,tilt5.546093degrees. No threshold-sized response was demonstrated
for either pose. The opposite pulse signs can speed up the swimmer, showing a
measurable but small local action effect under these exact conditions.

Unperturbed zero-goal branches averaged8.228710..8.294405mm/s over their final.5s
and traveled16.173080..16.327820mm in2s. Compared with matched continuing8branches,
requestingzero increased last.5s speed by0.074417..0.086021mm/s and displacement
by0.066956..0.079865mm in all6cases. These are coherent goal/anchor branches, not
an isolated neural-input substitution. The learned zero-goal behavior therefore
does not produce useful braking at these captured states.

The32s prefixes pass depth/lane/tilt in5/6cases. Case5(pose3,+.5mm) retains the
previous10.421790degree tilt failure; it is not erased by subsequent safe branch
behavior. Every recorded prefix motion/depth/lane/tilt/command field matches the
previous frozenB trajectory exactly(maximum difference0). Initial branch states
and original references/filters are preserved; original fixed depth remains-80mm.

## Accounting, checks and limitations

Started 2026-11-04T19:52:07.2495754Z; ended 2026-11-04T19:52:08.4873776Z;
exit0, native1.151771s/process1.237923s, below30s. Prefixes3840controls + branches
2400=6240controls/62400substeps. No physical preflight steps, retry, extra branch,
extension or optimizer update. Actor parameters unchanged. The attempt is closed.
The preflight CPU actor matched independent NumPy inference at16saved states
with maximum latent error4.172325e-7(tolerance2e-5). Build3.836613s and zero-step
preparation.171694s. Independent saved-trace analysis confirms all60gates, pulse
axes/durations/decoding, sample counts and metric arithmetic without integration.

These are fixed half-second one-axis pulses at six captured phases and one
frozen policy. This does NOT test coordinated counterstrokes, larger/longer
pulses, additional phases,16s stopping deadlines,32s stationary holding or anchor
recovery. No tested response satisfies the0.5mm/s+0.5mm local criterion. That does
not prove the model cannot brake. All four joints may move while holding.

Before more PPO, inspect coordinated/phase-dependent braking authority and choose
a new explicitly bounded protocol. The small response does not justify silently
increasing training length, reward weights, pulse amplitude or physical limits.
No next trial is launched or authorized by this recommendation.

## Exact reward in the preceding two-arm training trial

This diagnostic trains nothing and uses no reward to select actions. The preceding
trial's active-holding reward was r=-(0.05s)/g times the sum of normalized costs:

- Horizontal velocity: ||v_h-v_h*||^2/(0.001m/s)^2, where
  v_h*=cap_norm(-e_h/(2s),0.002m/s).
- Vertical velocity: (v_y-v_y*)^2/(0.001m/s)^2, where
  v_y*=clip(-e_y/(2s),-0.001,+0.001m/s).
- Original lane:0.05*(cross-track error/0.002m)^2.
- Horizontal anchor:0.05*||e_h||^2/(0.002m)^2.
- Depth:0.05*(e_y/0.001m)^2.
- Roll:0.05*(roll/10degrees)^2.
- Pitch:0.01*(pitch/10degrees)^2.
- Effort:0.002*mean((actual_torque/1e-5Nm)^2), over4axes x10substeps.

e_h is horizontal displacement from the hold anchor; e_y is signed actual minus
target depth(Yup). g=1 inA and1024 inB. These error ratios/costs are dimensionless;
the control interval integrates the negative cost. They activate only in active
holding after the32s zero request; the horizontal anchor is captured once, while
original depth/lane/heading remain fixed. Other swimming rewards are unchanged.
Scaling all holding terms together does not change their relative proportions.
At the exact target, desired velocities arezero; away from it, the reward asks
for motion BACK toward the point rather than freezing at the wrong location.

Training reward is soft. Separate full acceptance requires actual3Dspeed<=.5mm/s
at every50ms sample48..64s,3Danchor error<=2mm and forward excursion<=2mm during
56..64s, whole64sdepth<=1mm,lane<=2mm,combinedtilt<=10degrees, plus swimming
regressions. Combined tilt is not the same as separately weighted roll/pitch
reward terms. Improved reward/critic fit or a safe two-second branch is not a
full hold-test pass.

## Preservation and files

All4362prior evidence files and15unrelated Godot/exploration files retain their
recorded hashes; all56frozen current-run source/input hashes verified unchanged.
Checkpoint336SHA256:
810DBD593CE39475CDD6E95B709BEF3FDB34D50C9FCBBB2C345C7A8882A0BEF0.
FrozenB344SHA256:
8D705FC82A2B12D465E4FDCD48BC8D881EA0710192B030B262E4A26A6C6AF2B2.
Raw source/executable/logs remain inbuild/diagnostics/frozen-brake-B344-v1.
This folder preserves controls.csv, all60branch rows, six state snapshots,
protocol/input metadata and independent analysis. Source changes were limited
to adding the new CPU diagnostic target/runner; no motor/physics/NN code changed
during this diagnostic. Branch codex/minimal-foundation; changes remain local,
uncommitted, with no push/merge or viewer policy adoption.
