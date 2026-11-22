# Why the holding comparison failed: saved-data investigation

The best-supported diagnosis is that training continued to sample and reinforce
variations of swimming; it never experienced a stopped or reversing holding
state. The stop command is visible to the actor and holding gradients are
connected. Position saturation, weak learned position feedback and residual
value/objective issues are measured concerns, but no single root cause is proven.

This investigation performed only saved-data reads, network forward inference,
independent arithmetic and unapplied analytical backpropagation. It integrated
ZERO physics controls, took ZERO optimizer/training steps, changed no runtime,
and modified none of the previous evidence or source files. All data below are
from the completed comparison, not new policy evaluations.

## What the saved samples establish

| Active-holding pre-action samples, across8batches/arm | A | B |
|---|---:|---:|
| Samples |81,920 |81,920 |
| Actual3DCOMspeed<=.5mm/s |0 |0 |
| Negative forward velocity |0 |0 |
| Minimum speed observed |4.093mm/s |4.615mm/s |
| Position feature magnitude>=.99 |78.052% |78.967% |
| Position feature magnitude exactly1 |26.384% |28.522% |
| Positive forward hold-point error feature |0 |0 |

These are correlated50ms pre-action observations, not81,920independent trials;
they omit the final post-action endpoint. Endpoints in the deterministic tests
also fail stopping, as already recorded. All active-holding episodes begin after
32seconds of8mm/s swimming. No episode explicitly starts at rest with zero goal.
The samples therefore contain no successful example of the actual hold-speed
criterion or reversal needed to recover a passed fixed anchor. That is a strong
coverage limitation, not proof that more learning could never discover braking.

The final frozen B policy's mean joint-rateRMS is1.522rad/s during24..32s and
1.521rad/s during56..64s. Its continued8.09mm/s translation establishes failure
to hold. Joint movement itself is permitted and is not a failure: an accepted
policy may move all four joints while swimming in place. No posture clamp or
requirement to freeze the joints is recommended.

## Goal conditioning and action sensitivity

At the exact first stop-command states from the common first batch, I changed
only requested velocity inputs3..5 between8mm/s andzero. Heading, physical
observations, filter history and zero initial anchor error were held fixed.
All16holding worlds/both poses were included; no dynamics were advanced.

Decoded four-joint target differenceRMS:
copied336=0.112574rad;
Bafter344=0.114078rad.
Thus zero speed is not absent from the input path, and the network is not
mathematically insensitive to it. This does NOT mean that the response is
semantically correct braking: the336swimmer was trained on forward swimming,
and the physical tests demonstrate continued translation after the command.

On all10,240holding states in the common first batch, replacing only input40
withzero changes final B's decoded targets by0.009427radRMS,
versus0.594486radRMS target magnitude. Position feedback is active
but small on these states (about1.6% by that RMS comparison). The actor's new
columnRMS is0.00644790, rather thanzero. This is an input
sensitivity/ablation, not a physical-policy replay or evidence that a particular
target perturbation brakes the body. Late-state goal-only ablations in JSON can
form an unseen input combination (nonzero input40 plus8mm/s request); only the
zero-error command-boundary comparison above is used for the main goal finding.

## Position, time and value information

Input40=tanh(forward_error/20mm). At the2mm gate it is only0.099668; magnitude
.99 occurs at52.933mm. Farther overshoot loses distance resolution. Exact float
saturation occurs in28.522% of Bholding samples overall, and in93.516% of the
last batch's56..64s samples. It was already near-.99 through much of each trial.
Every nonzero observed hold error feature has the overshoot sign; there is no
coverage of targets ahead of the swimmer under zero goal.

Saturation does NOT disconnect parameter learning: a feature of-1 still produces
weight gradients. It prevents the controller/critic from distinguishing larger
overshoot distances through this scalar. It also cannot explain the absence of
braking immediately at32s, where the anchor error iszero and the stop goal changes.

Input34 iszero in every saved observation after2s; neither actor nor critic
explicitly sees the remaining64s episode time. In B's last batch, among exactly
saturated holding samples, critic-before residual correlates0.885 with remaining
episode time. Finite terminal returns change with time while forward distance is
largely saturated. This is evidence consistent with a remaining value-prediction
problem, not a proof of irreducible state aliasing or a justification to change
observations without a controlled test. The critic also omits explicit ARnoise
history; a state-only baseline is not thereby a disconnected or invalid PPO path.

## Learning signals and objective balance

The saved16rollouts reproduce the original conditional ARlogp with maximum
error1.33514404e-05,
below the existing1e-4gate. This checks the independently parsed weights,
observations, predecessor means/actions and probability arithmetic.
The completed run already verified actual returns, minibatch coverage, nonzero
late gradients and all2560Adam steps/network/arm; KL did not shorten an epoch.

Independent CPU reconstruction of each batch's INITIAL full-batch, unclipped
on-policy actor gradient gives nonzero holding signals on every row. B's new
input-column gradient norm ranges .0100..0225. Holding and other-task gradient
directions oppose in4/8 B batches (last cosine-.458), suggesting possible shared
objective interference. These are unapplied gradients at the collection policy,
not recorded actual later minibatch gradients or a replay of Adam/clipping.
Their norms/cosines do not prove which trained update caused failure.

At B's last update59.023% of still-swimming holding rows have positive normalized
advantage. Positive advantage means better than the current baseline estimate,
not passing the physical stop test. No sample in that group is actually stopped.
The aggressive scaling problem was reduced: active holding accounts for about
56.75% of B's last-batch cost, versus99.91% in A. B's conditioning improvement
therefore remains real under the prespecified diagnostic, but did not generate
successful braking/holding experience.

Within active holding, both arms retain the SAME component proportions: B's last
batch is72.155% combined lane/anchor cost,27.836% speed/recovery,0.006495% depth,
0.002472% posture, and about0.000002% effort. These proportions describe these
failed far-overshoot trajectories. They help explain why improving total reward
need not enforce the strict1mm depth/10degree tilt gates, but do not establish
that a different posture weight would solve it. Dividing all holding components
by1024 leaves their relative proportions unchanged.

The reward explicitly asks for return toward the fixed anchor: desired horizontal
velocity is -error/2s, capped at2mm/s. At large overshoot it asks for backward
motion, which none of the saved holding samples performs. With gamma.99 per50ms,
a reward16seconds later carries weight.04011 at the command. Immediate speed and
position costs still provide local signals, so discounting is another possible
limitation, not an explanation that removes all stopping incentive.

## Physical feasibility: what is and is not established

Model6's assemble() uses fully submerged uniform-density mass/volume centres,
mass=1000*volume and default water density1000. At rest with matched joint targets,
zero joint/filter rates and no current, its gravity/buoyancy term cancels and
drag/inertial velocity terms vanish. Thus the equations permit stationary
equilibria; the creature need not propel forward to counter unavoidable sinking.
This is a SOURCE-BASED INFERENCE, not a new simulated holding test. The old generic
physics buoyancy-offset code is not the model6composite movement assembly.

It does not establish that the existing four-joint controller can brake from
8mm/s, stay under10degrees, and return within2mm of the command-time anchor on
the required schedule. Internal joint targets, drag and gait phase make the
sign/size of physical braking response uncertain. Saved neural target changes
alone cannot answer that question. Passive slowdown alone also would not show
recovery to an anchor already overshot by many millimetres.

## Most informative next step: frozen braking-response diagnostic

Do NOT extend training or retune another reward based on these results alone.
First propose/approve a small controlled dynamics diagnostic, with no PPO or
runtime installation. It would separate useful action authority from the
learned goal response before selecting the next curriculum.

Proposed exact protocol, NOT EXECUTED:

1. Use a separate frozen copy of B344. Six original pose/depth cases, all four
   network outputs active. Run8mm/s for32s using unchanged physics/filters; capture
   body, joints, filters, task reference and heading at32s. This costs3840controls.
2. From each captured state run ten2s branches: continue8; requestzero; and eight
   zero-goal branches with a +/-.2latent diagnostic pulse on each one of four
   outputs for the first.5s, then unmodified learned outputs for1.5s. All other
   outputs always remain learned; original tanh/filter/torque constraints apply.
   Pulses are temporary research excitations, never a candidate controller,
   posture clamp, teacher, external thrust or runtime change.
3. Branches cost6*10*40=2400controls. Total maximum6240controls/62400substeps,
   zero optimizer steps, native/own-process30s caps, no retries/extensions.
   First hard numerical/integrity/physics fault stops the study; behavioral gate
   failures are recorded with no clamp or threshold relaxation.
4. Compare last.5s mean3Dspeed and2s forward displacement to each matched zero-goal
   baseline; also compare zero versus8. A useful local braking response must
   reduce speed by>=.5mm/s AND displacement by>=.5mm, with depth<=1mm,lane<=2mm,
   tilt<=10degrees throughout that branch. Report every branch, not only best.
   Finding such a response in at least one case of EACH pose supports accessible
   local braking authority. It does not count as passing full stop/hold gates.

If useful braking authority is demonstrated but the learned zero-goal response
does not use it, the next training proposal should focus on coverage: begin
zero-goal lessons near the anchor from rest/low speeds, then introduce higher
entry speeds while retaining swimming rehearsal and unchanged acceptance gates.
If no tested pulse brakes usefully, that is an inconclusive local probe, NOT
proof of physical impossibility; investigate coordinated/phase-dependent action
responses before choosing another PPO objective. No next training is authorized
or launched by this recommendation.

## Evidence and preservation

diagnosis.json stores per-batch coverage/saturation, matched-state sensitivity,
unapplied gradient diagnostics and frozen joint activity. diagnose_hold.py is the
reproducible analysis source; all source inputs refer to the preserved run.
34unique batch/checkpoint files were verified against the run's SHA256manifest.
All36prior binaries,15unrelated files,67frozen inputs and129archived evidence files
were verified unchanged. Original336 SHA256:
810DBD593CE39475CDD6E95B709BEF3FDB34D50C9FCBBB2C345C7A8882A0BEF0.
Branch codex/minimal-foundation. No source/runtime change, training, physical
integration, push or merge during this investigation. Existing local experiment
changes remain as previously reported; no new candidate is accepted.
