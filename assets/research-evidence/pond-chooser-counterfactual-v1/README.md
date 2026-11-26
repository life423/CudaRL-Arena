# Same-state chooser diagnostic — November 25, 2026

**Safe alternatives exist with the unchanged motor in5 of9 captured states.**
This establishes avoidable chooser-sequence failures in those cases, rather
than an inability of the body to execute every feasible escape. It does not
establish that every late state is recoverable or that the motor is globally
qualified for sharp turns. No motor/chooser weights, physics, active Godot file,
or mouse/keyboard input changed. No training or scripted avoidance is installed.

## Exact-state evidence

Three previously rejected research-policy failures are replayed once: frontier8
northwest_diagonal, frontier128 east_shelf_tangent and frontier128
northeast_to_center. They are not failures of the installed appended16 policy.
All2522 saved COM samples match exactly (maximum difference0m), all251 saved
decision actions match, accepted substep counts are exact, and the original
clearance faults reproduce. Three full-state captures per prefix are made at
the last chooser decision before20,12 and4 seconds remain until that original
failure. The labels are retrospective time, NOT the current sensor's TTC.

Each saved World includes body/momentum, joints, servo/filter state, current
heading/request, whole-beat and COM history, visit memory and control clock.
Every branch memcpy-copies these exact bytes and verifies the same checksum
before its first action. No branch reinitializes or restores its body. Policies
are feedforward; temporal input state lives in the captured World. All35 inputs
and both policies'15 logits at every fork are retained with full native captures.

At each of9 states,26 branches compare the original research chooser, the
installed chooser, and24 diagnostic request sequences. The library is requested
speeds6/8/12mm/s ×rates−8/−4/+4/+8deg/s ×initial durations4/12s. Each sequence
then holds zero course rate at the same speed for8s and returns to the original
research chooser for the rest of40s. The return is live frozen-policy inference
from the modified physical state, not replay of archived action commands.
No sequence is used for training or deployed as a route/controller. ±8 remains
unqualified as a general motor skill despite some successful local witnesses.

## Comparison

All original chooser continuations reproduce their failures (0/9 escapes).
The installed chooser escapes2/9 states created by the rejected policies; this
off-distribution probe does not replace its original12+4 and four30-minute tests.
Of216 request-sequence branches,29 demonstrate useful40s escapes across5 states.
Three other branches stay physically clear for40s but never meet the declared
sustained-inward criterion and are reported separately. Total useful branches
including installed-policy witnesses are31/234; total safe40s branches34/234.

| Original failed approach | Approx. seconds before original failure | Original policy escape | Installed policy escape | Successful diagnostic sequences |
| --- | ---: | ---: | ---: | ---: |
| Northwest,frontier8 |20|No|Yes|9/24|
| Northwest,frontier8 |12|No|No|8/24|
| Northwest,frontier8 |4|No|No|0/24|
| East shelf,frontier128 |20|No|No|1/24|
| East shelf,frontier128 |12|No|No|6/24|
| East shelf,frontier128 |4|No|No|0/24|
| Northeast,frontier128 |20|No|Yes|5/24|
| Northeast,frontier128 |12|No|No|0/24|
| Northeast,frontier128 |4|No|No|0/24|

A useful escape requires40s without a guard/reset, positive certified clearance,
tilt<=10deg,depth error<=1mm,mean half-second whole-body COM speed>=6mm/s,
maximum rolling-speed stall<=8.14s and inward COM motion>=6mm/s sustained1s.
Course uses1s COM displacement. Safe-but-non-inward cases remain separate.
No sequence found in this finite library is not proof that none could exist.

### A direct avoidable-choice example

Northwest at59.2s,about12.295s before the original failure, has exact state hash
14505488376059004609. Both source and installed choosers initially select action9:
requested8mm/s,+8deg/s. Original-policy continuation stops12.295s later without
sustained inward motion. Its mean actual COM speed is23.55mm/s, markedly above
the request. At the fork the bank sensor sees23.00mm/s,184.57mm forward range
and8.03s constant-shape TTC; the risk signal is already valid, not absent.

From those identical bytes, action8 (same requested8mm/s,+4deg/s) for4s,
then0deg/s for8s,then the SAME original chooser, survives the full40s. Minimum
certified clearance is14.38mm,mean COM speed18.61mm/s,zero stalls/tilt/depth
error,and sustained inward travel is reached15.28s after the fork. This is a
measured physical witness that a less aggressive initial turn and different
sequence can succeed where continuing the research chooser fails. It is not
a prescribed maneuver for the application.

On the east-shelf20s fork, a successful sequence even begins with the exact
same action14 as the original chooser. Their later requests differ, showing
why checking only the first turn direction or a constant40s turn is insufficient.
That case's reported minimum0.107mm includes the shared prefix's lifetime
minimum because diagnostic branching never clears controller/world history.
The successful witness is not considered a robust new policy or a clearance
margin suitable for automatic publication.

The fixed-shape TTC also changes with the moving course. On the east-shelf12s
fork it predicts20.91s straight travel, despite original-policy failure12.71s
later while continuing to turn. Thus sensor TTC is not the time to collision
under the chosen future maneuver. Both actual speed and future course matter;
raising one fixed deadline cannot identify all safe action sequences.

## Conclusion and limits

For the recoverable states, the next learning target is the chooser's sequence
of rate/speed requests and its credit for sustained safe progress. More motor
turning power is not shown to be necessary to avoid those particular failures.
The moderate±4 witnesses in northwest and northeast also show that the result
does not depend entirely on the unqualified±8 requests. Earlier action still
matters: no tested sequence escapes any of the three4s-before-failure states.

Recommend a focused PPO curriculum using these native bank-approach states,
with held-out cases and unchanged qualification. Use the diagnostic witnesses
to identify recoverable mistakes, not to install scripted turns or silently
switch to imitation learning. Investigate sequence-level risk/credit and the
requested-versus-actual speed discrepancy. No further training is started here.

## Execution and preservation

One native shared-physics CPU run completes in19.143s,19.194s externally, with
2989111 attempted/2988908 accepted substeps, below7776000. Prefixes account for
161546/161543;234 branches2827565/2827365. All200 branch failures and three
prefix failures are preserved; guard rejection is not independently confirmed
terrain contact. No nonfinite/native fault, reset within branch, optimizer
call, CUDA training or desktop automation occurs. The archived analysis checks
all branch counts, exact fork identity and work sums. All11 active Godot package
hashes still verify. Complete source/model/executable bindings, every decision,
all234 traces and9 binary captures are under`data`; build evidence remains in
local`build/flexible-tail/chooser-counterfactual-v1`. Godot stays appended16/motor88.
