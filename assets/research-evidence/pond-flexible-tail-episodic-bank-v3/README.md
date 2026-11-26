# Learning signal, repeated bank encounters and retention — closed

November25,2026. The user authorized all three: score safe/failed continuations,
provide repeated meaningful bank practice,and retain ordinary navigation.
Native C++/CUDA,one chooser,frozen motor88,no food/scripted avoidance/mouse.
**The workflow changes pass their checks,but no trained policy qualifies.**

## 1. What the reward and critic actually say

The first audit loads the exact matching actor/critic from curved-bank-v1 full
checkpoint80. It forks the two saved0.8s World captures into62 continuations:
policy and15 actions held2.4/8s,then8s straight,then policy. The collector's actual
macro reward is used,including curved reward once per accepted transition and
the existing terminal penalty.15 continuations pass the40s physical/movement
gate. Execution4.657s,747612 attempted/747565 accepted substeps,zero optimization.

| State | Failed policy discounted reward | Safe example discounted reward | Initial critic V | Same example40s return with estimated endpoint |
|---|---:|---:|---:|---:|
| Phase8/north/negative tangent | -4.3501 | +6.9775 | 7.8740 | 18.5921 |
| Phase9/north/positive tangent | -4.4219 | +9.1399 | 14.8706 | 20.4335 |

The endpoint column includes a critic estimate; it is not a measured infinite-
horizon return. The measured reward-only ranking already favors safe movement.
Keep the environment reward unchanged for this experiment. This does not rule
out reward defects in other states or all exploration/survival tradeoffs.

Credit is not equally strong in both directions. First8-step counterfactual
GAE-style advantages for three safe phase9 +4deg/s short requests are negative,
-0.642 to-0.664,despite superior40s outcomes. Longer8s +4 requests have weakly
positive0.320–0.573 advantages. Phase8 safe examples have2.12–5.09. These forced
sequences are diagnostic comparisons,not on-policy performance or training labels.

### Correcting a critic-comparison confound

The critic learns the **stochastic training policy**,whereas the existing Godot
and fixed evaluator use argmax. Therefore comparing V to one greedy failure
does **not alone** establish critic calibration error. The earlier greedy value
gap in `unchanged-failure-value-error.json` must be interpreted with that caveat.

A preregistered read-only supplement tests16 shared categorical RNG streams
per state/model,plus greedy controls,using original shared PPO sampling math.
The starting and final policies each score0/16 sampled40s continuations in both
states. All sampled episodes terminate at the native clearance guard,so their
returns need no endpoint bootstrap. The supplement runs68 cases in1.132s,
175943 attempted/175875 accepted substeps,zero optimizer calls.

| Model/state | Critic V | Empirical sampled discounted return | Sampled physical passes |
|---|---:|---:|---:|
| Initial0 / phase8 | 7.874 | about-4.36 | 0/16 |
| Initial0 / phase9 | 14.871 | about-4.40 | 0/16 |
| Final128 / phase8 | 19.378 | about-4.38 | 0/16 |
| Final128 / phase9 | 25.188 | about-4.39 | 0/16 |

This is evidence of poor empirical calibration on these difficult held-out
states,not a theorem about the critic everywhere. A zero-physics scan binds all
129 portable actors to matching full checkpoint actor/critic arrays and reports
258 value/action predictions. Phase8's final greedy trajectory remains identical
to the starting failure over61 logged rows excluding logits. Highest-probability
action remains12mm/s,-8deg/s,with probability0.814 at the first decision. Simply
switching the viewer to sampling is not qualified by these results.

## 2. Repeated practice with correct time-limit handling

Keep the39 inputs,15 actions,environment reward,PPO parameters,motor and physics
unchanged. Fork the full intact parent80 learned state,then explicitly initialize
new training worlds:50% varied15–30mm bank tangents,25% earlier approaches,25%
interior. Preserve body,servo,request and history through all decisions.

New TRAINING episodes last60s from the first decision. Their boundaries set
done+truncated,compute V from the real healthy pre-reset final observation,and
cut GAE propagation into the next episode. Real failures remain done without
truncated and bootstrap0. Raw rewards are not modified by a bootstrap bonus.
The existing native GAE implementation already supports these semantics; the
collector now supplies the correct masks and final values. No Godot reset or
live action override is introduced.

Full session format4 records the60s definition and per-world episode clock,
timeout count,close-decision count and decision count. Legacy Lane/Counts layouts
are preserved for byte-exact format3 parent import. The original checkpoints
and separate candidate files stay intact.

CPU/CUDA bootstrap algebra agrees exactly on continuing,terminal and truncated
fixtures:advantages0.0337601,-1,and2.976 respectively. Placement4096 checks agree
within0.246um. Three actual updates demonstrate exact full-state resume. That
short diagnostic does not itself reach a60s timeout. The final checkpoint then
passes a zero-update native import/roundtrip with all13,095 real timeout counters
and1,048,576 decision counters preserved,covering nonzero episode metadata.

## 3. Measured learning and retained skills

**128 updates,1,024 CUDA worlds,1,048,576 macro transitions,4,096 new actor and
4,096 critic Adam calls.** Ordinary navigation remains represented in training;
all original tests run every8 updates. Representation does not guarantee retention.

| Metric | Previous close-shore-v2 | New episodic-bank-v3 |
|---|---:|---:|
| Episode starts | 1,178 | 15,234 |
| Training time limits | 0 | 13,095 |
| Scored clearance guard terminals | 154 | 1,115 |
| Main transitions | 1,048,576 | 1,048,576 |
| Trainer native wall | 131.419s | 178.147s |
| Final original short tests | 4/12 | **1/12** |
| Final close-shore tests | 4/8 | **6/8** |

The new run records44,705 near-region decisions,4.2634% of collection. More
episode starts do not mean all transitions are meaningful near-bank practice.
Neither original failing start passes in this run. No trained checkpoint passes
all12 original short cases; update8 is best at11/12. All17 fixed and17 close
assessments are preserved. At update112 one case exceeds15s near shore,but full
close qualification still fails. Higher union coverage on unsafe runs is not
successful individual exploration. Zero new30-minute or Godot parity runs occur.

Main GPU physics/task collection169.570s,observation/inference0.874s,PPO1.499s,
evidence writing3.798s,native wall178.147s/external178.473s. Throughput5886 macro
transitions/s. Transfer/wait170.430s overlaps GPU work and is not additive. CPU
evaluation,compilation and the4.657s signal audit are separate. The resume
diagnostic takes5.252s externally and executes24,576 extra transitions.

## Preservation and stopping point

Godot still loads appended16/motor88;all11 package hashes pass. Motor88 hash
remains`01B74A18E83933E41734A29E9B6FB9ABDB9951DC2AB306D4F0CF4505015C561E`.
No deployed DLL,model,scene or habitat changes and no mouse/keyboard takeover.
No fresh standalone rest physical suite ran; unchanged motor/checkpoint evidence
is preserved without calling it a new rest measurement.

Final ledger counts **727559703 attempted /727558240 accepted substeps**,including
both signal diagnostics,resume,main and all fixed/close assessments. No native
numerical fault or silent physical rerun occurred. Compiler/prelaunch failures
were corrected before their corresponding native work and their logs remain.
The supplemental sampling diagnostic was declared before execution;it does not
alter the closed training budget or any adoption gate.

All129 portable actors,selected full0/8/112/128 checkpoints,exact sources and
bindings,raw assessments and diagnostics are archived under`data`;all129 full
checkpoints remain locally under`build/flexible-tail/episodic-bank-v3/training`.
`data/manifest.json` binds committed evidence. Original and supplemental scanner
sources are separated;the old ledger is retained before the supplement.

**No adoption.** Repeated encounters are implemented,but another actor run is
not justified without addressing the critic's measured near-bank value estimates
and weak early credit,while protecting navigation retention. Frozen-actor critic
calibration using native on-policy returns is the next focused proposal. It has
not run. No automatic extension or further optimizer work follows this result.
