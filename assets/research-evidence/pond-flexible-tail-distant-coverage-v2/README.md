# Distant visitation feedback — closed, not adopted

The new chooser receives three whole-map directional novelty means instead of
its three far point probes. Inputs21–23 summarize reachable unvisited water
farther than24cm in three course-relative120° sectors. The other29 inputs,
32→32→15 network, actions, reward, PPO settings and frozen motor are unchanged.
The geometric cache has338 floor-aligned sensing cells at80mm depth; evaluation
still uses the unchanged345-cell centered estimate. Neither map prescribes an
action, destination, route or turn. Other depths remain unqualified.

Model version4/schema3 binds the generated map checksum. CPU, CUDA and the new
native Godot code use the same observer and explicit model-mode decoding.
Version3 models retain all original inputs. Unsupported consumers reject the
new model instead of interpreting its inputs as the old format.

The first observation gate incorrectly demanded1e−6 agreement across unchanged
bank inputs as well as the new inputs. It failed before any physics or PPO.
Two read-only inspections isolated input9/lane976: CPU−0.366296798 versus
CUDA−0.3662990332. The inherited normalized difference2.235174179e−6 is exactly
the same under the legacy observer (about0.112µm clearance). New inputs agree
EXACTLY; all other GPU inputs are bit-identical to legacy, as are CPU inputs.
The corrected gate scopes1e−6 to the new inputs and preserves bit identity of
all other inputs per backend; it records the full inherited discrepancy too.
No physical equation, arithmetic setting or physical tolerance was changed.
Original failed receipts, source snapshot and executable hashes are preserved.

The corrected session imports coverage-v1 selected80 full state exactly,
including actor, critic, Adam, worlds, histories and visitation. Three actual
PPO updates establish exact full-state resume. Main training completes128 new
updates across1,024 CUDA worlds,1,048,576 macro transitions and4,096 new Adam
calls per optimizer. Native training takes157.555s (157.805s externally):
GPU physics149.423s, observation/inference0.676s, PPO1.518s, evidence3.633s.
Transfer/wait overlaps physical execution and must not be added to it.

| Checkpoint | Two-minute pond cases | Five-minute endurance | Guard stops |
| --- | ---: | ---: | ---: |
| New observer, before training | 12/12 | 4/4 | 0 |
| Update8 | 11/12 | Not run | 1 |
| Update24 | 11/12 | Not run | 1 |
| Update32 | 12/12 | 2/4 | 2 |
| Update96 | 3/12 | Not run | 9 |
| Update128 | 3/12 | Not run | 9 |

All17 scheduled assessments are retained. None of the trained candidates
passes the original12+4 gate. Therefore ZERO post-training30-minute coverage
cases ran and no individual-coverage improvement is established. Update96's
72.8% union is unsafe, combines separate starts, and is NOT a success result.
Original motor swimming capability is preserved by exact immutable weights
and requests; failed chooser bank decisions are not motor retraining.

**Godot remains on working coverage80.** No active weight, DLL, habitat, scene,
script or deployment-manifest file was changed by this experiment. Ten bundle
hashes still verify. In an isolated project, the newly built native extension
reproduces3,750 states of the old version3 working policy exactly, and another
3,750 states of the untrained version4 experimental policy exactly. These two
five-minute replays total480,000 substeps and zero optimizer calls. The first
isolated launch lacked Godot's generated extension registration and spent zero
physics steps; importing the isolated project fixed it, with failure retained.
The experimental DLL was not staged into the user's project.

This study changes input meaning, not network capacity or motor architecture.
The measured outcome is lost bank safety during learning. It does not prove
which local feature or reward interaction caused that loss. A focused next
comparison should preserve the complete existing32-input interface and append
the distant field with zero-initialized new input weights, with explicit actor,
critic and Adam migration. Also inspect whether terminating a failed episode
refreshes enough novelty to offset the fixed failure cost. Neither new training
nor a reward change has been started here. This128-update session is closed.

The archive contains executed source bindings, parent/resume proofs, the
geometric cache report, all fixed case summaries, learning curves, initial and
final candidate models/full checkpoints, native compatibility receipts and
actual work accounting. Full checkpoints and raw traces remain in local
`build/flexible-tail/distant-coverage-v2-corrected`. Original preflight evidence
remains in `build/flexible-tail/distant-coverage-v2` (zero physics/PPO).
