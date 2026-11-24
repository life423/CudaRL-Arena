# Obsolete source retirement — November 23, 2026

The user explicitly requested removing the smaller legacy pond limit and other
legacy code. This is the dependency-checked source cleanup accompanying the
new pond runtime; it is not a new training allocation or a claim of navigation.

## Removed executable paths

148 obsolete source files were retired, and the active learned-amplitude math
header was moved unchanged from `tools/` into `src/rl/`. The
[machine-readable manifest](LEGACY_SOURCE_RETIREMENT_2026-11-23.json) binds every
path to its exact source SHA256, original Git blob and base commit
`495caff3c7bc01fe4ab1cf89f27cb8ce50424928`. Where available it also identifies
existing byte-exact frozen snapshot and portable evidence-ZIP members. Git blob
verification allows CRLF/LF checkout normalization only; raw hashes identify the
preserved byte-exact copies. No new source archive was created.

Retired groups are the unused Godot `FrozenSwimmer` class and its 40-input
`FrozenController` runtime, including its near-wall viewing reset; old direct
comparison/braking drivers; historical holding/current/gait/turning study
executables, launchers and authorization/orchestration helpers; unrun obsolete
requested-turn preparation; and historical Godot study parsers and their tests.
These are removed from the active build as coherent dependency groups. Old
current/turning motor-task wrappers and parallel integrators are gone, including
the last unreferenced current-v2 integrator. Its measured results and force-code
sources remain reproducible from existing evidence and Git; there is no active
claim of current capability in the new pond runtime.

No historical result, protocol, original review, checkpoint or ignored experiment
directory was deleted. Saved `.bin` files and their associated metadata remain.
The existing source snapshots inside checkpoint/evidence directories are records
of completed or stopped experiments, not alternate production implementations.

## Preserved current foundation

Native CUDA MLP/GEMM/Adam/loss, Gaussian/PPO kernels and mathematics, checkpoint
code, shared body/water math, physical types and save support remain. There is no
currently authorized trainer executable: a future trainer must target the new
world under a fresh finite protocol. Removing obsolete study drivers does not
replace native PPO with a script or another training/runtime framework.

The qualified saved motor remains package2/method1 with unchanged weights and
two learned persistent-amplitude parameters. `saved_motor_contract` replaces
the old controller class solely as an immutable provenance decoder. Its12
original source hashes were extracted from the original package, checked against
the exact original source revision, and are recorded in
`assets/saved-motor-provenance.json`. They are not regenerated from changed code
and do not certify behavioral equivalence of a changed pond runtime. Current
runtime identity and retention results must be bound separately.

Shared movement/joint/request/observation/scoring and physical-state headers
still contain helpers used by current body math, saved-motor interpretation and
compatibility/regression fixtures. The old pond-ellipse guard and old untimed
benchmark recorder have also been retired after all callers were removed.
CUDA rollout/admission fixtures are also
retained because `test_physics_cuda.cu` depends on them. Those dependency edges
must be split deliberately before further deletions; a version label alone does
not establish obsolescence. The retained current UI/geometry checks are being
adapted to the new pond world.

## Verification performed by the cleanup agent

CPU configuration and `foundation_tests` compilation passed with `/W4 /WX`.
The first compilation reported a constant-expression warning in the new
provenance test; that assertion was corrected to a compile-time `static_assert`,
then the build passed. The focused `saved_motor:` test decoded the actual
original package, matched its12 provenance hashes and2532 finite weights, and
checked the stored amplitude mapping:1/1 passed. No other foundation test ran.
Physical controls/substeps, CUDA kernels and optimizer updates for this cleanup
verification were all zero. A CPU-only build's device message is not a GPU test.

The initial shell deletion batch was rejected before execution by automatic
policy review with the stated reason `blocked by policy`. Explicit literal
file deletions through the purpose-built patch editor succeeded after the
per-file provenance audit. No save/evidence path was targeted.

The cleanup agent did not launch or inspect Godot. Actual new-world physics,
retention, current app startup and rendered UI remain the root's separate checks.
