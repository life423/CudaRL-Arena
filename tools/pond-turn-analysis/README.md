# Recorded turning direction and completion analysis

Build this standalone native C++ utility, then run:

```
pond_turn_analysis.exe --self-test
pond_turn_analysis.exe build/turn32 assets/research-evidence/pond-turning-v2
pond_turn_analysis.exe build/turn34 assets/research-evidence/pond-turning-v4-repeated
pond_turn_analysis.exe --training build/turn33 assets/research-evidence/pond-turning-v4-repeated 32
pond_turn_analysis.exe --training build/turn34 assets/research-evidence/pond-turning-v4-repeated 32
```

It reads only completed existing v2, v3-course or v4-repeated assessments at updates0/8/16/24/32. It does not
include a simulator, neural network, CUDA runtime or optimizer. Raw CSV/JSON
SHA256s and derived-file SHA256s are recorded in `analysis.json`. Running again
after a new checkpoint completes refreshes the reproducible derived tables.

The mass-weighted COM velocities come from the stored10substeps per50ms control.
Direction is atan2(-vz,vx), positive/left and negative/right. Signed residual is
the shortest actual-minus-requested course angle. Underturn/overshoot/wrong-way
labels describe final angles, not unwrapped full revolutions or head alignment.

A completion is the earliest end of a complete rolling4s window wholly after
the request change at2s, with mean requested COM projection>=6mm/s and absolute
course error<=5degrees, while all preceding depth<=1mm and tilt<=10degrees.
Resolution is50ms; the earliest possible delay is4s. An early qualifying window
is labeled transient if the authoritative final case fails. No window means no
reported completion time. The original final20s qualification is unchanged and
is reconstructed against `cases.csv` for complete cases.

The last available4s diagnostic is explicitly separate from the registered
16–20s final window on prematurely stopped cases. Actual course change uses the
COM course during the final second before the request as its origin; achieved
turn also reports the angle from the original requested heading. Both are
separate from the requested angle and signed residual.

Retention margins summarize existing whole-body COM lane/depth/tilt/rest speed
and post-startup long-swim windows. Turn lane distances remain descriptive.
No pass requirement, learner, checkpoint or physical trajectory is changed.

Training mode reads the saved native batch files, verifies each checkpoint
wrapper checksum and dimensions, and binds raw-file SHA256s. It writes
`v3-training-requests.csv`/`v4-training-requests.csv` and their summary JSONs.
The final argument is the completed-update prefix to inspect (1–32).

The shared training completion definition requires at least100 post-onset ticks,
then20 consecutive qualifying instantaneous COM samples: course error≤5°,
requested projection≥6mm/s, depth≤1mm and tilt≤10°. The first possible completion
is tick120 (6s); comparable completion must occur by tick400 (20s). This differs
from the deterministic held-out four-second mean measurement above. Predicate
failure counts overlap; longest favorable streaks expose which requirement is
interrupting success. Native v4 events and cumulative per-world counters are
cross-checked at every recorded tick, including across rollout/checkpoint edges.

V3 exact target headings come from recorded post-onset task snapshots; its
nominal angle draws reproduce the frozen counter RNG. V3 did not replace failed
requests at400 ticks, so comparable misses are not actual timeouts. Episode-cap,
other task-terminal interruptions and final pending requests remain separate.
V4 additionally records actual repeats and assigned long holds. Training-minute
rates use native training child-process wall time; full elapsed rates include
evaluations, runner bookkeeping and any parser-repair gap, excluding compilation.
