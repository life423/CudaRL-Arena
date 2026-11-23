# Requested-turn capability check: prepared, not executed

The [finite probe](FROZEN_REQUEST_TURN_PROBE_2026-11-23.md) is compiled and
prepared. It has executed zero physical controls/substeps and zero optimizer
steps. No authorization exists. This is the narrow next measurement identified
by the [design audit](CONTINUOUS_POND_DESIGN_AUDIT_2026-11-23.md), not a new PPO
allocation or a claim that pond swimming is working.

The original motor receives one persistent requested-course change after four
seconds of actual swimming. Twenty-two cases cover two original poses and
zero/+/-5/15/30/60/90 degrees. The zero case preserves the original request and
lane exactly. Nonzero cases keep original world depth and all physical/filter
state. There are no action prescriptions, body rotations or learned wrappers.

The proposed one-shot limits are 7,360 total control attempts / 73,600 substeps,
120 seconds and 128 MiB including frozen inputs and evidence. A conservative
pre-step output projection is 76,866,896 bytes; frozen preparation occupies
12,686,122 bytes before the small independent proof. These fit the cap without
changing physics or acceptance to save space. The first 320 controls are a
matched-command CPU/CUDA precheck; the remaining 7,040 are the actual cases.

## Checks completed

- Standalone CUDA build passed `/W4 /WX`.
- Native preparation ran 26 CUDA initial-state/inference fixtures and 22 CPU
  plus 22 CUDA request-latch fixtures, with zero integration. Same-observation
  CPU/GPU latent/command differences were at most 1.78814e-7/1.19209e-7.
- Independent saved-fixture verification reconstructed initial bodies, all 41
  observation channels, the 2,532 saved weights, tanh decoding and both sets of
  request rotations. Zero-offset tasks are byte-identical. It also rehashed
  all 87 frozen source/library/executable/parent inputs and their snapshots.
- A direct execution attempt without approval exited 2 before creating a
  physical directory. Its error was “Fresh approval and the frozen input
  manifest are required.” No authorization was fabricated to test this.
- Independent source reviews checked unchanged physical tolerances, all 40
  substep torques, guards at every accepted 5 ms state, actual attempt counters,
  exact zero-offset behavior, preserved clocks/depth/filter state, and the
  distinction between oscillating head angle and COM travel course.

The native study is only descriptive. Bank/depth/tilt-stopped prefixes are not
heading-tracking passes or proof that the motor cannot turn. Head alignment
based on 20 control samples is labeled as sampled, not continuous alignment.
Undefined course/circular means and unavailable final windows are null. No
policy selection, Godot replay or actor adoption belongs to this probe.

## Evidence identity

Run root: `build/ppo/frozen-request-turn-2026-11-23-v1`.

| Artifact | SHA-256 |
|---|---|
| Frozen input manifest | `75f2bef0bc09144b72abe7c5607b5cb002ddc9561358191eff769f794c178b10` |
| Independent preparation proof | `285e1d5cbcc978c6fd090ea9033a85e07d61b0db1ba71150e48601dd91a251d1` |
| Native executable | `c18802cab8dcff6b3b3185083951e7bc2e67d2cc2ab9496ed5ca0836b7be4d30` |
| Source | `c880e57f92116073fc8bae400e25397e95ee8e417b081f52002dc208815b8f46` |
| Protocol document | `ea4110345009501e73958618956a40dd72076e3dd0bba22bcdec815535973c73` |

An initial offline parser omitted the holding checkpoint's separate protocol
field. That parser was corrected before proof creation; the saved native bytes,
source, thresholds and fixtures were unchanged. No simulation was repeated.

The separate offline result reader is prepared as
`tools/verify_frozen_requested_turn_result.py` (SHA-256
`973b4a8fe73f778b89d4a462cf599cd043a116dbd2e5538d2f40a9c0d7c76cfe`).
Syntax, course signs, zero-state guards, kinetic scaling and all 26 prepared
1,076-byte endpoint record round-trips pass. Nine imported source files are
snapshotted under
`build/frozen-request-turn-preparation-2026-11-23/offline-auditor-ready/`.
The full reader has no actual physical probe records to validate yet; its
synthetic/preparation checks must not be reported as a passed physical probe.

The running Godot app already shows the closed continuous-study result. Its
working actor, camera and zoom stay active. When validated turning behavior is
ready, update and verify the actual Godot app as the user requested; a separate
native tool or documentation alone does not complete the product request.
