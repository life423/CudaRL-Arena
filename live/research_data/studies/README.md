# Portable recorded study results

These small files travel with the source checkout and Godot project. They record
native C++/CUDA study results; opening the app does not run training or an
evaluation. `candidate-study.json` selects the latest recorded result and binds
the exact bytes of its sources. Every app dependency uses a `res://` path.

Each study folder contains exact native training/assessment JSON, the small case
and long-window tables, a derived movement summary and a checkpoint digest. The
digest identifies the evaluated checkpoint and records its counters; it is not a
replacement checkpoint and cannot drive the creature. Large raw control traces,
rollouts and intermediate checkpoints remain preserved research evidence outside
the app. A terminal checkpoint may separately be published under
`assets/research-evidence/`; the digest records that repository path when supplied.

Use `tools/publish-study-status.ps1` to export a terminal result. It reads existing
files and writes compact receipts; it never simulates, trains or adopts weights.
The native assessment is responsible for full checkpoint/behavior validation.
The exporter checks its checkpoint identity and records exact SHA-256 digests.

Recorded CPU/CUDA results remain visible even when a local rebuild has a different
DLL fingerprint. That rebuilt binary is not silently declared physically
qualified: the app distinguishes recorded checks from its installed-build binding.
Original motor336 remains the live actor unless separately qualified adoption is
explicitly authorized. Missing or changed receipts produce an unavailable status.

Verify source portability with `tests/godot/study_receipt_portability.gd` in a
copy of the prospective public `live` files without a native DLL or private build
folder. It reads the records without starting a world or an actor. Evidence files
use `-text` attributes so Git preserves the bytes addressed by their hashes.
The optional study label in the latest receipt identifies which recorded study
Details shows. It is covered by the same receipt hash and changes no selected
actor. New terminal records preserve earlier study folders instead of replacing
their evidence.
