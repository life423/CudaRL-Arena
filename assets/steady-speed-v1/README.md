# Constant-speed lesson1

The frozen NN passed10/10four-second constant requests from rest, in reset cases
0/3. Trained4/6/8mm/s;5/7withheld. Measured4.350/5.318/6.259/7.115/7.843mm/s.
Inference used the NN alone, with the supplied stroke generator and native model6.
This does not establish persistent speed changes, braking, turning or holding.
The previous8->4teacher switch gate remains FAILED. Live Godot still uses the
separate passing8mm/s actor in live/assets/; this directory is not auto-loaded.

checkpoint.bin is the exact30,644-byte selected actor+Adam native research snapshot,
SHA2560A19AF86FBBDCD189A330547A281666B4D8C781922A7743C870E7A58D9FC6411.
It carries canonical protocol and checksummed FP32 parameters, not machine pointers
or a physical-world resume. Format/shape/scope are in checkpoint-provenance.json.
Only the tested Windows x64/little-endian implementation is verified. Runtime
integration/export requires preserving the constant-from-rest restriction; the
existing unrestricted mid-run speed setter is not validated for this controller.

accepted-examples.bin contains960observations/actions from the six accepted
constant teacher profiles. teacher-gates.json intentionally also records the failed
switch; the source whole trial is never relabelled accepted. Original dataset and
row lineage hashes are in example-provenance.json. The accepted slice and gates
were verified against original evidence before curation. No bulk failed-run data
or generated executable is included. baseline-reference.csv retains the160saved
rows needed by the runner's case0/3baseline regression. evaluation.json preserves
the measured frozen results and scope. Ignored full source snapshots/traces remain
in build/speed-motor; this curated fixture lets the active runner work after checkout
without requiring those ignored folders. Launching the runner creates a NEW physical/
training trial and still requires an explicitly agreed budget.
