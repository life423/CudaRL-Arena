# Current swimming v1 — rejected before PPO

This is the immutable compact archive of the November 22 current-physics gate
that stopped after 2,634 controls / 26,340 physical substeps. It contains no
trained current candidate: zero PPO updates and zero Adam steps occurred.

Read [REPORT.md](REPORT.md) for the exact stop, scope, spend and preservation.
The active calm-water policy remains separate and unchanged.

The archive includes all 2,682 formal gate CSV/binary state records (including
resets), the first-failure snapshot, final zero-assembly probes, raw native and
aggregate terminal results, independent failed-arm verification, parent/full
initial policy states, executed source, launch/freeze helpers and prior failed
zero-preparation diagnostics. `manifest.json` hashes every other file exactly.

`independent-verification.json` having `passed=true` certifies that the failed
result is honestly recorded; it does not change `qualified=false`. Behavioral
current suites never ran, so their pass counts are null rather than 0/16.

Full ignored source/executable/library and preparation history remain in the
recorded source locations. Earlier budgets stay closed. This archive is not
authorization for another physical gate, training, a retry or production adoption.
