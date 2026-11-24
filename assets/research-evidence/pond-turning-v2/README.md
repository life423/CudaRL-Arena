# Food-free turning, exercise 2

[Measured outcomes and timing](../../../docs/POND_TURNING_PPO_V2_RESULTS_2026-11-23.md)

The `update-*` directories hold the five frozen skill assessments.
The `updates-*-*` directories contain the four native trainers' chunk results
and per-update timing rows from this single continued run. The final saved
research motor and its optimizer state are in [candidate-32.bin](candidate-32.bin).
The original viewer motor is a separate file.

`analysis.json` binds the derived course and timing tables to the original
CSV traces. The paths in the native study summary refer to the complete local
study directory. Large raw rollouts are preserved locally. The checkpoint and
the compact results in this directory are published with the source code.
