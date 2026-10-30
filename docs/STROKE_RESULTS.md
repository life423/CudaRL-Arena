# Fixed stroke: measurable model propulsion, diagnostic only

The single approved `STROKE_PROTOCOL.md` check is CLOSED.640controls/6400native
substeps executed,160ticks per condition/backend; no failures, retries or PPO.
Native time0.333s; preparation114.173s including the small diagnostic mode/build.
All87 frozen sources and917 previous evidence files remained hash-identical
during execution; original checkout stayed clean. Artifacts/source/executable:
`build/stroke/2026-10-30-v1/`. Runtime/contact/saves unchanged; target OFF.

CPU net whole-COM forward travel relative to rest: **1.08746055mm**;
CUDA1.08746137mm. Resting COM displacement/work exactly zero on both backends.
Stroke lateralZ displacement-.046708mm; verticalY displacement0. CPU net
forward advances in the two interior cycles:+.170932mm and+.391518mm. Total
exceeds frozen.1mm threshold and5times backend disagreement(.000000815mm).
Defined-field maximum normalized parity score.010828 (limit1). Minimum sampled
floor/other gap73.6mm, above10mm; no contact or torque/finite/domain failure.

Peak requested/applied torque1.020725microNm, below unchanged10microNm limit.
Positive internal actuator work3.006031microJ. A sinusoidal fit over2..6s gives
actual yaw amplitudes.323289/.162265rad for.5rad commanded amplitudes, with
49.354/70.459degree lag relative to the corresponding held target. Control-end
target-error RMS.222232/.275478rad. Thus servos move the joints but attenuate
and delay the coordinated shape. These are measured tracking limits for THIS
waveform, not proof of a missing torque limit or a globally bad actuator.

This demonstrates that the existing body/actuation/fluid equations can produce
forward propulsion from a coordinated, prescribed stroke. It is not learned
movement,20mm-in4s acceptance, validated real-world hydrodynamics or proof the
body is adequate at desired speed. Mean8s speed is only.136mm/s. The model uses
distributed drag/buoyancy rather than resolved wake hydrodynamics. External
forward force/impulse was not instrumented; torque/work must not be mislabeled
as that measurement or propulsion efficiency.

There is no present evidence requiring a body redesign merely because the PPO
policies failed. The concrete next learning issue is coordinated cyclic motion;
servo tracking is a separate measurable limitation with substantial unused torque
headroom. Further actuator/body/interface changes remain options requiring their
own bounded comparison. No further run, training, retuning, commit or adoption
follows automatically from this result.
