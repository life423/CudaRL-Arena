# Requested-speed trial5: frozen registration

One model6 motor,42observations, two outputs: stroke strength/frequency. No
physical parameters, torque limits, optimizer, exploration or oscillator changes.
Task5 removes the superseded arrival-task dependency and supports persistent
speed requests. Reward2 still scores2..4s tracking and all4s effort. Request
changes preserve physical/filter/phase/time/energy state; segment metrics and
their work baseline restart explicitly, with the original heading preserved.

Initialize the actor from the selected reward2/update25 parameters, checksum
1699f42d0090c042. Fresh critic and Adam; seed0x53574D32.24updates,16worlds,
80controls/episode, lambda1/gamma.99, unchanged Gaussian sigma exp(-2), two epochs.
Train4/6/8mm/s,128episodes each. Five-speed frozen evaluation4/5/6/7/8mm/s at
two poses (old cases0/3), plus one continuous6->4->8mm/s sequence, each segment4s.
Baseline and final candidate receive identical conditions.5/7 are held out of PPO.

Ceiling:32readiness+30,720training+2,080paired evaluation=32,832controls and
328,320native substeps, including failed/CPU/CUDA work. Native wall limit20s;
wrapper stops only its own process after25s. No retry, sweep or extension.
Before PPO, the refactored baseline8mm/s trajectory must match160saved rows
(19fields, tolerance1e-5+1e-5*abs(reference)). Readiness compares actual CPU/CUDA.

Every request: mean within10%; velocity RMSE<=31.25% of requested speed (same
2.5mm/s gate at8, proportionately stricter below); lateral<=2mm, depth<=1mm,
path angle<=5deg, tilt<=10deg, positive work/metre<=1mJ/m. Native finite/torque,
energy and10mm clearance guards apply throughout, including switch transients.
Head yaw is reported separately. This is not stopping, arrival or current holding.

Promote only10/10constant cases AND3/3switch segments, improved normalized speed
RMSE overall, and no worse8mm/s RMSE or work/metre against the selected baseline.
Otherwise preserve the verified8mm/s actor; no silent multi-speed UI adoption.
Source/executable/actor/reference/numeric hashes freeze before the sole run in
build/speed-motor/2026-10-30-variable-v1/. Publish result immediately, then docs.
