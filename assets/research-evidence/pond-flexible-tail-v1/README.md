# Flexible-tail physical feasibility

The separate native test uses the original head dimensions and a 52 mm tapered
tail. One joint supplies root yaw and pitch. Two passive bend coordinates
distribute curvature down the tail. The six tail sections integrate mass,
inertia, elastic response and water resistance. They have no independent
actuators. Godot sweeps a mesh along the measured centerline.

The bench supplies diagnostic root targets and measures whole-body center of
mass. No learner or live navigation code uses those inputs. The native servo
limit each root-axis torque to 10 micronewton metres. The bench uses the flat
central portion of the current pond and checks a conservative whole-body
envelope against the flat bed and water surface at every substep. It does not
model slope contact or surface crossing. A case leaving that fixture stops and
fails the check.

## Results

| Trial | Physics step | Moving turns | Process time |
| --- | ---: | ---: | ---: |
| [Fastest stroke](prototype-v1.csv) | 5 ms | 0/4 | 0.138 s |
| [Stroke near 8 mm/s](pond-speed-v2.csv) | 5 ms | 0/4 | 0.158 s |
| [Higher feedback gain, matched comparison](pond-speed-refined-v3.csv) | 1.25 ms | 2/4, lower gain 0/4 | 0.402 s |
| [12-second steering window](pond-speed-extended-v4.csv) | 1.25 ms | 4/4 | 0.270 s |

Process times include startup and evidence writing. In the initial fastest
fixture, the 34.6 mm/s swimmer left the flat test area before finishing each
turn. Its partial rows use the original reporting code and are preserved.
The later fixtures enter the turn at the measured swimming course, correct
the sign of the diagnostic root feedback and use the 1 Hz, 0.55 rad stroke
closest to the previous 8 mm/s forward target. The third fixture compares
both feedback gains with the same timestep and initial state. The fourth
runs the stronger gain for four more simulated seconds from the same state.

In the fourth fixture, straight swimming averaged 8.76 mm/s along the
requested course during the final two seconds. The 0.625 ms replay endpoint
was 0.154 mm away from the 1.25 ms result after six simulated seconds. The
unpowered centered rest check recorded zero COM travel in two seconds. The
passive bend check reproduced near-zero linear and angular momentum in zero
external force and dissipated its stored elastic energy.

| Requested | Achieved COM turn | Final projected speed |
| ---: | ---: | ---: |
| -30° | -26.03° | 10.37 mm/s |
| -15° | -13.76° | 10.30 mm/s |
| +15° | +13.73° | 10.31 mm/s |
| +30° | +25.88° | 10.38 mm/s |

The checks require final-two-second course error of at most 5°, forward
projection of at least 6 mm/s, and maximum 1 mm depth error and 10° tilt
throughout each case. The four passing turns were upright, at the initial
depth to recorded precision, and at least 68 mm from the bed and surface by
the conservative bound. Each began from the same continuous post-swim state.
The neutral root command still left 4.30 mm/s horizontal speed six seconds
later, missing the 0.5 mm/s stop criterion.

This establishes steering authority in calm center water using diagnostic
inputs. PPO learning, disturbance recovery, braking, 60–90° turns, repeated
alternating turns and long pond navigation remain open. The default Godot
scene, production body and saved motors use the existing runtime.

The optional CPU/CUDA spot check never reached execution. The CUDA assembler
used about 14.8 GB during optimization, so the scratch compile was stopped.
It produced zero physics substeps and no parity result. The separate Godot
playback used the RTX 5070 to render the CPU-recorded trajectory.

## Reproduce the final fixture

Run from the minimal-foundation worktree. Use a new output folder within
the existing build directory. The recorded measurements used a 4-second
external process cap and excluded compilation.

```powershell
cmake -S tools/pond-flexible-tail-check -B build/flexible-tail/native
cmake --build build/flexible-tail/native --config Release
& build/flexible-tail/native/Release/pond_flexible_tail_check.exe build/flexible-tail/another-fixture --extended
```

[Recorded +30° case](turn-plus-30.json) is bundled in the separate flexible
tail preview. The playback uses native positions and bend, with whole-body
COM shown by the path. To open the evidence file directly from the worktree,
use the following command.

```powershell
$trace = (Resolve-Path assets/research-evidence/pond-flexible-tail-v1/turn-plus-30.json).Path
& 'C:/Users/aiand/Godot/Godot_console.exe' --path flexible-tail-preview -- "--flex-tail-trace=$trace"
```

Final fixture bindings, SHA-256:

- flexible_tail.h: `463F33FB3E17FDCBAB9A6E9FB57514D6A38CE1A2A97091A6E1374EB406C92653`
- check.cpp: `57FB62534D1886DB378E3189649FB4F8D94A22BA75C602D18359C06F6965C649`
- Release executable: `638596B28873AFD8081DE8215EC79C59C0DBBBBB20F8C08FF6806C2C52EDD82C`
- Final case CSV: `1743D44A10AAF6B631D2010B67C69A2C8C94C0BED98DE6A0FD3E01AA98F07578`
- Recorded +30° case: `0494B4644B78AA967D6D5D8EA1C8AD5B784081BF309BC98485A526B882340478`
