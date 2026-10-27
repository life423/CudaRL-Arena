# Three-food foraging — October 26, 2026

The user approved three simultaneous food pieces and explicitly made increasing food collected per minute the goal. The live 3D pond starts a fresh policy and trains continuously with 1,024 background CUDA swimmers. It does not load the benchmark weights. CartPole, Racing and the archived one-food swimmer remain available in their existing scenes/tools.

## Learned control and shared dynamics

The new task uses a 13→32→32→2 tanh policy: six body-relative food coordinates, two joint angles, two joint rates and three body velocities. All three food pairs enter the network in stable slots. No distance sorting, target-following controller, food index selector or scripted swimming gait chooses a route. The policy's two outputs command the existing bounded internal tail torques. This is learned control conditioned on three goals, not a separate explicit path planner or a proof of optimal routes. Food coordinates are provided directly; this is not learned visual perception.

`foraging.h` supplies the same observation, placement, collection and reward rules to the CUDA environments and C++ visible pond. The articulated dynamics call the existing `swimmer::advance_physics`; inertia, resistance, body dimensions, timestep and actuation limits are unchanged. Godot displays the simulated pose and joints. Cosmetic eyes look forward rather than falsely indicating a target selected by game code.

Each world has three food pieces. A collection awards 3 reward and immediately replaces only that piece. Other positions, the body and momentum remain unchanged. Following the user's feedback about central clumps and farming one edge, the food ellipse now has semiaxes **3.9 × 2.8 model units**, inset from the rendered water (4.6 × 3.5). Food remains at least **2.2 units apart**. Each replacement must move at least **2.6 units from its previous location** and leave its previous quarter of the pond. Sampling prefers a quarter without other food; a constrained layout may use another occupied quarter while retaining the hard relocation, separation and body-clearance rules. Candidates are sampled across the whole area, without a preference for proximity to the animal. The maximum-spacing rule was removed because it could keep choosing almost the same vacant edge. No other food is moved and no target is selected for the policy.

Training episodes last 1,200 control steps (60 simulated seconds), with proper timeout bootstrapping; collecting food does not terminate the episode. The visible pond continues beyond that horizon. The shared PPO/GAE/Adam path is templated by task, preserving the old nine-input trainer and saved one-food policy fixture.

## Objective and reward

The principal reward is collection count. A small progress signal helps an untrained controller discover useful motion:

`Phi = log(sum(exp(-4 * distance_to_food))) / 4`

The implementation uses a numerically stable log-sum-exp over all three distances. Progress reward is `3 * (Phi_after_motion - Phi_before_motion)`, computed **before** respawning food. A time cost of 0.005 per step and small actuation/smoothness costs discourage idling and erratic commands. The minimum distance used for numerical stability never becomes a selected policy target. Reward shaping provides learning signal; it applies no physical force and cannot move the animal.

The user subsequently reported that the animal stopped moving while optimizer updates continued. A reproduced old policy held both joints beyond their nominal ±1.2 limits while issuing positive commands into the stops. This was a learned controller failure, not a stopped rendering/training loop. The recovery pass now samples resting joint angles throughout the valid ±1.2 range during initialization, adds a reward penalty of `0.05 * proximity_to_stop * outward_command` for pushing into a stop (proximity ramps from 80% to 100% of the joint limit), and uses a 0.005 entropy coefficient for the three-food task. The nine-input legacy task keeps its previous entropy setting. These changes train release/reversal rather than adding an escape animation, rescue teleport, automatic food move, minimum swim speed or periodic optimizer reset. Physics and mean-action display remain unchanged.

PPO optimizes discounted return; the external comparison measures actual sustained collection rate. Increasing scores and shorter trips are desired, but monotonic improvement and globally optimal food routes are not guaranteed.

## Honest progress display

The graph now reports **mean food per minute**, with an adaptive numeric axis rather than a percentage. At update zero and every 25 updates, the frozen current policy runs for a full simulated minute in 64 separate ponds, always using the same comparison seed `0x51A7E123` and placement rules. Collection continues throughout each trial. The score counts all food eaten and divides by 64; it does not report only successful trips. Evaluation uses deterministic mean actions, sends no experience to PPO and does not move the visible animal. The baseline is delivered with the first optimizer update. The graph connects actual measurements without smoothing.

The left sidebar distinguishes that population evaluation from the visible animal. Its Food / min is a rolling count over the previous 60 simulated seconds, with a dash until the first full minute. Pause learning preserves the optimizer while the animal keeps swimming. Speed controls retain the learner, and only Start over clears learning and the graph.

## Validation and limitations

### Current regional-spawn and recovery pass

The updated seed-1 run, using 1,024 swimmers for 1,000 updates, improved from zero to **11.84375 food/min** on the fixed 64-pond comparison and **11.71875** on separate layouts. These are development comparisons for the expanded, regional task; they are not directly comparable to the easier compact-pond scores below. Raw measurements are [three-food-regional-recovery.csv](experiments/swimmer/three-food-regional-recovery.csv).

The updated frozen policy released a folded tail and traveled at least 0.5 model units within 12 simulated seconds in **64/64** deliberately folded starts, covering all four sign combinations of ±1.25 rad joints. Its [snapshot](../tests/fixtures/foraging-recovery-policy.json) is a native regression fixture only; the live viewer starts random weights. The old snapshot passed 58/64 of those starts and subsequently stalled on long feeding runs. On the same current placement rules, the updated snapshot collected **11.6–13.2 food/min** across ten five-minute CPU ponds, with a longest food wait of **13.2 s**, versus waits up to **289.45 s** for the old snapshot. See the [updated probe](experiments/swimmer/three-food-recovery-probe.txt) and [old-policy comparison](experiments/swimmer/three-food-old-policy-probe.txt). These are bounded recovery checks, not proof that every future policy/state will recover or that routes are optimal. Food-slot preference remains visible in the probe and is a separate robustness target.

The sampler passed **24,000** replacement trials without placement failure: every replacement changed region, moved at least 2.6 units and left the other two pieces exactly fixed. A separate 6,000-position initial/replacement coverage check reached x≈±3.9 and y≈±2.8, with 73.7% outside the old central food ellipse. CPU/CUDA trajectory, collection and timeout parity tests pass.

### Earlier compact-pond development evidence

Three independent training seeds used the same 1,024 environments and 400-update budget. Each untrained baseline collected zero food. Final fixed-comparison and separately seeded evaluation means were:

| Training seed | Fixed comparison: food/min | Separate layouts: food/min |
| --- | ---: | ---: |
| 1 | 19.875 | 19.8125 |
| 2 | 25.09375 | 25.578125 |
| 3 | 23.859375 | 24.265625 |

Separate layouts used seed `0xB738019D`, inspected after training; these remain development evidence, not an untouched final benchmark. Five additional one-minute CPU ponds per training seed collected 10–27 food. These measurements verify improvement on this task, not biological realism or universal reliability. Raw curves are [three-food-study.csv](experiments/swimmer/three-food-study.csv).

A broader initial progress potential (sharpness 1) produced only 3.7–11.6 food/min at update 400; its [raw results](experiments/swimmer/three-food-broad-potential.csv) are retained. Sharpening the reward potential improved all three measured seeds without changing propulsion or food collection rules.

A longer five-minute CPU probe of the seed-1, update-400 snapshot found 9.8–23.6 food/min across ten ponds. It also exposed food-slot preference and rare long stalls, including a 177.75-second wait. Actual collection counts show the actor can eat all three slots, but seed 1 often prefers slots 1 and 2. The [probe output](experiments/swimmer/three-food-continuous-probe.txt) is retained. Robust food-order handling and long-run recovery are specific next evaluation targets; neither is claimed solved. The live learner continues beyond update 400.

Current checks completed:

- Nine native foraging tests cover food observation, broad coverage, distant region-changing replacements, stop-release incentives, recovery using a frozen regression policy, consumed-only replacement, unchanged physical trajectories, CPU/CUDA parity and evaluation isolation. A measured trainer's weights after two updates exactly match a trainer with evaluation disabled.
- Thirteen existing native swimmer tests and five PPO demo tests pass.
- The live Godot smoke test passes 138 checks, including 1,024 swimmers, 13-input metadata, progress past update 350, food/min arithmetic, baseline and evaluation history, stable uneaten pieces, actual distant region-changing respawns, rolling-minute UI count, keyboard pause/resume, speed controls, fresh restart and shutdown.
- Actual 1280×720, 768×1024, 390×844 and 844×390 renders were inspected; all three foods fit inside the pond view, controls meet the tested 44 px height and the graph does not cover the pond. The archived one-food scene passes its 100 checks.
- The menu passes 101 checks, including existing CartPole/Racing playback and learning, Tadpole Pond automatic training, keyboard navigation, Back to demos/Escape cleanup and reachable controls at the same four sizes. All three rows fit at 1280×720. Narrow screens use the existing scroll layout. Tadpole Pond offers **Watch it learn**, not saved-policy playback. Its thumbnail is captured by `tests/godot/pond_thumbnail.gd` from the actual scene.

Internal design review: usability 8 (short landscape still needs sidebar scrolling), hierarchy 9, accessibility 7 (keyboard and target sizes tested; native assistive technology and physical touch remain untested), consistency 9, responsiveness 8 (four measured sizes), feedback 8 (initial rolling-minute dash needs its tooltip), performance 8 (CUDA background evaluation; frame-time contention remains unprofiled). Priorities are assistive/device tests, measuring training/render contention and checking smaller windows. These are follow-ups, not authorization to add limbs, change physics or build the lifecycle game.

```powershell
cmake --build --preset release --target cudarl cudarl_tests foraging_experiment pond_probe
build/Release/cudarl_tests.exe foraging
build/Release/foraging_experiment.exe build/foraging-reproduce 1000 1
build/Release/pond_probe.exe build/foraging-reproduce/policy-seed-1.json
C:/Users/aiand/Godot/Godot.exe --path godot-project res://pond/pond.tscn
```

Training, inference, physics and evaluation remain native C++/CUDA. No Python, PyTorch or external RL framework is used.
