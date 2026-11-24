# Motor 336 input and actuator audit

This diagnostic performs read-only inference. It ran zero physical controls and zero optimizer calls. The historical checkpoint SHA-256 is `810dbd593ce39475cdd6e95b709bef3fdb34d50c9fcbbb2c345c7a8882a0bef0`. The live package SHA-256 is `4b6400c4db65873bd346c3e0c4fed384e3a77f9e5174d0dcea5d74073b94616d`. The historical 40-input actor's 2,500 weights/biases match the live 41-input actor after inserting 32 zero coefficients for input 40.

Sensitivity uses 3,174 recorded swimming states. These comprise 2,592 pre-action states from 108 straight-swimming worlds at 2.00–3.15 seconds in the first original-policy CUDA collection, plus 582 states sampled every 0.2 seconds from two original 60-second straight swims. All four initial poses are represented in the CUDA sample. Reconstructed long-swim observations reproduce 2,398 following recorded actuator commands with exactly zero absolute error. Range measurements use all 32,768 mixed-task rows of the original-policy collection plus the 582 long-swim observations. The combined audit took 0.813 seconds, excluding compilation. This sample establishes local neural responses and recorded-state asymmetry. It does not establish physical performance on new trajectories.

The numerical output is `u=tanh(actor_mean)`, the normalized joint-angle target before filtering and servo torque. The physical angle limit is 1 radian, so a change of 0.1 in this output corresponds to a change of 0.1 radian in its unfiltered angle target. The four output indices are front yaw, front pitch, rear yaw and rear pitch.

## All 41 inputs

Let `B(v)=body.orientation.inverse(v)`, `C` be whole-body COM in model coordinates, `C0` its original reference and `H` the unit requested heading. Model coordinates convert at 0.04 m/unit. Let `d=(C-C0)*0.04 m` and `e=H*(d·H)-d`. Holding retains this line/depth feedback, with longitudinal anchor error supplied separately by input 40. The heading-only turning adapter supplies only world-vertical error before the same body-frame transformation.

Source O is [pond_motor_observation.h](../src/nn/motor/pond_motor_observation.h), T is [pond_ppo_task.h](../src/rl/pond_ppo_task.h), and P is [dynamics.h](../src/physics/dynamics.h). The source column specifies exact one-based lines. Roles describe the information supplied. The fully connected network couples these inputs to all four outputs, as measured in the accompanying matrix. S means steering, W swimming, B balance/depth and R resting. “Finite” means the observation function enforces finiteness without imposing a fixed normalized magnitude bound.

| Index | Code expression | Physical meaning | Normalization | Expected range | Roles | Source |
|---:|---|---|---|---|---|---|
| 0 | `B(lane_error).x` | Body-forward component of correction to reference line | `B(e).x / 0.002 m` | Finite | S/W/R | O:82–83 |
| 1 | `B(lane_error).y` | Body-up component of correction, including depth | `B(e).y / 0.002 m` | Finite | B/W/R | O:82–83 |
| 2 | `B(lane_error).z` | Body-lateral component of correction | `B(e).z / 0.002 m` | Finite | S/W/R | O:82–83 |
| 3 | `B(request.terminal_velocity).x` | Requested forward velocity | `B(Vreq).x / 0.02 m/s` | ±0.4 at 8 mm/s; 0 at rest | S/W/R | O:84 |
| 4 | `B(request.terminal_velocity).y` | Requested up/down velocity in body axes | `B(Vreq).y / 0.02 m/s` | ±0.4 at 8 mm/s; 0 at rest | B/W/R | O:84 |
| 5 | `B(request.terminal_velocity).z` | Requested lateral velocity | `B(Vreq).z / 0.02 m/s` | ±0.4 at 8 mm/s; 0 at rest | S/W/R | O:84 |
| 6 | `B(request.heading).x` | Requested heading along body-forward axis | Unit-vector component | −1 to 1 | S/W/R | O:85 |
| 7 | `B(request.heading).y` | Requested heading along body-up axis | Unit-vector component | −1 to 1 | S/B/W/R | O:85 |
| 8 | `B(request.heading).z` | Signed lateral heading request | Unit-vector component | −1 to 1 | S/W/R | O:85 |
| 9 | `B({0,1,0}).x` | World up projected onto body-forward axis | Unit-vector component | −1 to 1 | B/S/W/R | O:86 |
| 10 | `B({0,1,0}).y` | Uprightness along body-up axis | Unit-vector component | −1 to 1 | B/W/R | O:86 |
| 11 | `B({0,1,0}).z` | World up projected onto body-lateral axis | Unit-vector component | −1 to 1 | B/S/W/R | O:86 |
| 12 | `B(current.velocity_m_s).x` | Actual forward whole-body COM speed | Divide by 0.02 m/s | Finite | S/W/R | O:18–24,87 |
| 13 | `B(current.velocity_m_s).y` | Actual vertical COM speed in body axes | Divide by 0.02 m/s | Finite | B/W/R | O:18–24,87 |
| 14 | `B(current.velocity_m_s).z` | Actual lateral COM speed | Divide by 0.02 m/s | Finite | S/W/R | O:18–24,87 |
| 15 | `B(body.velocity[3..5]).x` | Root angular velocity about body X, roll | Divide by 4 rad/s | Finite | B/S/W/R | O:88 |
| 16 | `B(body.velocity[3..5]).y` | Root angular velocity about body Y, yaw | Divide by 4 rad/s | Finite | S/B/W/R | O:88 |
| 17 | `B(body.velocity[3..5]).z` | Root angular velocity about body Z, pitch | Divide by 4 rad/s | Finite | B/S/W/R | O:88 |
| 18 | `body.angle[0]` | Front yaw joint angle | Divide by 1 rad | −1 to 1 in accepted state | S/W/B/R | O:90; P:10,79–81 |
| 19 | `body.angle[1]` | Front pitch joint angle | Divide by 1 rad | −1 to 1 in accepted state | B/W/S/R | O:90; P:10,79–81 |
| 20 | `body.angle[2]` | Rear yaw joint angle | Divide by 1 rad | −1 to 1 in accepted state | S/W/B/R | O:90; P:10,79–81 |
| 21 | `body.angle[3]` | Rear pitch joint angle | Divide by 1 rad | −1 to 1 in accepted state | B/W/S/R | O:90; P:10,79–81 |
| 22 | `body.velocity[6]` | Front yaw joint rate | Divide by 12 rad/s | Finite | S/W/B/R | O:91 |
| 23 | `body.velocity[7]` | Front pitch joint rate | Divide by 12 rad/s | Finite | B/W/S/R | O:91 |
| 24 | `body.velocity[8]` | Rear yaw joint rate | Divide by 12 rad/s | Finite | S/W/B/R | O:91 |
| 25 | `body.velocity[9]` | Rear pitch joint rate | Divide by 12 rad/s | Finite | B/W/S/R | O:91 |
| 26 | `last_actual_torques[0]` | Last applied front yaw torque | Already `τ / 10 µN·m` | −1 to 1 | S/W/B/R | O:92 |
| 27 | `last_actual_torques[1]` | Last applied front pitch torque | Already `τ / 10 µN·m` | −1 to 1 | B/W/S/R | O:92 |
| 28 | `last_actual_torques[2]` | Last applied rear yaw torque | Already `τ / 10 µN·m` | −1 to 1 | S/W/B/R | O:92 |
| 29 | `last_actual_torques[3]` | Last applied rear pitch torque | Already `τ / 10 µN·m` | −1 to 1 | B/W/S/R | O:92 |
| 30 | `held_targets[0]` | Filtered front yaw target angle | Divide by 1 rad | −1 to 1 | S/W/B/R | O:93 |
| 31 | `held_targets[1]` | Filtered front pitch target angle | Divide by 1 rad | −1 to 1 | B/W/S/R | O:93 |
| 32 | `held_targets[2]` | Filtered rear yaw target angle | Divide by 1 rad | −1 to 1 | S/W/B/R | O:93 |
| 33 | `held_targets[3]` | Filtered rear pitch target angle | Divide by 1 rad | −1 to 1 | B/W/S/R | O:93 |
| 34 | `state.controls` | Startup fraction over first 40 controls | `clamp(1-controls/40,0,1)` | 0 to 1 | W/B/R/S | O:96 |
| 35 | `joint_target_actuation` | Actuator representation flag | `true ? 1 : 0` | 0 or 1; currently always 1 | All, constant in current mode | O:97 |
| 36 | `reference_rate[0]` | Filtered front yaw target rate | Divide by 12 rad/s | −1 to 1 | S/W/B/R | O:98 |
| 37 | `reference_rate[1]` | Filtered front pitch target rate | Divide by 12 rad/s | −1 to 1 | B/W/S/R | O:98 |
| 38 | `reference_rate[2]` | Filtered rear yaw target rate | Divide by 12 rad/s | −1 to 1 | S/W/B/R | O:98 |
| 39 | `reference_rate[3]` | Filtered rear pitch target rate | Divide by 12 rad/s | −1 to 1 | B/W/S/R | O:98 |
| 40 | `holding / hold_anchor_model` | Signed longitudinal anchor error | Holding `tanh(((anchor-C)*0.04 m)·H / 0.02 m)`, otherwise 0 | −1 to 1; 0 while swimming | R intended; zero actor influence | O:99–102 |

The 12 rad/s denominator for actual joint rates at 22–25 is a scale, not a bound imposed on those measured rates. The separate filter rates at 36–39 are checked against 12 rad/s. World-up inputs 9–11 are not gravity with a downward sign. Heading components 6–8 encode intent, not measured course. Startup input 34 is not a remaining episode timer.

## Numerical sensitivity

Each matrix entry below is the mean signed response, followed by RMS response in parentheses. Define the response as `u(x+δe_i)-u(x-δe_i)` while holding every other input constant at the same recorded state. These are finite command differences, not derivatives. A half-step of 0.5 for 0–2 is ±1 mm. A half-step of 0.05 is ±1 mm/s for velocity channels and ±0.2 rad/s for angular velocity. Individual unit-vector component probes deliberately isolate neural sensitivity and can depart from physically realizable unit vectors.

| Input | Half-step δ | Front yaw | Front pitch | Rear yaw | Rear pitch |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.50 | -0.0422 (0.0532) | -0.3179 (0.3190) | -0.1181 (0.1416) | -0.3009 (0.3424) |
| 1 | 0.50 | +0.0096 (0.0272) | -0.3934 (0.3951) | +0.0070 (0.0219) | -0.1986 (0.2192) |
| 2 | 0.50 | +0.0843 (0.1026) | +0.1996 (0.2052) | +0.0913 (0.1120) | -0.0217 (0.0685) |
| 3 | 0.05 | +0.0158 (0.0202) | +0.0387 (0.0393) | +0.0175 (0.0213) | +0.0298 (0.0335) |
| 4 | 0.05 | +0.0318 (0.0391) | -0.2380 (0.2389) | -0.0308 (0.0357) | -0.1043 (0.1169) |
| 5 | 0.05 | +0.0177 (0.0209) | +0.0491 (0.0513) | +0.0080 (0.0124) | +0.0066 (0.0182) |
| 6 | 0.05 | -0.0044 (0.0073) | -0.0034 (0.0087) | -0.0212 (0.0266) | -0.0471 (0.0549) |
| 7 | 0.05 | +0.0047 (0.0187) | -0.3845 (0.3855) | -0.0613 (0.0769) | -0.1392 (0.1585) |
| 8 | 0.05 | +0.0244 (0.0294) | +0.0799 (0.0860) | -0.0196 (0.0264) | +0.0022 (0.0298) |
| 9 | 0.05 | +0.0048 (0.0079) | +0.0073 (0.0143) | -0.0207 (0.0249) | -0.0639 (0.0728) |
| 10 | 0.05 | -0.0017 (0.0101) | -0.0026 (0.0161) | +0.0065 (0.0101) | -0.0536 (0.0649) |
| 11 | 0.05 | -0.1768 (0.2114) | +0.0167 (0.1768) | -0.2043 (0.2446) | -0.1303 (0.2407) |
| 12 | 0.05 | +0.0229 (0.0336) | -0.1184 (0.1191) | -0.0181 (0.0266) | -0.1014 (0.1155) |
| 13 | 0.05 | +0.0264 (0.0401) | -0.1455 (0.1510) | +0.0054 (0.0144) | +0.0075 (0.0347) |
| 14 | 0.05 | +0.0548 (0.0660) | -0.0664 (0.0816) | +0.0414 (0.0526) | -0.0053 (0.0466) |
| 15 | 0.05 | +0.0358 (0.0437) | +0.1048 (0.1151) | +0.0612 (0.0770) | +0.0766 (0.1122) |
| 16 | 0.05 | -0.0235 (0.0296) | +0.0210 (0.0228) | -0.0049 (0.0114) | +0.0745 (0.0832) |
| 17 | 0.05 | -0.0095 (0.0116) | +0.1245 (0.1252) | +0.0560 (0.0654) | +0.0592 (0.0676) |

The complete 41-by-four matrix, including minima, maxima and RMS, is [input-sensitivity.csv](../assets/research-evidence/motor-336-input-audit-full/input-sensitivity.csv). Observed minimum/maximum values and incoming weight-column norms for all inputs are in [input-ranges.csv](../assets/research-evidence/motor-336-input-audit-full/input-ranges.csv).

## Positive and negative heading requests

Rotate the heading vector ±15° about world up expressed in body coordinates, holding the recorded body, joint, filter, torque, lane and velocity inputs fixed. Positive rotation is pond-left. Entries below are mean changes from each state's unchanged-request output. The coherent request also rotates requested velocity consistently, while actual velocity and all other inputs stay fixed.

| Request change | Degrees | Front yaw | Front pitch | Rear yaw | Rear pitch |
|---|---:|---:|---:|---:|---:|
| heading_only | -15 | +0.0632 | +0.2102 | -0.0409 | +0.0503 |
| heading_only | +15 | -0.0630 | -0.1981 | +0.0608 | +0.0123 |
| velocity_only | -15 | +0.0153 | +0.0450 | +0.0051 | +0.0053 |
| velocity_only | +15 | -0.0204 | -0.0546 | -0.0099 | -0.0103 |
| coherent_request | -15 | +0.0773 | +0.2525 | -0.0359 | +0.0639 |
| coherent_request | +15 | -0.0844 | -0.2451 | +0.0519 | +0.0098 |

The paired positive-minus-negative RMS responses for heading alone are front yaw 0.1490, front pitch 0.4204, rear yaw 0.1337 and rear pitch 0.1721. For the coherent heading-plus-velocity request they are 0.1894, 0.5078, 0.1218 and 0.1889. Front pitch responds more strongly than either yaw output in both probes.

The complete ±5°, ±15°, ±30°, ±60° and ±90° sweeps, including RMS and extrema for each output, are in [heading-sensitivity.csv](../assets/research-evidence/motor-336-input-audit-full/heading-sensitivity.csv) and [heading-paired.csv](../assets/research-evidence/motor-336-input-audit-full/heading-paired.csv).

## Saturation, asymmetry and conflicting cues

Across the sampled states, the first hidden layer has 0% of activations beyond absolute 0.95 or 0.99. The second layer has 7.29% beyond 0.95 and 0.791% beyond 0.99. Second-layer neuron 15 exceeds 0.95 in 43.82% of samples and 0.99 in 20.76%, reaching 0.99624. This is localized saturation.

Final command saturation beyond absolute 0.95 is front yaw 4.789%, front pitch 0%, rear yaw 0.0945%, rear pitch 0%. None exceeds 0.99. Final command saturation reduces physical sensitivity to latent changes. The implemented Gaussian PPO scores latent actions, so its policy gradient does not differentiate through that final tanh. [Per-neuron and per-output measurements](../assets/research-evidence/motor-336-input-audit-full/activations.csv)

The full reflection test reverses polar Z, axial X/Y and yaw-angle/rate/torque signs while preserving pitch signs. The RMS violation of the expected mirrored command is 0.1355 front yaw, 0.7089 front pitch, 0.7597 rear yaw and 0.9986 rear pitch. These input-reflection queries establish that motor 336 is not reflection-equivariant. Earlier matched physical diagnostics found right 10° passes 8/8, right 15° passes 4/8 and zero passes on their left counterparts, with exact mirrored physical response under prescribed mirrored diagnostic commands. [Actor reflection measurements](../assets/research-evidence/motor-336-input-audit-full/mirror-sensitivity.csv) · [Prior physical diagnostic](../assets/research-evidence/mirrored-turn-2026-11-24/result.json)

At a fixed 8 mm/s request, inputs 3–5 equal 0.4 times inputs 6–8. Changing heading alone makes those request cues inconsistent, which is why the isolated and coherent measurements are both shown.

The original lane uses a line through the initial COM. After 16 mm of forward travel, rotating it 15° introduces about 4.14 mm of cross-track error, about 2.07 normalized units before projection into body axes. The original straight reward penalizes lane and vector-velocity error. It contains no direct joint-yaw penalty. Current turning practice already replaces horizontal lane error with depth-only error, so the old lane correction is not active in that path. [Original reward](../src/nn/motor/straight_task.h:369) · [Current turning observation](../src/rl/pond_ppo_task.h:128)

The network has instantaneous requested direction, measured COM motion, yaw rate and joint/filter history. It lacks an explicit signed course-error history, shoreline clearance, current velocity and visitation information. The latter three are needed for pond exploration and current-aware navigation, rather than proving an input deficiency for a basic requested turn. Requested absolute destination and distance-to-destination are also absent from these 41 channels. Input 40 is zero-weighted and supplies no learned anchor correction.

The measurements support a straight-trained, directionally asymmetric motor with coupled pitch/yaw response. They do not establish that saturation or one particular input caused sustained turning failure. This diagnostic performed no training. The user subsequently resumed the separate residual experiments.

## Depth stabilization and directional learning

The successful original depth lesson uses `v_y_ref = clamp(-signed_depth / 2 seconds, -0.001, +0.001) m/s`. Its reward penalizes `((actual_v_y-v_y_ref)/0.008)^2` together with normalized depth error and posture costs. A creature 1 mm too high receives a reference of −0.5 mm/s. That reference changes the training objective. The actor still chooses the four joints.

The later turning lesson retains normalized depth observations but replaces this explicit vertical correction-velocity cost with `exp(-(depth/0.003)^2)`. At 1 mm depth error that factor is approximately 0.895. Actual vertical velocity toward the correct depth has no equivalent direct recovery term in this lesson. Heading quality uses actual horizontal course and a factor `1/(1+(1-cos(error))/(1-cos(15 degrees)))`. It is 1 at zero error and 0.5 at 15°. The extra experimental course input scales signed error by π. A 15° course error is 0.0833, while a 1 mm position error is 0.5 on the original depth scale.

| Setting | Original depth/swim training ending at motor 336 | Full-motor turning exercise v6 | Separate residual experiment |
|---|---:|---:|---:|
| Actor learning rate | 0.00025 | 0.00001 | 0.0003 for residual only |
| Critic learning rate | 0.001 | 0.00001 | 0.001 |
| Discount γ | 0.99 | 0.995 | 0.995 |
| GAE λ | 1.0 | 0.95 | 0.95 |
| Collected controls/world/update | 1,280 | 64 | 64 |
| Simulated collection span/world | 64 s | 3.2 s | 3.2 s |
| Parallel worlds | 32 | 512 | 512 |
| Optimizer passes | 2 | 4 | 4 |
| Minibatch | 256 | 4,096 | 4,096 |
| Latent exploration σ | exp(−1), approximately 0.368 | 0.1 | 0.1 on active residual axes |
| Temporal correlation | ρ=0.8 | Independent samples | Independent samples |

The final 336 continuation inherited its earlier actor and critic. The v6 actor inherited 336, while its critic started fresh. Under the short GAE collection, predictions at the next rollout edge determine most longer-horizon credit. These differences prevent treating the two learning setups as equivalent merely because both use PPO. [Historical experiment settings](../assets/ppo-four-axis-checkpoints/update-336-episode64/experiment.json) · [Depth reward](../src/nn/motor/straight_task.h:365) · [Turning reward](../src/rl/pond_ppo_task.h:143) · [Turning PPO parameters](../src/rl/pond_ppo_contract.h:48)

The transferable principle is signed error plus measured corrective motion, with separate depth, posture and movement requirements. A desired course-rate or lateral correction-velocity term could supply the directional counterpart to vertical recovery. Such a term belongs to the learning objective and should be measured independently. The matched residual architecture comparison preserves its reward across both arms.
