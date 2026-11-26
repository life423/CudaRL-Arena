# Moving approaches with frozen motor 88

This study tests whether a chooser can learn to turn earlier while retaining the three-depth parent's exploration coverage. Use the larger physical pond with the established body, fitted stroke, actuator constraints and food-free reward. Protect motor 88, the installed appended-16 chooser, the north-bank +112 full session, the three-depth +128 parent and the previous open-water +104 candidate.

## Frozen recovery comparison

Rebuild each of the twelve moving cases that failed for +104 from its archived start. Reproduce the baseline outcome and its first real chooser decision. Keep the whole native world, water, body momentum, servo state, stroke, chooser timing and visit history identical across branches. Compare autonomous choosers and bounded sequences of existing speed and turn requests followed by autonomous control. Count an exact-start recovery only when the branch passes the 180-second physical, movement and stall checks. Report any shorter safe exit separately. An unsuccessful finite action search leaves recoverability open. Moving a start farther inland measures the value of earlier warning and does not qualify the original start. Keep all twelve starts and their counterfactual actions out of PPO and checkpoint selection.

## One GPU continuation

Complete the frozen comparison first. Use the measured runway and signs to interpret the new training mix. Fork the full north-bank +112 chooser (it retained 11 of 12 ordinary navigation cases and three of four long swims). Keep its actor, critic, optimizer state, sensing, actions and reward. Collect 128 updates on 1,024 CUDA worlds at eight chooser decisions per world per update, up to 1,048,576 decisions. Use the existing four optimization passes and a 1,200-second native training deadline. Save each checkpoint. Report collected decisions by depth and time spent in simulation and learning.

Allocate half the worlds to unforced moving approaches of both orientations with short, intermediate and long runway. Include approaches with enough distance to change course before reaching the bank. Allocate one quarter to near-parallel bank swimming and one quarter to continuous interior exploration. Include 72, 75 and 80 mm throughout. The swimmer chooses every action during collection. No recovery sequences, actor labels or exact diagnostic states enter training.

## Selection and deployment

Use the independent selection fixtures, scores and tie-breaking rule in selection-protocol.md. Freeze one actor by content hash before final tests. For an update to Godot the same actor must finish all twelve original moving approaches, all 24 existing shoreline approaches, and the original twelve navigation and four five-minute follow-ups. Require sustained near-bank swimming at each depth by the existing gate (at least four of eight at 15 seconds, mean at least 10.96 seconds) and the 11.81-degree uninterrupted reference shoreline arc at 80 mm.

All four independent 30-minute swims must finish without a reset or clearance stop. Score individual stall limits using integer 20-ms controls, at most 407 per swimmer. Retain at least 185 Godot, 195 northeast, 218 west and 201 southwest cells out of 345; apply the existing aggregate comparison to the installed chooser. Record every failed gate. Only a full native qualifier proceeds to a separate staged, headless Godot/native playback comparison before changing the active project. Use no mouse input and do not rerank candidates on final results.
