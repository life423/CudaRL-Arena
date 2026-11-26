# Godot open-water clearance-stop replay

One read-only diagnostic. Bind motor 88, the fitted gait, the 35-input +112 chooser, the three-depth parent 128, and the original evaluator Godot-start trace. Confirm the failure at exactly 33,979 accepted and 33,980 attempted native substeps, including all greedy decisions on the path. Keep the exact replay starts out of PPO.

Capture full native decision states at 24.0 s, 30.4 s, 33.6 s and every decision from 36.0 through 42.4 s, spaced 0.8 s apart. From each, let frozen +112 continue, let frozen parent take control indefinitely, and compare all 15 speed/rate requests held for one or three chooser decisions, then return control to +112. Compare 120 seconds after each fork. Require zero stops or reset, strictly positive native whole-body clearance, at least 6 mm/s mean horizontal COM transport, at most 407 native 20 ms consecutive slow controls, tilt at most 10 degrees and depth error at most 1 mm. Compute slow controls from the existing 0.5-second displacement history. Record first policy action on return.

For one longer follow-up, take the latest fork with a passing forced request differing from +112's first request. Prefer the shortest intervention, fewer changed request components, then maximum minimum continuation clearance and lowest action index. Retest that branch uninterrupted for 1,800 seconds after its fork. This single longer diagnostic is supplemental. All actor choices and study outputs remain offline.

Bound each trial by its horizon and the grid by 384 short branches, one prefix and one longer follow-up (under 39 million attempted native substeps), and the invocation by 150 seconds process time. No optimizer updates.
