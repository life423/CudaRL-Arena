# Flexible-tail yaw feedback pilot

The matched yaw diagnostic found that the learned target stopped reversing
before forward speed fell below 6 mm/s. The physical reference continued
swimming from the same starts at the existing 80 ms action cadence. Fit a
yaw feedback initialization from training-only reference trajectories, then
run a single short native CUDA PPO pilot. Keep the native body, water,
controller limits, substeps, reward, starts and held-out gates from
[the straight-swimming pilot](LAUNCH_PROTOCOL.md).

- Fit at most eight stored rest starts for six simulated seconds each.
  Save command labels at the first decision and after two simulated seconds.
  Fit yaw target in Gaussian latent space against three neural features
  from actual yaw angle and angular rate. Normalize phase by its measured
  amplitude with an 0.08 rad floor. The third feature fades to zero as
  motion develops and lets the trained network initiate a stroke.
  The training-only reference retains the 1 Hz, 0.55 rad stroke.
  The actor observes joint measurements and receives no clock signal.
  Preserve the original fitted motor and all prior PPO checkpoints.
- Start the same 31 → 32 → 32 → 2 network with this fitted yaw readout.
  Initialize pitch at zero with all weights trainable. Run at most eight
  CUDA PPO updates of 256 worlds × 64 original physical controls,
  corresponding to 131,072 controls and 2,097,152 physics substeps.
  Sample once per four controls with a matched likelihood and discount.
  Use two passes, minibatch 1,024, seed 20261124 and actor/critic rates
  0.0001/0.0003. Yaw/pitch latent log standard deviations are −2.6/−3.8.
  Keep base gamma 0.998, lambda 0.9797, clip 0.2 and exact KL cap 0.05.
- Evaluate the deterministic seed and checkpoints 4 and 8 on the previous
  12 withheld starts. A pass requires ≥6 mm/s projected COM speed and
  ≤5° course error in each of the final two two-second windows, with
  ≤10° tilt, ≤1 mm depth error and positive clearance throughout.
  A candidate passes the full pilot if all 12 cases pass.
- Use one launch per stage, with a 4.000 s external wall cap and 3.5 s
  internal deadlines. Stop affected work at a physical, numerical,
  KL or time-limit fault. Save the raw receipts, exact input bindings,
  fit, checkpoints and evaluations. Keep every attempted result.
