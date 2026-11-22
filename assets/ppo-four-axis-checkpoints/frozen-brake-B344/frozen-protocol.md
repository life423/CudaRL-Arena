# Approved frozen braking-response diagnostic, 2026-11-04

Identity `build/diagnostics/frozen-brake-B344-v1`; branch
`codex/minimal-foundation`. This is a new, explicitly approved physical budget;
all old study budgets remain closed. No PPO, optimizer updates, runtime adoption,
push, merge, retries or extensions. Prior evidence and checkpoint336 preserved.

Frozen actor source:
`assets/ppo-four-axis-checkpoints/balanced-hold-scale-336to344/B/ppo-append-update-344.bin`
SHA256 `8D705FC82A2B12D465E4FDCD48BC8D881EA0710192B030B262E4A26A6C6AF2B2`.
Parent protocol SHA256
`B96BC59DD6F134946AA0FFCF5C371834347B54EAED7B6BB90322EDBCA48D8458`.
Original336 SHA256
`810DBD593CE39475CDD6E95B709BEF3FDB34D50C9FCBBB2C345C7A8882A0BEF0`.
Copy, checksum and explicitly decode the41input schema5 checkpoint; require
update344, old Adam metadata24260 and new-column age2560. Read optimizer metadata
only; the executable never constructs an optimizer or learner.

Use CPU inference and the existing native model6 physics equations, matching
the previous frozen assessments. Deterministic actor means, no exploration.
Independent NumPy inference on16actual saved stop-command states must agree
within2e-5 latent units before physical execution. Zero-step preparation performs
no simulator integration. Freeze all source/executable/input hashes before run.
Build preparation cap180s; zero-step preparation cap60s. These are not physical
or training attempts. The one physical attempt has30s native and own-process caps.

## Exact physical protocol

Six original cases: pose0/3 times original target-depth start offsets0,-.5,+.5mm.
For each, use unchanged1.2x.8m arena, body/water, initial pose, fixed -80mm depth,
lane and heading. Run frozenB344 at requested8mm/s for640controls(32s). Preserve
all four learned outputs and original filters. Capture complete body/joints,
filter angles/rates, task references, heading and counters once at32s.

From each captured state make ten independent40control(2s) branches:

| Branch | Goal | Diagnostic pulse |
|---|---|---|
|0 |continue8mm/s |none |
|1 |zero |none |
|2,3 |zero |axis0 latent -.2,+.2 |
|4,5 |zero |axis1 latent -.2,+.2 |
|6,7 |zero |axis2 latent -.2,+.2 |
|8,9 |zero |axis3 latent -.2,+.2 |

Each pulse is added to the current learned latent for controls1..10(first.5s)
only; then all four outputs are unmodified learned outputs for1.5s. Every other
axis remains learned at every control. Original tanh angle decoding,1Hz target
filter, angle/rate/acceleration checks and +/-10microNm internal torque limits
remain unchanged. No posture clamp, forced joint freeze, external propulsion,
teacher/gait substitution or runtime installation. A pulse is a temporary
research excitation, never an accepted controller.

Clone and compare semantic state fields before changing the goal. Zero branches
capture the horizontal COM anchor ONCE at32s with original target depth. Retain
original lane/depth/heading/time/filter state. Continuing8 never captures anchor.
Do not rebase targets, reset physical state or alter filters at the switch.

## Measurements and fixed decisions

Record all6240controls with actual3DCOMspeed, forward velocity/displacement,
signed depth, lane, combined tilt, roll/pitch, anchor error, learned/applied
latents, decoded commands, all joint angles/rates/last torques and peak applied
torque across all physics substeps. Save each32s complete state snapshot with
ABI sizes and checksummed payload. Report all60branches and each32s prefix.

At each branch, compare mean3DCOMspeed over the last10control endpoints(last.5s)
and2s forward displacement against that case's matched unperturbed zero-goal
branch1. Also report zero-versus8 differences. A useful local braking pulse must
reduce last.5s mean speed by>=.5mm/s AND2s forward displacement by>=.5mm, with
depth<=1mm,lane<=2mm,combined head tilt<=10degrees throughout the branch, including
the initial32s state. Require at least one useful pulse in a case of each pose
to support local braking authority across both poses. Report speed/displacement
improvements that fail the physical envelope separately. Prefix envelope failures
are also reported; they do not silently change branch acceptance or clamp state.

This diagnoses local response to fixed pulses at the six captured gait phases.
It does not test coordinated pulse combinations, other amplitudes/durations,
additional phases,16s stopping deadlines,32s hold stability or anchor recovery.
Passing is not full stopping/holding acceptance, and failure of these probes
does not prove physical impossibility. Joint motion while holding is allowed.

## Accounting and stopping

Prefixes6*640=3840controls; branches6*10*40=2400controls. Combined MAX6240controls,
62400physics substeps(.05s controls,.005s substeps). Copies do not integrate and
prefix costs are counted once. Native/own-process MAX30s including execution I/O.
Sticky attempt marker before any physical step; no automatic retry or reuse of
unspent budget. First hard numerical/integrity/physics/torque fault ends entire
diagnostic, preserving last state and attempted substeps. Behavioral threshold
failures remain measurements; no relaxation or extra branches. Frozen actor
parameters must remain exact. Record actual launch/end, executed controls,
substeps and any partial branch. No optimizer updates or new training afterward.
