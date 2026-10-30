# Fixed-A validation — reject the current 5 ms implementation

The one authorized run is CLOSED. Its accuracy prerequisite failed. A is not
accepted for motor training or adoption. This rejects the tested time-advance
implementation at5ms, not the possibility of using the fixed compliant law
with a more accurate integration method. B remains available, unchanged.

Executed source commit: `7d6b53ff06bf5b5b9cd647db94d9cfabbd26a3e8`.
Frozen registration: [protocol](CONTACT_VALIDATION_PROTOCOL.md). Complete raw
evidence, executable and86-file source snapshot are in ignored
`build/contact-validation/2026-10-30-v1/`. The earlier pilot remains closed;
none of its398 unused calls was reused. This run made no retries or tuning.

## Exact execution and stopping

| Stage | Actual physical attempts | Result |
| --- | ---: | --- |
| Independent geometry, mass and calibration preflight |0 physical;137 geometry,168 kinetic,60 matrix queries |3/3 reference states pass; includes8 registered fixture-placement queries |
| Scalar accuracy |2,100 |Support6/6 and separation6/6 pass; impact0/6 passes |
| Rotating accuracy |508 |Five complete short trajectories; sixth stops at its8th step |
| Original5s bent/pitched/rolled support |0 |Blocked, not passed or physically failed |
| Original-free versus A preservation |0 |Blocked; not inferred from prior A/B equality |
| Additional whole-body passive energy stage |0 |Blocked |
| Four fresh withheld5s support starts |0 |Registered but unrun |
| **Total** |**2,608 /18,200** |**Closed;15,592 unused advances unavailable for automatic reuse** |

CPU accounts for1,400 attempts; CUDA1,208. All2,608 native proposals committed
to their disposable local trial state, including the final proposal rejected by
the external numerical audit. No app or personal state was updated. Serial/input
hashes used the actual nonblocking CUDA path. There were2,135 continuous-oracle
queries, under12,000. All18 scalar trajectories finished; each backend used its
own evolving state. The finest0.625ms rotating runs never started.

The run stopped immediately at rotating1.25ms CUDA step8, approximately10ms:
the normalized impulse–momentum certificate was1.099127512 against its fixed
maximum1. It did not retry, relax the allowance or finish that trajectory.
The scalar fidelity failure already blocked longer work independently of this
later audit stop. All available traces and the first-failure binary are retained.

## Independent continuous-law comparison

The analytic reference uses the exact native FP32 coefficients promoted to
double:320N/m and1.9595917463302612N·s/m. Its scalar mass is exactly3g. The
impact starts25µm clear at50mm/s. The reference has geometric arrival at0.5ms,
maximum compression at3.561862ms and unilateral force release at6.623724ms.
It releases while still compressed: the law clips tensile force; it is not a
bilateral oscillator integrated until zero gap.

| Impact quantity | Continuous law |5ms |2.5ms |1.25ms |
| --- | ---: | ---: | ---: | ---: |
| Maximum compression,µm |56.3198 |20.6776 |27.3219 |31.4106 |
| Compression error,% low |— |63.29 |51.49 |44.23 |
| Restitution |0.135335 |0.030502 |0.046323 |0.063539 |
| Restitution absolute error |— |0.104833 |0.089012 |0.071796 |
| Total normal impulse,µN·s |170.3003 |154.5753 |156.9485 |159.5309 |
| Impulse error,% low |— |9.23 |7.84 |6.32 |
| Maximum time-aligned gap error,mm |— |1.25872 |1.06841 |0.86038 |
| Maximum time-aligned speed error,mm/s |— |14.5369 |9.1567 |5.9384 |

CPU and CUDA agree on these scalar measurements to roundoff. Every impact
fails all five preregistered accuracy gates at every tested timestep. Those
limits remain:compression error max(2µm,5%); restitution error0.02;
impulse error max(10nN·s,5%); gap error10µm; speed error1mm/s.
The observed trend improves with refinement, but1.25ms is still inadequate;
it was not adopted and does not justify a new unregistered refinement.

This is not only a missed sampled peak. At the same5ms instant, the continuous
reference is already moving outward at5.401mm/s. The5/2.5/1.25ms numerical
trajectories are still closing at9.136/3.755/0.378mm/s, respectively. Trajectory
timing and velocity are wrong as well as the peak. The5ms method returns only
5.08% of the continuous law's rebound kinetic energy;2.5/1.25ms return
11.72%/22.04%. Thus stability and low bounce concealed substantial additional
numerical dissipation.

The law's own characteristic scalar time1/sqrt(k/m) is about3.062ms, shorter
than the current5ms step. Backward implicit contact and discrete activation/
release are not accurately resolving this fast response. Less bounce could be
a deliberate alternative physical model, but it is not an accurate simulation
of the unchanged calibrated law. Accepting it would silently make effective
damping depend on the timestep. It is unacceptable under this validation.

Loaded scalar support does reach the analytic0.9375µm compression at0.3mN;
all support transient errors pass. Separating scalar contact exerts zero impulse
and remains nonadhesive. Correct equilibrium/separation therefore do not rescue
the independently failed impact transient.

## Reached physical and numerical evidence

All reached states were finite and within the unchanged500µm overlap rule.
Maximum reached compression across scalar/body traces was31.4106µm. This
is a reached-prefix statement, not a claim about longer support or final cases.

Scalar momentum residuals were at most0.002244 of the registered allowance;
constitutive residuals were at most7.80e−10m/s. Contact kinetic change and
impulse work agree to about2.12e−22J. Scalar total energy includes actual load
potential; actual load work and the split-kick ledger remain separate. Clipped
unilateral release can discard residual spring energy, so reference loss is
labeled dissipated-or-discarded, not solely dashpot dissipation. The scalar
accuracy failure is compatible with these successful passivity identities.

The5ms/2.5ms rotating CPU/CUDA trajectories completed0.25s with safe finite
states and reached equation checks passing. The1.25ms CPU trajectory also
completed; its CUDA partner stopped at10ms. These are completed short numerical
trials, NOT passes of the rotating accuracy gate: its required0.625ms comparison
was not reached. Formal timestep convergence remains unproved.

Descriptive comparisons of the saved CPU trajectories already show sensitivity:
5ms versus1.25ms differs by up to112.11µm head position,6.055mm/s head
velocity and1.962rad/s joint rate over the shared quarter second. These are
post-run comparisons of existing rows, not substitute acceptance tests or new
physical advances. At the failed1.25ms sample, CPU/CUDA head velocity differs
by only1.49e−8m/s and the gaps agree exactly; this small trajectory difference
must not be confused with the independently failed impulse certificate.

At the actual abort, clearance was−7.48038µm and the contact impulse was
positive61.2027nN·s. Constitutive residual7.1843e−10m/s passed. Contact
kinetic/work closure error was7.655e−14J, below1e−10J; spring energy fell
12.6198→8.9530nJ and whole mechanical energy fell9.3805nJ. The failure is
the momentum certificate, not observed penetration, adhesion or energy gain.

That certificate reconstructsM·delta_v from separately FP32-stored velocities,
andJᵀp from FP32-stored row impulses, after a double contact solve. Storage
quantization/cancellation is a plausible cause at a tiny impulse. The raw
per-coordinate residual, normalization scale and unrounded solve values were
not serialized, so that explanation is not proved. A1.099 normalized residual
alone does not establish a physical-law CPU/CUDA defect. Its gate remains
failed and unresolved; no tolerance was changed to excuse it.

## Decision and narrow next boundary

**Reject current A at5ms.** It cannot be accepted by treating numerical damping
as improved physical realism. No support/retention/learning claims follow from
its short safe trajectories or successful equilibrium calculation.

The smallest next decision is inside contact time advancement: an explicitly
bounded correction to activation/release and temporal integration of this same
fixed law, with an auditable small-impulse momentum certificate. It is not a
new neural network, reward change, actuator/body change, coefficient sweep or
whole-app/engine rewrite. Merely selecting1.25ms is not supported by the data.
No such correction or follow-up run has been authorized or executed here.

Before any motor training, a corrected candidate must pass the independent
accuracy gate and then the still-blocked original5s support, actual-impulse
force balance/settling, original-free preservation, energy and fresh withheld
checks. Contact readiness would still not demonstrate swimming, braking,
unsupported holding or goal selection.

All86 snapshotted source files were hashed unchanged through the run. Runtime
contact/dynamics and both candidate laws are unchanged. Post-run verification
again passed477 protected records,39 old contact artifacts and25 personal
saves plus25 backups. Original checkout stays clean at its baseline commit.
The validation CMake option is restored OFF; CTest enumeration still contains
only the small foundation suite. No default tests with unbudgeted physical
advances, PPO, adoption, merge, push or deployment ran.
