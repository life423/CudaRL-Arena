# Per-goal actor advantages: prepared, not launched

One focused change to the failed variable-speed trial: normalize actor advantages
by requested-speed group across the full rollout, before minibatch shuffling.
Group labels are captured with observations, before any reset or request change.
Population mean/std use double accumulation; flat groups produce zero signal;
invalid labels/nonfinite data stop before Adam. The Gaussian loss receives identity
normalization statistics to prevent a second global normalization. Critic returns
remain the original full-episode GAE targets. Per-goal raw count/mean/std are logged.
This explicitly equalizes actor signal scales across goals; it does not change
reward values, success criteria or physical safety. No compatibility flag added.

Evidence: saved baseline evaluation has gamma.99 mean returns per row of
-1.460/-0.248/-0.057 for4/6/8mm/s. Between-goal means account for84.89% of that
return variance. These are deterministic evaluation returns, NOT the unavailable
stochastic training advantages. Different advantage scales remain a hypothesis;
the correction may fail and does not resolve untrained continuous-state coverage.

Proposed sole trial identity: build/speed-motor/2026-10-31-goal-advantage-v1/.
Same selected8mm/s actor, fresh critic/Adam, seed0x53574D32,24updates,16worlds,
80controls, lambda1/gamma.99. Train4/6/8, frozen paired evaluation4/5/6/7/8 in
cases0/3, plus continuous6->4->8 switch sequence.5/7 are withheld from training.
Same body/water/oscillator/action limits, learning rate/exploration and gates.

Ceiling32,832controls/328,320native steps total, including readiness, both frozen
evaluations and all failures.20second native wall limit;25second wrapper watchdog.
No retry/sweep/extension. Expected few native seconds based on the prior3.347s,
not a promise. Only source/executable preparation and nonphysical checks authorized
now: notify readiness, then wait for authorization before this new trial starts.

Selection remains10/10constant AND3/3switch passes, improved normalized RMSE
overall, no worse8mm/s RMSE or positive work/metre. Otherwise retain the current
actor. Native finite/energy/torque/10mmclearance guards and each request's10% mean,
31.25% RMSE,2mmlateral/1mmdepth/5degpath/10degtilt/1mJpermetre limits unchanged.
No commit, live adoption, model change or additional learning implied by readiness.


## Approved execution outcome

After explicit launch approval, closed00:12:50UTC on2026-10-31. Full declared
32,832controls/328,320steps consumed in3.271native seconds, no faults/retries.
Candidate4/10constant and0/3switches; rejected. Requests4/5/6/7/8mm/s produced
7.495/7.501/7.507/7.513/7.519mm/s; the 7/8pass bands overlap, so the pass-count
increase is not useful conditioning. Selected8mm/s actor unchanged. No further run.
