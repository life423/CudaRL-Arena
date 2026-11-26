# Read-only policy-semantics supplement

Before interpreting critic predictions as calibration errors, compare the
stochastic policy the critic learns with the argmax policy used in Godot and
the fixed evaluator. The initial signal audit's fixed continuations are not
unbiased estimates of the critic's stochastic-policy expected value.

No optimizer updates or changes to the closed128-update session. From the same
two0.8s captures,use checkpoint0 and128 with each matching critic. Per pair run
one greedy continuation and16 fixed independent action-sampling streams using
the existing shared native log_softmax/sample_action/rng functions. Pair RNG
streams across models. Every branch runs at most40s,no resets.68 cases maximum,
2176000 substeps,45s native/50s external. Preserve all outcomes and seeds.
Report measured discounted reward separately from critic-bootstrapped40s return;
the latter still contains a critic estimate. This small conditional diagnostic
does not establish whole-pond safety or change any adoption criterion. Keep
Godot untouched regardless of these diagnostic results.
