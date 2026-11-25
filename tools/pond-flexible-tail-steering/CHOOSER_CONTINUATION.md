# Seven-stage chooser continuation

The first nine-action capacity diagnostic failed before collecting an action.
The reserved second diagnostic passed after the shared categorical capacity
repair. Both attempts and all five completed motor stages remain preserved.

The following chooser main invocation was interrupted after1.0926767 seconds:
the runner misclassified the evaluator's valid exit1 (unqualified6/12) as a
process fault and killed training. Only exact checkpoint0 was persisted. Actual
partial rollout/optimizer work is unknown; it must not be reported as zero.

The repaired runner accepts exit1 only with complete, internally consistent
evaluation evidence and no native fault or timeout. Guard stops remain failed
navigation cases. Future error cleanup requests a cooperative training stop.

Continue once from a byte-identical copy of checkpoint0 and actor0. Tighten the
copied budget from64 to63 complete updates, reserving one update for interrupted
work: at most8192 transitions,32 Adam calls per optimizer and10,485,760 physical
substeps including warmup. Charge the interrupted wall time to both cumulative
and native session budgets. Do not overwrite the original budget/checkpoint.

Reuse the passed second diagnostic and valid pretraining evaluation. The final
scheduled assessment is63 instead of64, leaving the same nine assessment slots.
Native task identity, binaries, inputs, physical settings and criteria remain
unchanged. No third diagnostic or extra main budget is permitted. Finish stage7
within its existing allowance, report the partial-work upper bound, then stop.
