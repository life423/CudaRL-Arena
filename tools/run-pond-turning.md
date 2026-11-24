# Run a native pond turning exercise

Use PowerShell 7 and the native Release trainer and evaluator. Build them once per source change. The runner uses the executable paths, hashes, physics receipts and budget in the study's sealed protocol.

To check a fresh protocol before launch, run the following command with its recorded SHA-256 hash in place of `<reviewed-SHA256>`.

```powershell
pwsh -NoProfile -File tools/run-pond-turning.ps1 `
  -Protocol build/turn33/protocol.json -ProtocolSha256 <reviewed-SHA256>
```

For an authorized exercise, the same command with `-Run` handles initialization, training, checkpointing and evaluation through the selected update.

```powershell
pwsh -NoProfile -File tools/run-pond-turning.ps1 `
  -Protocol build/turn33/protocol.json -ProtocolSha256 <reviewed-SHA256> `
  -StopAfterUpdate 32 -Run
```

**Use a fresh study directory.** An existing run retains its evidence and ledgers. For a shorter exercise within an approved budget, choose update 8, 16 or 24. The runner still evaluates the last checkpoint and records `INTENTIONAL_CHECKPOINT_FINISH`.

## What runs

1. Verify the executables, parent, protocol, long-swim reference and existing CPU/CUDA physics qualifications.
2. Initialize the candidate and check the frozen baseline. It must pass 14 rests, four short swims, both long swims and all 28 required long-swim windows.
3. Run eight PPO updates and assess the saved checkpoint. Swimming and resting must pass before the next chunk starts. Every checkpoint also attempts all 16 turns so incomplete turning remains measurable during training.
4. Continue immediately after a passing check. On a failed gate, preserve the native result, partial work counts and last checkpoint.

Each continuation requires a native receipt bound to the candidate, parent, protocol and evaluator. It covers 30,400 retention controls and 304,000 physical substeps. The trainer independently verifies the receipt before accepting the candidate. The course-sensitive study selects its one learning change through the sealed arguments `--turn-reward course-direction-v1`.

The runner checks both receipt seals: the outer checkpoint checksum and the native inner FNV64 over its 88 data bytes. The complete native payload is 96 bytes.

`-ResumeVerifiedBaseline` is an explicit recovery for the original receipt-parser stop only. It checks the unchanged protocol, preserved old runner hash, original stopped invocation with no training calls, candidate0 and a zero-update/zero-Adam native ledger, plus the genuine passing baseline and evaluation ledger. It skips initialization and baseline. New evidence goes in `runner-recovered-baseline/`, including old/new script and input hashes; the first `runner/` is preserved. This cannot resume a trained or physically failed allocation, and it does not reset any budget.

## Evidence and timing

The `runner/` directory contains the argument arrays, stdout, stderr, process exit codes and phase times. Child-process wall time includes process startup, execution and exit. The runner also measures saving stdout and stderr, validating outputs and handing off to the next process. Handoff time includes this bookkeeping, so the timing categories overlap. CUDA event timings are in the trainer's per-update metrics.

## Test the runner

```powershell
pwsh -NoProfile -File tools/run-pond-turning.ps1 -SelfTest
```

The self-test uses temporary fixtures and child processes that immediately exit. It checks full completion, an intentional stop at update eight, baseline rejection, mismatched receipts, corrupt inner seals, a failed retention check, partial training accounting and duplicate-run rejection. The output names the temporary directory containing its evidence. Passing `-Protocol` and `-ProtocolSha256` with `-SelfTest` also checks the actual saved native baseline receipt and zero-training recovery ledgers read-only.
