# Frozen requested-turn probe preparation

From the repository root:

```powershell
cmake -S tools/frozen-request-turn -B build/frozen-request-turn-native-2026-11-23 -G "Visual Studio 17 2022" -A x64
cmake --build build/frozen-request-turn-native-2026-11-23 --config Release --parallel 4
```

The recipe links the already reviewed static libraries from
`build/continuous-pond-native-2026-11-23/Release`. It changes no prior target.

Pure preparation only:

```powershell
build/frozen-request-turn-native-2026-11-23/Release/frozen_requested_turn_probe.exe --prepare build/ppo/frozen-request-turn-2026-11-23-v1 build/ppo/continuous-pond-2026-11-23-v1 docs/FROZEN_REQUEST_TURN_PROBE_2026-11-23.md
```

This copies parents, performs inference and intent-change fixtures, and writes
zero-control evidence. No physics control or optimizer is called. The run root
must be new. Physical execution requires a separate frozen-input manifest and
fresh user grant for this exact probe, plus an external 120 s process watchdog.
The protocol defines the finite budget; this README does not authorize a run.
