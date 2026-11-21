# CudaRL motor foundation research

This standalone research snapshot coexists with the original Tadpole Pond application,
website, installer and CI at the repository root. It does not replace, install, or
deploy the public application. Build this directory as its own CMake project, not
as an add_subdirectory of the legacy root.

## Verify the research build (Windows, Visual Studio 2022 C++ tools)

From the repository root:

```powershell
cmake -S research/minimal-foundation -B build/research-foundation-cpu -DCUDARL_ENABLE_CUDA=OFF -DCUDARL_BUILD_GODOT=OFF
cmake --build build/research-foundation-cpu --config Release
ctest --test-dir build/research-foundation-cpu -C Release --output-on-failure
```

See CMakePresets.json for a Windows CUDA build; select the nested project
directory as the working directory. The Godot extension requires matching,
built Godot 4.4 C++ bindings. Set CUDARL_GODOT_CPP_ROOT and, if needed,
CUDARL_GODOT_CPP_LIBRARY explicitly. The research scene does not ship an
installed controller and is expected to start paused without one.

The native controller learns four joint-angle targets. The research motor
restricts itself to calm, submerged, noncontact operation. Movement from
internal joints is not forced by the viewer. This code and data do not
establish successful learned holding, recovery from moving starts, or current
handling.

Two historical inputs are included for reproducibility: the old two-output
checkpoint fixture tests explicit schema rejection, and update 336 plus its
protocol are the sustained-swimming research baseline used by the optional
append/research tools. Neither is an active viewer policy. Rejected policy
weights, prior study traces, one-shot training release scripts, generated
build outputs, local absolute paths, and user editor changes are kept out of
this snapshot. The nested source includes PPO tools for research but running
them is a separate operation, not part of build or publication.
