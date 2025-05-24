# Godot Project Directory

This directory contains the Godot 4 project for CudaRL-Arena.

- `project.godot` and other Godot project files will be initialized here.
- `Scenes/` will contain Godot scenes (e.g., GridWorld.tscn).
- `gdextension/` will contain the C++ GDExtension plugin code.
- `godot-cpp/` will contain the Godot C++ bindings (submodule or downloaded).
- `bin/` will contain built GDExtension binaries.

## Initializing the Godot Project

1. Open the Godot 4 editor.
2. Select "Import" or "New Project" and choose this `godot/` directory.
3. Godot will create `project.godot` and other necessary files.
4. Create a `Scenes/` folder in the Godot editor if it does not already exist.
5. Save your first scene (e.g., `GridWorld.tscn`) in `Scenes/`.

You can now add nodes, scripts, and integrate the GDExtension plugin.

---

## Automated Build Instructions

The build system is fully automated using CMake and ExternalProject for godot-cpp.

### Prerequisites

- Godot 4.x executable (path set in CMakeLists.txt, currently: `C:/Users/aiand/Godot/Godot.exe`)
- Python and SCons installed (required for godot-cpp build)
- CMake 3.20+ and a C++17 compiler

### One-Command Build

From the project root:

```sh
mkdir build && cd build
cmake .. 
cmake --build . --target all
```

This will:
- Fetch and build the correct godot-cpp version (branch 4.5)
- Generate `extension_api.json` using your Godot executable
- Build the RL backend and the GDExtension plugin
- Output the plugin to `godot/bin/`

You can then open the Godot project and add the plugin to your scene.

---
