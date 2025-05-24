@echo off
echo Creating directories and copying GDExtension headers...

set BUILD_DIR=C:\Users\aiand\Documents\programming\CudaRL-Arena\build
set SRC_DIR=%BUILD_DIR%\_deps\godot_engine-src\modules\gdextension\include
set DEST_DIR=%BUILD_DIR%\_deps\godot_cpp-src\..\godot\modules\gdextension\include

mkdir "%DEST_DIR%" 2>nul
xcopy /E /I /Y "%SRC_DIR%" "%DEST_DIR%"

echo Done!