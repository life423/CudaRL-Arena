@echo off
echo ========================================
echo    CudaRL-Arena Build Script
echo ========================================

:: Check prerequisites
where cmake >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: CMake not found in PATH
    exit /b 1
)

where godot >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Godot not found in PATH
    exit /b 1
)

:: Build the GDExtension
echo Building GDExtension...
cd src\godot_bindings
if not exist build mkdir build
cd build

cmake .. -DCMAKE_BUILD_TYPE=Release
if %errorlevel% neq 0 (
    echo ERROR: CMake configuration failed
    exit /b 1
)

cmake --build . --config Release
if %errorlevel% neq 0 (
    echo ERROR: Build failed
    exit /b 1
)

:: Test the extension
echo Testing extension...
cd ..\..\..\godot-project
godot --headless --path . --script res://test_basic.gd
if %errorlevel% neq 0 (
    echo ERROR: Extension test failed
    exit /b 1
)

echo ========================================
echo    Build completed successfully!
echo ========================================