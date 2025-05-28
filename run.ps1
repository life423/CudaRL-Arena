param(
    [string]$Mode = "game",  # game, train, or test
    [switch]$Debug = $false
)

$godotPath = "C:\Users\aiand\Godot\Godot.exe"

function Build-CudaCore {
    Write-Host "Building CUDA core..." -ForegroundColor Yellow
    cmake --build build --config Release --target cudarl_core
    if ($LASTEXITCODE -ne 0) {
        Write-Error "CUDA core build failed!"
        exit 1
    }
}

function Build-MainApp {
    Write-Host "Building main application..." -ForegroundColor Yellow
    cmake --build build --config Release --target cudarl_app
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Main app build failed!"
        exit 1
    }
}

function Build-PythonBindings {
    Write-Host "Building Python bindings..." -ForegroundColor Yellow
    cmake --build build --config Release --target cudarl_core_python
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Python bindings build failed!"
        exit 1
    }
}

function Build-GodotExtension {
    Write-Host "Building Godot extension..." -ForegroundColor Yellow
    cmake --build build --config Release --target cudarl_godot
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Godot extension build failed!"
        exit 1
    }
}

function Build-Godot {
    Write-Host "Preparing Godot project..." -ForegroundColor Yellow
    if (-not (Test-Path $godotPath)) {
        Write-Warning "Godot not found at $godotPath"
        Write-Host "Please install Godot 4.2+ or update the path in this script" -ForegroundColor Yellow
        return $false
    }
    
    & $godotPath --headless --path godot --quit
    return ($LASTEXITCODE -eq 0)
}

function Run-Game {
    Write-Host "Starting CudaRL-Arena GUI..." -ForegroundColor Cyan
    if (-not (Test-Path $godotPath)) {
        Write-Warning "Godot not found at $godotPath"
        Write-Host "Please install Godot 4.2+ or update the path in this script" -ForegroundColor Yellow
        return
    }
    
    & $godotPath --path godot
}

function Run-Training {
    Write-Host "Running training without GUI..." -ForegroundColor Cyan
    
    # Add CUDA to PATH for this session
    $env:PATH += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin"
    
    # Add build directory to Python path
    $env:PYTHONPATH = "build/lib/Release;$env:PYTHONPATH"
    
    python python/scripts/train.py --episodes 1000 --render-every 0
}

function Run-Test {
    Write-Host "Running CUDA test..." -ForegroundColor Cyan
    
    if (-not (Test-Path "build/bin/Release/cudarl_app.exe")) {
        Write-Error "cudarl_app.exe not found. Please build first."
        return
    }
    
    .\build\bin\Release\cudarl_app.exe
}

function Run-PythonTest {
    Write-Host "Testing Python bindings..." -ForegroundColor Cyan
    
    # Add CUDA to PATH for this session
    $env:PATH += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin"
    
    # Test from build directory
    cd build/lib/Release
    python -c "import cudarl_core_python; print('Python bindings working!')"
    cd ../../..
}

# Ensure build directory exists
if (-not (Test-Path "build")) {
    Write-Host "Configuring CMake..." -ForegroundColor Yellow
    cmake -S . -B build
    if ($LASTEXITCODE -ne 0) {
        Write-Error "CMake configuration failed!"
        exit 1
    }
}

# Main execution
switch ($Mode) {
    "game" {
        Build-CudaCore
        Build-GodotExtension
        $godotResult = Build-Godot
        if ($godotResult) {
            Run-Game
        } else {
            Write-Error "Godot build failed, cannot run game"
        }
    }
    "train" {
        Build-CudaCore
        Build-PythonBindings
        Run-Training
    }
    "test" {
        Build-CudaCore
        Build-MainApp
        Run-Test
    }
    "python" {
        Build-CudaCore
        Build-PythonBindings
        Run-PythonTest
    }
    default {
        Write-Error "Unknown mode: $Mode. Use 'game', 'train', 'test', or 'python'"
        Write-Host "Examples:"
        Write-Host "  .\run.ps1 -Mode game    # Run GUI game"
        Write-Host "  .\run.ps1 -Mode train   # Run headless training"
        Write-Host "  .\run.ps1 -Mode test    # Run CUDA test"
        Write-Host "  .\run.ps1 -Mode python  # Test Python bindings"
    }
}
