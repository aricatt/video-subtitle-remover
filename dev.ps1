# ==========================================================
# Video Subtitle Remover - Development Environment Setup
#
# Usage:
#   .\dev.ps1              # Auto-detect GPU and CUDA version
#   .\dev.ps1 -Mode cpu   # Force CPU mode
#   .\dev.ps1 -Mode cuda  # Auto-detect CUDA version
#   .\dev.ps1 -Mode cuda11.8  # Force CUDA 11.8
#   .\dev.ps1 -Mode cuda12.6  # Force CUDA 12.6
#   .\dev.ps1 -Mode cuda12.8  # Force CUDA 12.8 (RTX 50 / Blackwell)
# ==========================================================

param(
    [ValidateSet("auto", "cpu", "cuda", "cuda11.8", "cuda12.6", "cuda12.8")]
    [string]$Mode = "auto"
)

$ErrorActionPreference = "Stop"

$REPO_ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
Write-Host "[dev.ps1] Repo root: $REPO_ROOT" -ForegroundColor Cyan

# --- Configuration ---
$VENV_DIR = Join-Path $REPO_ROOT "videoEnv"
$VENV_PYTHON = Join-Path $VENV_DIR "Scripts\python.exe"
$REQUIREMENTS_FILE = Join-Path $REPO_ROOT "requirements.txt"
$MIN_PYTHON_VERSION = [Version]"3.12.0"

# --- Function: Check Python Version ---
function Test-PythonVersion {
    param([string]$PythonPath)
    
    try {
        $versionOutput = & $PythonPath --version 2>&1
        if ($versionOutput -match "Python (\d+\.\d+\.\d+)") {
            $version = [Version]$matches[1]
            return $version
        }
    } catch {
        return $null
    }
    return $null
}

# --- Function: Detect NVIDIA GPU and CUDA ---
function Get-NvidiaInfo {
    $info = @{
        HasGPU = $false
        GPUName = $null
        ComputeCap = $null
        CUDAVersion = $null
        CUDASupported = @()
        IsBlackwell = $false
    }
    
    try {
        # Check for nvidia-smi
        $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
        if ($nvidiaSmi) {
            $smiOutput = & nvidia-smi 2>&1
            if ($LASTEXITCODE -eq 0) {
                $info.HasGPU = $true
                Write-Host "[dev.ps1] NVIDIA GPU detected" -ForegroundColor Green
                
                # Parse CUDA version from nvidia-smi
                try {
                    Write-Host "[dev.ps1] nvidia-smi output preview:" -ForegroundColor Cyan
                    Write-Host ($smiOutput -split "`n" | Select-Object -First 5) -ForegroundColor Gray
                    if ($smiOutput -match "CUDA Version: (\d+\.\d+)") {
                        $info.CUDAVersion = [Version]$matches[1]
                        Write-Host "[dev.ps1] CUDA Version: $($info.CUDAVersion)" -ForegroundColor Cyan
                    } else {
                        Write-Host "[dev.ps1] Could not parse CUDA version from nvidia-smi output" -ForegroundColor Yellow
                    }
                } catch {
                    Write-Host "[dev.ps1] Error parsing CUDA version: $($_.Exception.Message)" -ForegroundColor Yellow
                }

                try {
                    $gpuQuery = & nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>&1
                    if ($LASTEXITCODE -eq 0 -and $gpuQuery) {
                        $gpuLine = ($gpuQuery | Select-Object -First 1).ToString().Trim()
                        if ($gpuLine -match "^(.+),\s*(\d+\.\d+)$") {
                            $info.GPUName = $matches[1].Trim()
                            $info.ComputeCap = [Version]$matches[2]
                            Write-Host "[dev.ps1] GPU: $($info.GPUName), Compute Capability: $($info.ComputeCap)" -ForegroundColor Cyan
                        }
                    }
                } catch {
                    Write-Host "[dev.ps1] Error querying GPU compute capability: $($_.Exception.Message)" -ForegroundColor Yellow
                }
                
                # Determine supported CUDA versions based on GPU
                try {
                    $gpuName = if ($info.GPUName) { $info.GPUName } else { $smiOutput }
                    $computeCap = $info.ComputeCap

                    if ($gpuName -match "RTX 50" -or ($computeCap -and $computeCap.Major -ge 12)) {
                        # Blackwell (RTX 50 series): PyTorch stable cu126 lacks sm_120 kernels
                        $info.IsBlackwell = $true
                        $info.CUDASupported = @("12.8")
                        Write-Host "[dev.ps1] Blackwell GPU detected, requires PyTorch cu128 (nightly)" -ForegroundColor Yellow
                    } elseif ($gpuName -match "RTX 40") {
                        $info.CUDASupported = @("12.6", "11.8")
                    } elseif ($gpuName -match "RTX 30" -or $gpuName -match "RTX 20") {
                        $info.CUDASupported = @("12.6", "11.8")
                    } else {
                        # Unknown GPU, use conservative default
                        $info.CUDASupported = @("12.6", "11.8")
                    }
                } catch {
                    Write-Host "[dev.ps1] Error determining GPU support: $($_.Exception.Message)" -ForegroundColor Yellow
                    $info.CUDASupported = @("11.8")
                }
            }
        }
    } catch {
        Write-Host "[dev.ps1] Could not detect NVIDIA GPU: $($_.Exception.Message)" -ForegroundColor Yellow
    }
    
    return $info
}

# --- Function: Select CUDA Version ---
function Select-CUDAVersion {
    param([hashtable]$NvidiaInfo)
    
    if (-not $NvidiaInfo.HasGPU) {
        Write-Host "[dev.ps1] No NVIDIA GPU detected, will use CPU mode" -ForegroundColor Yellow
        return "cpu"
    }
    
    # Auto-select based on detected CUDA version
    if ($NvidiaInfo.CUDAVersion) {
        $detected = $NvidiaInfo.CUDAVersion.ToString(2)
        if ($NvidiaInfo.CUDASupported -contains $detected) {
            Write-Host "[dev.ps1] Auto-selecting CUDA $detected" -ForegroundColor Green
            return $detected
        }
    }
    
    # Default to latest supported (with safety check)
    if ($NvidiaInfo.CUDASupported -and $NvidiaInfo.CUDASupported.Count -gt 0) {
        $latest = $NvidiaInfo.CUDASupported[0]
        Write-Host "[dev.ps1] Using CUDA $latest (latest supported)" -ForegroundColor Green
        return $latest
    } else {
        Write-Host "[dev.ps1] Could not determine supported CUDA version, defaulting to 11.8" -ForegroundColor Yellow
        return "11.8"
    }
}

# --- Step 1: Check Python Version ---
Write-Host "`n[dev.ps1] Step 1: Checking Python version..." -ForegroundColor Cyan
$pythonVersion = Test-PythonVersion -PythonPath "python"
if (-not $pythonVersion) {
    Write-Host "[dev.ps1] Python not found in PATH. Please install Python 3.12+ from https://www.python.org/downloads/" -ForegroundColor Red
    exit 1
}

if ($pythonVersion -lt $MIN_PYTHON_VERSION) {
    Write-Host "[dev.ps1] Python version $pythonVersion is too old. Minimum required: $MIN_PYTHON_VERSION" -ForegroundColor Red
    exit 1
}
Write-Host "[dev.ps1] Python $pythonVersion OK" -ForegroundColor Green

# --- Step 2: Detect GPU and CUDA ---
Write-Host "`n[dev.ps1] Step 2: Detecting GPU and CUDA..." -ForegroundColor Cyan

# Handle mode parameter
if ($Mode -eq "cpu") {
    Write-Host "[dev.ps1] CPU mode forced by parameter" -ForegroundColor Yellow
    $cudaVersion = "cpu"
} elseif ($Mode -eq "cuda11.8") {
    Write-Host "[dev.ps1] CUDA 11.8 mode forced by parameter" -ForegroundColor Yellow
    $cudaVersion = "11.8"
} elseif ($Mode -eq "cuda12.6") {
    Write-Host "[dev.ps1] CUDA 12.6 mode forced by parameter" -ForegroundColor Yellow
    $cudaVersion = "12.6"
} elseif ($Mode -eq "cuda12.8") {
    Write-Host "[dev.ps1] CUDA 12.8 mode forced by parameter" -ForegroundColor Yellow
    $cudaVersion = "12.8"
} elseif ($Mode -eq "cuda") {
    Write-Host "[dev.ps1] CUDA mode forced by parameter, auto-detecting version..." -ForegroundColor Yellow
    $nvidiaInfo = Get-NvidiaInfo
    $cudaVersion = Select-CUDAVersion -NvidiaInfo $nvidiaInfo
} else {
    # Auto mode (default)
    $nvidiaInfo = Get-NvidiaInfo
    $cudaVersion = Select-CUDAVersion -NvidiaInfo $nvidiaInfo
}

# --- Step 3: Create/Update Virtual Environment ---
Write-Host "`n[dev.ps1] Step 3: Setting up virtual environment..." -ForegroundColor Cyan

if (-not (Test-Path $VENV_PYTHON)) {
    Write-Host "[dev.ps1] Creating virtual environment at $VENV_DIR ..." -ForegroundColor Yellow
    python -m venv $VENV_DIR
    & $VENV_PYTHON -m pip install --upgrade pip --quiet
    Write-Host "[dev.ps1] Virtual environment created" -ForegroundColor Green
} else {
    Write-Host "[dev.ps1] Virtual environment exists at $VENV_DIR" -ForegroundColor Green
}

# --- Step 4: Install Dependencies ---
Write-Host "`n[dev.ps1] Step 4: Installing dependencies..." -ForegroundColor Cyan

# Install base requirements first
Write-Host "[dev.ps1] Installing base requirements..." -ForegroundColor Yellow
& $VENV_PYTHON -m pip install -r $REQUIREMENTS_FILE

# Install PyTorch and PaddlePaddle based on CUDA version
if ($cudaVersion -eq "cpu") {
    Write-Host "[dev.ps1] Installing CPU versions of PyTorch and PaddlePaddle..." -ForegroundColor Yellow
    & $VENV_PYTHON -m pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
    & $VENV_PYTHON -m pip install torch==2.7.0 torchvision==0.22.0
    Write-Host "[dev.ps1] Installing CPU version of ONNX Runtime..." -ForegroundColor Yellow
    & $VENV_PYTHON -m pip install onnxruntime
} elseif ($cudaVersion -eq "11.8") {
    Write-Host "[dev.ps1] Installing CUDA 11.8 versions of PyTorch and PaddlePaddle..." -ForegroundColor Yellow
    & $VENV_PYTHON -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
    & $VENV_PYTHON -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu118
    Write-Host "[dev.ps1] Installing CUDA 11.8 version of ONNX Runtime..." -ForegroundColor Yellow
    & $VENV_PYTHON -m pip install onnxruntime-gpu==1.20.1 --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple/
} elseif ($cudaVersion -eq "12.6") {
    Write-Host "[dev.ps1] Installing CUDA 12.6 versions of PyTorch and PaddlePaddle..." -ForegroundColor Yellow
    & $VENV_PYTHON -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
    & $VENV_PYTHON -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126
    Write-Host "[dev.ps1] Installing CUDA 12.6 version of ONNX Runtime..." -ForegroundColor Yellow
    & $VENV_PYTHON -m pip install onnxruntime-gpu==1.22.0
} elseif ($cudaVersion -eq "12.8") {
    Write-Host "[dev.ps1] Installing CUDA 12.8 (Blackwell) stack..." -ForegroundColor Yellow
    Write-Host "[dev.ps1] PaddlePaddle still uses cu126; PyTorch uses cu128 nightly for sm_120" -ForegroundColor Yellow
    & $VENV_PYTHON -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
    & $VENV_PYTHON -m pip install -U --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
    Write-Host "[dev.ps1] Installing CUDA 12.6 version of ONNX Runtime..." -ForegroundColor Yellow
    & $VENV_PYTHON -m pip install onnxruntime-gpu==1.22.0
} else {
    Write-Host "[dev.ps1] Unknown CUDA version $cudaVersion, defaulting to CUDA 11.8" -ForegroundColor Yellow
    & $VENV_PYTHON -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
    & $VENV_PYTHON -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu118
    Write-Host "[dev.ps1] Installing CUDA 11.8 version of ONNX Runtime..." -ForegroundColor Yellow
    & $VENV_PYTHON -m pip install onnxruntime-gpu==1.20.1 --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple/
}

Write-Host "[dev.ps1] Dependencies installed successfully" -ForegroundColor Green

# --- Step 5: Verify Installation ---
Write-Host "`n[dev.ps1] Step 5: Verifying installation..." -ForegroundColor Cyan
try {
    $verifyScript = @'
import torch
import paddle
print("PyTorch:", torch.__version__)
print("Paddle:", paddle.__version__)
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability()
    arch = f"sm_{major}{minor}"
    arch_list = torch.cuda.get_arch_list() if hasattr(torch.cuda, "get_arch_list") else []
    print("GPU:", torch.cuda.get_device_name(0))
    print("Compute capability:", f"{major}.{minor}", f"({arch})")
    if arch_list and arch not in arch_list:
        raise RuntimeError(f"PyTorch build lacks kernels for {arch}; supported: {arch_list}")
    x = torch.randn(4, 4, device="cuda")
    _ = x @ x
    print("CUDA kernel test: OK")
'@
    & $VENV_PYTHON -c $verifyScript
    Write-Host "[dev.ps1] Installation verified successfully" -ForegroundColor Green
} catch {
    Write-Host "[dev.ps1] Warning: GPU verification failed. For RTX 50 series, rerun with: .\dev.ps1 -Mode cuda12.8" -ForegroundColor Yellow
}

# --- Step 6: Launch Options ---
Write-Host "`n[dev.ps1] =========================================" -ForegroundColor Cyan
Write-Host "[dev.ps1] Environment setup complete!" -ForegroundColor Green
Write-Host "[dev.ps1] =========================================`n" -ForegroundColor Cyan

Write-Host "To activate the virtual environment manually:" -ForegroundColor White
Write-Host "  $VENV_DIR\Scripts\activate`n" -ForegroundColor Gray

Write-Host "Launch options:" -ForegroundColor White
Write-Host "  1. GUI mode (recommended):" -ForegroundColor Gray
Write-Host "     & $VENV_PYTHON gui.py`n" -ForegroundColor Gray
Write-Host "  2. CLI mode:" -ForegroundColor Gray
Write-Host "     & $VENV_PYTHON backend\main.py -i <input_video> -o <output_video>`n" -ForegroundColor Gray

# Ask if user wants to launch GUI
$launchGUI = Read-Host "Launch GUI now? (Y/n)"
if ($launchGUI -ne "n" -and $launchGUI -ne "N") {
    Write-Host "[dev.ps1] Launching GUI..." -ForegroundColor Green
    & $VENV_PYTHON gui.py
}
