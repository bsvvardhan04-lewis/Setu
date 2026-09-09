# SETU first-time setup for a Snapdragon-powered Windows PC (ARM64).
#
# Run from an ordinary PowerShell prompt in the repo root:
#     powershell -ExecutionPolicy Bypass -File scripts\setup_windows_arm64.ps1
#
# The app runs on any machine without this - the runtime layer falls back to CPU. This
# script is what lights up the Hexagon NPU on a Snapdragon X / X2 device.

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

function Say($msg)  { Write-Host "  $msg" }
function Good($msg) { Write-Host "  [ok]   $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "  [warn] $msg" -ForegroundColor Yellow }

Write-Host "`nSETU setup" -ForegroundColor Cyan
Write-Host ("-" * 56)

# --------------------------------------------------------------- 1. architecture
$arch = $env:PROCESSOR_ARCHITECTURE
Say "Processor architecture: $arch"
$isArm = $arch -eq "ARM64"
if ($isArm) {
    Good "Windows on ARM detected - the QNN execution provider is available."
} else {
    Warn "Not an ARM64 machine. Setup continues, but SETU will run on CPU only."
    Warn "That is a supported configuration; the NPU path needs Snapdragon hardware."
}

# ------------------------------------------------------------------ 2. python
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) { throw "Python 3.10+ not found on PATH. Install the ARM64 build from python.org." }
$pyVersion = & python -c "import sys, platform; print(sys.version.split()[0], platform.machine())"
Say "Python: $pyVersion"
if ($isArm -and $pyVersion -notmatch "ARM64") {
    Warn "This is an x64 Python running under emulation. Install the native ARM64 build,"
    Warn "or onnxruntime-qnn will refuse to load and every model will fall back to CPU."
}

# ------------------------------------------------------------------- 3. venv
if (-not (Test-Path ".venv")) {
    Say "Creating virtual environment..."
    & python -m venv .venv
}
$venvPy = Join-Path $repo ".venv\Scripts\python.exe"
Good "Virtual environment ready"

# ------------------------------------------------------------- 4. dependencies
Say "Installing SETU and its dependencies..."
& $venvPy -m pip install --quiet --upgrade pip
& $venvPy -m pip install --quiet -e .

if ($isArm) {
    Say "Installing the QNN execution provider (Hexagon NPU)..."
    try {
        & $venvPy -m pip install --quiet onnxruntime-qnn
        Good "onnxruntime-qnn installed"
    } catch {
        Warn "onnxruntime-qnn failed to install; falling back to CPU onnxruntime."
        & $venvPy -m pip install --quiet onnxruntime
    }
    try {
        & $venvPy -m pip install --quiet onnxruntime-genai
        Good "onnxruntime-genai installed (NPU path for the LLM)"
    } catch {
        Warn "onnxruntime-genai unavailable; the LLM will use Genie or llama.cpp instead."
    }
} else {
    & $venvPy -m pip install --quiet onnxruntime
    Good "onnxruntime (CPU) installed"
}

# ---------------------------------------------------------------- 5. QAIRT check
$qairt = Get-Command genie-t2t-run -ErrorAction SilentlyContinue
if ($qairt) {
    Good "Genie runtime found: $($qairt.Source)"
} else {
    Warn "genie-t2t-run not on PATH. The LLM will use onnxruntime-genai or llama.cpp."
    Warn "For the full NPU LLM path, install the Qualcomm AI Runtime SDK (QAIRT) and"
    Warn "add its bin directory to PATH, or set SETU_GENIE_BIN to the executable."
}

# ------------------------------------------------------------------ 6. verify
Write-Host ""
Write-Host "Verifying..." -ForegroundColor Cyan
& $venvPy -m setu.cli doctor

Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Say "1. Download model assets:   .venv\Scripts\python.exe scripts\fetch_models.py --all"
Say "2. Benchmark this machine:  .venv\Scripts\python.exe scripts\run_bench.py --iters 30"
Say "3. Start SETU:              .venv\Scripts\python.exe -m setu.server.app"
Write-Host ""
