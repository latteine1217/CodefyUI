# CodefyUI 一鍵安裝腳本 (Windows / PowerShell)
# 用法：
#   powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/treeleaves30760/CodefyUI/main/install.ps1 | iex"
#
# 可選環境變數：
#   $env:CODEFYUI_DIR = 'D:\path\to\CodefyUI'   # 自訂安裝路徑（預設 $HOME\CodefyUI）

$ErrorActionPreference = 'Stop'

$Repo = 'https://github.com/treeleaves30760/CodefyUI.git'
$InstallDir = if ($env:CODEFYUI_DIR) { $env:CODEFYUI_DIR } else { Join-Path $HOME 'CodefyUI' }

# ── Helpers ───────────────────────────────────────────────────────────────────
function Step($msg) { Write-Host ""; Write-Host "==> $msg" -ForegroundColor Blue }
function Ok($msg)   { Write-Host "  OK  $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "  !   $msg" -ForegroundColor Yellow }
function Die($msg)  { Write-Host ""; Write-Host "  X Error: $msg" -ForegroundColor Red; exit 1 }

function Test-Cmd($name) {
    return $null -ne (Get-Command $name -ErrorAction SilentlyContinue)
}

function Refresh-Path {
    $machine = [System.Environment]::GetEnvironmentVariable('Path', 'Machine')
    $user    = [System.Environment]::GetEnvironmentVariable('Path', 'User')
    $env:Path = ($machine, $user, $env:Path | Where-Object { $_ }) -join ';'
}

function Install-Winget($id, $friendlyName) {
    if (-not (Test-Cmd winget)) {
        Die "winget not found. Install '$friendlyName' manually or install 'App Installer' from Microsoft Store, then re-run."
    }
    winget install --id $id --silent --accept-source-agreements --accept-package-agreements --exact
    if ($LASTEXITCODE -ne 0) { Die "winget install $id failed (exit $LASTEXITCODE)" }
    Refresh-Path
}

# ══════════════════════════════════════════════════════════════════════════════
Write-Host ""
Write-Host "+======================================+"
Write-Host "|        CodefyUI Installer (Windows)  |"
Write-Host "+======================================+"
Write-Host "  Install dir: $InstallDir"

# ── git ───────────────────────────────────────────────────────────────────────
Step "git"
if (-not (Test-Cmd git)) {
    Warn "Not installed, installing via winget..."
    Install-Winget 'Git.Git' 'Git'
}
Ok (git --version)

# ── uv ────────────────────────────────────────────────────────────────────────
# Use uv to manage Python — users don't need any pre-existing Python install.
Step "uv"
if (-not (Test-Cmd uv)) {
    Warn "Not installed, running standalone installer..."
    Invoke-WebRequest -UseBasicParsing -Uri 'https://astral.sh/uv/install.ps1' | Invoke-Expression
    Refresh-Path
    if (-not (Test-Cmd uv)) { Die "uv not found on PATH after install. Open a new shell and re-run." }
}
Ok "uv $(uv --version)"

# ── Python 3 (provided by uv) ─────────────────────────────────────────────────
Step "Python 3"
uv python install 3.11
if ($LASTEXITCODE -ne 0) { Die "uv python install 3.11 failed" }
$PythonCmd = (uv python find 3.11).Trim()
if (-not (Test-Path $PythonCmd)) { Die "uv python find returned invalid path: $PythonCmd" }
Ok "$(& $PythonCmd --version) ($PythonCmd)"

# ── pnpm ──────────────────────────────────────────────────────────────────────
Step "pnpm"
if (-not (Test-Cmd pnpm)) {
    Warn "Not installed, running standalone installer..."
    Invoke-WebRequest -UseBasicParsing -Uri 'https://get.pnpm.io/install.ps1' | Invoke-Expression
    # pnpm installer sets PNPM_HOME in the user env; pull it into this session
    $PnpmHome = [System.Environment]::GetEnvironmentVariable('PNPM_HOME', 'User')
    if (-not $PnpmHome) { $PnpmHome = Join-Path $env:LOCALAPPDATA 'pnpm' }
    $env:PNPM_HOME = $PnpmHome
    $env:Path = "$PnpmHome;$env:Path"
    if (-not (Test-Cmd pnpm)) { Die "pnpm not found on PATH after install. Open a new shell and re-run." }
}
Ok "pnpm $(pnpm --version)"

# ── Node.js (let pnpm manage it) ──────────────────────────────────────────────
# Require Node 24+; older versions are upgraded via pnpm-managed runtime.
Step "Node.js"
$NodeMin = 24
$NodeOk = $false
if (Test-Cmd node) {
    $currentMajor = ((node --version) -replace '^v','' -split '\.')[0]
    if ($currentMajor -match '^\d+$' -and [int]$currentMajor -ge $NodeMin) {
        $NodeOk = $true
    }
}
if (-not $NodeOk) {
    Warn "Not installed or version < $NodeMin, installing Node $NodeMin via 'pnpm env use --global $NodeMin'..."
    pnpm env use --global $NodeMin
    if ($LASTEXITCODE -ne 0) { Die "pnpm env use --global $NodeMin failed" }
    Refresh-Path
    if (-not (Test-Cmd node)) { Die "node not found on PATH after install. Open a new shell and re-run." }
}
Ok "Node.js $(node --version)"

# ── Clone / Update ────────────────────────────────────────────────────────────
Step "Downloading CodefyUI"
if (Test-Path (Join-Path $InstallDir '.git')) {
    Warn "Directory exists, updating..."
    git -C $InstallDir pull --ff-only
    if ($LASTEXITCODE -ne 0) { Die "git pull failed" }
    Ok "Updated"
} else {
    New-Item -ItemType Directory -Path (Split-Path -Parent $InstallDir) -Force | Out-Null
    git clone --depth 1 $Repo $InstallDir
    if ($LASTEXITCODE -ne 0) { Die "git clone failed" }
    Ok "Clone complete"
}

# ── Install project deps ──────────────────────────────────────────────────────
Step "Installing project dependencies"
Set-Location $InstallDir
& $PythonCmd scripts\dev.py install
if ($LASTEXITCODE -ne 0) { Die "scripts\dev.py install failed" }

# ── Install cdui launcher to PATH ─────────────────────────────────────────────
# Write a small forwarding stub at %USERPROFILE%\.local\bin\cdui.cmd (uv already
# adds this dir to user PATH; we add it defensively in case uv wasn't used).
Step "Installing cdui launcher to PATH"
$LauncherDir = Join-Path $env:USERPROFILE '.local\bin'
$Launcher = Join-Path $LauncherDir 'cdui.cmd'
New-Item -ItemType Directory -Path $LauncherDir -Force | Out-Null
$stub = @"
@echo off
rem CodefyUI launcher stub — forwards to the install at $InstallDir.
call "$InstallDir\cdui.cmd" %*
"@
Set-Content -Path $Launcher -Value $stub -Encoding ASCII
Ok "cdui -> $Launcher"

# Ensure LauncherDir is on user PATH for future shells
$userPath = [System.Environment]::GetEnvironmentVariable('Path', 'User')
if ($null -eq $userPath) { $userPath = '' }
$pathEntries = $userPath -split ';' | Where-Object { $_ }
if ($pathEntries -notcontains $LauncherDir) {
    $newUserPath = if ($userPath) { "$userPath;$LauncherDir" } else { $LauncherDir }
    [System.Environment]::SetEnvironmentVariable('Path', $newUserPath, 'User')
    Ok "Added $LauncherDir to user PATH"
}

# ── Done ──────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "+======================================+" -ForegroundColor Green
Write-Host "|         Installation complete!       |" -ForegroundColor Green
Write-Host "+======================================+" -ForegroundColor Green
Write-Host ""
Write-Host "  Restart PowerShell to pick up PATH, then:"
Write-Host "    cdui dev            # from any directory"
Write-Host ""
Write-Host "  Or from the current shell using the absolute path:"
Write-Host "    $InstallDir\cdui.cmd dev"
Write-Host ""
Write-Host "  Other commands: cdui update | stop | test | clean | uninstall"
Write-Host ""
