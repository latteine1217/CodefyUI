# Setup Guide

[![zh-TW](https://img.shields.io/badge/語言-繁體中文-blue)](./SETUP_zh-TW.md)

Two ways to get CodefyUI running:

1. **[Quick Install](#quick-install)** (all platforms) — one-liner, auto-installs all dependencies. Recommended for most users.
2. **[Dev Install](#dev-install)** (all platforms) — manual [uv](https://github.com/astral-sh/uv) + pnpm setup. Use this for development or contributing.

---

## Quick Install

Automatically installs git, Node.js, pnpm, and uv if missing. Python itself is then provided by uv — you don't need to install Python separately.

```bash
# macOS / Linux
curl -fsSL https://raw.githubusercontent.com/treeleaves30760/CodefyUI/main/install.sh | bash
```

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/treeleaves30760/CodefyUI/main/install.ps1 | iex"
```

By default installs to `~/CodefyUI` (macOS/Linux) or `%USERPROFILE%\CodefyUI` (Windows). Override with the `CODEFYUI_DIR` environment variable.

On Windows, `install.ps1` uses [winget](https://learn.microsoft.com/en-us/windows/package-manager/) to install `git` if missing. `winget` ships with Windows 11 and recent Windows 10 via the "App Installer" Microsoft Store package.

The installer places a `cdui` launcher at `~/.local/bin/cdui` (Windows: `%USERPROFILE%\.local\bin\cdui.cmd`). Restart your terminal, then from any directory:

```bash
cdui dev
```

Available commands: `install` / `update` / `dev` / `stop` / `test` / `clean` / `uninstall`. `cdui` is a tiny launcher that finds a Python (uv-managed or system) and forwards to `dev.py`.

Open [http://localhost:5173](http://localhost:5173). The frontend proxies API/WS to the backend at `:8000`.

> The default PyTorch install works on every platform (CPU / Apple Silicon MPS). For specific NVIDIA CUDA versions, AMD ROCm, or GPU verification, see [GPU Acceleration](#gpu-acceleration).

---

## Dev Install

Manual setup using [uv](https://github.com/astral-sh/uv) and pnpm — works on Windows, macOS, and Linux. Use this for development or contributing.

### 1. Clone the repository

```bash
git clone https://github.com/treeleaves30760/CodefyUI.git
cd CodefyUI
```

### 2. Install uv

```bash
uv --version
```

If not installed:

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or via pip
pip install uv
```

### 3. Install pnpm + Node.js

```bash
pnpm --version
```

If not installed:

```bash
# Windows (PowerShell)
iwr https://get.pnpm.io/install.ps1 -useb | iex

# macOS / Linux
curl -fsSL https://get.pnpm.io/install.sh | sh -
```

Then let pnpm install a Node.js runtime:

```bash
pnpm env use --global lts
```

Restart your terminal so PATH updates take effect, then verify:

```bash
node -v
```

### 4. Backend setup

```bash
cd backend

# Create virtual environment with Python 3.10+
uv venv --python 3.11

# Activate it
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# Install core dependencies + test tools
uv pip install -e ".[dev]"
```

### 5. Install PyTorch

Default install — works on every platform:

```bash
uv pip install torch torchvision
uv pip install gymnasium safetensors
```

This is enough to run the app and test models. macOS gets an MPS-capable build, Linux/Windows get the default PyPI build. Skip to [Start the Backend and Frontend](#start-the-backend-and-frontend) unless you need a specific GPU configuration.

---

## GPU Acceleration

Expand this only if you need a specific CUDA version, AMD ROCm/DirectML, or want to verify GPU detection.

<details>
<summary><b>NVIDIA CUDA / AMD / MPS verification</b> — click to expand</summary>

### NVIDIA CUDA (specific version)

First, check your installed CUDA version:

```bash
nvidia-smi
```

Look at the `CUDA Version:` field in the top-right. Then reinstall the matching wheel. PyTorch currently ships three stable CUDA wheel channels:

```bash
uv pip uninstall torch torchvision

# CUDA 12.8 — required for RTX 50 series (Blackwell, sm_120). Also works on RTX 30/40.
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# CUDA 12.6 — RTX 30 / 40 series, widely compatible default for modern drivers
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# CUDA 11.8 — GTX 10 / RTX 20 series, or older drivers
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

> RTX 5090 / 5080 / 5070 (Blackwell) **require** `cu128` — older wheels lack the `sm_120` kernels and will fail at runtime with `no kernel image is available for execution`.

Verify CUDA is working:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

### Apple Silicon (MPS) verification

The default install already uses the Metal Performance Shaders backend on M1/M2/M3/M4 Macs. Verify:

```bash
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
```

### AMD GPU

AMD support depends heavily on your OS.

#### Linux + AMD (ROCm, officially supported)

```bash
uv pip uninstall torch torchvision
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.3
```

Verify:

```bash
python -c "import torch; print('CUDA (ROCm):', torch.cuda.is_available())"
```

Note: On ROCm, `torch.cuda.is_available()` returns True because ROCm presents itself as a CUDA-compatible backend.

#### Windows + AMD (limited)

PyTorch does NOT ship an official Windows ROCm build. Your options:

**(a) DirectML** — uses the AMD GPU but with reduced performance and requires code changes (the built-in nodes default to `cuda`/`cpu`):

```bash
uv pip install torch-directml
```

**(b) CPU mode** — the default install above already works. Recommended for learning/prototyping on Windows with AMD.

</details>

---

## Start the Backend and Frontend

**Backend (terminal 1):**

```bash
cd backend
# Activate venv if not already
.venv\Scripts\activate       # Windows
source .venv/bin/activate    # macOS / Linux

uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

**Frontend (terminal 2):**

```bash
cd frontend
pnpm install
pnpm dev
```

Open [http://localhost:5173](http://localhost:5173).

Or start both at once from the project root:

```bash
cdui dev                 # if ~/.local/bin is on PATH
./cdui dev               # from the project root
python scripts/dev.py dev
```

## Verify Everything Works

1. Backend health check:
   ```bash
   curl http://127.0.0.1:8000/api/health
   ```
   Should return `{"status":"ok","nodes_loaded":62,"presets_loaded":3}`.

2. Verify device detection:
   ```bash
   curl -s http://127.0.0.1:8000/api/nodes/TrainingLoop | python -c "import sys,json; d=json.load(sys.stdin); print([p['options'] for p in d['params'] if p['name']=='device'][0])"
   ```
   Should print the available devices on your machine, e.g. `['cpu', 'cuda']` for NVIDIA, `['cpu']` if PyTorch isn't installed.

3. Open the frontend, load the **Train CNN on MNIST** example, and click **Run**. You should see training progress appear in the bottom panel.

## Troubleshooting

### Switching from CPU to CUDA (or vice versa)

```bash
uv pip uninstall torch torchvision
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### `uv pip install -e ".[ml]"` installs the wrong PyTorch version

The `[ml]` optional group in `pyproject.toml` does NOT specify an index URL, so uv installs whatever PyPI has as default — usually the CPU build on Windows, or a version that might not match your CUDA runtime. Always use the explicit `--index-url` command from the [GPU Acceleration](#gpu-acceleration) section above.

### CUDA version mismatch

If `torch.cuda.is_available()` returns False even with an NVIDIA GPU:
1. Run `nvidia-smi` to confirm the driver version
2. Make sure you installed the matching CUDA PyTorch wheel (e.g. don't install `cu128` on a driver that only supports up to CUDA 11.8)
3. If you have an RTX 50 series card and see `no kernel image is available for execution`, you're on an older wheel — reinstall with `cu128`
4. Update your NVIDIA driver if needed

### Device dropdown in the UI doesn't show CUDA

The frontend dynamically reads the available devices from the backend. If you don't see your GPU in the dropdown:
1. Confirm PyTorch can see it: `python -c "import torch; print(torch.cuda.is_available())"`
2. Reload node definitions via the toolbar **Reload Nodes** button
3. Reload the page

### `pnpm: command not found`

See [Dev Install → Install pnpm + Node.js](#3-install-pnpm--nodejs) above.

## Running Tests

```bash
cd backend
.venv\Scripts\activate       # or source .venv/bin/activate
pytest tests/ -v
```
