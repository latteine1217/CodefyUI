# Setup Guide

[![zh-TW](https://img.shields.io/badge/語言-繁體中文-blue)](./SETUP_zh-TW.md)

Full environment setup for CodefyUI using [uv](https://github.com/astral-sh/uv), with CPU, NVIDIA CUDA, Apple Silicon (MPS), and AMD options.

## Prerequisites

Verify uv is installed:

```bash
uv --version
```

Install uv if needed:

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or via pip
pip install uv
```

Verify pnpm is installed:

```bash
pnpm --version
```

Install pnpm if needed:

```bash
# Windows (PowerShell)
iwr https://get.pnpm.io/install.ps1 -useb | iex

# macOS / Linux
curl -fsSL https://get.pnpm.io/install.sh | sh -
```

Then let pnpm install a Node.js runtime (required before `pnpm install` can run project scripts):

```bash
pnpm env use --global lts
```

Restart your terminal so the updated PATH takes effect, then verify:

```bash
node -v
```

## Base Setup (Required for Everyone)

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

The backend will start at this point, but executing graphs will fail with `No module named 'torch'` until PyTorch is installed.

## Install PyTorch

Default install — works on every platform:

```bash
uv pip install torch torchvision
uv pip install gymnasium safetensors
```

This is enough to run the app and test models. macOS gets an MPS-capable build, Linux/Windows get the default PyPI build. Skip to **Start the Backend and Frontend** unless you need a specific GPU configuration.

<details>
<summary><b>GPU Acceleration (NVIDIA CUDA / AMD / MPS verification)</b> — click to expand</summary>

Expand this only if you need a specific CUDA version, AMD ROCm/DirectML, or want to verify GPU detection.

### NVIDIA CUDA (specific version)

First, check your installed CUDA version:

```bash
nvidia-smi
```

Look at the `CUDA Version:` field in the top-right. Then reinstall the matching wheel:

```bash
uv pip uninstall torch torchvision

# CUDA 12.4 (RTX 40 series, latest drivers)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# CUDA 12.1 (RTX 30 / 40 series)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8 (GTX 10 / RTX 20 series, older drivers)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

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
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.1
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
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### `uv pip install -e ".[ml]"` installs the wrong PyTorch version

The `[ml]` optional group in `pyproject.toml` does NOT specify an index URL, so uv installs whatever PyPI has as default — usually the CPU build on Windows, or a version that might not match your CUDA runtime. Always use the explicit `--index-url` command from the **GPU Acceleration** section above.

### CUDA version mismatch

If `torch.cuda.is_available()` returns False even with an NVIDIA GPU:
1. Run `nvidia-smi` to confirm the driver version
2. Make sure you installed the matching CUDA PyTorch wheel (e.g. don't install `cu124` on a driver that only supports up to CUDA 11.8)
3. Update your NVIDIA driver if needed

### Device dropdown in the UI doesn't show CUDA

The frontend dynamically reads the available devices from the backend. If you don't see your GPU in the dropdown:
1. Confirm PyTorch can see it: `python -c "import torch; print(torch.cuda.is_available())"`
2. Reload node definitions via the toolbar **Reload Nodes** button
3. Reload the page

### `pnpm: command not found`

See the [Prerequisites](#prerequisites) section above for the pnpm standalone installer.

## Running Tests

```bash
cd backend
.venv\Scripts\activate       # or source .venv/bin/activate
pytest tests/ -v
```
