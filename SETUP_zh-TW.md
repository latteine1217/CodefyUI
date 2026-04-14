# 環境設定指南

[![en](https://img.shields.io/badge/lang-English-red)](./SETUP.md)

使用 [uv](https://github.com/astral-sh/uv) 搭建 CodefyUI 完整環境的指南，涵蓋 CPU、NVIDIA CUDA、Apple Silicon (MPS) 與 AMD 選項。

## 前置需求

確認已安裝 uv：

```bash
uv --version
```

若未安裝：

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# 或使用 pip
pip install uv
```

確認已安裝 pnpm：

```bash
pnpm --version
```

若未安裝：

```bash
# Windows (PowerShell)
iwr https://get.pnpm.io/install.ps1 -useb | iex

# macOS / Linux
curl -fsSL https://get.pnpm.io/install.sh | sh -
```

接著請 pnpm 安裝 Node.js runtime（`pnpm install` 後執行專案 script 會用到）：

```bash
pnpm env use --global lts
```

安裝完後請重新開啟 terminal，讓 PATH 生效，然後驗證：

```bash
node -v
```

## 基本設定（所有人都需要）

```bash
cd backend

# 建立虛擬環境（Python 3.10+）
uv venv --python 3.11

# 啟用虛擬環境
# Windows：
.venv\Scripts\activate
# macOS / Linux：
source .venv/bin/activate

# 安裝核心依賴 + 測試工具
uv pip install -e ".[dev]"
```

此時後端已可啟動，但執行 graph 時會出現 `No module named 'torch'` 錯誤，需要先安裝 PyTorch。

## 安裝 PyTorch

預設安裝 —— 全平台通用：

```bash
uv pip install torch torchvision
uv pip install gymnasium safetensors
```

這樣就夠跑這個應用、測試模型。macOS 會拿到支援 MPS 的版本，Linux/Windows 會拿到 PyPI 預設版本。除非你需要特定 GPU 設定，否則直接跳到 **啟動後端與前端**。

<details>
<summary><b>GPU 加速（NVIDIA CUDA / AMD / MPS 驗證）</b> —— 點擊展開</summary>

只有在需要特定 CUDA 版本、AMD ROCm/DirectML、或想驗證 GPU 偵測時才需要展開。

### NVIDIA CUDA（特定版本）

先確認你的 CUDA 版本：

```bash
nvidia-smi
```

看右上角 `CUDA Version:` 欄位，然後重裝對應的 wheel：

```bash
uv pip uninstall torch torchvision

# CUDA 12.4（RTX 40 系列 + 最新驅動）
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# CUDA 12.1（RTX 30 / 40 系列）
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8（GTX 10 / RTX 20 系列，或舊驅動）
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

驗證 CUDA 可用：

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

### Apple Silicon (MPS) 驗證

M1/M2/M3/M4 Mac 上預設安裝就已經使用 Metal Performance Shaders 後端。驗證：

```bash
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
```

### AMD 顯卡

AMD 的支援度高度取決於你的作業系統。

#### Linux + AMD（ROCm，官方支援）

```bash
uv pip uninstall torch torchvision
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.1
```

驗證：

```bash
python -c "import torch; print('CUDA (ROCm):', torch.cuda.is_available())"
```

註：ROCm 環境下 `torch.cuda.is_available()` 會回傳 True，因為 ROCm 對外以相容於 CUDA 的介面呈現。

#### Windows + AMD（支援有限）

PyTorch 官方沒有提供 Windows ROCm 版本。可行選項：

**(a) DirectML** — 可使用 AMD 顯卡，但效能較差，而且節點程式碼預設使用 `cuda`/`cpu`，使用時需要改程式：

```bash
uv pip install torch-directml
```

**(b) CPU 模式** — 上方的預設安裝已經可用，建議 Windows + AMD 用戶採用，適合學習與原型開發。

</details>

## 啟動後端與前端

**後端（終端機 1）：**

```bash
cd backend
# 若尚未啟用虛擬環境
.venv\Scripts\activate       # Windows
source .venv/bin/activate    # macOS / Linux

uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

**前端（終端機 2）：**

```bash
cd frontend
pnpm install
pnpm dev
```

開啟瀏覽器至 [http://localhost:5173](http://localhost:5173)。

## 驗證環境是否正確

1. 後端健康檢查：
   ```bash
   curl http://127.0.0.1:8000/api/health
   ```
   應回傳 `{"status":"ok","nodes_loaded":62,"presets_loaded":3}`。

2. 驗證裝置偵測：
   ```bash
   curl -s http://127.0.0.1:8000/api/nodes/TrainingLoop | python -c "import sys,json; d=json.load(sys.stdin); print([p['options'] for p in d['params'] if p['name']=='device'][0])"
   ```
   應列出你機器上可用的裝置，例如 NVIDIA 環境會顯示 `['cpu', 'cuda']`，未安裝 PyTorch 時只會顯示 `['cpu']`。

3. 開啟前端，載入 **Train CNN on MNIST** 範例並點擊 **執行**。應該會在下方面板看到訓練進度。

## 疑難排解

### 從 CPU 切換到 CUDA（或反向切換）

```bash
uv pip uninstall torch torchvision
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### `uv pip install -e ".[ml]"` 裝到錯的 PyTorch 版本

`pyproject.toml` 中的 `[ml]` 選項群組沒有指定 index URL，所以 uv 會安裝 PyPI 的預設版本——通常 Windows 上是 CPU 版，或是版本不一定符合你的 CUDA runtime。請務必使用上方 **GPU 加速** 區塊裡對應硬體的 `--index-url` 指令。

### CUDA 版本不匹配

若 `torch.cuda.is_available()` 在有 NVIDIA 顯卡時仍回傳 False：
1. 執行 `nvidia-smi` 確認驅動版本
2. 確認安裝的 CUDA PyTorch wheel 與驅動匹配（例如不要在只支援 CUDA 11.8 的驅動上安裝 `cu124`）
3. 若需要請更新 NVIDIA 驅動

### UI 的 device 下拉選單沒有顯示 CUDA

前端會從後端動態讀取可用裝置。若你的 GPU 沒有出現：
1. 確認 PyTorch 能看見它：`python -c "import torch; print(torch.cuda.is_available())"`
2. 點擊工具列的 **重新載入節點** 按鈕
3. 重新整理頁面

### `pnpm: command not found`

請參考上方 [前置需求](#前置需求) 章節的 pnpm 獨立安裝方式。

## 執行測試

```bash
cd backend
.venv\Scripts\activate       # 或 source .venv/bin/activate
pytest tests/ -v
```
