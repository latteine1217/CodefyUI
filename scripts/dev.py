#!/usr/bin/env python3
"""Cross-platform task runner for CodefyUI.

Usage:
    uv run python scripts/dev.py <command>

Commands:
    install   安裝所有依賴（backend + frontend）
    dev       啟動開發伺服器（Ctrl+C 停止）
    stop      停止所有服務
    test      執行 backend 測試
    clean     移除虛擬環境與 node_modules
"""

import shutil
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT / "backend"
FRONTEND_DIR = ROOT / "frontend"
VENV = BACKEND_DIR / ".venv"
VENV_BIN = VENV / ("Scripts" if sys.platform == "win32" else "bin")


# ── Helpers ───────────────────────────────────────────────────────────────────

def run(cmd: list, cwd: Path = ROOT) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def _stream(proc: subprocess.Popen, prefix: str) -> None:
    assert proc.stdout is not None
    for raw in iter(proc.stdout.readline, b""):
        print(f"{prefix} {raw.decode(errors='replace').rstrip()}", flush=True)


# ── Commands ──────────────────────────────────────────────────────────────────

def install() -> None:
    print("=== Backend: 建立虛擬環境 ===")
    run(["uv", "venv", "--python", "3.11"], cwd=BACKEND_DIR)

    print("=== Backend: 安裝依賴 ===")
    run(["uv", "pip", "install", "-e", ".[dev]"], cwd=BACKEND_DIR)

    print("=== Backend: 安裝 PyTorch ===")
    run(["uv", "pip", "install", "torch", "torchvision", "gymnasium", "safetensors"],
        cwd=BACKEND_DIR)

    print("=== Frontend: 安裝依賴 ===")
    run(["pnpm", "install"], cwd=FRONTEND_DIR)

    print("=== 安裝完成 ===")


def dev() -> None:
    uvicorn = str(VENV_BIN / "uvicorn")
    backend_cmd = [uvicorn, "app.main:app", "--reload"]
    frontend_cmd = ["pnpm", "dev"]

    # Windows 需要 shell=True 才能找到 pnpm（非 .exe 的指令）
    shell = sys.platform == "win32"

    print("=== 啟動 CodefyUI（Ctrl+C 停止）===")
    print("    backend  → http://localhost:8000")
    print("    frontend → http://localhost:5173")
    print("")

    backend = subprocess.Popen(
        backend_cmd,
        cwd=BACKEND_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=shell,
    )
    frontend = subprocess.Popen(
        frontend_cmd,
        cwd=FRONTEND_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=shell,
    )

    threading.Thread(target=_stream, args=(backend, "[backend] "), daemon=True).start()
    threading.Thread(target=_stream, args=(frontend, "[frontend]"), daemon=True).start()

    try:
        backend.wait()
        frontend.wait()
    except KeyboardInterrupt:
        print("\n=== 停止服務 ===")
        backend.terminate()
        frontend.terminate()
        backend.wait()
        frontend.wait()


def stop() -> None:
    print("=== 停止所有服務 ===")
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/IM", "uvicorn.exe"], capture_output=True)
        subprocess.run(["taskkill", "/F", "/FI", "WINDOWTITLE eq vite*"], capture_output=True)
    else:
        subprocess.run(["pkill", "-f", "uvicorn app.main:app"], capture_output=True)
        subprocess.run(["pkill", "-f", "vite"], capture_output=True)
    print("=== 完成 ===")


def test() -> None:
    pytest = str(VENV_BIN / "pytest")
    run([pytest], cwd=BACKEND_DIR)


def clean() -> None:
    print("=== 清除虛擬環境與 node_modules ===")
    shutil.rmtree(VENV, ignore_errors=True)
    shutil.rmtree(FRONTEND_DIR / "node_modules", ignore_errors=True)
    print("=== 完成 ===")


# ── Entry point ───────────────────────────────────────────────────────────────

COMMANDS = {
    "install": install,
    "dev": dev,
    "stop": stop,
    "test": test,
    "clean": clean,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(1)
    COMMANDS[sys.argv[1]]()
