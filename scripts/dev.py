#!/usr/bin/env python3
"""CodefyUI 跨平台任務執行器。

用法（建議）：
    cdui <command>                 # 若已透過 install 腳本加到 PATH
    ./cdui <command>               # 從專案根目錄執行
    python scripts/dev.py <command>

指令：
    install     安裝所有依賴（backend + frontend）
    update      拉取最新版本並重新安裝依賴（git pull + install）
    dev         啟動開發伺服器（Ctrl+C 停止）
    stop        停止所有服務
    test        執行 backend 測試
    clean       移除虛擬環境與 node_modules
    uninstall   解除安裝：clean + 移除全域 cdui launcher
"""

import shutil
import subprocess
import sys
import threading
from pathlib import Path

# Force UTF-8 on Windows so we can print non-ASCII (Chinese headings etc.)
# without hitting cp1252 UnicodeEncodeError in CI / default consoles.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

ROOT = Path(__file__).resolve().parent.parent  # dev.py lives in <root>/scripts/
BACKEND_DIR = ROOT / "backend"
FRONTEND_DIR = ROOT / "frontend"
VENV = BACKEND_DIR / ".venv"
VENV_BIN = VENV / ("Scripts" if sys.platform == "win32" else "bin")
VENV_PY = VENV_BIN / ("python.exe" if sys.platform == "win32" else "python")


# ── Bootstrap ─────────────────────────────────────────────────────────────────

def _exec_into_venv_if_available() -> None:
    """Re-exec into backend/.venv's Python when it exists.

    Lets `python dev.py <cmd>` work transparently with any outer interpreter
    (uv-managed, system, or a temp env) — we hand off to the venv's Python so
    subprocess calls run against the installed deps.
    """
    if not VENV_PY.exists():
        return
    try:
        if Path(sys.executable).resolve() == VENV_PY.resolve():
            return
    except OSError:
        return
    import os
    os.execv(str(VENV_PY), [str(VENV_PY)] + sys.argv)


def _ensure_uv() -> None:
    if shutil.which("uv"):
        return
    print("=== uv 未安裝，正在自動安裝 ===")
    if sys.platform == "win32":
        subprocess.run(
            ["powershell", "-c", "irm https://astral.sh/uv/install.ps1 | iex"],
            check=True,
        )
    else:
        subprocess.run(
            "curl -LsSf https://astral.sh/uv/install.sh | sh",
            shell=True,
            check=True,
        )
    # 安裝後重新啟動自身，讓新 PATH 生效
    import os
    os.execv(sys.executable, [sys.executable] + sys.argv)


# ── Helpers ───────────────────────────────────────────────────────────────────

def run(cmd: list, cwd: Path = ROOT) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def _stream(proc: subprocess.Popen, prefix: str) -> None:
    assert proc.stdout is not None
    for raw in iter(proc.stdout.readline, b""):
        print(f"{prefix} {raw.decode(errors='replace').rstrip()}", flush=True)


# ── Commands ──────────────────────────────────────────────────────────────────

def install() -> None:
    if VENV.exists():
        print("=== Backend: 虛擬環境已存在，跳過建立 ===")
    else:
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


def update() -> None:
    """拉取 main branch 的最新版本並重新同步依賴。"""
    if not (ROOT / ".git").exists():
        print("錯誤：此目錄不是 git clone，無法 update", file=sys.stderr)
        sys.exit(1)
    print("=== 拉取最新版本（main）===")
    # Explicit remote/branch so the command works even on a detached HEAD or
    # a branch that doesn't track upstream.
    run(["git", "fetch", "origin", "main"], cwd=ROOT)
    run(["git", "checkout", "main"], cwd=ROOT)
    run(["git", "merge", "--ff-only", "origin/main"], cwd=ROOT)
    install()
    print("=== 更新完成 ===")


def dev() -> None:
    uvicorn = str(VENV_BIN / "uvicorn")
    backend_cmd = [uvicorn, "app.main:app", "--reload"]
    frontend_cmd = ["pnpm", "dev"]

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


def uninstall() -> None:
    """移除 venv、node_modules，以及全域 cdui launcher stub。"""
    clean()
    launcher = (
        Path.home() / ".local" / "bin" / ("cdui.cmd" if sys.platform == "win32" else "cdui")
    )
    if launcher.exists() or launcher.is_symlink():
        try:
            launcher.unlink()
            print(f"=== 已移除 launcher：{launcher} ===")
        except OSError as e:
            print(f"=== 無法移除 launcher {launcher}：{e} ===")
    else:
        print(f"=== 未發現 launcher（{launcher}），跳過 ===")
    print(f"=== 解除安裝完成。若要完全移除，請手動刪除：{ROOT} ===")


# ── Entry point ───────────────────────────────────────────────────────────────

COMMANDS = {
    "install": install,
    "update": update,
    "dev": dev,
    "stop": stop,
    "test": test,
    "clean": clean,
    "uninstall": uninstall,
}

# Commands that mutate or remove the venv must run from the outer interpreter,
# never from the venv's Python (Windows can't delete a running exe; update
# rewrites deps in-place).
_SKIP_VENV_EXEC = {"install", "update", "clean", "uninstall"}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(1)
    if sys.argv[1] not in _SKIP_VENV_EXEC:
        _exec_into_venv_if_available()
    _ensure_uv()
    COMMANDS[sys.argv[1]]()
