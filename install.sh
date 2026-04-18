#!/usr/bin/env bash
# CodefyUI 一鍵安裝腳本
# 用法：curl -fsSL https://raw.githubusercontent.com/treeleaves30760/CodefyUI/main/install.sh | bash
set -euo pipefail

# ── 顏色 ──────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'

REPO="https://github.com/treeleaves30760/CodefyUI.git"
INSTALL_DIR="${CODEFYUI_DIR:-$HOME/CodefyUI}"

step() { echo -e "\n${BLUE}==>${NC} ${BOLD}$*${NC}"; }
ok()   { echo -e "  ${GREEN}✓${NC} $*"; }
warn() { echo -e "  ${YELLOW}!${NC} $*"; }
die()  { echo -e "\n${RED}✗ 錯誤：$*${NC}" >&2; exit 1; }

# root 不需要 sudo
SUDO=""
[[ "$(id -u)" != "0" ]] && SUDO="sudo"

# ── OS 偵測 ───────────────────────────────────────────────────────────
if [[ "$OSTYPE" == "darwin"* ]]; then
  OS="macos"
elif [[ -f /etc/debian_version ]]; then
  OS="debian"
elif [[ -f /etc/redhat-release ]]; then
  OS="redhat"
else
  OS="unknown"
fi

# ── 套件安裝 helper ───────────────────────────────────────────────────
pkg_install() {
  case "$OS" in
    macos)
      if ! command -v brew &>/dev/null; then
        warn "安裝 Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
      fi
      brew install "$@" ;;
    debian)
      $SUDO apt-get update -qq
      $SUDO apt-get install -y "$@" ;;
    redhat)
      $SUDO yum install -y "$@" ;;
    *)
      die "不支援的作業系統，請手動安裝：$*" ;;
  esac
}

# ══════════════════════════════════════════════════════════════════════
echo ""
echo -e "${BOLD}╔══════════════════════════════════════╗${NC}"
echo -e "${BOLD}║        CodefyUI  安裝程式            ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════╝${NC}"
echo -e "  安裝目錄：${BOLD}$INSTALL_DIR${NC}"

# ── git ───────────────────────────────────────────────────────────────
step "git"
if ! command -v git &>/dev/null; then
  warn "未安裝，正在安裝..."
  pkg_install git
fi
ok "$(git --version)"

# ── uv ────────────────────────────────────────────────────────────────
# 用 uv 管理 Python —— 使用者不需要預先安裝任何 Python。
step "uv"
if ! command -v uv &>/dev/null; then
  warn "未安裝，正在安裝..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # uv 安裝器會寫入 shell rc 檔，但不更新當前 session 的 PATH
  export PATH="$HOME/.local/bin:$PATH"
fi
ok "$(uv --version)"

# ── Python 3（由 uv 提供）─────────────────────────────────────────────
step "Python 3"
uv python install 3.11
PYTHON="$(uv python find 3.11)"
[[ -x "$PYTHON" ]] || die "uv python find 回傳無效路徑：$PYTHON"
ok "$("$PYTHON" --version) ($PYTHON)"

# ── Node.js ───────────────────────────────────────────────────────────
step "Node.js"
if ! command -v node &>/dev/null; then
  warn "未安裝，透過 nvm 安裝 LTS..."
  export NVM_DIR="$HOME/.nvm"
  curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
  # 在目前 shell 載入 nvm（不依賴 .bashrc）
  [[ -s "$NVM_DIR/nvm.sh" ]] && source "$NVM_DIR/nvm.sh"
  nvm install --lts --no-progress
  nvm use --lts
fi
ok "Node.js $(node --version)"

# ── pnpm ──────────────────────────────────────────────────────────────
step "pnpm"
if ! command -v pnpm &>/dev/null; then
  warn "未安裝，正在安裝..."
  # 優先用 npm（已隨 Node.js 安裝）
  if command -v npm &>/dev/null; then
    npm install -g pnpm --silent
  else
    curl -fsSL https://get.pnpm.io/install.sh | sh -
    export PNPM_HOME="$HOME/.local/share/pnpm"
    export PATH="$PNPM_HOME:$PATH"
  fi
fi
ok "pnpm $(pnpm --version)"

# ── Clone / Update ────────────────────────────────────────────────────
step "下載 CodefyUI"
if [[ -d "$INSTALL_DIR/.git" ]]; then
  warn "目錄已存在，執行更新..."
  git -C "$INSTALL_DIR" pull --ff-only
  ok "已更新至最新版本"
else
  git clone --depth 1 "$REPO" "$INSTALL_DIR"
  ok "Clone 完成"
fi

# ── 安裝依賴 ──────────────────────────────────────────────────────────
step "安裝專案依賴"
cd "$INSTALL_DIR"
"$PYTHON" scripts/dev.py install

# ── 安裝 cdui 到 PATH ─────────────────────────────────────────────────
# 寫一支小的 forwarding stub 到 ~/.local/bin/cdui (uv 已經把這個目錄加到 PATH)。
step "安裝 cdui 到 PATH"
LAUNCHER_DIR="$HOME/.local/bin"
LAUNCHER="$LAUNCHER_DIR/cdui"
mkdir -p "$LAUNCHER_DIR"
cat > "$LAUNCHER" <<STUB
#!/usr/bin/env bash
# CodefyUI launcher stub — forwards to the install at $INSTALL_DIR.
exec "$INSTALL_DIR/cdui" "\$@"
STUB
chmod +x "$LAUNCHER"
ok "cdui → $LAUNCHER"

# ── 完成 ──────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}╔══════════════════════════════════════╗${NC}"
echo -e "${GREEN}${BOLD}║          安裝完成！                  ║${NC}"
echo -e "${GREEN}${BOLD}╚══════════════════════════════════════╝${NC}"
echo ""
echo -e "  ${BOLD}重新開啟 terminal${NC} 讓 PATH 生效，然後："
echo -e "    ${BOLD}cdui dev${NC}        # 在任何目錄都可用"
echo ""
echo -e "  或在當前 shell 直接用絕對路徑："
echo -e "    ${BOLD}$INSTALL_DIR/cdui dev${NC}"
echo ""
echo -e "  其他指令：${BOLD}cdui update | stop | test | clean | uninstall${NC}"
echo ""
