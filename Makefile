RUNNER := uv run python scripts/dev.py

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "CodefyUI"
	@echo ""
	@echo "  make install   安裝所有依賴（backend + frontend）"
	@echo "  make dev       啟動開發伺服器（backend :8000 + frontend :5173）"
	@echo "  make stop      停止所有服務"
	@echo "  make test      執行 backend 測試"
	@echo "  make clean     移除虛擬環境與 node_modules"
	@echo ""
	@echo "Windows 使用者（不需要 make）："
	@echo "  uv run python scripts/dev.py <command>"
	@echo ""

.PHONY: install
install:
	$(RUNNER) install

.PHONY: dev
dev:
	$(RUNNER) dev

.PHONY: stop
stop:
	$(RUNNER) stop

.PHONY: test
test:
	$(RUNNER) test

.PHONY: clean
clean:
	$(RUNNER) clean
