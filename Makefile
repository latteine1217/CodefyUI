RUNNER := python dev.py

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
	@echo "或直接（全平台通用）："
	@echo "  python dev.py <command>"
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
