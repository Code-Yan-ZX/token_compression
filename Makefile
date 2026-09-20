.PHONY: setup test lint clean install

PYTHON ?= python3

## 创建虚拟环境并安装依赖
setup:
	$(PYTHON) -m venv .venv
	.venv/bin/pip install -e ".[dev]"

## 安装依赖(无 venv 的情况下)
install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest tests/ -v

lint:
	$(PYTHON) -m ruff check . && $(PYTHON) -m black --check .

fmt:
	$(PYTHON) -m black .

clean:
	rm -rf .venv build dist *.egg-info .pytest_cache __pycache__

## 协议自检
verify:
	$(PYTHON) -m tcbench.cli verify-protocol