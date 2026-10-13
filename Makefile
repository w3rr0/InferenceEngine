all: build run

build: build-engine build-server

run: run-engine run-server

clean:
	cargo clean
	find . -path "./.venv" -prune -o \( -name "*.so" -o -name "*.pyd" -o -name "*.pyi" -o -name "__pycache__" \) -exec rm -rf {} +

format:
	uvx ruff format

check:
	uvx ruff check
	uvx typos
	cargo check

build-engine:
	uv run maturin develop --generate-stubs

build-server:
	cargo build -p server

run-engine:
	uv run inference_engine

run-server:
	cargo run -p server