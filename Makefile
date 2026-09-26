all: build run

build:
	uv run maturin develop --generate-stubs

run:
	uv run inference_engine

clean:
	cargo clean
	find . -path "./.venv" -prune -o \( -name "*.so" -o -name "*.pyd" -o -name "*.pyi" -o -name "__pycache__" \) -exec rm -rf {} +