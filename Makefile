all: build run

build:
	uv run maturin develop

run:
	uv run inference_engine