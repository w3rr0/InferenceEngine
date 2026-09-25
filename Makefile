all: build run

build:
	uv run maturin develop --generate-stubs

run:
	uv run inference_engine