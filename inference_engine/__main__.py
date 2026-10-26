from inference_engine.runner import EngineRunner
from inference_engine.server import Options, start

def main():
    runner = EngineRunner(total_blocks=128, block_size=16, max_batch_size=4)

    options = Options(
        http_addr = "0.0.0.0:1000",
        grpc_addr = "0.0.0.0:1001",
    )

    start(options)

    print("[Main] Waiting for HTTP requests", flush=True)

    runner.run_forever()

if __name__ == "__main__":
    main()