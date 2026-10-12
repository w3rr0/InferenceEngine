from inference_engine.runner import EngineRunner

def main():
    runner = EngineRunner(total_blocks=128, block_size=16, max_batch_size=4)

    runner.add_request("The capital of France is", max_tokens=20)
    runner.add_request("Once upon a time in a galaxy", max_tokens=100)
    runner.add_request("List the planets in our solar system:", max_tokens=150)

    print("[Main] Starting processing queue")

    runner.run()

if __name__ == "__main__":
    main()