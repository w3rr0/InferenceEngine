from inference_engine.runner import EngineRunner

def main():
    runner = EngineRunner(total_blocks=128, block_size=16, max_batch_size=4)

    runner.add_prompt("The capital of France is")
    runner.add_prompt("Once upon a time in a galaxy")

    # TODO: without new block allocation during decoding, the total length (prompt + num_steps) cannot yet exceed block_size num of tokens
    runner.run_steps(num_steps=8)

if __name__ == "__main__":
    main()