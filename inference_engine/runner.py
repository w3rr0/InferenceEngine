import torch
from transformers import PreTrainedTokenizerFast
import engine_core

from inference_engine.attention.flex import PagedKVCache
from inference_engine.models.weights import load_llama_model


class EngineRunner:
    def __init__(
        self,
        model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        total_blocks: int = 256,
        block_size: int = 16,
        max_batch_size: int = 4,
    ):
        # Automatic device selection
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.dtype = torch.float16 if self.device.type != "cpu" else torch.float32
        print(f"[Runner] Launching on device: {self.device} ({self.dtype})") # TODO: change to logger

        # Control Plane initialization in rust
        self.rust_engine = engine_core.Engine(total_blocks, block_size, max_batch_size)

        # Loading tokenizer and model
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(model_id)
        self.model = load_llama_model(model_id, self.device, self.dtype)

        # Physical PagedKVCache allocation
        cfg = self.model.config
        self.kv_cache = PagedKVCache(
            num_layers=cfg.num_hidden_layers,
            total_blocks=total_blocks,
            block_size=block_size,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.hidden_size // cfg.num_attention_heads,
            dtype=self.dtype,
            device=self.device,
        )

    def add_prompt(self, prompt_text: str):
        tokens = self.tokenizer.encode(prompt_text)
        self.rust_engine.add_dummy_request(tokens)

    @torch.inference_mode()
    def run_steps(self, num_steps: int = 10):
        outputs: dict[int, list[int]] = {}

        for step_idx in range(num_steps):
            batch = self.rust_engine.get_next()
            if batch is None:
                break

            req_ids, input_tokens, block_tables, seq_lens, is_prefill = batch

            flat_tokens_list = []
            positions_list = []

            for tokens, seq_len in zip(input_tokens, seq_lens):
                flat_tokens_list.extend(tokens)
                if is_prefill:
                    positions_list.extend(range(seq_len))
                else:
                    positions_list.append(seq_len - 1)

            flat_tokens = torch.tensor(flat_tokens_list, dtype=torch.long, device=self.device)
            positions = torch.tensor(positions_list, dtype=torch.long, device=self.device)

            # Passing through a model with PagedAttention
            logits = self.model(
                flat_tokens, positions, self.kv_cache, block_tables, seq_lens, is_prefill
            )

            next_tokens = torch.argmax(logits, dim=-1).tolist()

            # Saving generated tokens for preview and sending them back to Rust
            for req_id, token in zip(req_ids, next_tokens):
                outputs.setdefault(req_id, []).append(token)

            self.rust_engine.step(req_ids, next_tokens)

        for req_id, gen_tokens in outputs.items():
            text = self.tokenizer.decode(gen_tokens)
            print(f"\n[Request {req_id}] Generated: {text!r}")  # TODO: change to logger