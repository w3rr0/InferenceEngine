import torch
from transformers import PreTrainedTokenizerFast

from inference_engine.core import Engine
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
        self.rust_engine = Engine(total_blocks, block_size, max_batch_size)

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

    def add_request(self, prompt_text: str, max_tokens: int = 512):
        tokens = self.tokenizer.encode(prompt_text)
        self.rust_engine.add_request(
            prompt=tokens,
            max_tokens=max_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
        )

    @torch.inference_mode()
    def run(self):
        step = 0

        while self.rust_engine.has_unfinished_requests():
            batch = self.rust_engine.get_next()
            if batch is None:
                print("[Runner] All request stop because of lack of VRAM")
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

            logits = self.model(
                flat_tokens, positions, self.kv_cache, block_tables, seq_lens, is_prefill
            )

            temperature = 0.7
            top_p = 0.9

            if temperature <= 0:
                next_tokens = torch.argmax(logits, dim=-1).tolist()
            else:
                scaled_logits = logits / temperature
                probs = torch.softmax(scaled_logits, dim=-1)

                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                scaled_logits[indices_to_remove] = float('-inf')

                filtered_probs = torch.softmax(scaled_logits, dim=-1)
                next_tokens = torch.multinomial(filtered_probs, num_samples=1).squeeze(-1).tolist()

            # Show live
            if not is_prefill:
                for req_id, token in zip(req_ids, next_tokens):
                    word = self.tokenizer.decode([token])
                    print(f"[Req {req_id}]: {word}", flush=True)

            self.rust_engine.step(req_ids, next_tokens)
            step += 1

        print(f"\n[Runner] Finished after {step} iterations")