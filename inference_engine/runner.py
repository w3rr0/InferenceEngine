import torch
import time
from transformers import PreTrainedTokenizerFast
from uuid import UUID

from inference_engine.core import Engine
from inference_engine.server import poll_request, StreamCallback
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

        self.callbacks: dict[UUID, StreamCallback] = {}

    def add_request(self, request_id: UUID, prompt_text: str, callback: StreamCallback, max_tokens: int = 512):
        tokens = self.tokenizer.encode(prompt_text)
        self.callbacks[request_id] = callback
        self.rust_engine.add_request(
            id=request_id,
            prompt=tokens,
            max_tokens=max_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
        )

    @torch.inference_mode()
    def run_one_step(self):
        batch = self.rust_engine.get_next()
        if batch is None:
            print("[Runner] All request stop because of lack of VRAM")
            return

        req_ids, input_tokens, block_tables, seq_lens, is_prefill = batch

        # Convert block tables to device tensors
        block_tables = [
            torch.tensor(table, dtype=torch.long, device=self.device)
            for table in block_tables
        ]

        flat_tokens_list = []
        positions_list = []

        for tokens, seq_len in zip(input_tokens, seq_lens):
            flat_tokens_list.extend(tokens)
            if is_prefill:
                positions_list.extend(range(seq_len))
            else:
                positions_list.append(seq_len - 1)

        flat_tokens = torch.tensor(
            flat_tokens_list, dtype=torch.long, device=self.device
        )
        positions = torch.tensor(positions_list, dtype=torch.long, device=self.device)

        logits = self.model(
            flat_tokens, positions, self.kv_cache, block_tables, seq_lens, is_prefill
        )

        temperature = 0.7
        top_p = 0.9

        next_tokens = self.sample_next_token(logits, temperature, top_p)

        result = self.rust_engine.step(req_ids, next_tokens)

        # Show live
        if not is_prefill:
            for req_id, token in zip(req_ids, next_tokens):
                is_finished = req_id in result.finished
                self.callbacks[req_id].send_token(
                    text = self.tokenizer.decode([token]),
                    is_finished = is_finished
                )
                if is_finished:
                    del self.callbacks[req_id]

    @torch.inference_mode()
    def sample_next_token(
            self,
            logits: torch.Tensor,
            temperature: float,
            top_p: float
    ) -> list[int]:
        if temperature <= 0:
            return logits.argmax(dim=-1).tolist()

        logits = logits / temperature
        sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        remove = cumulative_probs > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False

        probs = sorted_probs.masked_fill(remove, 0.0)
        sampled = torch.multinomial(probs, num_samples=1)

        return sorted_indices.gather(1, sampled).squeeze(1).tolist()

    def accept_request(self):
        while True:
            request = poll_request()

            if request is None:
                break

            request_id, prompt, max_tokens, callback = request

            print(
                f"[Runner] Received HTTP request: "
                f"id={request_id}, prompt={prompt!r}, max_tokens={max_tokens}",
                flush=True,
            )

            self.add_request(
                request_id=request_id,
                prompt_text=prompt,
                max_tokens=max_tokens,
                callback=callback,
            )

    def run_forever(self):
        while True:
            self.accept_request()

            if not self.rust_engine.has_unfinished_requests():
                time.sleep(0.01)
                continue

            self.run_one_step()