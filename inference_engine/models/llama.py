import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaConfig
from inference_engine.attention.flex import PagedKVCache, paged_attention


class RotaryEmbedding(nn.Module):
    """Native RoPE for flattened 2D tensors [total_tokens, num_heads, head_dim]."""

    def __init__(self, head_dim: int, max_position: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        t = torch.arange(max_position, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos = self.cos_cached[positions].unsqueeze(1).to(q.dtype)
        sin = self.sin_cached[positions].unsqueeze(1).to(q.dtype)

        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed


class LlamaMLP(nn.Module):
    """SwiGLU MLP layer compatible with .safetensors weights."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // self.num_heads

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryEmbedding,
        positions: torch.Tensor,
        kv_cache: PagedKVCache,
        block_tables: list[list[int]],
        seq_lens: list[int],
        is_prefill: bool,
    ) -> torch.Tensor:
        q = self.q_proj(x).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(-1, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(-1, self.num_kv_heads, self.head_dim)

        q, k = rope(q, k, positions)

        attn_out = paged_attention(
            q, k, v, kv_cache, self.layer_idx, block_tables, seq_lens, is_prefill
        )
        return self.o_proj(attn_out.reshape(-1, self.num_heads * self.head_dim))


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.self_attn = LlamaAttention(config, layer_idx)
        self.mlp = LlamaMLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryEmbedding,
        positions: torch.Tensor,
        kv_cache: PagedKVCache,
        block_tables: list[list[int]],
        seq_lens: list[int],
        is_prefill: bool,
    ) -> torch.Tensor:
        x = x + self.self_attn(
            self.input_layernorm(x), rope, positions, kv_cache, block_tables, seq_lens, is_prefill
        )
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class LlamaForCausalLM(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.rope = RotaryEmbedding(
            head_dim=config.hidden_size // config.num_attention_heads,
            max_position=config.max_position_embeddings,
            base=getattr(config, "rope_theta", 10000.0),
        )

    def forward(
        self,
        flat_tokens: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: PagedKVCache,
        block_tables: list[list[int]],
        seq_lens: list[int],
        is_prefill: bool,
    ) -> torch.Tensor:
        x = self.embed_tokens(flat_tokens)
        for layer in self.layers:
            x = layer(x, self.rope, positions, kv_cache, block_tables, seq_lens, is_prefill)
        x = self.norm(x)

        if is_prefill:
            last_indices = torch.tensor(seq_lens, device=x.device).cumsum(0) - 1
            x = x[last_indices]

        return self.lm_head(x)