import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaMLP, LlamaRotaryEmbedding, apply_rotary_pos_emb
from inference_engine.attention.flex import PagedKVCache, paged_attention


class LlamaAttention(nn.Module):
    def __init__(self, config, layer_idx: int):
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
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        kv_cache: PagedKVCache,
        block_tables: list[list[int]],
        seq_lens: list[int],
        is_prefill: bool,
    ) -> torch.Tensor:
        q = self.q_proj(x).view(1, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(1, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(-1, self.num_kv_heads, self.head_dim)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        q = q.transpose(1, 2).squeeze(0)
        k = k.transpose(1, 2).squeeze(0)

        attn_out = paged_attention(
            q, k, v, kv_cache, self.layer_idx, block_tables, seq_lens, is_prefill
        )
        return self.o_proj(attn_out.reshape(-1, self.num_heads * self.head_dim))


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.self_attn = LlamaAttention(config, layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x, position_embeddings, kv_cache, block_tables, seq_lens, is_prefill):
        x = x + self.self_attn(
            self.input_layernorm(x),
            position_embeddings,
            kv_cache,
            block_tables,
            seq_lens,
            is_prefill,
        )
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class LlamaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

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

        position_embeddings = self.rotary_emb(x.unsqueeze(0), positions.unsqueeze(0))

        for layer in self.layers:
            x = layer(
                x, position_embeddings, kv_cache, block_tables, seq_lens, is_prefill
            )
        x = self.norm(x)

        if is_prefill:
            last_indices = torch.tensor(seq_lens, device=x.device).cumsum(0) - 1
            x = x[last_indices]

        return self.lm_head(x)