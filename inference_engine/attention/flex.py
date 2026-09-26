import torch
import torch.nn.functional as F


class PagedKVCache:
    """Physical VRAM memory buffer managed virtually by the rust engine."""

    def __init__(
        self,
        num_layers: int,
        total_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.block_size = block_size
        shape = (num_layers, total_blocks, block_size, num_kv_heads, head_dim)
        self.k_cache = torch.zeros(shape, dtype=dtype, device=device)
        self.v_cache = torch.zeros(shape, dtype=dtype, device=device)


def paged_attention(
    q: torch.Tensor,          # [total_tokens, num_heads, head_dim]
    k: torch.Tensor,          # [total_tokens, num_kv_heads, head_dim]
    v: torch.Tensor,          # [total_tokens, num_kv_heads, head_dim]
    kv_cache: PagedKVCache,
    layer_idx: int,
    block_tables: list[list[int]],
    seq_lens: list[int],
    is_prefill: bool,
) -> torch.Tensor:
    """Saves new K/V to blocks received by the rust engine and calculate attention."""
    block_size = kv_cache.block_size
    k_layer = kv_cache.k_cache[layer_idx]
    v_layer = kv_cache.v_cache[layer_idx]

    outputs = []
    token_offset = 0

    for req_idx, seq_len in enumerate(seq_lens):
        block_table = block_tables[req_idx]

        if is_prefill:
            # Prefill phase: whole prompt at once
            q_req = q[token_offset : token_offset + seq_len]
            k_req = k[token_offset : token_offset + seq_len]
            v_req = v[token_offset : token_offset + seq_len]
            token_offset += seq_len

            # Saves prompt tokens to assigned physical blocks
            for pos in range(seq_len):
                block_id = block_table[pos // block_size]
                block_offset = pos % block_size
                k_layer[block_id, block_offset] = k_req[pos]
                v_layer[block_id, block_offset] = v_req[pos]

            # Calculate causal attention for prompt
            # [seq_len, num_heads, head_dim] -> [1, num_heads, seq_len, head_dim]
            q_sdpa = q_req.transpose(0, 1).unsqueeze(0)
            k_sdpa = k_req.transpose(0, 1).unsqueeze(0)
            v_sdpa = v_req.transpose(0, 1).unsqueeze(0)

            out = F.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa, is_causal=True, enable_gqa=True
            )
            # Back to: [seq_len, num_heads, head_dim]
            outputs.append(out.squeeze(0).transpose(0, 1))

        else:
            # Decode phase: every request sends one new token
            q_req = q[token_offset : token_offset + 1]
            k_req = k[token_offset]
            v_req = v[token_offset]
            token_offset += 1

            # Saves a new token to the last position of the request (seq_len - 1)
            last_pos = seq_len - 1
            target_block = block_table[last_pos // block_size]
            target_offset = last_pos % block_size
            k_layer[target_block, target_offset] = k_req
            v_layer[target_block, target_offset] = v_req

            # Gather full K and V history from every request block
            blocks_tensor = torch.tensor(block_table, dtype=torch.long, device=q.device)
            # Extract the blocks and flatten them to [max_tokens_in_blocks, num_kv_heads, head_dim]
            k_history = k_layer[blocks_tensor].view(-1, k.size(1), k.size(2))[:seq_len]
            v_history = v_layer[blocks_tensor].view(-1, v.size(1), v.size(2))[:seq_len]

            q_sdpa = q_req.transpose(0, 1).unsqueeze(0)        # [1, num_heads, 1, head_dim]
            k_sdpa = k_history.transpose(0, 1).unsqueeze(0)    # [1, num_kv_heads, seq_len, head_dim]
            v_sdpa = v_history.transpose(0, 1).unsqueeze(0)

            out = F.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa, is_causal=False, enable_gqa=True
            )
            outputs.append(out.squeeze(0).transpose(0, 1))

    return torch.cat(outputs, dim=0)