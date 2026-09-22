import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb
from torch.nn.attention.flex_attention import flex_attention

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# Global engine state
BLOCK_SIZE = 16
NUM_BLOCKS = 1024

# Global KV Cache buffers
global_k_cache = None
global_v_cache = None

DEVICE = get_device()

def init_global_cache(config, dtype=torch.float16, device="cuda"):
    global global_k_cache, global_v_cache
    num_heads = config.num_key_value_heads
    head_dim = config.hidden_size // config.num_attention_heads

    global_k_cache = torch.zeros(
        (NUM_BLOCKS, num_heads, BLOCK_SIZE, head_dim), dtype=dtype, device=device
    )
    global_v_cache = torch.zeros(
        (NUM_BLOCKS, num_heads, BLOCK_SIZE, head_dim), dtype=dtype, device=device
    )

# Attention layer replacement
def patched_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask = None,
        position_ids = None,
        past_key_value = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position = None,
        position_embeddings = None,
        block_tables: list[list[int]] = None,
        seq_lens: list[int] = None,
        **kwargs,
):
    bsz, q_len, _ = hidden_states.size()

    num_heads = self.config.num_attention_heads
    num_key_value_heads = getattr(self.config, "num_key_value_heads", num_heads)
    head_dim = self.config.hidden_size // num_heads

    # Standard projections Q, K, V
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)

    #  Rotary Positional Embeddings (RoPE) on new input
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    num_key_value_groups = num_heads // num_key_value_heads
    attn_outputs = []

    # Writing new Q/K to the global physical buffer
    if block_tables is not None and seq_lens is not None:
        for i in range(bsz):
            seq_len = seq_lens[i]
            blocks = block_tables[i]

            # Gathering physical keys and values from fragmented memory
            physical_k = global_k_cache[blocks]
            physical_v = global_v_cache[blocks]

            # Flattening and trimming to actual length
            flat_k = physical_k.transpose(1, 2).reshape(1, num_key_value_heads, -1, head_dim)
            flat_v = physical_v.transpose(1, 2).reshape(1, num_key_value_heads, -1, head_dim)

            past_k = flat_k[:, :, :seq_len, :]
            past_v = flat_v[:, :, :seq_len, :]

            if num_key_value_groups > 1:
                past_k = past_k.repeat_interleave(num_key_value_groups, dim=1)
                past_v = past_v.repeat_interleave(num_key_value_groups, dim=1)

            # Calculating attention
            q_i = query_states[i:i + 1]

            out_i = flex_attention(q_i, past_k, past_v)
            attn_outputs.append(out_i)

    # Combining results
    attn_output = torch.cat(attn_outputs, dim=0)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.config.hidden_size)
    attn_output = self.o_proj(attn_output)

    # Dynamic construction of the output structure
    outputs = (attn_output,)
    if output_attentions:
        outputs += (None,)
    if use_cache:
        # Rust engine manages KV cache, so we return `None` for the Python model
        outputs += (None,)

    return outputs

def forward_mps_fallback(
        self,
        hidden_states: torch.Tensor,
        attention_mask = None,
        position_ids = None,
        past_key_value = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position = None,
        position_embeddings = None,
        block_tables: list[list[int]] = None,
        seq_lens: list[int] = None,
        **kwargs,
):
    bsz, q_len, _ = hidden_states.size()

    num_heads = self.config.num_attention_heads
    num_key_value_heads = getattr(self.config, "num_key_value_heads", num_heads)
    head_dim = self.config.hidden_size // num_heads

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    num_key_value_groups = num_heads // num_key_value_heads
    attn_outputs = []

    if block_tables is not None and seq_lens is not None:
        for i in range(bsz):
            seq_len = seq_lens[i]
            blocks = block_tables[i]

            physical_k = global_k_cache[blocks]
            physical_v = global_v_cache[blocks]

            flat_k = physical_k.transpose(1, 2).reshape(1, num_key_value_heads, -1, head_dim)
            flat_v = physical_v.transpose(1, 2).reshape(1, num_key_value_heads, -1, head_dim)

            past_k = flat_k[:, :, :seq_len, :]
            past_v = flat_v[:, :, :seq_len, :]

            if num_key_value_groups > 1:
                past_k = past_k.repeat_interleave(num_key_value_groups, dim=1)
                past_v = past_v.repeat_interleave(num_key_value_groups, dim=1)

            q_i = query_states[i:i + 1]
            out_i = F.scaled_dot_product_attention(q_i, past_k, past_v)
            attn_outputs.append(out_i)

    attn_output = torch.cat(attn_outputs, dim=0)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.config.hidden_size)

    attn_output = self.o_proj(attn_output)

    outputs = (attn_output,)
    if output_attentions:
        outputs += (None,)
    if use_cache:
        outputs += (None,)

    return outputs

# Rust API - FFI
def load_engine():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    print("Loading model")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        # device_map={"": DEVICE}   # Not working with macos, bring back later
        low_cpu_mem_usage=True,     # Delete later
        attn_implementation = "eager",
    )

    model.to(DEVICE)                # Delete later

    # Empty KV cache initialization in VRAM
    init_global_cache(model.config, dtype=torch.float16, device=DEVICE)

    # Replace attention implementation
    if DEVICE == "cuda":
        LlamaAttention.forward = patched_forward
    else:
        LlamaAttention.forward = forward_mps_fallback

    return model

def forward_step(model, input_ids: list[list[int]], block_tables: list[list[int]], seq_lens: list[int]):
    device = model.device

    # Converting lists received from Rust into tensors
    input_tensor = torch.tensor(input_ids, device=device)

    # Calculating position_ids for incoming tokens
    position_ids = torch.tensor([[l - 1] for l in seq_lens], device=device)

    with torch.no_grad():
        # Model graph traversal
        outputs = model(
            input_ids=input_tensor,
            position_ids=position_ids,
            block_tables=block_tables,
            seq_lens=seq_lens
        )

        # Retrieving logits from last token
        next_token_logits = outputs.logits[:, -1, :]

        next_tokens = torch.argmax(next_token_logits, dim=-1)

    return next_tokens.tolist()


if __name__ == "__main__":
    model = load_engine()

    mock_input_ids = [[312], [845]]
    mock_block_tables = [[5, 12], [100, 101, 102]]
    mock_seq_lens = [20, 35]

    # Simulation of a single cycle
    new_tokens = forward_step(model, mock_input_ids, mock_block_tables, mock_seq_lens)

    print(f"Generated tokens for the batch: {new_tokens}")
