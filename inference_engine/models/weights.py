import glob
import os
import torch
from huggingface_hub import snapshot_download
from huggingface_hub.utils import disable_progress_bars
from safetensors.torch import load_file
from transformers import AutoConfig
from inference_engine.models.llama import LlamaForCausalLM


def load_llama_model(
        model_id: str, device: torch.device, dtype: torch.dtype
) -> LlamaForCausalLM:
    disable_progress_bars()     # TODO: enable in dev profile with terminal emulation (disable when writing to file)

    config = AutoConfig.from_pretrained(model_id)

    # Model initialization on the target type and device
    with torch.device(device):
        torch.set_default_dtype(dtype)
        model = LlamaForCausalLM(config)
        torch.set_default_dtype(torch.float32)

    model_path = snapshot_download(repo_id=model_id, allow_patterns=["*.safetensors"])
    safetensor_files = glob.glob(os.path.join(model_path, "*.safetensors"))

    state_dict = {}
    for file in safetensor_files:
        tensors = load_file(file, device=str(device))
        for k, v in tensors.items():
            # Delete 'model.' prefix used by HuggingFace inside CausalLM classes
            clean_key = k.replace("model.", "") if k.startswith("model.") else k
            state_dict[clean_key] = v.to(dtype)

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model