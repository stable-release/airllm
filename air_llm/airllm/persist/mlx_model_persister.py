import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from mlx.utils import tree_unflatten

from .model_persister import ModelPersister


# Increment whenever the on-disk MLX representation changes in a way that can affect inference.
# v2 preserves Qwen3.5-family Gated DeltaNet A_log tensors in fp32 instead of casting them to fp16.
MLX_SHARD_FORMAT_VERSION = "airllm-mlx-v2"


def map_torch_to_mlx(model):
    """Map the legacy Llama MLX backend's Hugging Face names to its local module names."""
    model = {k.replace("model.", ""): v for k, v in model.items()}
    model = {k.replace("mlp", "feed_forward"): v for k, v in model.items()}
    model = {k.replace("down_proj", "w2"): v for k, v in model.items()}
    model = {k.replace("up_proj", "w3"): v for k, v in model.items()}
    model = {k.replace("gate_proj", "w1"): v for k, v in model.items()}
    model = {k.replace("input_layernorm", "attention_norm"): v for k, v in model.items()}
    model = {k.replace("post_attention_layernorm", "ffn_norm"): v for k, v in model.items()}
    model = {k.replace("lm_head", "output"): v for k, v in model.items()}
    model = {k.replace("embed_tokens", "tok_embeddings"): v for k, v in model.items()}
    model = {k.replace("self_attn", "attention"): v for k, v in model.items()}
    model = {k.replace("q_proj", "wq"): v for k, v in model.items()}
    model = {k.replace("k_proj", "wk"): v for k, v in model.items()}
    model = {k.replace("v_proj", "wv"): v for k, v in model.items()}
    model = {k.replace("o_proj", "wo"): v for k, v in model.items()}
    return model


def _tensor_to_numpy(name, tensor):
    """Store Mac shards compactly without corrupting architecture-specific state.

    AirLLM's old MLX path cast every tensor to fp16. Qwen3.5-family Gated DeltaNet layers keep
    A_log in fp32, so preserve it. Integer/bool tensors are also kept in their native dtype.
    """
    tensor = tensor.detach().cpu()
    if not torch.is_floating_point(tensor):
        return tensor.numpy()
    if name.endswith("A_log"):
        return tensor.to(torch.float32).numpy()
    return tensor.to(torch.float16).numpy()


class MlxModelPersister(ModelPersister):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def model_persist_exist(self, layer_name, saving_path):
        npz_path = saving_path / (layer_name + "mlx.npz")
        done_path = saving_path / (layer_name + "mlx.done")
        if not npz_path.exists() or not done_path.exists():
            return False

        try:
            marker = done_path.read_text(encoding="utf-8").strip()
        except OSError:
            return False

        if marker == MLX_SHARD_FORMAT_VERSION:
            return True

        # Pre-v2 markers were empty. v2 only changes storage for Qwen's A_log tensors, so do not
        # force a costly rewrite of embeddings, full-attention layers, norm or lm_head. Inspect the
        # existing NPZ metadata: if the shard has no A_log it is representation-compatible and can
        # simply be adopted into v2. Linear-attention shards do contain A_log and must be rebuilt
        # from the original HF checkpoint to recover the fp32 values lost by the old fp16 cast.
        if marker == "":
            try:
                arrays = mx.load(str(npz_path))
                has_a_log = any(key.endswith("A_log") for key in arrays.keys())
                del arrays
                if not has_a_log:
                    done_path.write_text(MLX_SHARD_FORMAT_VERSION, encoding="utf-8")
                    return True
            except Exception:
                return False

        return False

    def persist_model(self, state_dict, layer_name, saving_path):
        weights = {k: _tensor_to_numpy(k, v) for k, v in state_dict.items()}
        np.savez(saving_path / (layer_name + "mlx"), **weights)
        print(f"saved as: {saving_path / (layer_name + 'mlx')}")
        (saving_path / (layer_name + "mlx.done")).write_text(
            MLX_SHARD_FORMAT_VERSION, encoding="utf-8"
        )

    def load_model_flat(self, layer_name, path):
        """Load original Hugging Face weight names without Llama-specific rewriting.

        Architecture-specific MLX backends (currently Qwen3.5/Qwen3.8) consume this form directly.
        """
        to_load_path = Path(path) / (layer_name + ".mlx.npz")
        return mx.load(str(to_load_path))

    def load_model(self, layer_name, path):
        """Legacy Llama loader retained for backwards compatibility."""
        try:
            layer_state_dict = self.load_model_flat(layer_name, path)
            layer_state_dict = map_torch_to_mlx(layer_state_dict)
            return tree_unflatten(list(layer_state_dict.items()))
        except Exception:
            print(f"error: {layer_name}, {path}")
            raise
