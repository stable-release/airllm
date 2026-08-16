"""Direct block-FP8 checkpoint ingestion for Qwen3.5/Qwen3.8 on Apple Silicon.

This path is for pre-quantized Hugging Face checkpoints whose linear weights are stored as
float8_e4m3fn plus per-128x128 ``weight_scale_inv`` tensors (the scheme used by Qwen3.8 FP8
checkpoints and OrcaRouter's Qwen3.8-27B-Uncensored-FP8).

Unlike the normal AirLLM splitter, it never writes an intermediate dense FP16 copy. Each streamed
component is read directly from the source safetensors shards, block-dequantized in bounded CPU
memory, converted immediately to the existing MLX affine-4bit sidecar format, then released.
"""

import gc
import json
import os
from pathlib import Path

import huggingface_hub
import mlx.core as mx
import mlx.nn as nn
import psutil
import torch
from mlx.utils import tree_flatten, tree_unflatten
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer

from mlx_lm.models.qwen3_5 import DecoderLayer, TextModelArgs

from .airllm_qwen35_mlx_fast import AirLLMQwen35Mlx as _FastQwen35Mlx


_FP8_SOURCE_FORMAT_VERSION = 1
_FP8_BLOCK = (128, 128)


class AirLLMQwen35MlxFp8(_FastQwen35Mlx):
    """Prepare and stream Qwen block-FP8 checkpoints through MLX 4-bit weights."""

    def __init__(
        self,
        model_local_path_or_repo_id,
        device=None,
        dtype=None,
        max_seq_len=8192,
        layer_shards_saving_path=None,
        profiling_mode=False,
        compression=None,
        hf_token=None,
        prefetching=True,
        test_nonlayered=False,
        show_memory_util=False,
        delete_original=False,
        mlx_sync_mode=None,
        mlx_resident_gib=None,
        **kwargs,
    ):
        normalized_compression = compression.lower() if isinstance(compression, str) else compression
        if normalized_compression not in ("4bit", "mlx4", "mlx-4bit"):
            raise NotImplementedError(
                "Block-FP8 Qwen checkpoints are currently supported on macOS through the "
                "direct FP8 -> MLX affine-4bit preparation path. Pass compression='4bit'."
            )

        # Mirror the optimized wrapper's runtime controls without invoking its parent constructor,
        # because the parent would first create a wasteful dense FP16 AirLLM split.
        import os as _os

        mode = mlx_sync_mode or _os.environ.get("AIRLLM_MLX_SYNC_MODE", "eval-clear")
        mode = str(mode).strip().lower()
        if mode not in {"eval-clear", "safe"}:
            raise ValueError("mlx_sync_mode must be 'eval-clear' or 'safe'.")
        if mlx_resident_gib is None:
            mlx_resident_gib = _os.environ.get("AIRLLM_MLX_RESIDENT_GIB", "0")
        resident_gib = float(mlx_resident_gib)
        if resident_gib < 0:
            raise ValueError("mlx_resident_gib must be >= 0.")

        self.mlx_sync_mode = mode
        self.mlx_resident_gib = resident_gib
        self.resident_layer_names = set()
        self.resident_planned_bytes = 0
        self._resident_embedding = None
        self._resident_layers = {}
        self._resident_lm_head = None
        self._runtime_cleanup_enabled = False

        self.mlx_quantized = True
        self.quant_bits = 4
        self.quant_group_size = 64
        self.quant_mode = "affine"
        self.hf_token = hf_token
        self.max_seq_len = max_seq_len
        self.show_memory_util = show_memory_util
        self.initial_available = psutil.virtual_memory().available / 1024 / 1024
        self.least_available = self.initial_available
        self.set_layer_names_dict()

        if hasattr(mx, "set_cache_limit"):
            mx.set_cache_limit(0)
        if hasattr(mx, "reset_peak_memory"):
            mx.reset_peak_memory()

        self.source_repo_id = None
        self.model_local_path = self._resolve_metadata_path(model_local_path_or_repo_id)

        config_kwargs = {"trust_remote_code": True}
        if hf_token is not None:
            config_kwargs["token"] = hf_token
        self.config = AutoConfig.from_pretrained(self.model_local_path, **config_kwargs)
        text_config = getattr(self.config, "text_config", self.config)
        if hasattr(text_config, "to_dict"):
            text_config = text_config.to_dict()
        self.model_args = TextModelArgs.from_dict(text_config)

        self.layer_names = [self.layer_names_dict["embed"]] + [
            f'{self.layer_names_dict["layer_prefix"]}.{i}'
            for i in range(self.model_args.num_hidden_layers)
        ] + [self.layer_names_dict["norm"]]
        if not self.model_args.tie_word_embeddings:
            self.layer_names.append(self.layer_names_dict["lm_head"])

        output_root = Path(layer_shards_saving_path) if layer_shards_saving_path else Path(self.model_local_path)
        self.dense_checkpoint_path = None
        self.checkpoint_path = str(self._ensure_direct_fp8_quantized_shards(output_root))

        tokenizer_kwargs = {"trust_remote_code": True}
        if hf_token is not None:
            tokenizer_kwargs["token"] = hf_token
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_local_path, **tokenizer_kwargs)

        self._runtime_cleanup_enabled = True
        self._configure_resident_budget()
        print(
            f"using direct block-FP8 -> MLX affine {self.quant_bits}-bit streamed weights "
            f"from {self.checkpoint_path}"
        )
        print(f"MLX runtime cleanup mode: {self.mlx_sync_mode}")
        if self.resident_layer_names:
            print(
                f"MLX resident weight budget: {self.mlx_resident_gib:.2f} GiB; "
                f"planned {self.resident_planned_bytes / 1024**3:.2f} GiB across "
                f"{len(self.resident_layer_names)} components"
            )

    def _resolve_metadata_path(self, model_local_path_or_repo_id):
        path = Path(model_local_path_or_repo_id)
        if path.exists():
            return path

        self.source_repo_id = str(model_local_path_or_repo_id)
        return Path(
            huggingface_hub.snapshot_download(
                self.source_repo_id,
                token=self.hf_token,
                ignore_patterns=["*.safetensors", "*.bin"],
            )
        )

    def _source_index(self):
        index_path = Path(self.model_local_path) / "model.safetensors.index.json"
        if not index_path.exists():
            raise FileNotFoundError(
                f"Direct block-FP8 ingestion expects model.safetensors.index.json under "
                f"{self.model_local_path}."
            )
        return json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]

    def _ensure_source_shard(self, filename):
        path = Path(self.model_local_path) / filename
        if path.exists():
            return path
        if self.source_repo_id is None:
            raise FileNotFoundError(path)
        huggingface_hub.snapshot_download(
            self.source_repo_id,
            token=self.hf_token,
            allow_patterns=[filename],
        )
        if not path.exists():
            raise FileNotFoundError(f"Hugging Face download completed but shard is missing: {path}")
        return path

    def _load_source_component(self, layer_name, weight_map):
        prefix = layer_name + "."
        keys = [key for key in weight_map if key.startswith(prefix)]
        if not keys:
            raise KeyError(f"No source tensors found for {layer_name}")

        by_file = {}
        for key in keys:
            by_file.setdefault(weight_map[key], []).append(key)

        out = {}
        for filename, file_keys in by_file.items():
            shard_path = self._ensure_source_shard(filename)
            with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
                for key in file_keys:
                    out[key] = handle.get_tensor(key)
        return out

    @staticmethod
    def _is_fp8_tensor(tensor):
        return str(tensor.dtype).startswith("torch.float8")

    @staticmethod
    def _dequantize_block_fp8(weight, scale_inv, block_size=_FP8_BLOCK):
        """Reconstruct a 2-D block-FP8 tensor without expanding the scale grid to full FP32 size."""
        if weight.ndim != 2 or scale_inv.ndim != 2:
            raise ValueError(
                f"Expected 2-D block-FP8 weight/scale, got weight={tuple(weight.shape)} "
                f"scale={tuple(scale_inv.shape)}"
            )

        rows, cols = weight.shape
        block_m, block_n = block_size
        expected = ((rows + block_m - 1) // block_m, (cols + block_n - 1) // block_n)
        if tuple(scale_inv.shape) != expected:
            raise ValueError(
                f"FP8 scale shape {tuple(scale_inv.shape)} does not match weight {tuple(weight.shape)} "
                f"for block size {block_size}; expected {expected}."
            )

        # Keep only the final dense result in FP16. Each temporary FP32 conversion is one 128-row
        # block, which prevents a 100M-parameter matrix from briefly requiring multiple full-size
        # FP32 buffers on an 8 GB machine.
        dense = torch.empty((rows, cols), dtype=torch.float16, device="cpu")
        for block_row in range(scale_inv.shape[0]):
            row_start = block_row * block_m
            row_end = min(row_start + block_m, rows)
            scale_cols = (
                scale_inv[block_row]
                .to(torch.float32)
                .repeat_interleave(block_n)[:cols]
            )
            chunk = weight[row_start:row_end].to(torch.float32)
            dense[row_start:row_end] = (chunk * scale_cols.unsqueeze(0)).to(torch.float16)
            del chunk, scale_cols
        return dense

    def _densify_source_component(self, source, layer_name):
        """Strip the HF prefix and fold each FP8 weight + weight_scale_inv pair into dense FP16."""
        prefix = layer_name + "."
        local = {}
        scale_keys = {key for key in source if key.endswith(".weight_scale_inv")}

        for full_key, tensor in source.items():
            if full_key in scale_keys:
                continue
            if not full_key.startswith(prefix):
                continue
            local_key = full_key[len(prefix):]

            scale_key = None
            if full_key.endswith(".weight"):
                scale_key = full_key[: -len(".weight")] + ".weight_scale_inv"

            if scale_key in source and self._is_fp8_tensor(tensor):
                local[local_key] = self._dequantize_block_fp8(tensor, source[scale_key])
            else:
                # Match the existing AirLLM MLX dense representation: A_log remains fp32, normal
                # floating tensors become fp16, integer/bool tensors keep their native dtype.
                tensor = tensor.detach().cpu()
                if torch.is_floating_point(tensor):
                    if local_key.endswith("A_log"):
                        tensor = tensor.to(torch.float32)
                    else:
                        tensor = tensor.to(torch.float16)
                local[local_key] = tensor
        return local

    @staticmethod
    def _torch_local_to_mlx(weights):
        out = {}
        for key, tensor in weights.items():
            # numpy has no portable float8, but every FP8 tensor has already been dequantized above.
            out[key] = mx.array(tensor.numpy())
        return out

    def _quantize_local_component_to_file(self, layer_name, local_torch, output_path):
        weights = self._torch_local_to_mlx(local_torch)

        if layer_name == self.layer_names_dict["embed"]:
            module = nn.Embedding(self.model_args.vocab_size, self.model_args.hidden_size)
            module.update(tree_unflatten(list(weights.items())))
            module = module.to_quantized(
                group_size=self.quant_group_size,
                bits=self.quant_bits,
                mode=self.quant_mode,
            )
        elif layer_name == self.layer_names_dict["norm"]:
            if "weight" in weights and weights["weight"].ndim == 1:
                weights["weight"] = weights["weight"] + 1.0
            module = nn.RMSNorm(self.model_args.hidden_size, eps=self.model_args.rms_norm_eps)
            module.update(tree_unflatten(list(weights.items())))
        elif layer_name == self.layer_names_dict["lm_head"]:
            module = nn.Linear(self.model_args.hidden_size, self.model_args.vocab_size, bias=False)
            module.update(tree_unflatten(list(weights.items())))
            module = module.to_quantized(
                group_size=self.quant_group_size,
                bits=self.quant_bits,
                mode=self.quant_mode,
            )
        else:
            prefix = self.layer_names_dict["layer_prefix"] + "."
            index = int(layer_name[len(prefix):])
            weights = self._sanitize_qwen_layer(weights)
            module = DecoderLayer(self.model_args, index)
            module.update(tree_unflatten(list(weights.items())))
            nn.quantize(
                module,
                group_size=self.quant_group_size,
                bits=self.quant_bits,
                mode=self.quant_mode,
            )

        flat = dict(tree_flatten(module.parameters()))
        mx.eval(list(flat.values()))
        tmp_path = Path(str(output_path) + ".tmp.npz")
        if tmp_path.exists():
            tmp_path.unlink()
        mx.savez(str(tmp_path), **flat)
        os.replace(tmp_path, output_path)

        del flat, module, weights
        gc.collect()
        mx.clear_cache()

    def _ensure_direct_fp8_quantized_shards(self, output_root):
        quant_path = output_root / (
            f"splitted_model.mlxq-fp8src-v{_FP8_SOURCE_FORMAT_VERSION}-"
            f"{self.quant_mode}-{self.quant_bits}bit-g{self.quant_group_size}"
        )
        quant_path.mkdir(parents=True, exist_ok=True)
        metadata_path = quant_path / "airllm_mlx_quantization.json"
        expected_metadata = {
            "format_version": _FP8_SOURCE_FORMAT_VERSION,
            "architecture": "qwen3_5",
            "source_format": "block-fp8-e4m3-128x128",
            "bits": self.quant_bits,
            "group_size": self.quant_group_size,
            "mode": self.quant_mode,
            "sanitized": True,
        }

        expected_files = [quant_path / f"{name}.mlx.npz" for name in self.layer_names]
        metadata_ok = False
        if metadata_path.exists():
            try:
                metadata_ok = json.loads(metadata_path.read_text(encoding="utf-8")) == expected_metadata
            except Exception:
                metadata_ok = False
        if metadata_ok and all(path.exists() for path in expected_files):
            return quant_path

        weight_map = self._source_index()
        print(
            f"Preparing direct block-FP8 -> MLX {self.quant_bits}-bit sidecar in {quant_path}. "
            "No dense FP16 AirLLM split will be written."
        )
        total = len(self.layer_names)
        for ordinal, (layer_name, output_path) in enumerate(zip(self.layer_names, expected_files), 1):
            if output_path.exists():
                print(f"[fp8->mlx4 {ordinal}/{total}] reuse {layer_name}")
                continue
            print(f"[fp8->mlx4 {ordinal}/{total}] convert {layer_name}")
            source = self._load_source_component(layer_name, weight_map)
            local = self._densify_source_component(source, layer_name)
            del source
            self._quantize_local_component_to_file(layer_name, local, output_path)
            del local
            gc.collect()

        metadata_path.write_text(json.dumps(expected_metadata, indent=2, sort_keys=True) + "\n")
        return quant_path
