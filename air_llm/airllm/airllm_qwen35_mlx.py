"""Text-only AirLLM streaming backend for Qwen3.5-family models on Apple Silicon.

Qwen3.8 checkpoints use the Qwen3.5-family text architecture internally. Rather than duplicating
that architecture, this backend reuses MLX-LM's decoder layer and cache implementations while
AirLLM remains responsible for storing/loading one decoder layer at a time.

The first implementation intentionally ignores the vision tower and MTP head. Its goal is a
minimal, low-memory text-generation path for machines that cannot keep Qwen3.8-27B resident.
"""

import gc
import json
import os
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import psutil
from mlx.utils import tree_flatten, tree_unflatten
from transformers import AutoConfig, AutoTokenizer

from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.qwen3_5 import DecoderLayer, TextModelArgs

from .persist import ModelPersister
from .utils import find_or_create_local_splitted_path


_MLX_QUANT_FORMAT_VERSION = 1
_MLX_QUANT_BITS = 4
_MLX_QUANT_GROUP_SIZE = 64
_MLX_QUANT_MODE = "affine"


class AirLLMQwen35Mlx:
    """Stream Qwen3.5-family decoder layers from disk using MLX.

    Qwen3.8-27B currently reports ``Qwen3_5ForConditionalGeneration`` and uses the same hybrid
    Gated-DeltaNet/full-attention text decoder implemented by MLX-LM's qwen3_5 module.

    On macOS ``compression='4bit'`` means MLX-native affine 4-bit weight quantization. The known-good
    FP16 AirLLM split is retained and a separate quantized sidecar split is prepared one module at a
    time, so the 27B model never needs to be resident just to quantize it.
    """

    def set_layer_names_dict(self):
        self.layer_names_dict = {
            "embed": "model.language_model.embed_tokens",
            "layer_prefix": "model.language_model.layers",
            "norm": "model.language_model.norm",
            "lm_head": "lm_head",
        }

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
        **kwargs,
    ):
        normalized_compression = compression.lower() if isinstance(compression, str) else compression
        if normalized_compression in (None, "none"):
            self.mlx_quantized = False
        elif normalized_compression in ("4bit", "mlx4", "mlx-4bit"):
            self.mlx_quantized = True
        else:
            raise NotImplementedError(
                "AirLLMQwen35Mlx currently supports either FP16 streaming (compression=None) or "
                "MLX-native affine 4-bit streaming (compression='4bit')."
            )

        self.quant_bits = _MLX_QUANT_BITS
        self.quant_group_size = _MLX_QUANT_GROUP_SIZE
        self.quant_mode = _MLX_QUANT_MODE
        self.hf_token = hf_token
        self.max_seq_len = max_seq_len
        self.show_memory_util = show_memory_util
        self.initial_available = psutil.virtual_memory().available / 1024 / 1024
        self.least_available = self.initial_available
        self.set_layer_names_dict()

        # A normal MLX-LM process benefits from keeping freed Metal buffers cached because the full
        # model remains resident and similarly-sized allocations are reused. AirLLM has the opposite
        # lifecycle: each decoder-layer allocation should leave Metal as soon as that layer has run.
        if hasattr(mx, "set_cache_limit"):
            mx.set_cache_limit(0)
        if hasattr(mx, "reset_peak_memory"):
            mx.reset_peak_memory()

        # Always create/reuse the dense split first. The generic AirLLM compression path is
        # bitsandbytes/CUDA-only; MLX 4-bit is prepared as a sidecar from these dense per-layer files.
        self.model_local_path, dense_checkpoint_path = find_or_create_local_splitted_path(
            model_local_path_or_repo_id,
            layer_shards_saving_path,
            compression=None,
            layer_names=self.layer_names_dict,
            hf_token=hf_token,
            delete_original=delete_original,
        )
        self.dense_checkpoint_path = Path(dense_checkpoint_path)

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

        if self.mlx_quantized:
            self.checkpoint_path = str(self._ensure_quantized_shards())
            print(
                f"using MLX affine {self.quant_bits}-bit streamed weights "
                f"(group_size={self.quant_group_size}) from {self.checkpoint_path}"
            )
        else:
            self.checkpoint_path = str(self.dense_checkpoint_path)

        tokenizer_kwargs = {"trust_remote_code": True}
        if hf_token is not None:
            tokenizer_kwargs["token"] = hf_token
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_local_path, **tokenizer_kwargs)

    def record_memory(self, msg=None):
        if not self.show_memory_util:
            return
        available = psutil.virtual_memory().available / 1024 / 1024
        self.least_available = min(self.least_available, available)
        consumed = self.initial_available - available
        max_consumed = self.initial_available - self.least_available

        mlx_parts = []
        for label, fn_name in (
            ("mlx_active", "get_active_memory"),
            ("mlx_cache", "get_cache_memory"),
            ("mlx_peak", "get_peak_memory"),
        ):
            fn = getattr(mx, fn_name, None)
            if fn is not None:
                try:
                    mlx_parts.append(f"{label}={fn() / 1024 / 1024:.02f}MB")
                except Exception:
                    pass

        suffix = " " + " ".join(mlx_parts) if mlx_parts else ""
        print(
            f"[{msg}] available={available:.02f}MB consumed={consumed:.02f}MB "
            f"max_consumed={max_consumed:.02f}MB{suffix}"
        )

    @staticmethod
    def _strip_prefix(weights, prefix):
        prefix = prefix + "."
        return {
            key[len(prefix):]: value
            for key, value in weights.items()
            if key.startswith(prefix)
        }

    @staticmethod
    def _sanitize_qwen_layer(weights):
        """Apply the raw-checkpoint transforms MLX-LM uses for Qwen3.5-family models."""
        sanitized = dict(weights)
        norm_suffixes = (
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "q_norm.weight",
            "k_norm.weight",
        )
        for key, value in list(sanitized.items()):
            if key.endswith("conv1d.weight") and value.ndim == 3 and value.shape[-1] != 1:
                sanitized[key] = value.moveaxis(2, 1)
            if any(key.endswith(suffix) for suffix in norm_suffixes) and value.ndim == 1:
                sanitized[key] = value + 1.0
        return sanitized

    def _load_flat_from(self, layer_name, path):
        persister = ModelPersister.get_model_persister()
        if not hasattr(persister, "load_model_flat"):
            raise RuntimeError("Qwen MLX backend requires MlxModelPersister.load_model_flat().")
        return persister.load_model_flat(layer_name, path)

    def _load_dense_component(self, layer_name):
        return self._strip_prefix(
            self._load_flat_from(layer_name, self.dense_checkpoint_path), layer_name
        )

    def _load_component(self, layer_name):
        weights = self._load_flat_from(layer_name, self.checkpoint_path)
        # Quantized sidecar files are already sanitized and store module-local names. Dense split
        # files retain their original HF prefix and are stripped/sanitized at load time below.
        if self.mlx_quantized:
            return dict(weights)
        return self._strip_prefix(weights, layer_name)

    def _quantized_checkpoint_dir(self):
        suffix = (
            f".mlxq-v{_MLX_QUANT_FORMAT_VERSION}-{self.quant_mode}-"
            f"{self.quant_bits}bit-g{self.quant_group_size}"
        )
        return self.dense_checkpoint_path.with_name(self.dense_checkpoint_path.name + suffix)

    def _quantize_and_save_component(self, layer_name, output_path):
        """Quantize one dense AirLLM component and atomically save local MLX weights."""
        weights = self._load_dense_component(layer_name)

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
            module = nn.Linear(
                self.model_args.hidden_size,
                self.model_args.vocab_size,
                bias=False,
            )
            module.update(tree_unflatten(list(weights.items())))
            module = module.to_quantized(
                group_size=self.quant_group_size,
                bits=self.quant_bits,
                mode=self.quant_mode,
            )
        else:
            prefix = self.layer_names_dict["layer_prefix"] + "."
            if not layer_name.startswith(prefix):
                raise ValueError(f"Unknown Qwen streamed component: {layer_name}")
            index = int(layer_name[len(prefix):])
            weights = self._sanitize_qwen_layer(weights)
            module = DecoderLayer(self.model_args, index)
            module.update(tree_unflatten(list(weights.items())))
            # Qwen3.8 dense has no special quantization predicate in MLX-LM. The MLX default
            # quantizes every leaf that supports to_quantized(), including Linear; Conv1d, norms,
            # A_log, and recurrent-state parameters remain in their native representation.
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

        del flat
        del module
        del weights
        self._cleanup()

    def _ensure_quantized_shards(self):
        quant_path = self._quantized_checkpoint_dir()
        quant_path.mkdir(parents=True, exist_ok=True)
        metadata_path = quant_path / "airllm_mlx_quantization.json"
        expected_metadata = {
            "format_version": _MLX_QUANT_FORMAT_VERSION,
            "architecture": "qwen3_5",
            "bits": self.quant_bits,
            "group_size": self.quant_group_size,
            "mode": self.quant_mode,
            "sanitized": True,
        }

        metadata_ok = False
        if metadata_path.exists():
            try:
                metadata_ok = json.loads(metadata_path.read_text()) == expected_metadata
            except Exception:
                metadata_ok = False

        expected_files = [quant_path / f"{name}.mlx.npz" for name in self.layer_names]
        if metadata_ok and all(path.exists() for path in expected_files):
            return quant_path

        print(
            f"Preparing MLX {self.quant_bits}-bit sidecar split in {quant_path}. "
            "Existing FP16 shards are left untouched."
        )
        total = len(self.layer_names)
        for index, (layer_name, output_path) in enumerate(zip(self.layer_names, expected_files), 1):
            # Files are written via os.replace, so an existing file is a completed conversion. This
            # makes the long one-time preparation resumable after an interruption.
            if output_path.exists():
                print(f"[mlx4 {index}/{total}] reuse {layer_name}")
                continue
            print(f"[mlx4 {index}/{total}] quantize {layer_name}")
            self._quantize_and_save_component(layer_name, output_path)

        metadata_path.write_text(json.dumps(expected_metadata, indent=2, sort_keys=True) + "\n")
        return quant_path

    def _load_embedding(self):
        name = self.layer_names_dict["embed"]
        weights = self._load_component(name)
        if self.mlx_quantized:
            embedding = nn.QuantizedEmbedding(
                self.model_args.vocab_size,
                self.model_args.hidden_size,
                group_size=self.quant_group_size,
                bits=self.quant_bits,
                mode=self.quant_mode,
            )
        else:
            embedding = nn.Embedding(self.model_args.vocab_size, self.model_args.hidden_size)
        embedding.update(tree_unflatten(list(weights.items())))
        return embedding

    def _load_layer(self, index):
        name = f'{self.layer_names_dict["layer_prefix"]}.{index}'
        weights = self._load_component(name)
        layer = DecoderLayer(self.model_args, index)
        if self.mlx_quantized:
            nn.quantize(
                layer,
                group_size=self.quant_group_size,
                bits=self.quant_bits,
                mode=self.quant_mode,
            )
        else:
            weights = self._sanitize_qwen_layer(weights)
        layer.update(tree_unflatten(list(weights.items())))
        return layer

    def _load_norm(self):
        name = self.layer_names_dict["norm"]
        weights = self._load_component(name)
        if not self.mlx_quantized and "weight" in weights and weights["weight"].ndim == 1:
            weights["weight"] = weights["weight"] + 1.0
        norm = nn.RMSNorm(self.model_args.hidden_size, eps=self.model_args.rms_norm_eps)
        norm.update(tree_unflatten(list(weights.items())))
        return norm

    def _project_logits(self, hidden):
        if self.model_args.tie_word_embeddings:
            embedding = self._load_embedding()
            logits = embedding.as_linear(hidden)
            mx.eval(logits)
            mx.synchronize()
            del embedding
            self._cleanup()
            return logits

        name = self.layer_names_dict["lm_head"]
        weights = self._load_component(name)
        if self.mlx_quantized:
            output = nn.QuantizedLinear(
                self.model_args.hidden_size,
                self.model_args.vocab_size,
                bias=False,
                group_size=self.quant_group_size,
                bits=self.quant_bits,
                mode=self.quant_mode,
            )
        else:
            output = nn.Linear(self.model_args.hidden_size, self.model_args.vocab_size, bias=False)
        output.update(tree_unflatten(list(weights.items())))
        logits = output(hidden)
        mx.eval(logits)
        mx.synchronize()
        del output
        self._cleanup()
        return logits

    @staticmethod
    def _cleanup():
        # Make sure submitted Metal work is retired before dropping references to a streamed layer.
        synchronize = getattr(mx, "synchronize", None)
        if synchronize is not None:
            synchronize()
        gc.collect()
        clear_cache = getattr(mx, "clear_cache", None)
        if clear_cache is not None:
            clear_cache()
        if synchronize is not None:
            synchronize()

    def _new_cache(self, layer_index):
        is_linear = (layer_index + 1) % self.model_args.full_attention_interval != 0
        return ArraysCache(size=2) if is_linear else KVCache()

    def _run_layers(self, hidden, caches):
        for index in range(self.model_args.num_hidden_layers):
            self.record_memory(f"before layer {index}")
            layer = self._load_layer(index)
            cache = caches[index]
            mask = create_ssm_mask(hidden, cache) if layer.is_linear else create_attention_mask(hidden, cache)
            hidden = layer(hidden, mask=mask, cache=cache)

            # MLX is lazy. Materialize both activation and state before evicting layer weights.
            mx.eval([hidden, cache.state])
            mx.synchronize()

            del layer
            self._cleanup()
            self.record_memory(f"after layer {index}")
        return hidden

    @staticmethod
    def _sample(logits, temperature=0.0):
        if temperature is None or temperature <= 0:
            return mx.argmax(logits, axis=-1)
        return mx.random.categorical(logits / temperature)

    def model_generate(self, x, temperature=0.0):
        if not isinstance(x, mx.array):
            x = mx.array(x)
        if x.ndim == 1:
            x = x[None, :]
        if x.shape[0] != 1:
            raise ValueError("AirLLMQwen35Mlx currently supports batch size 1 only.")
        if x.shape[1] > self.max_seq_len:
            raise ValueError(
                f"Prompt has {x.shape[1]} tokens but max_seq_len is {self.max_seq_len}. "
                "Increase max_seq_len only after confirming memory usage on the target Mac."
            )

        caches = [self._new_cache(i) for i in range(self.model_args.num_hidden_layers)]

        embedding = self._load_embedding()
        hidden = embedding(x)
        mx.eval(hidden)
        mx.synchronize()
        del embedding
        self._cleanup()

        hidden = self._run_layers(hidden, caches)
        self.record_memory("before final norm")
        norm = self._load_norm()
        hidden = norm(hidden)
        mx.eval(hidden)
        mx.synchronize()
        self.record_memory("after final norm")
        del norm
        self._cleanup()

        self.record_memory("before logits")
        logits = self._project_logits(hidden[:, -1, :])
        self.record_memory("after logits")
        token = self._sample(logits, temperature)
        mx.eval(token)
        mx.synchronize()
        yield token

        while True:
            embedding = self._load_embedding()
            hidden = embedding(token[:, None])
            mx.eval(hidden)
            mx.synchronize()
            del embedding
            self._cleanup()

            hidden = self._run_layers(hidden, caches)
            norm = self._load_norm()
            hidden = norm(hidden)
            mx.eval(hidden)
            mx.synchronize()
            del norm
            self._cleanup()

            logits = self._project_logits(hidden[:, -1, :])
            token = self._sample(logits, temperature)
            mx.eval(token)
            mx.synchronize()
            yield token

    def generate(self, x, temperature=0.0, max_new_tokens=128, **kwargs):
        if max_new_tokens is None:
            max_new_tokens = 128

        token_ids = []
        eos_ids = self.tokenizer.eos_token_id
        if eos_ids is None:
            eos_ids = set()
        elif isinstance(eos_ids, int):
            eos_ids = {eos_ids}
        else:
            eos_ids = set(eos_ids)

        for token in self.model_generate(x, temperature=temperature):
            token_id = int(token.item())
            token_ids.append(token_id)
            if token_id in eos_ids or len(token_ids) >= max_new_tokens:
                break

        return self.tokenizer.decode(token_ids, skip_special_tokens=True)