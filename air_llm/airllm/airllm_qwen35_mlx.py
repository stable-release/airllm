"""Text-only AirLLM streaming backend for Qwen3.5-family models on Apple Silicon.

Qwen3.8 checkpoints use the Qwen3.5-family text architecture internally.  Rather than duplicating
that architecture, this backend reuses MLX-LM's decoder layer and cache implementations while
AirLLM remains responsible for storing/loading one decoder layer at a time.

The first implementation intentionally ignores the vision tower and MTP head.  Its goal is a
minimal, low-memory text-generation path for machines that cannot keep Qwen3.8-27B resident.
"""

import gc

import mlx.core as mx
import mlx.nn as nn
import psutil
from mlx.utils import tree_unflatten
from transformers import AutoConfig, AutoTokenizer

from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.qwen3_5 import DecoderLayer, TextModelArgs

from .persist import ModelPersister
from .utils import find_or_create_local_splitted_path


class AirLLMQwen35Mlx:
    """Stream Qwen3.5-family decoder layers from disk using MLX.

    Qwen3.8-27B currently reports ``Qwen3_5ForConditionalGeneration`` and uses the same hybrid
    Gated-DeltaNet/full-attention text decoder implemented by MLX-LM's qwen3_5 module.
    """

    def set_layer_names_dict(self):
        # Qwen3.8 is a multimodal ConditionalGeneration checkpoint.  Text weights live below
        # model.language_model; vision weights are deliberately excluded from this text-only path.
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
        if compression is not None:
            raise NotImplementedError(
                "AirLLMQwen35Mlx does not yet support AirLLM bitsandbytes compression on macOS. "
                "Use the original Qwen3.8 weights for the first prototype."
            )

        self.hf_token = hf_token
        self.max_seq_len = max_seq_len
        self.show_memory_util = show_memory_util
        self.initial_available = psutil.virtual_memory().available / 1024 / 1024
        self.least_available = self.initial_available
        self.set_layer_names_dict()

        self.model_local_path, self.checkpoint_path = find_or_create_local_splitted_path(
            model_local_path_or_repo_id,
            layer_shards_saving_path,
            compression=None,
            layer_names=self.layer_names_dict,
            hf_token=hf_token,
            delete_original=delete_original,
        )

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
        print(
            f"[{msg}] available={available:.02f}MB consumed={consumed:.02f}MB "
            f"max_consumed={max_consumed:.02f}MB"
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
        """Apply the same raw-checkpoint transforms MLX-LM uses for Qwen3.5-family models.

        Official Qwen3.8 weights use the pre-MLX Conv1D layout and shifted RMSNorm convention.
        MLX-LM normally detects this while sanitizing the complete checkpoint.  AirLLM only sees
        one layer at a time, so the transformation is made explicit here.
        """
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

    def _load_flat(self, layer_name):
        persister = ModelPersister.get_model_persister()
        if not hasattr(persister, "load_model_flat"):
            raise RuntimeError("Qwen MLX backend requires MlxModelPersister.load_model_flat().")
        return persister.load_model_flat(layer_name, self.checkpoint_path)

    def _load_embedding(self):
        name = self.layer_names_dict["embed"]
        weights = self._strip_prefix(self._load_flat(name), name)
        embedding = nn.Embedding(self.model_args.vocab_size, self.model_args.hidden_size)
        embedding.update(tree_unflatten(list(weights.items())))
        return embedding

    def _load_layer(self, index):
        name = f'{self.layer_names_dict["layer_prefix"]}.{index}'
        weights = self._strip_prefix(self._load_flat(name), name)
        weights = self._sanitize_qwen_layer(weights)
        layer = DecoderLayer(self.model_args, index)
        layer.update(tree_unflatten(list(weights.items())))
        return layer

    def _load_norm(self):
        name = self.layer_names_dict["norm"]
        weights = self._strip_prefix(self._load_flat(name), name)
        # Qwen3.8's final RMSNorm follows the same shifted-weight convention as decoder norms.
        if "weight" in weights and weights["weight"].ndim == 1:
            weights["weight"] = weights["weight"] + 1.0
        norm = nn.RMSNorm(self.model_args.hidden_size, eps=self.model_args.rms_norm_eps)
        norm.update(tree_unflatten(list(weights.items())))
        return norm

    def _project_logits(self, hidden):
        if self.model_args.tie_word_embeddings:
            embedding = self._load_embedding()
            logits = embedding.as_linear(hidden)
            mx.eval(logits)
            del embedding
            self._cleanup()
            return logits

        name = self.layer_names_dict["lm_head"]
        weights = self._strip_prefix(self._load_flat(name), name)
        output = nn.Linear(self.model_args.hidden_size, self.model_args.vocab_size, bias=False)
        output.update(tree_unflatten(list(weights.items())))
        logits = output(hidden)
        mx.eval(logits)
        del output
        self._cleanup()
        return logits

    @staticmethod
    def _cleanup():
        gc.collect()
        clear_cache = getattr(mx, "clear_cache", None)
        if clear_cache is not None:
            clear_cache()

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

            # MLX is lazy.  Evaluating only `hidden` is not sufficient for Qwen's hybrid decoder:
            # ArraysCache/KVCache updates can remain as unevaluated graphs that still reference the
            # just-loaded layer weights.  Materialize both the activation and this layer's cache
            # before deleting the layer so streaming actually releases those weights.  MLX-LM does
            # the same during hybrid-model prefill by explicitly evaluating cache state.
            mx.eval([hidden, cache.state])

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
        del embedding
        self._cleanup()

        hidden = self._run_layers(hidden, caches)
        norm = self._load_norm()
        hidden = norm(hidden)
        mx.eval(hidden)
        del norm
        self._cleanup()

        logits = self._project_logits(hidden[:, -1, :])
        token = self._sample(logits, temperature)
        mx.eval(token)
        yield token

        while True:
            embedding = self._load_embedding()
            hidden = embedding(token[:, None])
            mx.eval(hidden)
            del embedding
            self._cleanup()

            hidden = self._run_layers(hidden, caches)
            norm = self._load_norm()
            hidden = norm(hidden)
            mx.eval(hidden)
            del norm
            self._cleanup()

            logits = self._project_logits(hidden[:, -1, :])
            token = self._sample(logits, temperature)
            mx.eval(token)
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