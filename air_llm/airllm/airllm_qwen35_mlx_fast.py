"""Optimized runtime lifecycle for the Qwen3.5-family MLX streaming backend.

The base backend intentionally uses an extremely conservative cleanup sequence after every
streamed module because it was developed while diagnosing Metal watchdog failures. Profiling on
Qwen3.8-27B showed that the Python GC + redundant synchronization in that sequence costs roughly
4.3 seconds per token across 64 layers on an 8 GB Mac.

This wrapper keeps the conservative behavior while model preparation is happening, then switches
runtime cleanup to MLX cache eviction only. The already-blocking ``mx.eval`` calls and cache-state
materialization in the base class remain unchanged. Set ``AIRLLM_MLX_SYNC_MODE=safe`` to restore
the original cleanup path without changing code.

A second optional optimization keeps a budgeted subset of repeatedly-used 4-bit modules resident
in unified memory. Set ``mlx_resident_gib`` (or ``AIRLLM_MLX_RESIDENT_GIB``) to a positive GiB
budget. The largest reusable components are selected first; the remainder continue to stream.
"""

import os
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from .airllm_qwen35_mlx import AirLLMQwen35Mlx as _SafeAirLLMQwen35Mlx


class AirLLMQwen35Mlx(_SafeAirLLMQwen35Mlx):
    """Qwen MLX streamer with low-overhead cleanup and optional resident weights."""

    def __init__(self, *args, mlx_sync_mode=None, mlx_resident_gib=None, **kwargs):
        mode = mlx_sync_mode or os.environ.get("AIRLLM_MLX_SYNC_MODE", "eval-clear")
        mode = str(mode).strip().lower()
        if mode not in {"eval-clear", "safe"}:
            raise ValueError(
                "mlx_sync_mode must be 'eval-clear' or 'safe' "
                "(or set AIRLLM_MLX_SYNC_MODE to one of those values)."
            )

        if mlx_resident_gib is None:
            mlx_resident_gib = os.environ.get("AIRLLM_MLX_RESIDENT_GIB", "0")
        try:
            resident_gib = float(mlx_resident_gib)
        except (TypeError, ValueError) as exc:
            raise ValueError("mlx_resident_gib / AIRLLM_MLX_RESIDENT_GIB must be a number.") from exc
        if resident_gib < 0:
            raise ValueError("mlx_resident_gib must be >= 0.")

        self.mlx_sync_mode = mode
        self.mlx_resident_gib = resident_gib
        self.resident_layer_names = set()
        self.resident_planned_bytes = 0
        self._resident_embedding = None
        self._resident_layers = {}
        self._resident_lm_head = None

        # Quantization preparation can create very large temporary Python/MLX objects. Keep the
        # original conservative cleanup until the base constructor has finished creating/reusing
        # all split files; only inference uses the measured low-overhead path.
        self._runtime_cleanup_enabled = False
        super().__init__(*args, **kwargs)
        self._runtime_cleanup_enabled = True
        self._configure_resident_budget()

        print(f"MLX runtime cleanup mode: {self.mlx_sync_mode}")
        if self.resident_layer_names:
            print(
                f"MLX resident weight budget: {self.mlx_resident_gib:.2f} GiB; "
                f"planned {self.resident_planned_bytes / 1024**3:.2f} GiB across "
                f"{len(self.resident_layer_names)} components"
            )

    def _configure_resident_budget(self):
        """Choose reusable components to pin without exceeding the requested file-size budget."""
        if self.mlx_resident_gib <= 0:
            return
        if not getattr(self, "mlx_quantized", False):
            print("MLX resident caching is currently enabled only for the 4-bit sidecar; ignoring budget.")
            return

        budget = int(self.mlx_resident_gib * 1024**3)
        checkpoint = Path(self.checkpoint_path)
        norm_name = self.layer_names_dict["norm"]
        embed_name = self.layer_names_dict["embed"]
        head_name = self.layer_names_dict["lm_head"]

        candidates = []
        for name in self.layer_names:
            if name == norm_name:
                continue
            path = checkpoint / f"{name}.mlx.npz"
            if path.exists():
                candidates.append((name, path.stat().st_size))

        # Embedding and lm_head are touched every token outside the decoder loop and are especially
        # attractive to pin. After those, choose the largest decoder shards for maximum bytes saved.
        priority = {embed_name: 2, head_name: 2}
        candidates.sort(key=lambda item: (priority.get(item[0], 1), item[1]), reverse=True)

        used = 0
        selected = set()
        for name, size in candidates:
            if size > budget - used:
                continue
            selected.add(name)
            used += size

        self.resident_layer_names = selected
        self.resident_planned_bytes = used

    @staticmethod
    def _materialize_module(module):
        """Force lazy file-backed parameter arrays to be evaluated while a strong ref is retained."""
        params = [value for _, value in tree_flatten(module.parameters())]
        if params:
            mx.eval(params)

    def _load_embedding(self):
        name = self.layer_names_dict["embed"]
        if name not in self.resident_layer_names:
            return super()._load_embedding()

        if self._resident_embedding is None:
            module = super()._load_embedding()
            self._materialize_module(module)
            self._resident_embedding = module
        return self._resident_embedding

    def _load_layer(self, index):
        name = f'{self.layer_names_dict["layer_prefix"]}.{index}'
        if name not in self.resident_layer_names:
            return super()._load_layer(index)

        module = self._resident_layers.get(index)
        if module is None:
            module = super()._load_layer(index)
            self._materialize_module(module)
            self._resident_layers[index] = module
        return module

    def _project_logits(self, hidden):
        name = self.layer_names_dict["lm_head"]
        if name not in self.resident_layer_names:
            return super()._project_logits(hidden)

        if self.model_args.tie_word_embeddings:
            embedding = self._load_embedding()
            logits = embedding.as_linear(hidden)
            mx.eval(logits)
            mx.synchronize()
            self._cleanup()
            return logits

        if self._resident_lm_head is None:
            weights = self._load_component(name)
            if self.mlx_quantized:
                module = nn.QuantizedLinear(
                    self.model_args.hidden_size,
                    self.model_args.vocab_size,
                    bias=False,
                    group_size=self.quant_group_size,
                    bits=self.quant_bits,
                    mode=self.quant_mode,
                )
            else:
                module = nn.Linear(self.model_args.hidden_size, self.model_args.vocab_size, bias=False)
            module.update(tree_unflatten(list(weights.items())))
            self._materialize_module(module)
            self._resident_lm_head = module

        logits = self._resident_lm_head(hidden)
        mx.eval(logits)
        mx.synchronize()
        self._cleanup()
        return logits

    def _cleanup(self):
        if not self._runtime_cleanup_enabled or self.mlx_sync_mode == "safe":
            return _SafeAirLLMQwen35Mlx._cleanup()

        # mx.eval() in the base backend has already materialized the activation/cache state before
        # each streamed module is dropped. On CPython, deleting the module releases ordinary
        # acyclic objects immediately; a full cyclic GC on every layer was measured at ~67 ms/layer.
        # Keep MLX's allocator cache empty so streamed weights do not accumulate across layers.
        clear_cache = getattr(mx, "clear_cache", None)
        if clear_cache is not None:
            clear_cache()
