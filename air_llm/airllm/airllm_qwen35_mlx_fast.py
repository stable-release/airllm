"""Optimized runtime lifecycle for the Qwen3.5-family MLX streaming backend.

The base backend intentionally uses an extremely conservative cleanup sequence after every
streamed module because it was developed while diagnosing Metal watchdog failures. Profiling on
Qwen3.8-27B showed that the Python GC + redundant synchronization in that sequence costs roughly
4.3 seconds per token across 64 layers on an 8 GB Mac.

This wrapper keeps the conservative behavior while model preparation is happening, then switches
runtime cleanup to MLX cache eviction only. The already-blocking ``mx.eval`` calls and cache-state
materialization in the base class remain unchanged. Set ``AIRLLM_MLX_SYNC_MODE=safe`` to restore
the original cleanup path without changing code.
"""

import os

import mlx.core as mx

from .airllm_qwen35_mlx import AirLLMQwen35Mlx as _SafeAirLLMQwen35Mlx


class AirLLMQwen35Mlx(_SafeAirLLMQwen35Mlx):
    """Qwen MLX streamer with low-overhead runtime cleanup and a safe fallback."""

    def __init__(self, *args, mlx_sync_mode=None, **kwargs):
        mode = mlx_sync_mode or os.environ.get("AIRLLM_MLX_SYNC_MODE", "eval-clear")
        mode = str(mode).strip().lower()
        if mode not in {"eval-clear", "safe"}:
            raise ValueError(
                "mlx_sync_mode must be 'eval-clear' or 'safe' "
                "(or set AIRLLM_MLX_SYNC_MODE to one of those values)."
            )

        self.mlx_sync_mode = mode
        # Quantization preparation can create very large temporary Python/MLX objects. Keep the
        # original conservative cleanup until the base constructor has finished creating/reusing
        # all split files; only inference uses the measured low-overhead path.
        self._runtime_cleanup_enabled = False
        super().__init__(*args, **kwargs)
        self._runtime_cleanup_enabled = True
        print(f"MLX runtime cleanup mode: {self.mlx_sync_mode}")

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
