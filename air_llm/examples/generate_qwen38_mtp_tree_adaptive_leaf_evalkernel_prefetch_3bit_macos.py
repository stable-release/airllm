"""Probe sustained leaf verification with eval recurrence and one-layer lookahead.

This isolated wrapper keeps the validated leaf and eval-kernel generators unchanged.  It overlaps
materialization of streamed target layer ``i + 1`` with computation of layer ``i`` using one
background worker.  The worker retains the complete next-layer module through its ``Future``, so
the main thread's MLX cache cleanup cannot release parameters that are still being prefetched.

Resident target components remain compatible with the lookahead path: loading an already-resident
layer simply returns and re-materializes the retained module.  Prompt prefill, every speculative
verification cycle, and the final linear validation all use the same eval-mode loader.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import time

from airllm.airllm_qwen35_mlx_fp8_3bit import AirLLMQwen35MlxFp8ThreeBit
import generate_qwen38_mtp_tree_adaptive_leaf_evalkernel_3bit_macos as eval_generator


@dataclass
class _ReadyLayer:
    index: int
    module: object
    load_s: float


class _OneLayerLookahead:
    """Double-buffer one streamed decoder layer without changing target semantics."""

    def __init__(self):
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="airllm-mlx-layer-prefetch",
        )
        self._target = None
        self._future = None
        self._future_index = None
        self._closed = False

        self.calls = 0
        self.prefetch_submitted = 0
        self.prefetch_hits = 0
        self.foreground_loads = 0
        self.sequence_resets = 0
        self.cancelled = 0
        self.foreground_load_s = 0.0
        self.prefetch_load_s = 0.0
        self.prefetch_wait_s = 0.0
        self.unused_prefetch_s = 0.0

    @staticmethod
    def _load_and_materialize(target, index):
        started = time.perf_counter()
        # Reuse the isolated eval wrapper's loader so both verification and cache reconstruction
        # retain the already-validated fused-kernel recurrence semantics.
        layer = eval_generator._load_layer_eval(target, index)
        target._materialize_module(layer)
        return _ReadyLayer(
            index=int(index),
            module=layer,
            load_s=time.perf_counter() - started,
        )

    def _bind(self, target):
        if self._target is None:
            self._target = target
        elif self._target is not target:
            raise RuntimeError("Layer lookahead supports one streamed target instance")

    def _drain_unexpected(self):
        """Release a stale speculative load before restarting a non-sequential traversal."""
        future = self._future
        self._future = None
        self._future_index = None
        if future is None:
            return
        if future.cancel():
            self.cancelled += 1
            return
        ready = future.result()
        self.unused_prefetch_s += ready.load_s
        # ``ready`` is deliberately kept alive until the worker has fully materialized it.  Dropping
        # this final strong reference now is safe because no target computation can use the stale
        # layer after a traversal reset.

    def _submit(self, target, index):
        self._future_index = int(index)
        self._future = self._executor.submit(
            self._load_and_materialize,
            target,
            int(index),
        )
        self.prefetch_submitted += 1

    def load(self, target, index):
        if self._closed:
            raise RuntimeError("Layer lookahead loader is already closed")
        self._bind(target)

        index = int(index)
        self.calls += 1
        if self._future is not None and self._future_index == index:
            waited = time.perf_counter()
            ready = self._future.result()
            self.prefetch_wait_s += time.perf_counter() - waited
            self.prefetch_load_s += ready.load_s
            self.prefetch_hits += 1
            self._future = None
            self._future_index = None
        else:
            if self._future is not None:
                self.sequence_resets += 1
                self._drain_unexpected()
            ready = self._load_and_materialize(target, index)
            self.foreground_load_s += ready.load_s
            self.foreground_loads += 1

        if ready.index != index:
            raise RuntimeError(
                f"Prefetched layer index drift: requested={index}, ready={ready.index}"
            )

        next_index = index + 1
        if next_index < int(target.model_args.num_hidden_layers):
            self._submit(target, next_index)

        # The caller's local ``layer`` reference becomes the owner of the current module before
        # the Future is replaced.  The new Future similarly owns the next module across cleanup.
        return ready.module

    def close(self):
        """Cancel queued work, join running work, and print non-invasive timing telemetry."""
        if self._closed:
            return
        self._closed = True

        # Never let shutdown errors mask the generator's original exception.  A worker failure that
        # matters to inference is raised by ``load`` when that prefetched layer is consumed.
        try:
            self._drain_unexpected()
        except BaseException as exc:
            print(f"layer lookahead shutdown warning: {type(exc).__name__}: {exc}")
        finally:
            self._executor.shutdown(wait=True, cancel_futures=True)

        estimated_hidden_s = max(self.prefetch_load_s - self.prefetch_wait_s, 0.0)
        print(
            "layer lookahead: "
            f"calls={self.calls} submitted={self.prefetch_submitted} "
            f"hits={self.prefetch_hits} foreground={self.foreground_loads} "
            f"resets={self.sequence_resets} cancelled={self.cancelled}"
        )
        print(
            "layer lookahead timing: "
            f"foreground_load={self.foreground_load_s:.3f}s "
            f"prefetch_load={self.prefetch_load_s:.3f}s "
            f"foreground_wait={self.prefetch_wait_s:.3f}s "
            f"estimated_hidden={estimated_hidden_s:.3f}s "
            f"unused_prefetch={self.unused_prefetch_s:.3f}s"
        )


_lookahead = None


def _load_layer_eval_prefetched(self, index):
    if _lookahead is None:
        raise RuntimeError("Layer lookahead wrapper was not initialized")
    return _lookahead.load(self, index)


def main():
    global _lookahead
    if _lookahead is not None:
        raise RuntimeError("Layer lookahead wrapper cannot be entered twice")

    previous_loader = AirLLMQwen35MlxFp8ThreeBit._load_layer
    _lookahead = _OneLayerLookahead()
    AirLLMQwen35MlxFp8ThreeBit._load_layer = _load_layer_eval_prefetched
    try:
        print("streamed target recurrence: eval/fused-kernel + one-layer lookahead")
        eval_generator.leaf_generator.main()
    finally:
        try:
            _lookahead.close()
        finally:
            AirLLMQwen35MlxFp8ThreeBit._load_layer = previous_loader
            _lookahead = None


if __name__ == "__main__":
    main()
