"""Experimental direct block-FP8 -> MLX affine 3-bit Qwen backend.

This intentionally leaves the validated 4-bit backend unchanged.  The parent direct-FP8
loader already parameterizes shard naming, module construction, and runtime loading by
``self.quant_bits``; its public constructor currently hard-gates the production path to
4-bit.  This experimental subclass reuses that implementation while switching the
quantization bit-width immediately before direct sidecar preparation.
"""

from .airllm_qwen35_mlx_fp8 import AirLLMQwen35MlxFp8


class AirLLMQwen35MlxFp8ThreeBit(AirLLMQwen35MlxFp8):
    """Direct FP8 ingestion using MLX affine 3-bit streamed target weights."""

    def __init__(self, *args, compression="3bit", **kwargs):
        normalized = compression.lower() if isinstance(compression, str) else compression
        if normalized not in ("3bit", "mlx3", "mlx-3bit"):
            raise NotImplementedError(
                "Experimental block-FP8 Qwen 3-bit backend requires compression='3bit'."
            )

        # The validated parent constructor performs all metadata resolution, bounded
        # FP8 dequantization, cache/runtime setup, and direct component preparation.
        # Pass its accepted public mode through the gate; dynamic dispatch below changes
        # quant_bits before any output sidecar is selected or written.
        super().__init__(*args, compression="4bit", **kwargs)

    def _ensure_direct_fp8_quantized_shards(self, output_root):
        # Parent __init__ sets 4 first. Override it at the exact point where the
        # bit-width starts affecting output names and quantization. The resulting
        # sidecar is separate from and cannot overwrite the known-good 4-bit split.
        self.quant_bits = 3
        return super()._ensure_direct_fp8_quantized_shards(output_root)
