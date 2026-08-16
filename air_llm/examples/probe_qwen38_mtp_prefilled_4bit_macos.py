"""Run the reference-aligned native Qwen3.8 MTP probe against the 4-bit streamed target.

This reuses the validated prefilled-MTP probe but swaps its target backend from the
experimental 3-bit requantization back to the higher-fidelity 4-bit AirLLM target.
The purpose is to measure whether native MTP agreement was degraded by moving the
verification target farther away from the checkpoint the MTP head was trained with.
"""

from airllm.airllm_qwen35_mlx_fp8 import AirLLMQwen35MlxFp8
import probe_qwen38_mtp_prefilled_macos as probe


class FourBitTarget(AirLLMQwen35MlxFp8):
    """Compatibility wrapper for the existing probe's 3-bit constructor call."""

    def __init__(self, *args, compression="3bit", **kwargs):
        super().__init__(*args, compression="4bit", **kwargs)


if __name__ == "__main__":
    probe.AirLLMQwen35MlxFp8ThreeBit = FourBitTarget
    probe.main()
