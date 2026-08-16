"""Run the checkpointed speculative Qwen3.8 benchmark with a 3-bit target.

The main checkpointed benchmark intentionally keeps its known-good production target at
4-bit. This wrapper swaps only target construction so the prompt, BF16 draft model,
checkpoint/restore logic, counters, and output remain directly comparable.
"""

import speculative_qwen38_checkpointed_macos as benchmark

from airllm.airllm_qwen35_mlx_fp8_3bit import AirLLMQwen35MlxFp8ThreeBit


class _ThreeBitAutoModel:
    @classmethod
    def from_pretrained(cls, model_id, *args, **kwargs):
        # The baseline benchmark passes compression='4bit'. Ignore that fixed
        # baseline value here and change only the target backend under test.
        kwargs.pop("compression", None)
        return AirLLMQwen35MlxFp8ThreeBit(
            model_id,
            *args,
            compression="3bit",
            **kwargs,
        )


if __name__ == "__main__":
    benchmark.AutoModel = _ThreeBitAutoModel
    benchmark.main()
