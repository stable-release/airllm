"""Profile the experimental 3-bit streamed Qwen3.8 target without changing the 4-bit profiler."""

import profile_qwen38_macos as profiler

from airllm.airllm_qwen35_mlx_fp8_3bit import AirLLMQwen35MlxFp8ThreeBit


class _ThreeBitAutoModel:
    @classmethod
    def from_pretrained(cls, model_id, *args, **kwargs):
        # The baseline profiler passes compression='4bit'. Replace only the target backend.
        kwargs.pop("compression", None)
        return AirLLMQwen35MlxFp8ThreeBit(
            model_id,
            *args,
            compression="3bit",
            **kwargs,
        )


if __name__ == "__main__":
    profiler.AutoModel = _ThreeBitAutoModel
    profiler.main()
