"""Probe sustained leaf-batched Qwen3.8 generation with consistent eval/kernel recurrence.

The streamed target backend constructs fresh MLX decoder layers in training mode even though all
generation is deterministic inference.  Qwen3.5 DeltaNet therefore selects its ops recurrence
instead of the fused Metal kernel.  This isolated wrapper marks every loaded target layer as eval
before use, consistently affecting prompt prefill, leaf verification, cache reconstruction, and
the final fresh linear reference.

The validated default backend and sustained leaf generator remain unchanged.
"""

from airllm.airllm_qwen35_mlx_fp8_3bit import AirLLMQwen35MlxFp8ThreeBit
import generate_qwen38_mtp_tree_adaptive_leaf_3bit_macos as leaf_generator


_load_layer_default = AirLLMQwen35MlxFp8ThreeBit._load_layer


def _load_layer_eval(self, index):
    layer = _load_layer_default(self, index)
    layer.eval()
    return layer


def main():
    AirLLMQwen35MlxFp8ThreeBit._load_layer = _load_layer_eval
    print("streamed target recurrence: eval/fused-kernel")
    leaf_generator.main()


if __name__ == "__main__":
    main()
