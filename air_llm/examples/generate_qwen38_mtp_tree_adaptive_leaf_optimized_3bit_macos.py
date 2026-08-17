"""Combine the isolated exact leaf-verifier optimizations for sustained Qwen3.8 generation.

This entry point composes:

* adaptive stop-before-prune native-MTP trees;
* unique-node affine/MLP work with leaf-shaped causal recurrence;
* consistent eval-mode fused DeltaNet kernels;
* one-layer streamed-weight lookahead;
* optional bounded target residency through ``--resident-gib``.

All individual baseline and diagnostic entry points remain available unchanged.  The inherited
fresh full-output target validation stays enabled unless ``--no-final-reference`` is requested.
"""

import generate_qwen38_mtp_tree_adaptive_leaf_evalkernel_prefetch_3bit_macos as prefetch
import probe_qwen38_mtp_leaf_verify_macos as baseline_verifier
from probe_qwen38_mtp_leaf_verify_uniquenode_macos import _verify_leaves_unique


def main():
    baseline_verifier._verify_leaves = _verify_leaves_unique
    print("target leaf affines: unique-node deduplicated")
    prefetch.main()


if __name__ == "__main__":
    main()
