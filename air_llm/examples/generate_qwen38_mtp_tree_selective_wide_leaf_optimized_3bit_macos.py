"""Compose selective-wide MTP drafting with the optimized exact leaf verifier."""

import generate_qwen38_mtp_tree_adaptive_leaf_3bit_macos as leaf_generator
import generate_qwen38_mtp_tree_adaptive_leaf_optimized_3bit_macos as optimized
from generate_qwen38_mtp_tree_selective_wide_3bit_macos import (
    _build_margin_tree_selective_wide,
)


def main():
    # ``leaf_generator.main`` installs this module-level symbol into the sustained generator after
    # the outer optimized wrappers have applied verifier/eval/prefetch patches.  Patch this symbol,
    # rather than the base generator directly, so the selective policy survives composition order.
    leaf_generator._build_margin_tree_stop_before_prune = (
        _build_margin_tree_selective_wide
    )
    print("native MTP tree policy: selective-wide global frontier")
    optimized.main()


if __name__ == "__main__":
    main()
