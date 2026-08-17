"""Sustained adaptive leaf generation with unique-node target affines.

This isolated wrapper preserves the validated adaptive tree builder, canonical leaf selector,
selected-cache recovery, hidden-history commit, cleanup, and final fresh linear validation.  It
replaces only target leaf verification with the unique-node affine / complete-leaf recurrence
probe.
"""

import generate_qwen38_mtp_tree_adaptive_leaf_3bit_macos as leaf_generator
import probe_qwen38_mtp_leaf_verify_macos as baseline_verifier
from probe_qwen38_mtp_leaf_verify_uniquenode_macos import _verify_leaves_unique


def main():
    baseline_verifier._verify_leaves = _verify_leaves_unique
    print("sustained target verifier: adaptive unique-node affine / leaf recurrence")
    leaf_generator.main()


if __name__ == "__main__":
    main()
