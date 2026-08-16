"""Run the leaf-batched Qwen3.8 verifier with canonical greedy-path leaf selection.

The underlying leaf verifier intentionally remains isolated.  This wrapper replaces only its
selection routine: target predictions are followed from the common root prefix, filtering the
candidate leaves at each speculative depth.  This mirrors the proven node-tree verifier and avoids
mistaking a wrong branch that is self-consistent under its own conditional context for the target's
canonical greedy path.
"""

import probe_qwen38_mtp_leaf_verify_macos as probe


def _select_canonical_leaf(leaf_paths, predictions):
    if not leaf_paths:
        raise ValueError("Canonical leaf selection requires at least one leaf")

    depths = {len(tokens) - 1 for tokens in leaf_paths}
    if len(depths) != 1:
        raise RuntimeError(f"Expected uniform leaf depth, got {sorted(depths)}")
    depth = next(iter(depths))

    candidate_rows = list(range(len(leaf_paths)))
    accepted = 0

    for position in range(depth):
        # All rows still alive share the exact same target-consistent input prefix, so the target
        # prediction at this position must be identical across those rows.
        wanted_values = {int(predictions[row][position]) for row in candidate_rows}
        if len(wanted_values) != 1:
            raise RuntimeError(
                f"Target logits disagree across leaves sharing the canonical prefix at "
                f"position {position}: {sorted(wanted_values)}"
            )
        wanted = next(iter(wanted_values))

        next_rows = [
            row
            for row in candidate_rows
            if int(leaf_paths[row][position + 1]) == wanted
        ]
        if not next_rows:
            # No speculative child matches the target.  Any surviving row has the same valid prefix
            # and therefore the same cache state through ``accepted`` speculative tokens.
            return candidate_rows[0], accepted, wanted

        candidate_rows = next_rows
        accepted += 1

    # Every speculative position matched.  The logits after the final proposal predict the target
    # correction/bonus token and must again agree for leaves sharing the full canonical path.
    correction_values = {int(predictions[row][depth]) for row in candidate_rows}
    if len(correction_values) != 1:
        raise RuntimeError(
            "Target correction disagrees across leaves sharing the full canonical path: "
            f"{sorted(correction_values)}"
        )
    correction = next(iter(correction_values))
    return candidate_rows[0], accepted, correction


if __name__ == "__main__":
    probe._select_leaf = _select_canonical_leaf
    probe.main()
