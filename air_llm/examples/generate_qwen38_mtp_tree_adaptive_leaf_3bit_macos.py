"""Sustained adaptive native-MTP generation with leaf-batched target verification.

This keeps the validated sustained generator loop and adaptive stop-before-prune MTP policy, but
replaces the depth-batched target verifier with the one-call-per-layer leaf verifier proven by
``probe_qwen38_mtp_leaf_verify_exactselect_macos.py``.

Each cycle:

* builds the same compact adaptive MTP margin tree;
* verifies every complete root-to-leaf sequence in one target call per streamed layer;
* follows target predictions only along the canonical greedy prefix;
* reconstructs only the selected DeltaNet cache from captured recurrence inputs;
* appends the selected attention KV suffix and copies normalized hidden history out of the batch;
* releases all leaf-batch allocations before the next cycle.

The final fresh linear validation in the underlying generator remains enabled by default.
"""

from dataclasses import dataclass

import mlx.core as mx

import generate_qwen38_mtp_tree_3bit_macos as generator
from generate_qwen38_mtp_tree_adaptive_3bit_macos import (
    _build_margin_tree_stop_before_prune,
)
import probe_qwen38_mtp_leaf_verify_macos as leaf_verifier
from probe_qwen38_mtp_leaf_verify_exactselect_macos import _select_canonical_leaf
import probe_qwen38_mtp_tree_verify_macos as node_verifier


@dataclass
class _LeafCycle:
    leaf_ids: list
    leaf_paths: list
    predictions: list
    normalized_hidden: object
    linear_captures: list
    attention_caches: list
    selected_row: int | None = None
    accepted: int | None = None


def _final_leaf_ids(nodes):
    """Return compact node ids at the uniform final frontier."""
    if not nodes:
        raise ValueError("Leaf-batched verification requires a non-empty tree")
    for index, node in enumerate(nodes):
        if int(node.node_id) != index:
            raise RuntimeError("Expected compact tree node ids to match list positions")
    final_depth = max(int(node.depth) for node in nodes)
    leaf_ids = [int(node.node_id) for node in nodes if int(node.depth) == final_depth]
    if not leaf_ids:
        raise RuntimeError("Adaptive MTP tree has no final frontier")
    parent_ids = {
        int(node.parent) for node in nodes if node.parent is not None
    }
    terminal_ids = {
        int(node.node_id) for node in nodes if int(node.node_id) not in parent_ids
    }
    if terminal_ids != set(leaf_ids):
        raise RuntimeError(
            "Leaf verifier requires a uniform final frontier; got terminal nodes "
            f"{sorted(terminal_ids)} across multiple depths"
        )
    return leaf_ids


def _verify_leaves_with_hidden(target, nodes, base_caches):
    """Adapt the leaf verifier to the sustained generator's verifier interface."""
    leaf_ids = _final_leaf_ids(nodes)
    (
        leaf_paths,
        predictions,
        normalized_hidden,
        linear_captures,
        attention_caches,
        layer_times,
    ) = leaf_verifier._verify_leaves(target, nodes, leaf_ids, base_caches)

    cycle = _LeafCycle(
        leaf_ids=leaf_ids,
        leaf_paths=leaf_paths,
        predictions=predictions,
        normalized_hidden=normalized_hidden,
        linear_captures=linear_captures,
        attention_caches=attention_caches,
    )

    # The sustained loop assigns six verifier outputs.  The first three aliases deliberately point
    # to one cycle object; the final two are unused placeholders retained for interface isolation.
    return cycle, cycle, cycle, None, None, layer_times


def _path_node_ids(nodes, leaf_id):
    path = []
    node_id = int(leaf_id)
    while node_id != 0:
        path.append(node_id)
        parent = nodes[node_id].parent
        if parent is None:
            raise RuntimeError("Leaf path terminated before reaching the root")
        node_id = int(parent)
    path.reverse()
    return path


def _select_target_leaf_path(nodes, cycle):
    """Return canonical accepted node ids plus an opaque leaf-cache selection."""
    if not isinstance(cycle, _LeafCycle):
        raise TypeError("Leaf target selection received an unexpected verifier result")

    row, accepted, correction = _select_canonical_leaf(
        cycle.leaf_paths,
        cycle.predictions,
    )
    full_path = _path_node_ids(nodes, cycle.leaf_ids[row])
    expected_depth = len(cycle.leaf_paths[row]) - 1
    if len(full_path) != expected_depth:
        raise RuntimeError(
            f"Leaf node path has {len(full_path)} proposals, expected {expected_depth}"
        )
    if accepted < 0 or accepted > len(full_path):
        raise RuntimeError(f"Invalid accepted leaf prefix length: {accepted}")

    cycle.selected_row = int(row)
    cycle.accepted = int(accepted)
    return full_path[:accepted], (int(row), int(accepted)), int(correction)


def _commit_selected_leaf(
    target_caches,
    cycle,
    nodes,
    unused_depth_positions,
    selection,
):
    """Recover root plus accepted proposals into the persistent target caches."""
    del nodes, unused_depth_positions
    if not isinstance(cycle, _LeafCycle):
        raise TypeError("Leaf cache commit received an unexpected verifier result")
    row, accepted = selection
    if cycle.selected_row != row or cycle.accepted != accepted:
        raise RuntimeError("Leaf cache selection drifted between target selection and commit")

    leaf_verifier._recover_selected_caches(
        target_caches,
        cycle.linear_captures,
        cycle.attention_caches,
        selected_row=row,
        keep_tokens=1 + accepted,
    )


def _append_selected_leaf_hidden(hidden_history, cycle, path_node_ids):
    """Append normalized root-plus-accepted hidden vectors without retaining the leaf batch."""
    if not isinstance(cycle, _LeafCycle):
        raise TypeError("Leaf hidden commit received an unexpected verifier result")
    if cycle.selected_row is None or cycle.accepted is None:
        raise RuntimeError("Leaf hidden commit ran before canonical target selection")

    keep_tokens = len(path_node_ids)
    if keep_tokens != 1 + cycle.accepted:
        raise RuntimeError(
            f"Hidden commit requested {keep_tokens} tokens after accepting {cycle.accepted}"
        )

    selected = cycle.normalized_hidden[
        cycle.selected_row : cycle.selected_row + 1,
        :keep_tokens,
        :,
    ]
    selected = leaf_verifier._copy_array(selected)
    combined = mx.concatenate([hidden_history, selected], axis=1)
    mx.eval(combined)
    return combined


def main():
    generator._build_margin_tree_batched = _build_margin_tree_stop_before_prune
    generator._verify_tree_batched_with_hidden = _verify_leaves_with_hidden
    generator._commit_selected_node = _commit_selected_leaf
    generator._append_selected_hidden = _append_selected_leaf_hidden
    node_verifier._select_target_path = _select_target_leaf_path

    print("sustained target verifier: adaptive leaf-batched exact-select")
    generator.main()


if __name__ == "__main__":
    main()
