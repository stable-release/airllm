"""Adaptive sustained native-MTP tree generation for streamed 3-bit Qwen3.8.

This wraps the sustained depth-5 generator with one tree-policy change motivated by the
multi-cycle benchmark: if expanding the current MTP frontier would exceed ``max_active``,
do not score-prune that depth and continue.  Instead, stop at the previous fully-preserved
frontier and verify the smaller tree.

The policy is non-oracle.  It uses only MTP branch count/score information; target logits are
not consulted while drafting.  The goal is to avoid the measured failure mode where budget
pruning discarded the target-consistent branch, acceptance collapsed to 1-2/5, and the full
deep-tree target verification cost was still paid.

All target verification/cache/hidden-history logic is reused from
``generate_qwen38_mtp_tree_3bit_macos.py`` unchanged.
"""

import time

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import KVCache

import generate_qwen38_mtp_tree_3bit_macos as generator
import probe_qwen38_mtp_tree_verify_macos as verifier
import probe_qwen38_mtp_tree_verify_fastbuild_macos as fastbuild


def _build_margin_tree_stop_before_prune(
    target,
    mtp,
    shifted_ids,
    target_hidden,
    guaranteed_token,
    *,
    steps,
    branch_margin,
    max_active,
):
    """Build a margin tree, stopping before the first frontier-budget overflow."""
    started = time.perf_counter()
    nodes = [verifier.TreeNode(0, None, int(guaranteed_token), 0, 0.0)]
    generated = 0
    budget_pruned = False
    reached_depth = 0

    with fastbuild._PinnedSharedIO(target) as io:
        mtp_cache = KVCache()
        mtp_hidden = fastbuild._mtp_forward_resident(
            target,
            mtp,
            mtp_cache,
            shifted_ids,
            target_hidden,
            io.embedding,
        )
        active_hidden = mtp_hidden[:, -1:, :]
        active_cache = mtp_cache
        active = [(0.0, 0)]  # (cumulative log probability, node id)

        for depth in range(1, steps + 1):
            logits_mx = io.logits(active_hidden[:, -1, :])
            logits_batch = np.asarray(logits_mx, dtype=np.float32)
            del logits_mx

            # First collect candidate metadata without mutating the tree.  If this depth
            # would exceed the frontier budget, the previous frontier remains complete.
            candidates_by_parent = []
            candidate_count = 0
            for parent_row, (score, parent_id) in enumerate(active):
                row = logits_batch[parent_row]
                top2 = fastbuild._topk(row, 2)
                t1, t2 = int(top2[0]), int(top2[1])
                margin = float(row[t1] - row[t2])
                tokens = [t1]
                if margin <= branch_margin:
                    tokens.append(t2)

                z = verifier._log_z(row)
                children = [
                    (score + float(row[token]) - z, int(token))
                    for token in tokens
                ]
                candidates_by_parent.append((parent_row, parent_id, children))
                candidate_count += len(children)

            del logits_batch

            if candidate_count > max_active:
                budget_pruned = True
                print(
                    f"adaptive MTP stop-before-prune: requested_depth={steps} "
                    f"stopped_before_depth={depth} reached_depth={reached_depth} "
                    f"candidate_frontier={candidate_count} budget={max_active}"
                )
                break

            expanded = []  # (score, parent row, child node id)
            for parent_row, parent_id, children in candidates_by_parent:
                for child_score, token in children:
                    node_id = len(nodes)
                    nodes.append(
                        verifier.TreeNode(
                            node_id=node_id,
                            parent=parent_id,
                            token=token,
                            depth=depth,
                            score=child_score,
                        )
                    )
                    generated += 1
                    expanded.append((child_score, parent_row, node_id))

            active = [(score, node_id) for score, _, node_id in expanded]
            reached_depth = depth

            if depth == steps:
                break

            parent_rows = [parent_row for _, parent_row, _ in expanded]
            child_tokens = [nodes[node_id].token for _, _, node_id in expanded]
            parent_idx = mx.array(parent_rows, dtype=mx.int32)
            parent_hidden = active_hidden[parent_idx]
            child_cache = fastbuild._gather_mtp_cache(active_cache, parent_rows)
            active_hidden = fastbuild._mtp_forward_resident(
                target,
                mtp,
                child_cache,
                mx.array(child_tokens, dtype=mx.int32)[:, None],
                parent_hidden,
                io.embedding,
            )
            active_cache = child_cache
            mx.eval(active_hidden, active_cache.state)

    active_node_ids = [node_id for _, node_id in active]
    compact, active_new = fastbuild._compact_tree(nodes, active_node_ids)
    elapsed = time.perf_counter() - started
    print(
        f"adaptive batched MTP builder: {elapsed:.3f}s; requested_depth={steps}, "
        f"actual_depth={reached_depth}, generated={generated}, "
        f"final_active={len(active_new)}"
    )
    return compact, active_new, generated, budget_pruned, elapsed


if __name__ == "__main__":
    generator._build_margin_tree_batched = _build_margin_tree_stop_before_prune
    generator.main()
