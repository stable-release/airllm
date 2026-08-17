"""Selective-wide native-MTP tree builder for streamed Qwen3.8 generation.

This isolated policy keeps the familiar uncertainty-triggered top-2 branch and selectively widens
very flat MTP distributions:

* rank 2 is included when ``top1 - top2 <= branch_margin``;
* ranks 2 through 4 are included when ``top1 - top4 <= branch_margin``;
* rank 5 is included only when ``top1 - top5 <= min(2.0, branch_margin)``.

Every depth first collects children from all current parents, then retains the globally best
cumulative-score paths up to ``max_active``.  Unlike the adaptive stop-before-prune policy, this
always reaches the requested depth and therefore leaves a uniform final frontier for leaf-batched
target verification.
"""

import time

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import KVCache

import generate_qwen38_mtp_tree_3bit_macos as generator
import probe_qwen38_mtp_tree_verify_fastbuild_macos as fastbuild
import probe_qwen38_mtp_tree_verify_macos as verifier


def _select_wide_tokens(row, branch_margin):
    """Return ordered token ids selected by the local rank/margin policy."""
    row = np.asarray(row)
    if row.ndim != 1 or row.size < 5:
        raise ValueError(
            f"Selective-wide MTP policy expects at least 5 logits, got {row.shape}"
        )
    if not np.isfinite(float(branch_margin)):
        raise ValueError("branch_margin must be finite")
    if not bool(np.isfinite(row).all()):
        raise RuntimeError("Selective-wide MTP policy received non-finite logits")

    top = fastbuild._topk(row, 5)
    if len(top) < 5:
        raise RuntimeError("Selective-wide MTP policy requires a vocabulary of at least 5")

    top = [int(token) for token in top]
    top1 = float(row[top[0]])
    gap2 = top1 - float(row[top[1]])
    gap4 = top1 - float(row[top[3]])
    gap5 = top1 - float(row[top[4]])
    rank5_margin = min(2.0, float(branch_margin))

    if gap5 <= rank5_margin:
        width = 5
    elif gap4 <= branch_margin:
        width = 4
    elif gap2 <= branch_margin:
        width = 2
    else:
        width = 1
    return top[:width], (gap2, gap4, gap5)


def _candidate_order(candidate):
    """Total deterministic ordering for globally competing child paths."""
    return (
        -float(candidate[0]),
        int(candidate[2]),
        int(candidate[1]),
        int(candidate[4]),
        int(candidate[3]),
    )


def _keep_diverse_global(candidates, max_active, active_parent_ids):
    """Reserve every parent's rank-1 child, then fill by cumulative score."""
    if max_active < 1:
        raise ValueError("max_active must be >= 1")
    if not candidates:
        raise RuntimeError("Selective-wide MTP expansion produced no candidates")
    if any(not np.isfinite(float(candidate[0])) for candidate in candidates):
        raise RuntimeError("Selective-wide MTP expansion produced a non-finite score")

    active_parent_ids = [int(parent_id) for parent_id in active_parent_ids]
    if len(active_parent_ids) != len(set(active_parent_ids)):
        raise RuntimeError("Selective-wide active frontier contains duplicate parent ids")
    if len(active_parent_ids) > max_active:
        raise RuntimeError(
            f"Cannot reserve {len(active_parent_ids)} parent lineages with cap {max_active}"
        )

    active_parent_set = set(active_parent_ids)
    candidate_parent_set = {int(candidate[2]) for candidate in candidates}
    if candidate_parent_set != active_parent_set:
        raise RuntimeError(
            "Selective-wide candidates do not cover the exact active parent frontier"
        )

    rank1_by_parent = {}
    for candidate in candidates:
        parent_id = int(candidate[2])
        if int(candidate[4]) != 1:
            continue
        if parent_id in rank1_by_parent:
            raise RuntimeError(
                f"Parent node {parent_id} produced more than one local-rank-1 child"
            )
        rank1_by_parent[parent_id] = candidate
    if set(rank1_by_parent) != active_parent_set:
        missing = sorted(active_parent_set - set(rank1_by_parent))
        raise RuntimeError(
            f"Active parents missing exactly one local-rank-1 child: {missing}"
        )

    ordered = sorted(candidates, key=_candidate_order)
    if len(ordered) <= max_active:
        return ordered, 0

    reserved = [rank1_by_parent[parent_id] for parent_id in active_parent_ids]
    remaining_slots = max_active - len(reserved)
    optional = [candidate for candidate in ordered if int(candidate[4]) != 1]
    kept = reserved + optional[:remaining_slots]
    kept = sorted(kept, key=_candidate_order)
    if len(kept) != max_active:
        raise RuntimeError(
            f"Diversity pruning kept {len(kept)} children for cap {max_active}"
        )
    kept_rank1_parents = {
        int(candidate[2]) for candidate in kept if int(candidate[4]) == 1
    }
    if kept_rank1_parents != active_parent_set:
        raise RuntimeError("Diversity pruning failed to retain every parent lineage")
    return kept, len(reserved)


def _build_margin_tree_selective_wide(
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
    """Build a uniform-depth selective-wide tree with global frontier pruning."""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if max_active < 1:
        raise ValueError("max_active must be >= 1")
    if not np.isfinite(float(branch_margin)):
        raise ValueError("branch_margin must be finite")

    started = time.perf_counter()
    nodes = [verifier.TreeNode(0, None, int(guaranteed_token), 0, 0.0)]
    generated = 0
    budget_pruned = False

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
            if not active:
                raise RuntimeError(
                    f"Selective-wide MTP frontier is empty before depth {depth}"
                )
            if len(active) > max_active:
                raise RuntimeError(
                    f"Selective-wide parent frontier {len(active)} exceeds cap {max_active}"
                )
            logits_mx = io.logits(active_hidden[:, -1, :])
            logits_batch = np.asarray(logits_mx, dtype=np.float32)
            del logits_mx
            if logits_batch.ndim != 2 or logits_batch.shape[0] != len(active):
                raise RuntimeError(
                    "Selective-wide MTP logits/frontier mismatch: "
                    f"logits={logits_batch.shape} parents={len(active)}"
                )

            candidates = []  # (score, parent row, parent node id, token, local rank)
            local_width_counts = {1: 0, 2: 0, 4: 0, 5: 0}
            for parent_row, (parent_score, parent_id) in enumerate(active):
                if not np.isfinite(float(parent_score)):
                    raise RuntimeError(
                        f"Non-finite cumulative score at parent node {parent_id}"
                    )
                row = logits_batch[parent_row]
                tokens, _ = _select_wide_tokens(row, branch_margin)
                local_width_counts[len(tokens)] += 1
                z = verifier._log_z(row)
                if not np.isfinite(z):
                    raise RuntimeError(
                        f"Non-finite MTP log normalization at parent node {parent_id}"
                    )
                for local_rank, token in enumerate(tokens, 1):
                    score = parent_score + float(row[token]) - z
                    if not np.isfinite(score):
                        raise RuntimeError(
                            f"Non-finite child score at parent node {parent_id}, "
                            f"local rank {local_rank}"
                        )
                    candidates.append(
                        (score, parent_row, parent_id, int(token), local_rank)
                    )

            del logits_batch
            candidate_count = len(candidates)
            generated += candidate_count
            kept, reserved_count = _keep_diverse_global(
                candidates,
                max_active,
                [parent_id for _, parent_id in active],
            )
            if not kept:
                raise RuntimeError(
                    f"Selective-wide MTP frontier is empty after pruning depth {depth}"
                )
            if len(kept) < candidate_count:
                budget_pruned = True

            widths = ",".join(
                f"w{width}:{count}"
                for width, count in local_width_counts.items()
                if count
            )
            print(
                f"selective-wide MTP depth={depth}: parents={len(active)} "
                f"candidates={candidate_count} kept={len(kept)} "
                f"reserved={reserved_count} "
                f"pruned={candidate_count - len(kept)} widths={widths}"
            )

            expanded = []  # (score, parent row, child node id)
            for score, parent_row, parent_id, token, _ in kept:
                node_id = len(nodes)
                nodes.append(
                    verifier.TreeNode(
                        node_id=node_id,
                        parent=parent_id,
                        token=token,
                        depth=depth,
                        score=score,
                    )
                )
                expanded.append((score, parent_row, node_id))

            active = [(score, node_id) for score, _, node_id in expanded]
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
    if not active_node_ids:
        raise RuntimeError("Selective-wide MTP builder produced an empty final frontier")
    compact, active_new = fastbuild._compact_tree(nodes, active_node_ids)
    if not active_new or {compact[node_id].depth for node_id in active_new} != {steps}:
        raise RuntimeError(
            "Selective-wide MTP compaction did not preserve a uniform requested-depth frontier"
        )
    elapsed = time.perf_counter() - started
    print(
        f"selective-wide batched MTP builder: {elapsed:.3f}s; "
        f"requested_depth={steps}, generated={generated}, "
        f"compact_nodes={len(compact)}, final_active={len(active_new)}, "
        f"budget_pruned={'yes' if budget_pruned else 'no'}"
    )
    return compact, active_new, generated, budget_pruned, elapsed


if __name__ == "__main__":
    generator._build_margin_tree_batched = _build_margin_tree_selective_wide
    generator.main()
