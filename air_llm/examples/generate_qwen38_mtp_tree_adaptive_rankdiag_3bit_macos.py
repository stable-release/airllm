"""Diagnose native-MTP correction ranks during adaptive sustained Qwen3.8 generation.

This keeps the validated adaptive target verifier/generator unchanged and instruments only
MTP tree construction.  For every active MTP parent it retains a CPU copy of that parent's
logits until target verification chooses the exact path.  If the target rejects before the
actual tree depth, the wrapper reports the correction token's exact rank in the MTP
distribution at that target-consistent parent plus its local top candidates.

The diagnostic answers whether later sustained-generation misses are cheap top-3/top-4
misses that a wider local branch could rescue, or whether the native MTP head substantially
disagrees with the 3-bit target.  The retained NumPy logits are diagnostic-only and are
cleared once the next tree begins.
"""

import time

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import KVCache

import generate_qwen38_mtp_tree_3bit_macos as generator
import probe_qwen38_mtp_tree_verify_macos as verifier
import probe_qwen38_mtp_tree_verify_fastbuild_macos as fastbuild


# Maps the MTP path already consumed (excluding the guaranteed root token) to the
# full CPU logits that predict the next speculative token from that parent.
_PARENT_LOGITS = {}


def _path_tuple(nodes, node_id):
    tokens = []
    while node_id is not None and node_id != 0:
        node = nodes[node_id]
        tokens.append(int(node.token))
        node_id = node.parent
    tokens.reverse()
    return tuple(tokens)


def _build_margin_tree_stop_before_prune_rankdiag(
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
    """Adaptive stop-before-prune tree builder with per-parent MTP logits retained."""
    _PARENT_LOGITS.clear()
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
        active = [(0.0, 0)]  # cumulative score, node id

        for depth in range(1, steps + 1):
            logits_mx = io.logits(active_hidden[:, -1, :])
            logits_batch = np.asarray(logits_mx, dtype=np.float32)
            del logits_mx

            candidates_by_parent = []
            candidate_count = 0
            for parent_row, (score, parent_id) in enumerate(active):
                row = logits_batch[parent_row]
                # Keep an independent CPU row because logits_batch is released after this depth.
                _PARENT_LOGITS[_path_tuple(nodes, parent_id)] = row.copy()

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

            expanded = []
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
        f"rankdiag adaptive MTP builder: {elapsed:.3f}s; requested_depth={steps}, "
        f"actual_depth={reached_depth}, generated={generated}, "
        f"final_active={len(active_new)}"
    )
    return compact, active_new, generated, budget_pruned, elapsed


_ORIGINAL_SELECT = verifier._select_target_path


def _select_target_path_rankdiag(nodes, predictions):
    accepted, current, wanted = _ORIGINAL_SELECT(nodes, predictions)
    actual_depth = max(node.depth for node in nodes)

    # A correction after accepting the deepest proposal is the normal target bonus token,
    # not an MTP rejection.  Diagnose only early termination inside the built tree.
    if len(accepted) < actual_depth:
        path = tuple(int(nodes[node_id].token) for node_id in accepted)
        row = _PARENT_LOGITS.get(path)
        if row is None:
            print(
                f"MTP rejection diagnostic: accepted={len(accepted)}/{actual_depth}; "
                "no retained parent logits found"
            )
        else:
            wanted = int(wanted)
            wanted_logit = float(row[wanted])
            rank = 1 + int(np.count_nonzero(row > wanted_logit))
            top = fastbuild._topk(row, 8)
            top1 = int(top[0])
            gap = float(row[top1] - wanted_logit)
            tok = getattr(generator, "_DIAG_TOKENIZER", None)
            print(
                f"MTP rejection diagnostic: accepted={len(accepted)}/{actual_depth} "
                f"target_correction={wanted} rank={rank} top1_gap={gap:.4f}"
            )
            print("  local MTP top candidates:")
            for i, token in enumerate(top, start=1):
                token = int(token)
                marker = " <-- target correction" if token == wanted else ""
                if tok is None:
                    piece = ""
                else:
                    piece = f" piece={tok.decode([token], skip_special_tokens=False)!r}"
                print(
                    f"    {i:2d}. id={token:6d} logit={float(row[token]):9.4f}"
                    f"{piece}{marker}"
                )

    return accepted, current, wanted


if __name__ == "__main__":
    # The generator creates the tokenizer inside main.  Wrap the target constructor so the
    # diagnostic formatter can discover it without changing the validated generation loop.
    original_target_cls = generator.AirLLMQwen35MlxFp8ThreeBit

    class _DiagTarget(original_target_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            generator._DIAG_TOKENIZER = self.tokenizer

    generator.AirLLMQwen35MlxFp8ThreeBit = _DiagTarget
    generator._build_margin_tree_batched = _build_margin_tree_stop_before_prune_rankdiag
    verifier._select_target_path = _select_target_path_rankdiag
    generator.main()
