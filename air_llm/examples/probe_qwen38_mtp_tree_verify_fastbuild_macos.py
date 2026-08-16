"""Depth-batched target-tree verification with a batched resident native-MTP builder.

This keeps the validated linear speculative decoder untouched and replaces only the slow
prompt-replay tree builder used by ``probe_qwen38_mtp_tree_verify_macos.py``.

The builder:
  * temporarily pins the shared target embedding and lm_head while native MTP is drafting;
  * prefills the one-layer MTP cache once over the aligned prompt;
  * scores every active path in one batched lm_head projection per tree depth;
  * advances every selected child in one batched MTP step per tree depth by gathering its
    parent's KV row;
  * releases the shared embedding/lm_head before the streamed 27B tree verifier runs.

The target-side verifier and its exact linear-reference check are reused unchanged.
"""

import time

import mlx.core as mx
import numpy as np
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import KVCache

import probe_qwen38_mtp_tree_verify_macos as verifier
from probe_qwen38_mtp_rank_macos import _topk


class _PinnedSharedIO:
    """Temporarily keep target embedding/lm_head resident during MTP tree construction."""

    def __init__(self, target):
        self.target = target
        self.previous_names = set(target.resident_layer_names)
        self.embed_name = target.layer_names_dict["embed"]
        self.head_name = target.layer_names_dict["lm_head"]
        self.embedding = None
        self.head = None

    def __enter__(self):
        names = set(self.previous_names)
        names.add(self.embed_name)
        if not self.target.model_args.tie_word_embeddings:
            names.add(self.head_name)
        self.target.resident_layer_names = names

        self.embedding = self.target._load_embedding()
        self.target._materialize_module(self.embedding)

        if not self.target.model_args.tie_word_embeddings:
            # Trigger construction/materialization of the fast backend's resident head once.
            dummy = mx.zeros((1, self.target.model_args.hidden_size), dtype=mx.float16)
            logits = self.target._project_logits(dummy)
            mx.eval(logits)
            del logits, dummy
            self.head = self.target._resident_lm_head
            if self.head is None:
                raise RuntimeError("Failed to materialize resident lm_head for MTP tree builder")
        return self

    def logits(self, hidden):
        if self.target.model_args.tie_word_embeddings:
            logits = self.embedding.as_linear(hidden)
        else:
            logits = self.head(hidden)
        mx.eval(logits)
        return logits

    def __exit__(self, exc_type, exc, tb):
        self.target.resident_layer_names = set(self.previous_names)
        if self.embed_name not in self.previous_names:
            self.target._resident_embedding = None
        if self.head_name not in self.previous_names:
            self.target._resident_lm_head = None
        self.embedding = None
        self.head = None
        clear_cache = getattr(mx, "clear_cache", None)
        if clear_cache is not None:
            clear_cache()


def _gather_mtp_cache(cache, rows):
    if not isinstance(cache, KVCache):
        raise TypeError(f"Expected MTP KVCache, got {type(cache).__name__}")
    idx = mx.array(list(map(int, rows)), dtype=mx.int32)
    out = KVCache()
    out.offset = cache.offset
    if cache.keys is not None:
        out.keys = cache.keys[idx]
        out.values = cache.values[idx]
    return out


def _mtp_forward_resident(target, mtp, cache, token_ids, hidden_states, embedding):
    """Run resident native MTP for either one sequence or a batch of one-token paths."""
    tokens = mx.array(token_ids, dtype=mx.int32)
    if tokens.ndim == 1:
        tokens = tokens[None, :]
    if hidden_states.shape[0] != tokens.shape[0] or hidden_states.shape[1] != tokens.shape[1]:
        raise ValueError(
            "Batched MTP alignment mismatch: "
            f"tokens={tokens.shape}, hidden={hidden_states.shape[:2]}"
        )

    embedded = embedding(tokens)
    embedded = mtp.pre_fc_norm_embedding(embedded)
    hidden = mtp.pre_fc_norm_hidden(hidden_states)
    x = mtp.fc(mx.concatenate([embedded, hidden], axis=-1))

    layer = mtp.layers[0]
    mask = create_attention_mask(x, cache)
    h = layer(x, mask=mask, cache=cache)
    h = mtp.norm(h)
    mx.eval(h, cache.state)
    return h


def _compact_tree(nodes, active_node_ids):
    keep = {0}
    for leaf in active_node_ids:
        node_id = leaf
        while node_id is not None:
            keep.add(node_id)
            node_id = nodes[node_id].parent

    old_to_new = {}
    compact = []
    for old in sorted(keep, key=lambda i: (nodes[i].depth, i)):
        old_to_new[old] = len(compact)
        node = nodes[old]
        compact.append(
            verifier.TreeNode(
                node_id=len(compact),
                parent=None,
                token=node.token,
                depth=node.depth,
                score=node.score,
            )
        )

    for old, new in old_to_new.items():
        parent = nodes[old].parent
        compact[new].parent = None if parent is None else old_to_new[parent]

    return compact, [old_to_new[node_id] for node_id in active_node_ids]


def _build_margin_tree_batched(
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
    """Build the same margin tree without replaying MTP state from the prompt per branch."""
    started = time.perf_counter()
    nodes = [verifier.TreeNode(0, None, int(guaranteed_token), 0, 0.0)]
    generated = 0
    budget_pruned = False

    with _PinnedSharedIO(target) as io:
        # Reference-aligned first MTP pass.  This is the only prompt-length MTP call.
        mtp_cache = KVCache()
        mtp_hidden = _mtp_forward_resident(
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

            expanded = []  # (score, parent row, child node id)
            for parent_row, (score, parent_id) in enumerate(active):
                row = logits_batch[parent_row]
                top2 = _topk(row, 2)
                t1, t2 = int(top2[0]), int(top2[1])
                margin = float(row[t1] - row[t2])
                candidates = [t1]
                if margin <= branch_margin:
                    candidates.append(t2)

                z = verifier._log_z(row)
                for token in candidates:
                    child_score = score + float(row[token]) - z
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

            del logits_batch

            if len(expanded) > max_active:
                budget_pruned = True
                expanded.sort(key=lambda item: item[0], reverse=True)
                expanded = expanded[:max_active]

            active = [(score, node_id) for score, _, node_id in expanded]

            if depth == steps:
                break

            # Every selected child has exactly one additional token relative to its parent,
            # so the whole next frontier can advance through the resident MTP layer as one batch.
            parent_rows = [parent_row for _, parent_row, _ in expanded]
            child_tokens = [nodes[node_id].token for _, _, node_id in expanded]
            parent_idx = mx.array(parent_rows, dtype=mx.int32)
            parent_hidden = active_hidden[parent_idx]
            child_cache = _gather_mtp_cache(active_cache, parent_rows)
            active_hidden = _mtp_forward_resident(
                target,
                mtp,
                child_cache,
                mx.array(child_tokens, dtype=mx.int32)[:, None],
                parent_hidden,
                io.embedding,
            )
            active_cache = child_cache

            # All outputs/cache rows must be materialized before the next gather/prune.
            mx.eval(active_hidden, active_cache.state)

    active_node_ids = [node_id for _, node_id in active]
    compact, active_new = _compact_tree(nodes, active_node_ids)
    elapsed = time.perf_counter() - started
    print(
        f"batched resident-MTP builder: {elapsed:.3f}s; "
        f"generated={generated}, final_active={len(active_new)}"
    )
    return compact, active_new, generated, budget_pruned, elapsed


if __name__ == "__main__":
    verifier._build_margin_tree = _build_margin_tree_batched
    verifier.main()
