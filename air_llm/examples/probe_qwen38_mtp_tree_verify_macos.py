"""Depth-batched native-MTP tree verification for streamed Qwen3.8 on macOS.

This is the first target-side tree verifier probe.  It intentionally leaves the validated
linear checkpointed generator untouched.

The probe:
  * builds a non-oracle uncertainty-triggered MTP tree;
  * keeps only nodes that are ancestors of the final active frontier;
  * loads each streamed target layer once;
  * evaluates tree nodes in batches by depth, gathering each child's parent recurrent/KV
    state into the batch dimension;
  * selects the exact target-consistent path from the resulting per-node logits;
  * compares those logits against a normal linear target traversal on the selected path.

The MTP tree builder still replays branch state from the prompt for simplicity.  Its timing
is therefore diagnostic overhead, not a production drafting cost.  The target verifier is
the part under test here.
"""

import argparse
import math
import time
from dataclasses import dataclass

import mlx.core as mx
import numpy as np
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.cache import ArraysCache, KVCache

from airllm.airllm_qwen35_mlx_fp8_3bit import AirLLMQwen35MlxFp8ThreeBit
from probe_qwen38_mtp_macos import _load_native_mtp, _mlx_memory_mb, _target_forward
from probe_qwen38_mtp_beam_survival_macos import _replay_path
from probe_qwen38_mtp_rank_macos import _topk
from speculative_qwen38_checkpointed_macos import _clone_caches


@dataclass
class TreeNode:
    node_id: int
    parent: int | None
    token: int
    depth: int
    score: float


def _piece(tokenizer, token_id):
    return tokenizer.decode([int(token_id)], skip_special_tokens=False)


def _path_text(tokenizer, nodes, node_id):
    toks = []
    while node_id is not None and node_id != 0:
        node = nodes[node_id]
        toks.append(node.token)
        node_id = node.parent
    toks.reverse()
    return tokenizer.decode(toks, skip_special_tokens=False)


def _log_z(logits):
    m = float(np.max(logits))
    return m + math.log(float(np.exp(logits - m).sum(dtype=np.float64)))


def _build_margin_tree(
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
    """Build a non-oracle margin-triggered MTP tree using safe prompt replays."""
    nodes = [TreeNode(0, None, int(guaranteed_token), 0, 0.0)]
    active = [(0.0, tuple(), 0)]  # score, MTP path tokens, leaf node id
    generated = 0
    budget_pruned = False

    started = time.perf_counter()
    for depth in range(1, steps + 1):
        expanded = []
        for score, path, parent_id in active:
            logits = _replay_path(target, mtp, shifted_ids, target_hidden, path)
            top2 = _topk(logits, 2)
            t1, t2 = int(top2[0]), int(top2[1])
            margin = float(logits[t1] - logits[t2])
            candidates = [t1]
            if margin <= branch_margin:
                candidates.append(t2)

            z = _log_z(logits)
            for tok in candidates:
                child_score = score + float(logits[tok]) - z
                node_id = len(nodes)
                nodes.append(TreeNode(node_id, parent_id, tok, depth, child_score))
                generated += 1
                expanded.append((child_score, path + (tok,), node_id))

        if len(expanded) > max_active:
            budget_pruned = True
            expanded.sort(key=lambda x: x[0], reverse=True)
            expanded = expanded[:max_active]
        active = expanded

    # Pruned nodes never need target verification.  Keep only ancestors of the final
    # frontier and rebuild compact, depth-ordered ids.
    keep = {0}
    for _, _, leaf in active:
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
            TreeNode(
                node_id=len(compact),
                parent=None,  # filled below
                token=node.token,
                depth=node.depth,
                score=node.score,
            )
        )
    for old, new in old_to_new.items():
        parent = nodes[old].parent
        compact[new].parent = None if parent is None else old_to_new[parent]

    active_new = [old_to_new[leaf] for _, _, leaf in active]
    return compact, active_new, generated, budget_pruned, time.perf_counter() - started


def _depth_layout(nodes):
    max_depth = max(node.depth for node in nodes)
    by_depth = [[] for _ in range(max_depth + 1)]
    for node in nodes:
        by_depth[node.depth].append(node.node_id)

    positions = []
    for ids in by_depth:
        positions.append({node_id: i for i, node_id in enumerate(ids)})
    return by_depth, positions


def _index_array(indices):
    return mx.array(list(map(int, indices)), dtype=mx.int32)


def _gather_arrays_cache(cache, indices):
    idx = _index_array(indices)
    out = ArraysCache(len(cache.cache))
    out.cache = [None if value is None else value[idx] for value in cache.cache]
    if cache.left_padding is not None:
        out.left_padding = cache.left_padding[idx]
    if cache.lengths is not None:
        out.lengths = cache.lengths[idx]
    return out


def _gather_kv_cache(cache, indices):
    idx = _index_array(indices)
    out = KVCache()
    out.offset = cache.offset
    if cache.keys is not None:
        out.keys = cache.keys[idx]
        out.values = cache.values[idx]
    return out


def _gather_cache(cache, indices):
    if isinstance(cache, ArraysCache):
        return _gather_arrays_cache(cache, indices)
    if isinstance(cache, KVCache):
        return _gather_kv_cache(cache, indices)
    raise TypeError(f"Unsupported Qwen tree cache type: {type(cache).__name__}")


def _compact_kv_cache(cache):
    if cache.keys is None:
        return cache
    keys, values = cache.state
    compact = KVCache()
    compact.keys = keys
    compact.values = values
    compact.offset = cache.offset
    return compact


def _slice_cache_row(cache, row):
    if isinstance(cache, ArraysCache):
        out = ArraysCache(len(cache.cache))
        out.cache = [
            None if value is None else value[row : row + 1]
            for value in cache.cache
        ]
        if cache.left_padding is not None:
            out.left_padding = cache.left_padding[row : row + 1]
        if cache.lengths is not None:
            out.lengths = cache.lengths[row : row + 1]
        return out

    if isinstance(cache, KVCache):
        out = KVCache()
        out.offset = cache.offset
        if cache.keys is not None:
            out.keys = cache.keys[row : row + 1]
            out.values = cache.values[row : row + 1]
        return out

    raise TypeError(f"Unsupported Qwen tree cache type: {type(cache).__name__}")


def _assign_cache(dst, src):
    if isinstance(dst, ArraysCache) and isinstance(src, ArraysCache):
        dst.cache = list(src.cache)
        dst.left_padding = src.left_padding
        dst.lengths = src.lengths
        return
    if isinstance(dst, KVCache) and isinstance(src, KVCache):
        dst.keys = src.keys
        dst.values = src.values
        dst.offset = src.offset
        return
    raise TypeError(
        f"Cannot assign tree cache {type(src).__name__} to {type(dst).__name__}"
    )


def _verify_tree_batched(target, nodes, base_caches):
    """Run one streamed target traversal over the candidate tree.

    Returns per-node greedy predictions plus enough layer history to commit any node.
    """
    by_depth, depth_positions = _depth_layout(nodes)

    token_ids = mx.array([node.token for node in nodes], dtype=mx.int32)[:, None]
    embedding = target._load_embedding()
    all_hidden = embedding(token_ids)
    mx.eval(all_hidden)
    del embedding
    target._cleanup()

    hidden_by_depth = [mx.take(all_hidden, _index_array(ids), axis=0) for ids in by_depth]
    del all_hidden

    # histories[layer][depth] is the batched cache after all nodes at that depth.
    histories = []
    layer_times = []

    for layer_index in range(target.model_args.num_hidden_layers):
        layer_started = time.perf_counter()
        layer = target._load_layer(layer_index)
        base = base_caches[layer_index]
        states_by_depth = []

        for depth, ids in enumerate(by_depth):
            if depth == 0:
                work = _gather_cache(base, [0])
            else:
                prev_state = states_by_depth[depth - 1]
                prev_pos = depth_positions[depth - 1]
                parent_rows = [prev_pos[nodes[node_id].parent] for node_id in ids]
                work = _gather_cache(prev_state, parent_rows)

            hidden = hidden_by_depth[depth]
            mask = (
                create_ssm_mask(hidden, work)
                if layer.is_linear
                else create_attention_mask(hidden, work)
            )
            hidden = layer(hidden, mask=mask, cache=work)
            mx.eval([hidden, work.state])

            if isinstance(work, KVCache):
                work = _compact_kv_cache(work)
                mx.eval(work.state)

            hidden_by_depth[depth] = hidden
            states_by_depth.append(work)

        del layer
        target._cleanup()
        histories.append(states_by_depth)
        layer_times.append(time.perf_counter() - layer_started)

    # Node ids are compacted in depth order, so concatenating depth batches recreates
    # exact node-id order.
    ordered_ids = [node_id for ids in by_depth for node_id in ids]
    if ordered_ids != list(range(len(nodes))):
        raise RuntimeError("Tree nodes are not compacted in depth order")

    hidden = mx.concatenate(hidden_by_depth, axis=0)
    norm = target._load_norm()
    hidden = norm(hidden)
    mx.eval(hidden)
    del norm
    target._cleanup()

    logits = target._project_logits(hidden[:, 0, :])
    predictions = [int(x) for x in mx.argmax(logits, axis=-1).tolist()]
    mx.eval(logits)
    del logits, hidden
    target._cleanup()

    return predictions, histories, by_depth, depth_positions, layer_times


def _select_target_path(nodes, predictions):
    children = {node.node_id: [] for node in nodes}
    for node in nodes[1:]:
        children[node.parent].append(node.node_id)

    current = 0
    accepted = []
    while True:
        wanted = predictions[current]
        match = next(
            (child for child in children[current] if nodes[child].token == wanted),
            None,
        )
        if match is None:
            return accepted, current, wanted
        accepted.append(match)
        current = match


def _commit_tree_node(base_caches, histories, nodes, depth_positions, node_id):
    node = nodes[node_id]
    row = depth_positions[node.depth][node_id]
    for base, layer_history in zip(base_caches, histories):
        selected = _slice_cache_row(layer_history[node.depth], row)
        _assign_cache(base, selected)
    mx.eval([cache.state for cache in base_caches])


def main():
    parser = argparse.ArgumentParser(
        description="Depth-batched native-MTP tree verification for streamed Qwen3.8."
    )
    parser.add_argument("--model", default="orcarouter/Qwen3.8-27B-Uncensored-FP8")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--branch-margin", type=float, default=5.0)
    parser.add_argument("--max-active", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument(
        "--prompt",
        default="Explain why Rust's ownership system prevents use-after-free bugs, using a short concrete example.",
    )
    args = parser.parse_args()

    if args.steps < 1:
        raise ValueError("--steps must be >= 1")

    print("loading streamed 3-bit target...")
    target = AirLLMQwen35MlxFp8ThreeBit(
        args.model,
        compression="3bit",
        max_seq_len=args.max_seq_len,
        mlx_resident_gib=0,
    )
    print("loading native MTP weights...")
    mtp, mtp_bytes = _load_native_mtp(target)

    prompt = target.tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_ids = list(target.tokenizer.encode(prompt, add_special_tokens=False))

    target_caches = [
        target._new_cache(i) for i in range(target.model_args.num_hidden_layers)
    ]

    print("prefilling target and capturing normalized prompt hidden states...")
    started = time.perf_counter()
    target_hidden, target_logits = _target_forward(target, prompt_ids, target_caches)
    target_prefill_s = time.perf_counter() - started
    guaranteed = int(mx.argmax(target_logits[:, -1, :], axis=-1).item())

    shifted_ids = prompt_ids[1:] + [guaranteed]

    print("building non-oracle MTP margin tree...")
    nodes, active, generated_nodes, budget_pruned, tree_build_s = _build_margin_tree(
        target,
        mtp,
        shifted_ids,
        target_hidden,
        guaranteed,
        steps=args.steps,
        branch_margin=args.branch_margin,
        max_active=args.max_active,
    )

    print(f"target:              {args.model}")
    print(f"prompt tokens:       {len(prompt_ids)}")
    print(f"guaranteed token:    {guaranteed} {_piece(target.tokenizer, guaranteed)!r}")
    print(f"MTP resident bytes:  {mtp_bytes / 1024**2:.2f} MiB")
    print(f"tree depth:          {args.steps}")
    print(f"compact tree nodes:  {len(nodes) - 1} proposals + root")
    print(f"final active paths:  {len(active)}")
    print(f"generated pre-prune: {generated_nodes}")
    print(f"budget pruned:       {'YES' if budget_pruned else 'NO'}")
    print(f"tree build/replay:   {tree_build_s:.3f} s (diagnostic, not production)")

    for node in nodes[1:]:
        print(
            f"  node={node.node_id:2d} depth={node.depth} parent={node.parent:2d} "
            f"score={node.score:8.4f} token={node.token:6d} "
            f"path={_path_text(target.tokenizer, nodes, node.node_id)!r}"
        )

    # Preserve an untouched prompt cache for an exact linear-reference check.
    reference_caches, clone_s = _clone_caches(target_caches)

    print("\nverifying tree in one depth-batched target traversal...")
    verify_started = time.perf_counter()
    predictions, histories, by_depth, depth_positions, layer_times = _verify_tree_batched(
        target, nodes, target_caches
    )
    verify_s = time.perf_counter() - verify_started

    accepted, selected_node, correction = _select_target_path(nodes, predictions)
    accepted_tokens = [nodes[node_id].token for node_id in accepted]

    print("\n=== target-selected tree path ===")
    print(f"accepted MTP nodes:   {len(accepted_tokens)}/{args.steps}")
    for i, node_id in enumerate(accepted):
        node = nodes[node_id]
        print(
            f"  [{i}] node={node_id} token={node.token} "
            f"piece={_piece(target.tokenizer, node.token)!r}"
        )
    print(
        f"correction token:     {correction} {_piece(target.tokenizer, correction)!r}"
    )
    continuation = accepted_tokens + [correction]
    print(
        "verified continuation: "
        + repr(target.tokenizer.decode(continuation, skip_special_tokens=False))
    )

    print("\nrunning exact linear reference on selected path...")
    ref_started = time.perf_counter()
    _, ref_logits = _target_forward(
        target,
        [guaranteed] + accepted_tokens,
        reference_caches,
    )
    ref_s = time.perf_counter() - ref_started
    ref_predictions = [int(x) for x in mx.argmax(ref_logits[0], axis=-1).tolist()]
    tree_path_nodes = [0] + accepted
    tree_predictions = [predictions[node_id] for node_id in tree_path_nodes]
    reference_ok = tree_predictions == ref_predictions

    print(f"tree predictions:      {tree_predictions}")
    print(f"linear predictions:    {ref_predictions}")
    print(f"exact prediction match:{' YES' if reference_ok else ' NO'}")

    if not reference_ok:
        raise RuntimeError(
            "Depth-batched tree verifier disagrees with linear target reference; "
            "do not use its cache history."
        )

    # Commit the already-computed state after root + accepted proposals.  The correction
    # remains the next unconsumed token, matching the checkpointed linear verifier model.
    commit_started = time.perf_counter()
    _commit_tree_node(target_caches, histories, nodes, depth_positions, selected_node)
    commit_s = time.perf_counter() - commit_started

    print("\n=== depth-batched MTP tree verifier ===")
    print(f"target prefill:             {target_prefill_s:.3f} s")
    print(f"target cache clone:         {clone_s:.3f} s")
    print(f"diagnostic MTP tree build:  {tree_build_s:.3f} s")
    print(f"tree target verification:   {verify_s:.3f} s")
    print(f"linear reference pass:      {ref_s:.3f} s")
    print(f"tree cache commit:          {commit_s:.3f} s")
    print(f"tree nodes incl root:       {len(nodes)}")
    print("nodes by depth:             " + ", ".join(str(len(ids)) for ids in by_depth))
    print(f"accepted MTP tokens:        {len(accepted_tokens)}")
    print(f"useful continuation tokens: {len(continuation)}")
    if continuation:
        print(
            f"verify sec/useful token:    {verify_s / len(continuation):.3f} s"
        )
        print(
            f"verify-only throughput:     {len(continuation) / verify_s:.3f} tok/s"
        )
    print(f"mean streamed layer time:   {sum(layer_times) / len(layer_times):.3f} s")

    active_mb = _mlx_memory_mb("get_active_memory")
    peak_mb = _mlx_memory_mb("get_peak_memory")
    if active_mb is not None:
        print(f"MLX active memory:          {active_mb:.2f} MiB")
    if peak_mb is not None:
        print(f"MLX peak memory:            {peak_mb:.2f} MiB")


if __name__ == "__main__":
    main()
