"""Sustained native-MTP tree generation benchmark for streamed 3-bit Qwen3.8 on macOS.

This is intentionally isolated from the validated linear speculative decoder.  It turns the
working single-cycle native-MTP tree probe into a repeated generation loop while preserving the
same exact target semantics:

* the 3-bit target cache contains every token consumed so far;
* ``current_token`` is the target-guaranteed next token, already emitted but not yet consumed;
* native MTP is rebuilt from the accumulated post-target-norm hidden history each cycle, avoiding
  persistent MTP-cache alignment assumptions;
* an uncertainty-triggered MTP tree is built with the batched resident one-layer MTP builder;
* one depth-batched 27B traversal verifies the whole tree;
* the selected target-consistent cache row is COPIED out of the batched history before history is
  released, so future cycles do not retain an entire tree through a slice/view;
* accepted MTP tokens plus the target correction are emitted; the correction becomes the next
  guaranteed token.

The default depth is 5 because the measured 20-node tree was the best single-cycle point on the
8 GB test Mac (~0.412 tok/s production-relevant cycle versus ~0.399 at depth 4 and ~0.336 at depth
8).  A final one-pass linear reference check is enabled by default and is timed separately.
"""

import argparse
import time

import mlx.core as mx
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.cache import ArraysCache, KVCache

from airllm.airllm_qwen35_mlx_fp8_3bit import AirLLMQwen35MlxFp8ThreeBit
import probe_qwen38_mtp_tree_verify_macos as verifier
from probe_qwen38_mtp_macos import _load_native_mtp, _mlx_memory_mb, _target_forward
from probe_qwen38_mtp_tree_verify_fastbuild_macos import _build_margin_tree_batched


def _copy_array(value):
    if value is None:
        return None
    # Match the checkpointed linear verifier's copy primitive.  Materializing a new expression
    # avoids retaining a large batched tree allocation through a selected-row view.
    copied = value + mx.zeros((), dtype=value.dtype)
    mx.eval(copied)
    return copied


def _copy_selected_cache_row(cache, row):
    """Return an independent batch-size-1 cache for one selected tree node."""
    if isinstance(cache, ArraysCache):
        out = ArraysCache(len(cache.cache))
        out.cache = [
            None if value is None else _copy_array(value[row : row + 1])
            for value in cache.cache
        ]
        if cache.left_padding is not None:
            out.left_padding = _copy_array(cache.left_padding[row : row + 1])
        if cache.lengths is not None:
            out.lengths = _copy_array(cache.lengths[row : row + 1])
        return out

    if isinstance(cache, KVCache):
        out = KVCache()
        out.offset = cache.offset
        if cache.keys is not None:
            # Only the logical prefix matters.  Do not copy KVCache's capacity padding.
            out.keys = _copy_array(cache.keys[row : row + 1, ..., : cache.offset, :])
            out.values = _copy_array(cache.values[row : row + 1, ..., : cache.offset, :])
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
        f"Cannot assign selected cache {type(src).__name__} to {type(dst).__name__}"
    )


def _verify_tree_batched_with_hidden(target, nodes, base_caches):
    """Verify one MTP tree and retain normalized hidden states for every tree node.

    This is the proven depth-batched target verifier with one additional return value: the
    post-final-RMSNorm hidden vector of each node in compact node-id order.  Those selected hidden
    vectors are the exact target-side history required to rebuild reference-aligned MTP next cycle.
    """
    by_depth, depth_positions = verifier._depth_layout(nodes)

    token_ids = mx.array([node.token for node in nodes], dtype=mx.int32)[:, None]
    embedding = target._load_embedding()
    all_hidden = embedding(token_ids)
    mx.eval(all_hidden)
    del embedding
    target._cleanup()

    hidden_by_depth = [all_hidden[ids] for ids in by_depth]
    del all_hidden

    histories = []
    layer_times = []

    for layer_index in range(target.model_args.num_hidden_layers):
        layer_started = time.perf_counter()
        layer = target._load_layer(layer_index)
        base = base_caches[layer_index]
        states_by_depth = []

        for depth, ids in enumerate(by_depth):
            if depth == 0:
                work = verifier._gather_cache(base, [0])
            else:
                prev_state = states_by_depth[depth - 1]
                prev_pos = depth_positions[depth - 1]
                parent_rows = [prev_pos[nodes[node_id].parent] for node_id in ids]
                work = verifier._gather_cache(prev_state, parent_rows)

            hidden = hidden_by_depth[depth]
            mask = (
                create_ssm_mask(hidden, work)
                if layer.is_linear
                else create_attention_mask(hidden, work)
            )
            hidden = layer(hidden, mask=mask, cache=work)
            mx.eval([hidden, work.state])

            if isinstance(work, KVCache):
                work = verifier._compact_kv_cache(work)
                mx.eval(work.state)

            hidden_by_depth[depth] = hidden
            states_by_depth.append(work)

        del layer
        target._cleanup()
        histories.append(states_by_depth)
        layer_times.append(time.perf_counter() - layer_started)

    ordered_ids = [node_id for ids in by_depth for node_id in ids]
    if ordered_ids != list(range(len(nodes))):
        raise RuntimeError("Tree nodes are not compacted in depth order")

    hidden = mx.concatenate(hidden_by_depth, axis=0)
    norm = target._load_norm()
    normalized_hidden = norm(hidden)
    mx.eval(normalized_hidden)
    del norm, hidden, hidden_by_depth
    target._cleanup()

    logits = target._project_logits(normalized_hidden[:, 0, :])
    predictions = [int(x) for x in mx.argmax(logits, axis=-1).tolist()]
    mx.eval(logits)
    del logits
    target._cleanup()

    return (
        predictions,
        normalized_hidden,
        histories,
        by_depth,
        depth_positions,
        layer_times,
    )


def _commit_selected_node(base_caches, histories, nodes, depth_positions, node_id):
    """Copy the selected node's per-layer cache out of the batched tree history."""
    node = nodes[node_id]
    row = depth_positions[node.depth][node_id]
    for base, layer_history in zip(base_caches, histories):
        selected = _copy_selected_cache_row(layer_history[node.depth], row)
        _assign_cache(base, selected)
    mx.eval([cache.state for cache in base_caches])


def _append_selected_hidden(hidden_history, normalized_hidden, node_ids):
    """Append target hidden vectors for consumed root+accepted tree nodes in path order."""
    idx = mx.array(list(map(int, node_ids)), dtype=mx.int32)
    # normalized_hidden is [tree_nodes, 1, H]; history is [1, sequence, H].
    selected = normalized_hidden[idx].transpose(1, 0, 2)
    selected = _copy_array(selected)
    combined = mx.concatenate([hidden_history, selected], axis=1)
    mx.eval(combined)
    return combined


def _eos_ids(tokenizer):
    eos = tokenizer.eos_token_id
    if eos is None:
        return set()
    if isinstance(eos, int):
        return {int(eos)}
    return {int(x) for x in eos}


def _validate_full_output(target, prompt_ids, generated):
    """Validate all generated tokens with one fresh linear full-sequence target pass."""
    if not generated:
        return True, [], 0.0

    caches = [
        target._new_cache(i) for i in range(target.model_args.num_hidden_layers)
    ]
    # To predict every generated token, consume the prompt plus all generated tokens except the
    # final one.  Logits at positions prompt[-1], generated[0], ... predict generated[0..].
    sequence = list(map(int, prompt_ids)) + list(map(int, generated[:-1]))
    started = time.perf_counter()
    _, logits = _target_forward(target, sequence, caches)
    elapsed = time.perf_counter() - started
    start = len(prompt_ids) - 1
    predictions = [
        int(x)
        for x in mx.argmax(logits[0, start : start + len(generated), :], axis=-1).tolist()
    ]
    return predictions == list(map(int, generated)), predictions, elapsed


def main():
    parser = argparse.ArgumentParser(
        description="Sustained depth-batched native-MTP tree generation for streamed Qwen3.8."
    )
    parser.add_argument("--model", default="orcarouter/Qwen3.8-27B-Uncensored-FP8")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--branch-margin", type=float, default=5.0)
    parser.add_argument("--max-active", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=31)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument(
        "--no-final-reference",
        action="store_true",
        help="Skip the final one-pass exact linear output validation.",
    )
    parser.add_argument(
        "--prompt",
        default="Explain why Rust's ownership system prevents use-after-free bugs, using a short concrete example.",
    )
    args = parser.parse_args()

    if args.steps < 1:
        raise ValueError("--steps must be >= 1")
    if args.max_active < 1:
        raise ValueError("--max-active must be >= 1")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be >= 1")

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

    print("prefilling target and capturing full normalized hidden history...")
    prefill_started = time.perf_counter()
    hidden_history, target_logits = _target_forward(target, prompt_ids, target_caches)
    prefill_s = time.perf_counter() - prefill_started
    current_token = int(mx.argmax(target_logits[:, -1, :], axis=-1).item())
    mx.eval(hidden_history)

    context_ids = list(map(int, prompt_ids))
    generated = [current_token]
    eos_ids = _eos_ids(target.tokenizer)

    print(f"target:                 {args.model}")
    print(f"prompt tokens:          {len(prompt_ids)}")
    print(f"tree depth:             {args.steps}")
    print(f"branch margin:          {args.branch_margin:.3f}")
    print(f"max active paths:       {args.max_active}")
    print(f"MTP resident bytes:     {mtp_bytes / 1024**2:.2f} MiB")
    print(
        f"initial guaranteed:     {current_token} "
        f"{target.tokenizer.decode([current_token], skip_special_tokens=False)!r}"
    )

    cycle_count = 0
    accepted_total = 0
    useful_total = 0
    build_total_s = 0.0
    verify_total_s = 0.0
    commit_total_s = 0.0
    cleanup_total_s = 0.0
    total_tree_nodes = 0
    max_tree_nodes = 0
    max_frontier = 0
    stopped = current_token in eos_ids

    generation_started = time.perf_counter()

    while len(generated) < args.max_new_tokens and not stopped:
        remaining = args.max_new_tokens - len(generated)
        # One cycle emits accepted MTP tokens plus one correction.  Avoid knowingly overshooting the
        # requested cap; with one slot remaining there is no reason to run another full tree cycle.
        step_budget = min(args.steps, max(remaining - 1, 0))
        if step_budget <= 0:
            break

        if hidden_history.shape[1] != len(context_ids):
            raise RuntimeError(
                f"Target hidden/token history drift: hidden={hidden_history.shape[1]} "
                f"tokens={len(context_ids)}"
            )

        cycle_count += 1
        shifted_ids = context_ids[1:] + [current_token]

        build_started = time.perf_counter()
        nodes, active, generated_nodes, budget_pruned, _ = _build_margin_tree_batched(
            target,
            mtp,
            shifted_ids,
            hidden_history,
            current_token,
            steps=step_budget,
            branch_margin=args.branch_margin,
            max_active=args.max_active,
        )
        build_s = time.perf_counter() - build_started
        build_total_s += build_s
        total_tree_nodes += len(nodes)
        max_tree_nodes = max(max_tree_nodes, len(nodes))
        max_frontier = max(max_frontier, len(active))
        actual_depth = max(int(node.depth) for node in nodes)

        verify_started = time.perf_counter()
        (
            predictions,
            normalized_hidden,
            histories,
            by_depth,
            depth_positions,
            layer_times,
        ) = _verify_tree_batched_with_hidden(target, nodes, target_caches)
        verify_s = time.perf_counter() - verify_started
        verify_total_s += verify_s

        accepted, selected_node, correction = verifier._select_target_path(nodes, predictions)
        accepted_tokens = [nodes[node_id].token for node_id in accepted]
        path_node_ids = [0] + accepted
        consumed_tokens = [current_token] + accepted_tokens

        commit_started = time.perf_counter()
        _commit_selected_node(
            target_caches,
            histories,
            nodes,
            depth_positions,
            selected_node,
        )
        hidden_history = _append_selected_hidden(
            hidden_history,
            normalized_hidden,
            path_node_ids,
        )
        context_ids.extend(map(int, consumed_tokens))
        commit_s = time.perf_counter() - commit_started
        commit_total_s += commit_s

        emitted = accepted_tokens + [int(correction)]
        room = args.max_new_tokens - len(generated)
        emitted = emitted[:room]
        generated.extend(map(int, emitted))
        accepted_total += len(accepted_tokens)
        useful_total += len(emitted)

        print(
            f"\n[tree {cycle_count}] requested_depth={step_budget} actual_depth={actual_depth} "
            f"nodes={len(nodes)} frontier={len(active)} "
            f"accepted={len(accepted_tokens)}/{actual_depth} "
            f"budget_pruned={'yes' if budget_pruned else 'no'}"
        )
        print(
            f"  build={build_s:.3f}s verify={verify_s:.3f}s commit={commit_s:.3f}s "
            f"cycle={(build_s + verify_s + commit_s):.3f}s"
        )
        print(
            "  emitted="
            + repr(target.tokenizer.decode(emitted, skip_special_tokens=False))
        )
        if layer_times:
            print(f"  mean streamed layer={sum(layer_times) / len(layer_times):.3f}s")

        active_mb = _mlx_memory_mb("get_active_memory")
        peak_mb = _mlx_memory_mb("get_peak_memory")
        if active_mb is not None:
            print(f"  MLX active={active_mb:.2f} MiB", end="")
            if peak_mb is not None:
                print(f" peak={peak_mb:.2f} MiB")
            else:
                print()

        current_token = int(correction)
        if current_token in eos_ids or any(int(tok) in eos_ids for tok in emitted):
            stopped = True

        # The target cache and compact hidden history are now independent of the tree histories.
        # Drop the large branch allocations before the next cycle.
        cleanup_started = time.perf_counter()
        del histories, normalized_hidden, predictions, nodes, active, by_depth, depth_positions
        clear_cache = getattr(mx, "clear_cache", None)
        if clear_cache is not None:
            clear_cache()
        mx.eval([cache.state for cache in target_caches], hidden_history)
        cleanup_total_s += time.perf_counter() - cleanup_started

    generation_s = time.perf_counter() - generation_started

    final_reference_s = 0.0
    reference_ok = None
    if not args.no_final_reference:
        print("\nrunning one-pass exact linear reference over the full generated continuation...")
        reference_ok, reference_predictions, final_reference_s = _validate_full_output(
            target, prompt_ids, generated
        )
        print(f"full-output exact match: {'YES' if reference_ok else 'NO'}")
        if not reference_ok:
            print("generated ids: ", generated)
            print("reference ids: ", reference_predictions)
            raise RuntimeError(
                "Sustained tree generation diverged from a fresh linear target reference."
            )

    decoded = target.tokenizer.decode(generated, skip_special_tokens=True)
    steady_outputs = max(len(generated) - 1, 0)
    tree_cycle_s = build_total_s + verify_total_s + commit_total_s

    print("\ndecoded:", decoded)
    print("\n=== sustained Qwen3.8 native-MTP tree benchmark ===")
    print(f"target prefill:                {prefill_s:.3f} s")
    print(f"generated tokens incl initial:{len(generated):6d}")
    print(f"steady outputs after prefill: {steady_outputs:6d}")
    print(f"tree cycles:                  {cycle_count:6d}")
    print(f"accepted MTP tokens:          {accepted_total:6d}")
    print(f"useful tree outputs:          {useful_total:6d}")
    print(f"MTP tree build total:         {build_total_s:.3f} s")
    print(f"target tree verify total:     {verify_total_s:.3f} s")
    print(f"tree cache/history commit:    {commit_total_s:.3f} s")
    print(f"post-cycle cleanup total:     {cleanup_total_s:.3f} s")
    print(f"measured generation wall:     {generation_s:.3f} s")
    print(f"tree compute subtotal:        {tree_cycle_s:.3f} s")
    if steady_outputs:
        print(
            f"sustained throughput:         {steady_outputs / generation_s:.3f} tok/s"
        )
        print(
            f"tree-compute throughput:      {steady_outputs / tree_cycle_s:.3f} tok/s"
        )
    print(f"avg compact tree nodes:       {total_tree_nodes / max(cycle_count, 1):.2f}")
    print(f"max compact tree nodes:       {max_tree_nodes}")
    print(f"max final frontier:           {max_frontier}")
    if reference_ok is not None:
        print(f"final linear validation:      {'PASS' if reference_ok else 'FAIL'}")
        print(f"final validation pass:        {final_reference_s:.3f} s (excluded above)")

    active_mb = _mlx_memory_mb("get_active_memory")
    peak_mb = _mlx_memory_mb("get_peak_memory")
    if active_mb is not None:
        print(f"MLX active memory:            {active_mb:.2f} MiB")
    if peak_mb is not None:
        print(f"MLX peak memory:              {peak_mb:.2f} MiB")


if __name__ == "__main__":
    main()
