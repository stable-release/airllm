"""Leaf-batched native-MTP tree verifier probe for streamed 3-bit Qwen3.8.

The existing exact tree verifier invokes each streamed target layer once per tree depth so it can
retain a recurrent cache checkpoint for every prefix.  This probe tests a different target-side
strategy:

* build the same native-MTP margin tree and keep only its final leaves;
* duplicate each complete root->leaf token path into the batch dimension;
* invoke every streamed target layer exactly once over the full leaf sequences;
* for DeltaNet layers, retain compact pre-recurrence tensors (q/k/v/a/b plus masked pre-conv qkv)
  instead of retaining a multi-megabyte recurrent state for every tree node;
* after target logits select the exact greedy prefix, reconstruct only that prefix's DeltaNet cache
  with ``gated_delta_update`` (no decoder weights need to be reloaded);
* for full-attention layers, retain only new leaf KV suffixes and append the selected prefix;
* compare selected predictions to an ordinary linear target pass, then consume the correction token
  through both recovered and reference caches and require the next greedy token to match.

This is an isolated correctness/performance probe.  It does not modify the validated sustained
native-MTP generator.
"""

import argparse
from dataclasses import dataclass
import time

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.gated_delta import gated_delta_update

from airllm.airllm_qwen35_mlx_fp8_3bit import AirLLMQwen35MlxFp8ThreeBit
from probe_qwen38_mtp_macos import _load_native_mtp, _mlx_memory_mb, _target_forward
import probe_qwen38_mtp_tree_verify_macos as verifier
from probe_qwen38_mtp_tree_verify_fastbuild_macos import _build_margin_tree_batched
from speculative_qwen38_checkpointed_macos import _clone_caches


def _copy_array(value):
    if value is None:
        return None
    out = value + mx.zeros((), dtype=value.dtype)
    mx.eval(out)
    return out


def _path_tokens(nodes, node_id):
    tokens = []
    while node_id is not None and node_id != 0:
        node = nodes[node_id]
        tokens.append(int(node.token))
        node_id = node.parent
    tokens.reverse()
    return tokens


def _repeat_base_cache(cache, batch_size):
    return verifier._gather_cache(cache, [0] * int(batch_size))


@dataclass
class _KVSuffix:
    base_offset: int
    keys: object
    values: object


def _copy_kv_suffix(cache, base_offset):
    """Retain only leaf tokens added after the shared persistent KV prefix."""
    if cache.keys is None:
        raise RuntimeError("Leaf attention verification produced an empty KV cache")
    if base_offset < 0 or base_offset > cache.offset:
        raise RuntimeError(
            f"Invalid KV suffix range: base={base_offset}, final={cache.offset}"
        )
    keys, values = cache.state
    return _KVSuffix(
        base_offset=int(base_offset),
        keys=_copy_array(keys[..., base_offset : cache.offset, :]),
        values=_copy_array(values[..., base_offset : cache.offset, :]),
    )


def _linear_layer_capture(layer, hidden, mask, cache):
    """Run one Qwen3.5 DeltaNet decoder layer and retain inputs needed to rebuild its cache."""
    attn = layer.linear_attn
    if getattr(attn, "sharding_group", None) is not None:
        raise RuntimeError("Leaf verifier probe does not support distributed sharding")
    if cache.lengths is not None or cache.left_padding is not None:
        raise RuntimeError("Leaf verifier probe expects normal generation caches without padding metadata")

    B, S, _ = hidden.shape
    x = layer.input_layernorm(hidden)

    qkv = attn.in_proj_qkv(x)
    z = attn.in_proj_z(x).reshape(B, S, attn.num_v_heads, attn.head_v_dim)
    b = attn.in_proj_b(x)
    a = attn.in_proj_a(x)

    if cache[0] is not None:
        conv_state = cache[0]
    else:
        conv_state = mx.zeros(
            (B, attn.conv_kernel_size - 1, attn.conv_dim), dtype=hidden.dtype
        )

    if mask is not None:
        qkv = mx.where(mask[..., None], qkv, 0)

    conv_input = mx.concatenate([conv_state, qkv], axis=1)
    n_keep = attn.conv_kernel_size - 1
    cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
    conv_out = nn.silu(attn.conv1d(conv_input))

    q, k, v = [
        tensor.reshape(B, S, heads, dim)
        for tensor, heads, dim in zip(
            mx.split(conv_out, [attn.key_dim, 2 * attn.key_dim], -1),
            [attn.num_k_heads, attn.num_k_heads, attn.num_v_heads],
            [attn.head_k_dim, attn.head_k_dim, attn.head_v_dim],
        )
    ]

    inv_scale = k.shape[-1] ** -0.5
    q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
    k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

    state_in = cache[1]
    use_kernel = not attn.training
    out, state_out = gated_delta_update(
        q,
        k,
        v,
        a,
        b,
        attn.A_log,
        attn.dt_bias,
        state_in,
        mask,
        use_kernel=use_kernel,
    )
    cache[1] = state_out
    cache.advance(S)

    r = attn.norm(out, z)
    r = attn.out_proj(r.reshape(B, S, -1))
    h = hidden + r
    result = h + layer.mlp(layer.post_attention_layernorm(h))

    # A_log/dt_bias are tiny but belong to the soon-to-be-evicted streamed layer.  Keep independent
    # evaluated arrays so cache recovery needs no decoder-weight reload.
    A_log = _copy_array(attn.A_log)
    dt_bias = _copy_array(attn.dt_bias)
    capture = {
        "qkv": qkv,
        "q": q,
        "k": k,
        "v": v,
        "a": a,
        "b": b,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "mask": mask,
        "use_kernel": use_kernel,
        "n_keep": n_keep,
        "conv_dim": attn.conv_dim,
    }
    mx.eval(result, cache.state, qkv, q, k, v, a, b, A_log, dt_bias)
    return result, capture


def _verify_leaves(target, nodes, active_leaf_ids, base_caches):
    """Verify every final tree leaf as a complete sequence, one call per streamed layer."""
    if not active_leaf_ids:
        raise ValueError("Leaf verifier requires at least one active leaf")

    depths = {nodes[node_id].depth for node_id in active_leaf_ids}
    if len(depths) != 1:
        raise RuntimeError(f"Expected uniform leaf depth, got {sorted(depths)}")
    depth = next(iter(depths))

    root = int(nodes[0].token)
    leaf_paths = []
    for leaf in active_leaf_ids:
        path = _path_tokens(nodes, leaf)
        if len(path) != depth:
            raise RuntimeError("Compacted tree leaf path length disagrees with node depth")
        leaf_paths.append([root] + path)

    batch_size = len(leaf_paths)
    token_matrix = mx.array(leaf_paths, dtype=mx.int32)

    embedding = target._load_embedding()
    hidden = embedding(token_matrix)
    mx.eval(hidden)
    del embedding
    target._cleanup()

    linear_captures = [None] * target.model_args.num_hidden_layers
    attention_caches = [None] * target.model_args.num_hidden_layers
    layer_times = []

    for layer_index in range(target.model_args.num_hidden_layers):
        started = time.perf_counter()
        layer = target._load_layer(layer_index)
        work = _repeat_base_cache(base_caches[layer_index], batch_size)
        mask = create_ssm_mask(hidden, work) if layer.is_linear else create_attention_mask(hidden, work)

        if layer.is_linear:
            hidden, capture = _linear_layer_capture(layer, hidden, mask, work)
            linear_captures[layer_index] = capture
            # Do not retain the large batch recurrent state in ``work``.
        else:
            hidden = layer(hidden, mask=mask, cache=work)
            mx.eval(hidden, work.state)
            attention_caches[layer_index] = _copy_kv_suffix(
                work,
                base_caches[layer_index].offset,
            )

        mx.eval(hidden)
        del layer, work
        target._cleanup()
        layer_times.append(time.perf_counter() - started)

    norm = target._load_norm()
    normalized_hidden = norm(hidden)
    mx.eval(normalized_hidden)
    del norm, hidden
    target._cleanup()

    B, T, H = normalized_hidden.shape
    flat_logits = target._project_logits(normalized_hidden.reshape(B * T, H))
    predictions = mx.argmax(flat_logits, axis=-1).reshape(B, T)
    predictions = [[int(x) for x in row] for row in predictions.tolist()]
    del flat_logits
    target._cleanup()

    return (
        leaf_paths,
        predictions,
        normalized_hidden,
        linear_captures,
        attention_caches,
        layer_times,
    )


def _select_leaf(leaf_paths, predictions):
    """Choose the leaf with the longest exact greedy-target proposal prefix."""
    best = None
    for row, (tokens, preds) in enumerate(zip(leaf_paths, predictions)):
        depth = len(tokens) - 1
        accepted = 0
        while accepted < depth and int(preds[accepted]) == int(tokens[accepted + 1]):
            accepted += 1
        correction = int(preds[accepted])
        candidate = (accepted, row, correction)
        if best is None or candidate[0] > best[0]:
            best = candidate

    accepted, row, correction = best

    # All leaves tied for the maximum valid prefix must imply the same accepted token prefix and
    # target correction.  If not, the leaf-selection logic is not representing greedy decoding.
    wanted_prefix = leaf_paths[row][1 : 1 + accepted]
    for other_row, (tokens, preds) in enumerate(zip(leaf_paths, predictions)):
        other_accepted = 0
        depth = len(tokens) - 1
        while other_accepted < depth and int(preds[other_accepted]) == int(tokens[other_accepted + 1]):
            other_accepted += 1
        if other_accepted == accepted and other_accepted == best[0]:
            if tokens[1 : 1 + accepted] != wanted_prefix or int(preds[accepted]) != correction:
                raise RuntimeError(
                    f"Ambiguous greedy leaf selection between rows {row} and {other_row}"
                )

    return row, accepted, correction


def _recover_selected_caches(
    target_caches,
    linear_captures,
    attention_caches,
    selected_row,
    keep_tokens,
):
    """Commit root + accepted proposals without retaining batched tree state."""
    for layer_index, base in enumerate(target_caches):
        capture = linear_captures[layer_index]
        if capture is not None:
            if not isinstance(base, ArraysCache):
                raise TypeError("DeltaNet capture paired with non-ArraysCache")
            if base.lengths is not None or base.left_padding is not None:
                raise RuntimeError("Cache recovery expects normal generation ArraysCache metadata")

            qkv = capture["qkv"][selected_row : selected_row + 1, :keep_tokens]
            if base[0] is None:
                conv_state = mx.zeros(
                    (1, capture["n_keep"], capture["conv_dim"]), dtype=qkv.dtype
                )
            else:
                conv_state = base[0]
            conv_input = mx.concatenate([conv_state, qkv], axis=1)
            new_conv = mx.contiguous(conv_input[:, -capture["n_keep"] :, :])

            q = capture["q"][selected_row : selected_row + 1, :keep_tokens]
            k = capture["k"][selected_row : selected_row + 1, :keep_tokens]
            v = capture["v"][selected_row : selected_row + 1, :keep_tokens]
            a = capture["a"][selected_row : selected_row + 1, :keep_tokens]
            b = capture["b"][selected_row : selected_row + 1, :keep_tokens]
            mask = capture["mask"]
            if mask is not None:
                mask = mask[selected_row : selected_row + 1, :keep_tokens]

            _, new_state = gated_delta_update(
                q,
                k,
                v,
                a,
                b,
                capture["A_log"],
                capture["dt_bias"],
                base[1],
                mask,
                use_kernel=capture["use_kernel"],
            )
            base[0] = new_conv
            base[1] = new_state
            base.advance(keep_tokens)
            mx.eval(base.state)
            continue

        saved = attention_caches[layer_index]
        if not isinstance(saved, _KVSuffix) or not isinstance(base, KVCache):
            raise TypeError("Full-attention layer missing saved KV batch")
        if base.offset != saved.base_offset:
            raise RuntimeError(
                f"Persistent KV offset changed during leaf verification: "
                f"base={base.offset}, captured={saved.base_offset}"
            )
        if keep_tokens > saved.keys.shape[-2]:
            raise RuntimeError(
                f"KV commit requested {keep_tokens} tokens from a "
                f"{saved.keys.shape[-2]}-token suffix"
            )

        selected_keys = saved.keys[
            selected_row : selected_row + 1,
            ...,
            :keep_tokens,
            :,
        ]
        selected_values = saved.values[
            selected_row : selected_row + 1,
            ...,
            :keep_tokens,
            :,
        ]
        if base.keys is None:
            base.keys = _copy_array(selected_keys)
            base.values = _copy_array(selected_values)
        else:
            prefix_keys = base.keys[..., : base.offset, :]
            prefix_values = base.values[..., : base.offset, :]
            base.keys = mx.concatenate([prefix_keys, selected_keys], axis=-2)
            base.values = mx.concatenate([prefix_values, selected_values], axis=-2)
        base.offset += keep_tokens
        mx.eval(base.keys, base.values)

    mx.eval([cache.state for cache in target_caches])


def _piece(tokenizer, token_id):
    return tokenizer.decode([int(token_id)], skip_special_tokens=False)


def main():
    parser = argparse.ArgumentParser(
        description="Probe one-call-per-layer leaf-batched verification for native Qwen3.8 MTP trees."
    )
    parser.add_argument("--model", default="orcarouter/Qwen3.8-27B-Uncensored-FP8")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--branch-margin", type=float, default=5.0)
    parser.add_argument("--max-active", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument(
        "--prompt",
        default="Explain why Rust's ownership system prevents use-after-free bugs, using a short concrete example.",
    )
    args = parser.parse_args()

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

    target_caches = [target._new_cache(i) for i in range(target.model_args.num_hidden_layers)]

    print("prefilling target and capturing normalized prompt hidden states...")
    prefill_started = time.perf_counter()
    target_hidden, target_logits = _target_forward(target, prompt_ids, target_caches)
    prefill_s = time.perf_counter() - prefill_started
    guaranteed = int(mx.argmax(target_logits[:, -1, :], axis=-1).item())

    shifted_ids = prompt_ids[1:] + [guaranteed]
    print("building fast native-MTP margin tree...")
    nodes, active, generated_nodes, budget_pruned, build_s = _build_margin_tree_batched(
        target,
        mtp,
        shifted_ids,
        target_hidden,
        guaranteed,
        steps=args.steps,
        branch_margin=args.branch_margin,
        max_active=args.max_active,
    )

    reference_caches, clone_s = _clone_caches(target_caches)

    print("\nverifying final leaves with one call per streamed target layer...")
    verify_started = time.perf_counter()
    (
        leaf_paths,
        predictions,
        normalized_hidden,
        linear_captures,
        attention_caches,
        layer_times,
    ) = _verify_leaves(target, nodes, active, target_caches)
    verify_s = time.perf_counter() - verify_started

    selected_row, accepted, correction = _select_leaf(leaf_paths, predictions)
    selected_tokens = leaf_paths[selected_row]
    accepted_tokens = selected_tokens[1 : 1 + accepted]
    keep_tokens = 1 + accepted

    print("\n=== leaf-selected target path ===")
    print(f"leaf rows:             {len(leaf_paths)}")
    print(f"leaf sequence length:  {len(selected_tokens)}")
    print(f"selected row:          {selected_row}")
    print(f"accepted MTP tokens:   {accepted}/{args.steps}")
    print(
        "accepted text:         "
        + repr(target.tokenizer.decode(accepted_tokens, skip_special_tokens=False))
    )
    print(f"correction token:      {correction} {_piece(target.tokenizer, correction)!r}")
    continuation = accepted_tokens + [correction]
    print(
        "verified continuation: "
        + repr(target.tokenizer.decode(continuation, skip_special_tokens=False))
    )

    print("\nrunning exact linear reference on selected consumed prefix...")
    ref_started = time.perf_counter()
    _, ref_logits = _target_forward(
        target,
        [guaranteed] + accepted_tokens,
        reference_caches,
    )
    ref_s = time.perf_counter() - ref_started
    ref_predictions = [int(x) for x in mx.argmax(ref_logits[0], axis=-1).tolist()]
    leaf_predictions = predictions[selected_row][:keep_tokens]
    prediction_ok = leaf_predictions == ref_predictions
    print(f"leaf predictions:      {leaf_predictions}")
    print(f"linear predictions:    {ref_predictions}")
    print(f"exact prediction match:{' YES' if prediction_ok else ' NO'}")
    if not prediction_ok:
        raise RuntimeError("Leaf-batched verifier logits disagree with linear target reference")

    print("\nreconstructing selected target caches from captured DeltaNet recurrence inputs...")
    recover_started = time.perf_counter()
    _recover_selected_caches(
        target_caches,
        linear_captures,
        attention_caches,
        selected_row,
        keep_tokens,
    )
    recover_s = time.perf_counter() - recover_started

    # Strong cache correctness test: both recovered and ordinary reference caches are now after
    # root + accepted proposals.  Consume the target correction in each and compare the NEXT greedy
    # token.  This catches state-reconstruction errors that current-block logits cannot reveal.
    print("validating reconstructed caches on the next target step...")
    next_tree_started = time.perf_counter()
    _, tree_next_logits = _target_forward(target, [correction], target_caches)
    tree_next_s = time.perf_counter() - next_tree_started
    next_ref_started = time.perf_counter()
    _, ref_next_logits = _target_forward(target, [correction], reference_caches)
    ref_next_s = time.perf_counter() - next_ref_started
    tree_next = int(mx.argmax(tree_next_logits[:, -1, :], axis=-1).item())
    ref_next = int(mx.argmax(ref_next_logits[:, -1, :], axis=-1).item())
    cache_ok = tree_next == ref_next
    print(
        f"next token recovered:  {tree_next} {_piece(target.tokenizer, tree_next)!r}"
    )
    print(f"next token reference:  {ref_next} {_piece(target.tokenizer, ref_next)!r}")
    print(f"reconstructed cache match:{' YES' if cache_ok else ' NO'}")
    if not cache_ok:
        raise RuntimeError("Recovered leaf-batched target cache diverges on the next greedy token")

    print("\n=== Qwen3.8 leaf-batched MTP verifier probe ===")
    print(f"target prefill:             {prefill_s:.3f} s")
    print(f"MTP resident bytes:        {mtp_bytes / 1024**2:.2f} MiB")
    print(f"MTP tree build:            {build_s:.3f} s")
    print(f"target cache clone:        {clone_s:.3f} s")
    print(f"compact tree nodes:        {len(nodes)}")
    print(f"final leaves:              {len(active)}")
    print(f"generated tree nodes:      {generated_nodes}")
    print(f"budget pruned:             {'YES' if budget_pruned else 'NO'}")
    print(f"leaf target verification:  {verify_s:.3f} s")
    print(f"selected cache recovery:   {recover_s:.3f} s")
    print(f"linear reference pass:     {ref_s:.3f} s (validation only)")
    print(f"recovered next-token pass: {tree_next_s:.3f} s (validation only)")
    print(f"reference next-token pass: {ref_next_s:.3f} s (validation only)")
    print(f"useful continuation tokens:{len(continuation):6d}")
    if continuation:
        production_cycle = build_s + verify_s + recover_s
        print(f"production-like cycle:     {production_cycle:.3f} s")
        print(
            f"production-like throughput:{len(continuation) / production_cycle:7.3f} tok/s"
        )
    print(f"mean streamed layer time:  {sum(layer_times) / len(layer_times):.3f} s")

    active_mb = _mlx_memory_mb("get_active_memory")
    peak_mb = _mlx_memory_mb("get_peak_memory")
    if active_mb is not None:
        print(f"MLX active memory:         {active_mb:.2f} MiB")
    if peak_mb is not None:
        print(f"MLX peak memory:           {peak_mb:.2f} MiB")


if __name__ == "__main__":
    main()
