"""Leaf verifier with unique-node affine work and unchanged leaf recurrence.

Complete root-to-leaf sequences are still carried through every streamed target layer, preserving
the validated causal convolution, DeltaNet recurrence, attention-cache capture, canonical target
selection, and selected-cache recovery paths.  The only optimization is inside tokenwise work:
duplicate occurrences of the same compact tree node share one embedding, affine projection, MLP,
and lm-head evaluation before their result is expanded back to the leaf layout.

The Qwen3.5 DeltaNet recurrence remains a normal ``B x T`` leaf call.  This makes the optimization
substantially less invasive than a custom tree-state recurrence while removing the duplicated
large matrix multiplications which dominate each linear-attention decoder layer.
"""

from dataclasses import dataclass
import time

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.gated_delta import gated_delta_update

import probe_qwen38_mtp_leaf_verify_exactselect_macos as exact_select
import probe_qwen38_mtp_leaf_verify_macos as baseline


@dataclass(frozen=True)
class _UniqueNodeLayout:
    """Map every duplicated leaf position to one compact tree node."""

    node_ids: object
    representative_flat: object
    node_count: int
    leaf_rows: int
    sequence_length: int

    @property
    def leaf_positions(self):
        return self.leaf_rows * self.sequence_length


def _path_node_ids(nodes, node_id):
    path = []
    node_id = int(node_id)
    while node_id != 0:
        path.append(node_id)
        parent = nodes[node_id].parent
        if parent is None:
            raise RuntimeError("Leaf node path terminated before reaching the root")
        node_id = int(parent)
    path.reverse()
    return path


def _build_unique_layout(nodes, active_leaf_ids):
    if not nodes:
        raise ValueError("Unique-node leaf verification requires a non-empty tree")
    if not active_leaf_ids:
        raise ValueError("Unique-node leaf verification requires at least one active leaf")
    for position, node in enumerate(nodes):
        if int(node.node_id) != position:
            raise RuntimeError("Unique-node verifier requires compact positional node ids")

    depths = {int(nodes[node_id].depth) for node_id in active_leaf_ids}
    if len(depths) != 1:
        raise RuntimeError(f"Expected uniform leaf depth, got {sorted(depths)}")
    depth = next(iter(depths))

    leaf_node_paths = []
    for leaf_id in active_leaf_ids:
        path = _path_node_ids(nodes, leaf_id)
        if len(path) != depth:
            raise RuntimeError("Leaf node path length disagrees with final frontier depth")
        leaf_node_paths.append([0] + path)

    representatives = [None] * len(nodes)
    flat_position = 0
    for row in leaf_node_paths:
        for node_id in row:
            if representatives[node_id] is None:
                representatives[node_id] = flat_position
            flat_position += 1

    missing = [node_id for node_id, position in enumerate(representatives) if position is None]
    if missing:
        raise RuntimeError(
            "Compacted tree contains nodes absent from the active leaf paths: "
            f"{missing}"
        )

    node_ids = mx.array(leaf_node_paths, dtype=mx.int32)
    representative_flat = mx.array(representatives, dtype=mx.int32)
    layout = _UniqueNodeLayout(
        node_ids=node_ids,
        representative_flat=representative_flat,
        node_count=len(nodes),
        leaf_rows=len(leaf_node_paths),
        sequence_length=depth + 1,
    )
    mx.eval(node_ids, representative_flat)
    return layout, leaf_node_paths


def _unique_values(value, layout):
    if tuple(value.shape[:2]) != (layout.leaf_rows, layout.sequence_length):
        raise RuntimeError(
            "Leaf tensor/layout shape mismatch: "
            f"tensor={value.shape[:2]} layout="
            f"{(layout.leaf_rows, layout.sequence_length)}"
        )
    flat = value.reshape(layout.leaf_positions, *value.shape[2:])
    return flat[layout.representative_flat]


def _expand_unique(value, layout):
    if value.shape[0] != layout.node_count:
        raise RuntimeError(
            f"Unique tensor has {value.shape[0]} rows for {layout.node_count} nodes"
        )
    return value[layout.node_ids]


def _assert_node_aligned(value, layout, label):
    """Require every duplicate occurrence of a compact node to be bit-identical."""
    unique = _unique_values(value, layout)
    reconstructed = _expand_unique(unique, layout)
    if not bool(mx.array_equal(value, reconstructed).item()):
        raise RuntimeError(
            f"Unique-node optimization is unsafe at {label}: duplicate leaf positions "
            "for one tree node have different hidden values"
        )


def _linear_layer_capture_unique(layer, hidden, mask, cache, layout):
    """Run tokenwise DeltaNet affines once per node and recurrence once per leaf."""
    attn = layer.linear_attn
    if getattr(attn, "sharding_group", None) is not None:
        raise RuntimeError("Unique-node leaf verifier does not support distributed sharding")
    if cache.lengths is not None or cache.left_padding is not None:
        raise RuntimeError(
            "Unique-node leaf verifier expects generation caches without padding metadata"
        )

    B, S, _ = hidden.shape
    if (B, S) != (layout.leaf_rows, layout.sequence_length):
        raise RuntimeError("DeltaNet leaf batch disagrees with unique-node layout")

    # These operations are tokenwise.  Evaluate their large matrices once for every compact node,
    # then restore the complete leaf layout before causal convolution and recurrence.
    unique_hidden = _unique_values(hidden, layout)
    N = unique_hidden.shape[0]
    x_unique = layer.input_layernorm(unique_hidden)

    qkv_unique = attn.in_proj_qkv(x_unique)
    z_unique = attn.in_proj_z(x_unique).reshape(
        N, attn.num_v_heads, attn.head_v_dim
    )
    b_unique = attn.in_proj_b(x_unique)
    a_unique = attn.in_proj_a(x_unique)

    qkv = _expand_unique(qkv_unique, layout)
    z = _expand_unique(z_unique, layout)
    b = _expand_unique(b_unique, layout)
    a = _expand_unique(a_unique, layout)

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

    # Recurrence output for a compact node is path-determined and duplicate rows were checked at
    # the preceding attention boundary.  The remaining norm/out projection/MLP are tokenwise.
    out_unique = _unique_values(out, layout)
    r_unique = attn.norm(out_unique, z_unique)
    r_unique = attn.out_proj(r_unique.reshape(N, -1))
    h_unique = unique_hidden + r_unique
    result_unique = h_unique + layer.mlp(layer.post_attention_layernorm(h_unique))
    result = _expand_unique(result_unique, layout)

    # Keep the established leaf-shaped recurrence captures so selected-cache recovery is reused
    # byte-for-byte.  Only verification-side affine evaluation changes in this probe.
    A_log = baseline._copy_array(attn.A_log)
    dt_bias = baseline._copy_array(attn.dt_bias)
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


def _verify_leaves_unique(target, nodes, active_leaf_ids, base_caches):
    """Verify final leaves while deduplicating tokenwise work by compact node id."""
    layout, leaf_node_paths = _build_unique_layout(nodes, active_leaf_ids)
    leaf_paths = [
        [int(nodes[node_id].token) for node_id in node_path]
        for node_path in leaf_node_paths
    ]

    duplicate_positions = layout.leaf_positions - layout.node_count
    print(
        "unique-node target layout: "
        f"depth={layout.sequence_length - 1}, leaves={layout.leaf_rows}, "
        f"leaf_positions={layout.leaf_positions}, nodes={layout.node_count}, "
        f"duplicate_positions={duplicate_positions}, "
        f"affine_rows_saved={duplicate_positions / layout.leaf_positions:.1%}"
    )

    # Embedding lookup is tokenwise too.  Expanding unique embeddings establishes exact duplicate
    # alignment at the first decoder layer without relying on batch-kernel behavior.
    unique_token_ids = mx.array(
        [int(node.token) for node in nodes], dtype=mx.int32
    )
    embedding = target._load_embedding()
    unique_hidden = embedding(unique_token_ids)
    hidden = _expand_unique(unique_hidden, layout)
    mx.eval(hidden)
    del embedding, unique_hidden, unique_token_ids
    target._cleanup()

    linear_captures = [None] * target.model_args.num_hidden_layers
    attention_caches = [None] * target.model_args.num_hidden_layers
    layer_times = []
    alignment_proven = True
    alignment_checks = 0

    for layer_index in range(target.model_args.num_hidden_layers):
        started = time.perf_counter()
        layer = target._load_layer(layer_index)
        work = baseline._repeat_base_cache(base_caches[layer_index], layout.leaf_rows)
        mask = (
            create_ssm_mask(hidden, work)
            if layer.is_linear
            else create_attention_mask(hidden, work)
        )

        if layer.is_linear:
            # Optimized linear layers explicitly expand one result per node, so alignment remains
            # proven until a standard full-attention layer runs.
            if not alignment_proven:
                _assert_node_aligned(hidden, layout, f"layer {layer_index} input")
                alignment_checks += 1
                alignment_proven = True
            hidden, capture = _linear_layer_capture_unique(
                layer,
                hidden,
                mask,
                work,
                layout,
            )
            linear_captures[layer_index] = capture
        else:
            hidden = layer(hidden, mask=mask, cache=work)
            mx.eval(hidden, work.state)
            attention_caches[layer_index] = baseline._copy_kv_suffix(
                work,
                base_caches[layer_index].offset,
            )
            alignment_proven = False

        mx.eval(hidden)
        del layer, work
        target._cleanup()
        layer_times.append(time.perf_counter() - started)

    norm = target._load_norm()
    normalized_hidden = norm(hidden)
    mx.eval(normalized_hidden)
    del norm, hidden
    target._cleanup()

    # The final Qwen3.5 layer is full attention.  Validate its duplicate outputs before sharing one
    # lm-head projection per compact node.
    _assert_node_aligned(normalized_hidden, layout, "final normalized hidden")
    alignment_checks += 1
    unique_normalized = _unique_values(normalized_hidden, layout)
    unique_logits = target._project_logits(unique_normalized)
    node_predictions = mx.argmax(unique_logits, axis=-1)
    predictions = _expand_unique(node_predictions, layout)
    predictions = [[int(x) for x in row] for row in predictions.tolist()]
    del unique_logits, unique_normalized, node_predictions
    target._cleanup()

    print(f"unique-node alignment checks: {alignment_checks} passed")
    return (
        leaf_paths,
        predictions,
        normalized_hidden,
        linear_captures,
        attention_caches,
        layer_times,
    )


def main():
    baseline._verify_leaves = _verify_leaves_unique
    baseline._select_leaf = exact_select._select_canonical_leaf
    print("leaf target verifier: unique-node affine / leaf recurrence")
    baseline.main()


if __name__ == "__main__":
    main()
