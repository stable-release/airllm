"""Probe Qwen3.8's native MTP head against the streamed 3-bit target on macOS.

This is intentionally an acceptance/latency probe, not a production speculative decoder.
It keeps the validated checkpointed generator untouched and answers one question first:
how accurately and cheaply can the checkpoint's own one-layer MTP head predict this exact
target's greedy continuation?

Alignment used here follows vLLM's Qwen3.5 MTP implementation:
  * the main model supplies its post-final-norm hidden state h_t;
  * the already-known target token c_{t+1} is embedded with the shared target embedding;
  * MTP RMS-normalizes embedding + h_t separately, concatenates them, projects with mtp.fc,
    runs the dedicated full-attention MTP decoder layer, and applies mtp.norm;
  * the shared target lm_head produces the next MTP proposal;
  * subsequent MTP steps feed the previous MTP hidden state plus the newly proposed token.

The MTP attention KV cache contains only speculative tokens, but RoPE uses their absolute
positions in the target sequence. MLX-LM's stock Qwen attention couples RoPE offset to the
cache length, so this probe calls the attention submodules directly to keep those two notions
separate without inserting fake prompt KV entries.
"""

import argparse
import time

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.cache import KVCache
from mlx_lm.models.qwen3_5 import DecoderLayer

from airllm.airllm_qwen35_mlx_fp8_3bit import AirLLMQwen35MlxFp8ThreeBit


_MTP_NORM_KEYS = {
    "norm.weight",
    "pre_fc_norm_embedding.weight",
    "pre_fc_norm_hidden.weight",
    "layers.0.input_layernorm.weight",
    "layers.0.post_attention_layernorm.weight",
    "layers.0.self_attn.q_norm.weight",
    "layers.0.self_attn.k_norm.weight",
}


class NativeMTP(nn.Module):
    def __init__(self, args):
        super().__init__()
        h = args.hidden_size
        self.fc = nn.Linear(h * 2, h, bias=False)
        # MLX-LM chooses Qwen3.5 layer type from the layer index. The final index
        # in one attention interval is a full-attention layer, matching Qwen MTP.
        full_attention_index = args.full_attention_interval - 1
        self.layers = [DecoderLayer(args, full_attention_index)]
        if self.layers[0].is_linear:
            raise RuntimeError("Synthetic MTP layer index did not select full attention")
        self.norm = nn.RMSNorm(h, eps=args.rms_norm_eps)
        self.pre_fc_norm_hidden = nn.RMSNorm(h, eps=args.rms_norm_eps)
        self.pre_fc_norm_embedding = nn.RMSNorm(h, eps=args.rms_norm_eps)


def _mlx_memory_mb(name):
    fn = getattr(mx, name, None)
    if fn is None:
        return None
    try:
        return fn() / 1024 / 1024
    except Exception:
        return None


def _load_native_mtp(target):
    text_config = getattr(target.config, "text_config", target.config)
    if getattr(text_config, "mtp_num_hidden_layers", 1) != 1:
        raise NotImplementedError("This probe currently expects exactly one Qwen MTP layer")
    if getattr(text_config, "mtp_use_dedicated_embeddings", False):
        raise NotImplementedError("Dedicated MTP embeddings are not implemented in this probe")

    weight_map = target._source_index()
    source = target._load_source_component("mtp", weight_map)
    local_torch = target._densify_source_component(source, "mtp")
    weights = target._torch_local_to_mlx(local_torch)

    # Qwen3.5 uses Gemma-style RMSNorm checkpoint weights: runtime scale is 1 + w.
    # MLX nn.RMSNorm expects the actual multiplicative scale, so shift every MTP norm.
    for key in _MTP_NORM_KEYS:
        if key in weights:
            weights[key] = weights[key] + 1.0

    module = NativeMTP(target.model_args)

    module.fc.update(tree_unflatten([
        (key[len("fc."):], value)
        for key, value in weights.items()
        if key.startswith("fc.")
    ]))
    module.layers[0].update(tree_unflatten([
        (key[len("layers.0."):], value)
        for key, value in weights.items()
        if key.startswith("layers.0.")
    ]))
    module.norm.update(tree_unflatten([
        (key[len("norm."):], value)
        for key, value in weights.items()
        if key.startswith("norm.")
    ]))
    module.pre_fc_norm_hidden.update(tree_unflatten([
        (key[len("pre_fc_norm_hidden."):], value)
        for key, value in weights.items()
        if key.startswith("pre_fc_norm_hidden.")
    ]))
    module.pre_fc_norm_embedding.update(tree_unflatten([
        (key[len("pre_fc_norm_embedding."):], value)
        for key, value in weights.items()
        if key.startswith("pre_fc_norm_embedding.")
    ]))

    params = [value for _, value in tree_flatten(module.parameters())]
    mx.eval(params)
    logical_bytes = sum(value.nbytes for value in params)

    del source, local_torch, weights, params
    return module, logical_bytes


def _target_forward(target, token_ids, caches):
    """Consume token_ids and return post-final-norm hidden states plus logits."""
    if not token_ids:
        raise ValueError("target forward requires at least one token")

    tokens = mx.array(token_ids, dtype=mx.int32)[None, :]

    embedding = target._load_embedding()
    hidden = embedding(tokens)
    mx.eval(hidden)
    del embedding
    target._cleanup()

    hidden = target._run_layers(hidden, caches)

    norm = target._load_norm()
    hidden = norm(hidden)
    mx.eval(hidden)
    del norm
    target._cleanup()

    logits = target._project_logits(hidden)
    mx.eval(logits)
    return hidden, logits


def _mtp_attention_step(attn, x, cache, absolute_position):
    """One Qwen3.5 full-attention step with absolute RoPE and suffix-only KV cache."""
    B, L, _ = x.shape
    if L != 1:
        raise ValueError("MTP probe currently advances one speculative token at a time")

    q_proj_output = attn.q_proj(x)
    queries, gate = mx.split(
        q_proj_output.reshape(B, L, attn.num_attention_heads, -1), 2, axis=-1
    )
    gate = gate.reshape(B, L, -1)

    keys = attn.k_proj(x)
    values = attn.v_proj(x)

    queries = attn.q_norm(queries).transpose(0, 2, 1, 3)
    keys = attn.k_norm(
        keys.reshape(B, L, attn.num_key_value_heads, -1)
    ).transpose(0, 2, 1, 3)
    values = values.reshape(B, L, attn.num_key_value_heads, -1).transpose(0, 2, 1, 3)

    # vLLM passes the real sequence position into the MTP layer. Keep cache.offset
    # solely as the count of MTP suffix tokens and use the real position for RoPE.
    queries = attn.rope(queries, offset=int(absolute_position))
    keys = attn.rope(keys, offset=int(absolute_position))
    keys, values = cache.update_and_fetch(keys, values)

    output = scaled_dot_product_attention(
        queries,
        keys,
        values,
        cache=None,
        scale=attn.scale,
        mask=None,
    )
    output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
    return attn.o_proj(output * mx.sigmoid(gate))


def _mtp_step(target, mtp, mtp_cache, token_id, hidden_state, absolute_position):
    started = time.perf_counter()

    token = mx.array([[int(token_id)]], dtype=mx.int32)
    embedding_module = target._load_embedding()
    embedding = embedding_module(token)
    mx.eval(embedding)
    del embedding_module
    target._cleanup()

    embedding = mtp.pre_fc_norm_embedding(embedding)
    hidden = mtp.pre_fc_norm_hidden(hidden_state)
    x = mtp.fc(mx.concatenate([embedding, hidden], axis=-1))

    layer = mtp.layers[0]
    r = _mtp_attention_step(
        layer.self_attn,
        layer.input_layernorm(x),
        mtp_cache,
        absolute_position,
    )
    h = x + r
    h = h + layer.mlp(layer.post_attention_layernorm(h))
    h = mtp.norm(h)
    mx.eval(h, mtp_cache.state)

    logits = target._project_logits(h[:, -1, :])
    next_token = int(mx.argmax(logits, axis=-1).item())
    mx.eval(logits)

    return next_token, h, time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser(
        description="Probe native Qwen3.8 MTP acceptance against the streamed 3-bit target."
    )
    parser.add_argument(
        "--model",
        default="orcarouter/Qwen3.8-27B-Uncensored-FP8",
    )
    parser.add_argument("--mtp-tokens", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument(
        "--prompt",
        default="Explain why Rust's ownership system prevents use-after-free bugs, using a short concrete example.",
    )
    args = parser.parse_args()

    if args.mtp_tokens < 1:
        raise ValueError("--mtp-tokens must be >= 1")

    print("loading streamed 3-bit target...")
    target = AirLLMQwen35MlxFp8ThreeBit(
        args.model,
        compression="3bit",
        max_seq_len=args.max_seq_len,
        mlx_resident_gib=0,
    )

    prompt = target.tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_ids = list(target.tokenizer.encode(prompt, add_special_tokens=False))

    print("loading native MTP weights from target checkpoint...")
    mtp_started = time.perf_counter()
    mtp, mtp_bytes = _load_native_mtp(target)
    mtp_load_s = time.perf_counter() - mtp_started

    print(f"target:              {args.model}")
    print(f"prompt tokens:       {len(prompt_ids)}")
    print(f"MTP layers:          1 full-attention layer")
    print(f"MTP resident bytes:  {mtp_bytes / 1024**2:.2f} MiB")
    print(f"MTP load/materialize:{mtp_load_s:8.3f} s")

    target_caches = [
        target._new_cache(i) for i in range(target.model_args.num_hidden_layers)
    ]

    print("prefilling target and capturing final hidden state...")
    prefill_started = time.perf_counter()
    hidden, logits = _target_forward(target, prompt_ids, target_caches)
    prefill_s = time.perf_counter() - prefill_started

    base_hidden = hidden[:, -1:, :]
    current_target_token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
    print(
        f"target next token:   id={current_target_token} "
        f"piece={target.tokenizer.decode([current_target_token], skip_special_tokens=False)!r}"
    )

    mtp_cache = KVCache()
    proposals = []
    mtp_times = []
    mtp_hidden = base_hidden
    mtp_input = current_target_token

    print("\nMTP proposals:")
    for step in range(args.mtp_tokens):
        absolute_position = len(prompt_ids) + step
        proposal, mtp_hidden, step_s = _mtp_step(
            target,
            mtp,
            mtp_cache,
            mtp_input,
            mtp_hidden,
            absolute_position,
        )
        proposals.append(proposal)
        mtp_times.append(step_s)
        print(
            f"  [{step}] id={proposal} "
            f"piece={target.tokenizer.decode([proposal], skip_special_tokens=False)!r} "
            f"time={step_s:.3f}s"
        )
        mtp_input = proposal

    # To verify N MTP proposals, the target consumes the guaranteed target token
    # followed by the first N-1 proposals. Each output logit then predicts one
    # of the N proposed tokens.
    verify_block = [current_target_token] + proposals[:-1]
    print(f"\nverifying {len(proposals)} MTP proposals in one target traversal...")
    verify_started = time.perf_counter()
    _, verify_logits = _target_forward(target, verify_block, target_caches)
    verify_s = time.perf_counter() - verify_started
    target_predictions = [
        int(x) for x in mx.argmax(verify_logits[0], axis=-1).tolist()
    ]

    prefix_accept = 0
    print("\nMTP vs target:")
    for index, (proposal, target_prediction) in enumerate(
        zip(proposals, target_predictions)
    ):
        match = proposal == target_prediction
        if match and prefix_accept == index:
            prefix_accept += 1
        print(
            f"  [{index}] {'MATCH' if match else 'MISS ':5s} "
            f"mtp={proposal:6d} {target.tokenizer.decode([proposal], skip_special_tokens=False)!r} "
            f"target={target_prediction:6d} "
            f"{target.tokenizer.decode([target_prediction], skip_special_tokens=False)!r}"
        )

    print("\n=== Qwen3.8 native MTP probe ===")
    print(f"target prefill:             {prefill_s:.3f} s")
    print(f"MTP proposals:              {len(proposals)}")
    print(f"MTP accepted prefix:        {prefix_accept}/{len(proposals)}")
    print(f"MTP total proposal time:    {sum(mtp_times):.3f} s")
    print(f"MTP average/step:           {sum(mtp_times) / len(mtp_times):.3f} s")
    print(f"target verification pass:   {verify_s:.3f} s")
    print(
        f"useful tokens if integrated:{1 + prefix_accept}" 
        " (guaranteed target token + accepted MTP prefix)"
    )

    active = _mlx_memory_mb("get_active_memory")
    peak = _mlx_memory_mb("get_peak_memory")
    if active is not None:
        print(f"MLX active memory:          {active:.2f} MiB")
    if peak is not None:
        print(f"MLX peak memory:            {peak:.2f} MiB")


if __name__ == "__main__":
    main()
