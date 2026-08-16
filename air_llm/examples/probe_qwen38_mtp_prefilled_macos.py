"""Reference-aligned native Qwen3.8 MTP acceptance probe on macOS.

Unlike the earlier suffix-only probe, this mirrors vLLM's EAGLE/MTP first-pass alignment:

* run the streamed target over the full prompt and keep every post-final-norm hidden state;
* shift prompt token IDs left by one and place the target's sampled next token in the final slot;
* keep the target hidden states and positions unchanged;
* run the native one-layer MTP over that whole aligned sequence, thereby prefilling MTP's own KV cache;
* sample the first MTP proposal from the final MTP hidden state;
* recursively generate later MTP proposals from the previous MTP hidden state + previous proposal;
* verify all proposals in one streamed 3-bit target traversal.

This remains a probe only; the validated checkpointed speculative generator is untouched.
"""

import argparse
import time

import mlx.core as mx
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import KVCache

from airllm.airllm_qwen35_mlx_fp8_3bit import AirLLMQwen35MlxFp8ThreeBit
from probe_qwen38_mtp_macos import _load_native_mtp, _mlx_memory_mb, _target_forward


def _mtp_forward(target, mtp, cache, token_ids, hidden_states):
    """Run MTP for an aligned token/target-hidden block and advance its KV cache."""
    if not token_ids:
        raise ValueError("MTP forward requires at least one token")
    if hidden_states.shape[1] != len(token_ids):
        raise ValueError(
            f"MTP alignment mismatch: {len(token_ids)} tokens vs "
            f"{hidden_states.shape[1]} hidden states"
        )

    tokens = mx.array(token_ids, dtype=mx.int32)[None, :]

    embedding_module = target._load_embedding()
    embedding = embedding_module(tokens)
    mx.eval(embedding)
    del embedding_module
    target._cleanup()

    embedding = mtp.pre_fc_norm_embedding(embedding)
    hidden = mtp.pre_fc_norm_hidden(hidden_states)
    x = mtp.fc(mx.concatenate([embedding, hidden], axis=-1))

    layer = mtp.layers[0]
    mask = create_attention_mask(x, cache)
    h = layer(x, mask=mask, cache=cache)
    h = mtp.norm(h)
    mx.eval(h, cache.state)
    return h


def _mtp_sample(target, hidden):
    logits = target._project_logits(hidden[:, -1, :])
    token = int(mx.argmax(logits, axis=-1).item())
    mx.eval(logits)
    return token


def main():
    parser = argparse.ArgumentParser(
        description="Probe native Qwen3.8 MTP with reference-aligned prompt KV prefill."
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
    load_started = time.perf_counter()
    mtp, mtp_bytes = _load_native_mtp(target)
    mtp_load_s = time.perf_counter() - load_started

    print(f"target:              {args.model}")
    print(f"prompt tokens:       {len(prompt_ids)}")
    print("MTP alignment:       full target-query prefill (vLLM/EAGLE style)")
    print("MTP layers:          1 full-attention layer")
    print(f"MTP resident bytes:  {mtp_bytes / 1024**2:.2f} MiB")
    print(f"MTP load/materialize:{mtp_load_s:8.3f} s")

    target_caches = [
        target._new_cache(i) for i in range(target.model_args.num_hidden_layers)
    ]

    print("prefilling target and capturing all normalized hidden states...")
    target_prefill_started = time.perf_counter()
    target_hidden, target_logits = _target_forward(target, prompt_ids, target_caches)
    target_prefill_s = time.perf_counter() - target_prefill_started

    guaranteed_token = int(mx.argmax(target_logits[:, -1, :], axis=-1).item())
    print(
        f"target next token:   id={guaranteed_token} "
        f"piece={target.tokenizer.decode([guaranteed_token], skip_special_tokens=False)!r}"
    )

    # vLLM's first MTP pass shifts input IDs by one but leaves target hidden states
    # and positions unchanged. For one request:
    #   token inputs = prompt[1:] + [target_next_token]
    #   hidden input = h(prompt[0]), ..., h(prompt[-1])
    #   positions    = 0, ..., len(prompt)-1
    # Running the whole block also constructs MTP's own causal prompt KV history.
    shifted_ids = prompt_ids[1:] + [guaranteed_token]
    mtp_cache = KVCache()

    print("prefilling native MTP KV cache with aligned target hidden states...")
    mtp_prefill_started = time.perf_counter()
    mtp_hidden = _mtp_forward(target, mtp, mtp_cache, shifted_ids, target_hidden)
    first_proposal = _mtp_sample(target, mtp_hidden)
    mtp_prefill_s = time.perf_counter() - mtp_prefill_started

    proposals = [first_proposal]
    proposal_times = []
    print("\nMTP proposals:")
    print(
        f"  [0] id={first_proposal} "
        f"piece={target.tokenizer.decode([first_proposal], skip_special_tokens=False)!r} "
        f"source=prompt-prefill"
    )

    current_hidden = mtp_hidden[:, -1:, :]
    current_token = first_proposal

    for step in range(1, args.mtp_tokens):
        started = time.perf_counter()
        current_hidden = _mtp_forward(
            target,
            mtp,
            mtp_cache,
            [current_token],
            current_hidden,
        )
        proposal = _mtp_sample(target, current_hidden)
        step_s = time.perf_counter() - started
        proposals.append(proposal)
        proposal_times.append(step_s)
        print(
            f"  [{step}] id={proposal} "
            f"piece={target.tokenizer.decode([proposal], skip_special_tokens=False)!r} "
            f"time={step_s:.3f}s"
        )
        current_token = proposal

    # To verify N MTP proposals, target consumes the guaranteed target token plus
    # the first N-1 proposals. The resulting N logits predict proposals 0..N-1.
    verify_block = [guaranteed_token] + proposals[:-1]
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

    recursive_total = sum(proposal_times)
    print("\n=== Qwen3.8 native MTP prefilled probe ===")
    print(f"target prefill:             {target_prefill_s:.3f} s")
    print(f"MTP prompt prefill:         {mtp_prefill_s:.3f} s")
    print(f"MTP KV cache positions:     {mtp_cache.offset}")
    print(f"MTP proposals:              {len(proposals)}")
    print(f"MTP accepted prefix:        {prefix_accept}/{len(proposals)}")
    print(f"recursive MTP proposal time:{recursive_total:8.3f} s")
    if proposal_times:
        print(
            f"recursive MTP average/step: {recursive_total / len(proposal_times):.3f} s"
        )
    print(f"target verification pass:   {verify_s:.3f} s")
    print(
        f"useful tokens if integrated:{1 + prefix_accept} "
        "(guaranteed target token + accepted MTP prefix)"
    )

    active = _mlx_memory_mb("get_active_memory")
    peak = _mlx_memory_mb("get_peak_memory")
    if active is not None:
        print(f"MLX active memory:          {active:.2f} MiB")
    if peak is not None:
        print(f"MLX peak memory:            {peak:.2f} MiB")


if __name__ == "__main__":
    main()
