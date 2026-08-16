"""Measure native Qwen3.8 MTP rank coverage along the target's canonical greedy path.

The ordinary rank probe follows MTP's own greedy branch. That is useful for strict-prefix
acceptance, but after the first mismatch later ranks are conditioned on the wrong branch.
This diagnostic instead teacher-forces the streamed target's actual greedy token back into
MTP after every step. It therefore answers the question needed for narrow tree speculation:
under the *correct target prefix*, how often is the next target token in MTP top-1/top-2/
top-3/top-4/...?

This is intentionally diagnostic and expensive: it performs one streamed target decode pass
per measured token. It does not modify the validated speculative generators.
"""

import argparse
import time

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import KVCache

from probe_qwen38_mtp_macos import _load_native_mtp, _mlx_memory_mb, _target_forward
from probe_qwen38_mtp_prefilled_macos import _mtp_forward
from probe_qwen38_mtp_rank_macos import _make_target, _mtp_logits, _topk


def _token_piece(tokenizer, token_id):
    return tokenizer.decode([int(token_id)], skip_special_tokens=False)


def _rank(arr, token_id):
    # Ranking by descending logit without materializing a full argsort is enough here:
    # rank = 1 + number of vocabulary logits strictly larger than the target logit.
    target_logit = float(arr[token_id])
    return 1 + int(np.count_nonzero(arr > target_logit))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Teacher-force native Qwen3.8 MTP along the streamed target's greedy path "
            "and report target-token ranks."
        )
    )
    parser.add_argument("--model", default="orcarouter/Qwen3.8-27B-Uncensored-FP8")
    parser.add_argument("--target-bits", type=int, choices=(3, 4), default=3)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument(
        "--prompt",
        default="Explain why Rust's ownership system prevents use-after-free bugs, using a short concrete example.",
    )
    args = parser.parse_args()

    if args.steps < 1:
        raise ValueError("--steps must be >= 1")

    print(f"loading streamed {args.target_bits}-bit target...")
    target = _make_target(args.model, args.target_bits, args.max_seq_len)

    prompt = target.tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_ids = list(target.tokenizer.encode(prompt, add_special_tokens=False))

    print("loading native MTP weights...")
    mtp, mtp_bytes = _load_native_mtp(target)

    target_caches = [
        target._new_cache(i) for i in range(target.model_args.num_hidden_layers)
    ]

    print("prefilling target and capturing normalized hidden states...")
    started = time.perf_counter()
    target_hidden, target_logits = _target_forward(target, prompt_ids, target_caches)
    target_prefill_s = time.perf_counter() - started
    guaranteed_token = int(mx.argmax(target_logits[:, -1, :], axis=-1).item())

    # Reference-aligned MTP first pass: shift token IDs left by one, put the
    # target's already-sampled token in the final slot, keep hidden states and
    # positions aligned to the original target query.
    shifted_ids = prompt_ids[1:] + [guaranteed_token]
    mtp_cache = KVCache()

    print("prefilling MTP prompt KV on the canonical prefix...")
    started = time.perf_counter()
    mtp_hidden = _mtp_forward(target, mtp, mtp_cache, shifted_ids, target_hidden)
    mtp_prefill_s = time.perf_counter() - started
    current_mtp_hidden = mtp_hidden[:, -1:, :]

    # The target cache currently ends at the prompt. Consuming guaranteed_token
    # yields the canonical token that MTP is trying to predict at teacher-forced step 0.
    current_target_input = guaranteed_token

    ranks = []
    target_decode_times = []
    mtp_step_times = []
    canonical_tokens = [guaranteed_token]

    print("\n=== teacher-forced MTP target-rank diagnostic ===")
    print(f"target bits:          {args.target_bits}")
    print(f"prompt tokens:        {len(prompt_ids)}")
    print(
        f"guaranteed token:     {guaranteed_token} "
        f"{_token_piece(target.tokenizer, guaranteed_token)!r}"
    )

    for step in range(args.steps):
        # Advance the real target by exactly one canonical token.
        started = time.perf_counter()
        target_step_hidden, target_step_logits = _target_forward(
            target, [current_target_input], target_caches
        )
        target_s = time.perf_counter() - started
        target_decode_times.append(target_s)
        target_next = int(mx.argmax(target_step_logits[:, -1, :], axis=-1).item())
        canonical_tokens.append(target_next)

        # Score the target token in the MTP distribution under the same canonical prefix.
        arr = _mtp_logits(target, current_mtp_hidden)
        mtp_top1 = int(np.argmax(arr))
        rank = _rank(arr, target_next)
        ranks.append(rank)
        top = _topk(arr, args.top_k)
        gap = float(arr[mtp_top1]) - float(arr[target_next])

        print(
            f"\nstep {step}: {'MATCH' if mtp_top1 == target_next else 'MISS '}  "
            f"mtp={mtp_top1} {_token_piece(target.tokenizer, mtp_top1)!r}  "
            f"target={target_next} {_token_piece(target.tokenizer, target_next)!r}"
        )
        print(
            f"  target rank in MTP: {rank}   "
            f"top1-target logit gap: {gap:.4f}   target pass: {target_s:.3f}s"
        )
        print("  MTP top candidates:")
        for j, tok in enumerate(top, start=1):
            tok = int(tok)
            marker = " <-- target" if tok == target_next else ""
            print(
                f"    {j:2d}. id={tok:6d} logit={float(arr[tok]):9.4f} "
                f"piece={_token_piece(target.tokenizer, tok)!r}{marker}"
            )

        if step + 1 >= args.steps:
            break

        # Teacher-force the target's actual token into MTP. The hidden state passed
        # to the next MTP step remains the previous MTP hidden, matching vLLM's
        # recurrent MTP/EAGLE loop; only the sampled token is replaced by target truth.
        started = time.perf_counter()
        current_mtp_hidden = _mtp_forward(
            target,
            mtp,
            mtp_cache,
            [target_next],
            current_mtp_hidden,
        )
        mtp_step_times.append(time.perf_counter() - started)
        current_target_input = target_next

    print("\n=== summary ===")
    print(f"MTP resident bytes:       {mtp_bytes / 1024**2:.2f} MiB")
    print(f"target prefill:           {target_prefill_s:.3f} s")
    print(f"MTP prompt prefill:       {mtp_prefill_s:.3f} s")
    print(f"MTP KV cache positions:   {mtp_cache.offset}")
    print(f"measured target tokens:   {len(ranks)}")
    if mtp_step_times:
        print(
            f"teacher-forced MTP step:  {sum(mtp_step_times) / len(mtp_step_times):.3f} s avg"
        )
    print(
        f"target decode pass:       {sum(target_decode_times) / len(target_decode_times):.3f} s avg"
    )

    for width in (1, 2, 3, 4, 8, 16):
        covered = sum(rank <= width for rank in ranks)
        print(
            f"top-{width:<2d} target coverage:   {covered}/{len(ranks)} "
            f"({100.0 * covered / len(ranks):5.1f}%)"
        )

    print(f"ranks:                    {ranks}")
    print(
        "canonical continuation:  "
        + repr(target.tokenizer.decode(canonical_tokens, skip_special_tokens=False))
    )

    active = _mlx_memory_mb("get_active_memory")
    peak = _mlx_memory_mb("get_peak_memory")
    if active is not None:
        print(f"MLX active memory:        {active:.2f} MiB")
    if peak is not None:
        print(f"MLX peak memory:          {peak:.2f} MiB")


if __name__ == "__main__":
    main()
