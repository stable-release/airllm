"""Measure whether the 3-bit target's canonical continuation survives non-oracle native-MTP beam search.

Teacher-forced rank coverage tells us whether the correct target token is locally present in
MTP top-k, but it does not tell us whether ordinary cumulative-score beam pruning would keep
the correct branch alive. This diagnostic generates a short canonical continuation with the
streamed target, then runs native MTP beam search without target guidance and reports whether
the canonical prefix remains in the beam at each depth.

For correctness and isolation, branch states are replayed from the reference-aligned MTP
prompt prefill rather than sharing mutable KV-cache arrays between beams. This is slower than
a production tree drafter, but avoids branch-state aliasing in a diagnostic whose purpose is
only to measure survival.
"""

import argparse
import math
import time

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import KVCache

from probe_qwen38_mtp_macos import _load_native_mtp, _target_forward
from probe_qwen38_mtp_prefilled_macos import _mtp_forward
from probe_qwen38_mtp_rank_macos import _make_target, _mtp_logits, _topk


def _piece(tokenizer, token_id):
    return tokenizer.decode([int(token_id)], skip_special_tokens=False)


def _log_softmax_topk(logits, k):
    k = min(int(k), logits.shape[0])
    top = _topk(logits, k)
    m = float(np.max(logits))
    log_z = m + math.log(float(np.exp(logits - m).sum(dtype=np.float64)))
    return [(int(tok), float(logits[tok]) - log_z) for tok in top]


def _replay_path(target, mtp, shifted_ids, target_hidden, path):
    """Rebuild an MTP branch from prompt state and return logits for its next token."""
    cache = KVCache()
    hidden = _mtp_forward(target, mtp, cache, shifted_ids, target_hidden)
    current = hidden[:, -1:, :]
    for token in path:
        current = _mtp_forward(target, mtp, cache, [int(token)], current)
    return _mtp_logits(target, current)


def main():
    parser = argparse.ArgumentParser(
        description="Check whether canonical target tokens survive native-MTP beam search."
    )
    parser.add_argument("--model", default="orcarouter/Qwen3.8-27B-Uncensored-FP8")
    parser.add_argument("--target-bits", type=int, choices=(3, 4), default=3)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--beam-width", type=int, default=2)
    parser.add_argument("--branch-k", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument(
        "--prompt",
        default="Explain why Rust's ownership system prevents use-after-free bugs, using a short concrete example.",
    )
    args = parser.parse_args()

    if args.steps < 1:
        raise ValueError("--steps must be >= 1")
    if args.beam_width < 1:
        raise ValueError("--beam-width must be >= 1")
    branch_k = args.branch_k or args.beam_width
    if branch_k < 1:
        raise ValueError("--branch-k must be >= 1")

    print(f"loading streamed {args.target_bits}-bit target...")
    target = _make_target(args.model, args.target_bits, args.max_seq_len)
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

    print("prefilling target and capturing prompt hidden states...")
    started = time.perf_counter()
    target_hidden, target_logits = _target_forward(target, prompt_ids, target_caches)
    target_prefill_s = time.perf_counter() - started
    guaranteed = int(mx.argmax(target_logits[:, -1, :], axis=-1).item())

    # Build the canonical greedy target continuation after the guaranteed token.
    canonical = []
    current = guaranteed
    target_decode_s = 0.0
    print(f"guaranteed token: {guaranteed} {_piece(target.tokenizer, guaranteed)!r}")
    print("generating canonical target continuation...")
    for _ in range(args.steps):
        started = time.perf_counter()
        _, logits = _target_forward(target, [current], target_caches)
        target_decode_s += time.perf_counter() - started
        current = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        canonical.append(current)

    shifted_ids = prompt_ids[1:] + [guaranteed]

    # Beam entries: (cumulative log probability, token tuple).
    beams = [(0.0, tuple())]
    replay_s = 0.0
    survival_depth = 0

    print("\n=== native MTP beam survival ===")
    print(f"target bits:       {args.target_bits}")
    print(f"prompt tokens:     {len(prompt_ids)}")
    print(f"steps:             {args.steps}")
    print(f"beam width:        {args.beam_width}")
    print(f"branch k:          {branch_k}")
    print(
        "canonical:         "
        + repr(target.tokenizer.decode(canonical, skip_special_tokens=False))
    )

    for depth in range(args.steps):
        expanded = []
        for score, path in beams:
            started = time.perf_counter()
            logits = _replay_path(target, mtp, shifted_ids, target_hidden, path)
            replay_s += time.perf_counter() - started
            for token, logp in _log_softmax_topk(logits, branch_k):
                expanded.append((score + logp, path + (token,)))

        expanded.sort(key=lambda x: x[0], reverse=True)
        beams = expanded[: args.beam_width]

        wanted = tuple(canonical[: depth + 1])
        canonical_rank = None
        for rank, (_, path) in enumerate(beams, start=1):
            if path == wanted:
                canonical_rank = rank
                break
        if canonical_rank is not None:
            survival_depth = depth + 1

        print(
            f"\ndepth {depth + 1}: canonical "
            + (f"SURVIVES at beam rank {canonical_rank}" if canonical_rank else "PRUNED")
        )
        print(
            f"  wanted: {target.tokenizer.decode(wanted, skip_special_tokens=False)!r}"
        )
        for rank, (score, path) in enumerate(beams, start=1):
            marker = " <-- canonical" if path == wanted else ""
            print(
                f"  {rank:2d}. score={score:9.4f} "
                f"text={target.tokenizer.decode(path, skip_special_tokens=False)!r}{marker}"
            )

    print("\n=== summary ===")
    print(f"MTP resident bytes:        {mtp_bytes / 1024**2:.2f} MiB")
    print(f"target prefill:            {target_prefill_s:.3f} s")
    print(f"target decode total:       {target_decode_s:.3f} s")
    print(f"diagnostic MTP replay:     {replay_s:.3f} s")
    print(f"canonical survival depth:  {survival_depth}/{args.steps}")
    print(
        "canonical full survival: "
        + ("YES" if survival_depth == args.steps else "NO")
    )


if __name__ == "__main__":
    main()
