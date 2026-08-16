"""Probe uncertainty-triggered native-MTP tree survival against the 3-bit target.

Unlike global beam search, this keeps local alternatives when the MTP distribution is
uncertain. Each active path follows top-1, and also keeps top-2 when the top1-top2
logit margin is <= --branch-margin. This is a non-oracle policy: target tokens are used
only after the tree is built to report whether the canonical target path survived.

Branch state is replayed from the reference-aligned prompt MTP prefill for correctness.
This makes the diagnostic slower than a production implementation, but avoids KV-cache
aliasing between branches.
"""

import argparse
import time

import mlx.core as mx
import numpy as np

from probe_qwen38_mtp_macos import _load_native_mtp, _target_forward
from probe_qwen38_mtp_prefilled_macos import _mtp_forward
from probe_qwen38_mtp_rank_macos import _make_target, _topk
from probe_qwen38_mtp_beam_survival_macos import _replay_path


def _piece(tok, path):
    return tok.decode(list(path), skip_special_tokens=False)


def main():
    p = argparse.ArgumentParser(
        description="Check canonical survival in a margin-triggered native-MTP tree."
    )
    p.add_argument("--model", default="orcarouter/Qwen3.8-27B-Uncensored-FP8")
    p.add_argument("--target-bits", type=int, choices=(3, 4), default=3)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--branch-margin", type=float, default=5.0)
    p.add_argument("--max-active", type=int, default=32)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument(
        "--prompt",
        default="Explain why Rust's ownership system prevents use-after-free bugs, using a short concrete example.",
    )
    a = p.parse_args()

    target = _make_target(a.model, a.target_bits, a.max_seq_len)
    mtp, mtp_bytes = _load_native_mtp(target)

    prompt = target.tokenizer.apply_chat_template(
        [{"role": "user", "content": a.prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_ids = list(target.tokenizer.encode(prompt, add_special_tokens=False))

    target_caches = [
        target._new_cache(i) for i in range(target.model_args.num_hidden_layers)
    ]
    started = time.perf_counter()
    target_hidden, target_logits = _target_forward(target, prompt_ids, target_caches)
    target_prefill_s = time.perf_counter() - started
    guaranteed = int(mx.argmax(target_logits[:, -1, :], axis=-1).item())

    canonical = []
    current = guaranteed
    target_decode_s = 0.0
    for _ in range(a.steps):
        started = time.perf_counter()
        _, logits = _target_forward(target, [current], target_caches)
        target_decode_s += time.perf_counter() - started
        current = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        canonical.append(current)

    shifted_ids = prompt_ids[1:] + [guaranteed]

    # Active entries are (score, path). Score is only used if max-active forces pruning.
    active = [(0.0, tuple())]
    replay_s = 0.0
    total_nodes = 0
    budget_pruned = False
    survival_depth = 0

    print("=== margin-triggered native MTP tree ===")
    print(f"target bits:        {a.target_bits}")
    print(f"prompt tokens:      {len(prompt_ids)}")
    print(f"steps:              {a.steps}")
    print(f"branch margin:      {a.branch_margin:.3f}")
    print(f"max active paths:   {a.max_active}")
    print(f"canonical:          {_piece(target.tokenizer, canonical)!r}")

    for depth in range(a.steps):
        expanded = []
        branch_events = 0
        for score, path in active:
            started = time.perf_counter()
            logits = _replay_path(target, mtp, shifted_ids, target_hidden, path)
            replay_s += time.perf_counter() - started

            top2 = _topk(logits, 2)
            t1, t2 = int(top2[0]), int(top2[1])
            margin = float(logits[t1] - logits[t2])
            candidates = [t1]
            if margin <= a.branch_margin:
                candidates.append(t2)
                branch_events += 1

            m = float(np.max(logits))
            z = m + float(np.log(np.exp(logits - m, dtype=np.float64).sum()))
            for tok in candidates:
                expanded.append((score + float(logits[tok]) - z, path + (tok,)))
                total_nodes += 1

        if len(expanded) > a.max_active:
            budget_pruned = True
            expanded.sort(key=lambda x: x[0], reverse=True)
            expanded = expanded[: a.max_active]

        active = expanded
        wanted = tuple(canonical[: depth + 1])
        canonical_rank = next(
            (i for i, (_, path) in enumerate(active, start=1) if path == wanted),
            None,
        )
        if canonical_rank is not None:
            survival_depth = depth + 1

        print(
            f"\ndepth {depth + 1}: active={len(active)} branched_parents={branch_events} "
            + (
                f"canonical SURVIVES (slot {canonical_rank})"
                if canonical_rank is not None
                else "canonical PRUNED"
            )
        )
        print(f"  wanted: {_piece(target.tokenizer, wanted)!r}")
        for i, (score, path) in enumerate(active[: min(10, len(active))], start=1):
            marker = " <-- canonical" if path == wanted else ""
            print(f"  {i:2d}. score={score:9.4f} text={_piece(target.tokenizer, path)!r}{marker}")

    print("\n=== summary ===")
    print(f"MTP resident bytes:        {mtp_bytes / 1024**2:.2f} MiB")
    print(f"target prefill:            {target_prefill_s:.3f} s")
    print(f"target decode total:       {target_decode_s:.3f} s")
    print(f"diagnostic MTP replay:     {replay_s:.3f} s")
    print(f"total generated tree nodes:{total_nodes}")
    print(f"final active paths:        {len(active)}")
    print(f"budget pruning occurred:   {'YES' if budget_pruned else 'NO'}")
    print(f"canonical survival depth:  {survival_depth}/{a.steps}")
    print(
        "canonical full survival: "
        + ("YES" if survival_depth == a.steps else "NO")
    )


if __name__ == "__main__":
    main()
