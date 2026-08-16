"""Inspect where the streamed target's greedy token ranks in native Qwen3.8 MTP logits.

This is a diagnostic for narrow tree speculation.  It uses the same reference-aligned
MTP prompt prefill as probe_qwen38_mtp_prefilled_macos.py, records the full MTP logits
for each greedy proposal, verifies the greedy MTP branch with one target traversal, and
reports the target token's rank/gap in the MTP distribution at every position.
"""

import argparse
import time

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import KVCache

from airllm.airllm_qwen35_mlx_fp8 import AirLLMQwen35MlxFp8
from airllm.airllm_qwen35_mlx_fp8_3bit import AirLLMQwen35MlxFp8ThreeBit
from probe_qwen38_mtp_macos import _load_native_mtp, _mlx_memory_mb, _target_forward
from probe_qwen38_mtp_prefilled_macos import _mtp_forward


def _target_cls(bits):
    return AirLLMQwen35MlxFp8ThreeBit if bits == 3 else AirLLMQwen35MlxFp8


def _make_target(model, bits, max_seq_len):
    cls = _target_cls(bits)
    return cls(
        model,
        compression=f"{bits}bit",
        max_seq_len=max_seq_len,
        mlx_resident_gib=0,
    )


def _mtp_logits(target, hidden):
    logits = target._project_logits(hidden[:, -1, :])
    mx.eval(logits)
    return np.asarray(logits[0], dtype=np.float32)


def _topk(arr, k):
    k = min(k, arr.shape[0])
    idx = np.argpartition(arr, -k)[-k:]
    idx = idx[np.argsort(arr[idx])[::-1]]
    return idx


def main():
    parser = argparse.ArgumentParser(
        description="Report streamed-target token ranks in native Qwen3.8 MTP logits."
    )
    parser.add_argument("--model", default="orcarouter/Qwen3.8-27B-Uncensored-FP8")
    parser.add_argument("--target-bits", type=int, choices=(3, 4), default=4)
    parser.add_argument("--mtp-tokens", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument(
        "--prompt",
        default="Explain why Rust's ownership system prevents use-after-free bugs, using a short concrete example.",
    )
    args = parser.parse_args()

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

    print("prefilling target...")
    started = time.perf_counter()
    target_hidden, target_logits = _target_forward(target, prompt_ids, target_caches)
    target_prefill_s = time.perf_counter() - started
    guaranteed_token = int(mx.argmax(target_logits[:, -1, :], axis=-1).item())

    shifted_ids = prompt_ids[1:] + [guaranteed_token]
    mtp_cache = KVCache()

    print("prefilling MTP prompt KV...")
    started = time.perf_counter()
    mtp_hidden = _mtp_forward(target, mtp, mtp_cache, shifted_ids, target_hidden)
    mtp_prefill_s = time.perf_counter() - started

    proposal_logits = []
    proposals = []
    proposal_times = []

    arr = _mtp_logits(target, mtp_hidden)
    proposal_logits.append(arr)
    proposals.append(int(np.argmax(arr)))

    current_hidden = mtp_hidden[:, -1:, :]
    current_token = proposals[0]

    for _ in range(1, args.mtp_tokens):
        started = time.perf_counter()
        current_hidden = _mtp_forward(
            target, mtp, mtp_cache, [current_token], current_hidden
        )
        arr = _mtp_logits(target, current_hidden)
        proposal_times.append(time.perf_counter() - started)
        proposal_logits.append(arr)
        current_token = int(np.argmax(arr))
        proposals.append(current_token)

    verify_block = [guaranteed_token] + proposals[:-1]
    print(f"verifying {len(proposals)} greedy MTP proposals...")
    started = time.perf_counter()
    _, verify_logits = _target_forward(target, verify_block, target_caches)
    verify_s = time.perf_counter() - started
    target_predictions = [
        int(x) for x in mx.argmax(verify_logits[0], axis=-1).tolist()
    ]

    prefix_accept = 0
    print("\n=== MTP target-rank diagnostic ===")
    print(f"target bits:          {args.target_bits}")
    print(f"prompt tokens:        {len(prompt_ids)}")
    print(
        f"guaranteed token:     {guaranteed_token} "
        f"{target.tokenizer.decode([guaranteed_token], skip_special_tokens=False)!r}"
    )

    for i, (proposal, target_id, arr) in enumerate(
        zip(proposals, target_predictions, proposal_logits)
    ):
        order = np.argsort(arr)[::-1]
        rank = int(np.flatnonzero(order == target_id)[0]) + 1
        top = _topk(arr, args.top_k)
        best_logit = float(arr[proposal])
        target_logit = float(arr[target_id])
        gap = best_logit - target_logit
        match = proposal == target_id
        if match and prefix_accept == i:
            prefix_accept += 1

        print(
            f"\nstep {i}: {'MATCH' if match else 'MISS'}  "
            f"mtp={proposal} {target.tokenizer.decode([proposal], skip_special_tokens=False)!r}  "
            f"target={target_id} {target.tokenizer.decode([target_id], skip_special_tokens=False)!r}"
        )
        print(f"  target rank in MTP: {rank}   top1-target logit gap: {gap:.4f}")
        print("  MTP top candidates:")
        for j, tok in enumerate(top, start=1):
            marker = " <-- target" if int(tok) == target_id else ""
            print(
                f"    {j:2d}. id={int(tok):6d} "
                f"logit={float(arr[tok]):9.4f} "
                f"piece={target.tokenizer.decode([int(tok)], skip_special_tokens=False)!r}{marker}"
            )

    recursive_s = sum(proposal_times)
    print("\n=== summary ===")
    print(f"MTP resident bytes:       {mtp_bytes / 1024**2:.2f} MiB")
    print(f"target prefill:           {target_prefill_s:.3f} s")
    print(f"MTP prompt prefill:       {mtp_prefill_s:.3f} s")
    print(f"recursive MTP time:       {recursive_s:.3f} s")
    print(f"target verification:      {verify_s:.3f} s")
    print(f"greedy accepted prefix:   {prefix_accept}/{len(proposals)}")

    active = _mlx_memory_mb("get_active_memory")
    peak = _mlx_memory_mb("get_peak_memory")
    if active is not None:
        print(f"MLX active memory:        {active:.2f} MiB")
    if peak is not None:
        print(f"MLX peak memory:          {peak:.2f} MiB")


if __name__ == "__main__":
    main()
