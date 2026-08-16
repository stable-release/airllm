"""Experimental speculative decoding for streamed Qwen3.8 on macOS.

This intentionally lives outside the production backend. Qwen3.5/3.8 mixes
Gated-DeltaNet ArraysCache entries with ordinary KVCache entries. MLX-LM's generic
speculative decoder requires every target cache to be trimmable, but ArraysCache is
recurrent state and is not trimmable. This experiment therefore uses copy-on-verify:

* canonical target/draft caches lag by one accepted token;
* a cheap resident draft model proposes N greedy tokens;
* a cloned target cache verifies current_token + N drafts in one streamed 27B pass;
* if every draft matches, the verified cache becomes canonical with no replay;
* on the first mismatch, discard the speculative target cache and replay only the
  accepted prefix into the canonical cache (one extra streamed pass).

The first version is greedy-only by design. That makes acceptance exact and keeps the
benchmark focused on whether speculative decoding can amortize AirLLM's streamed
weight traversal.
"""

import argparse
import time

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.utils import load as mlx_load

from airllm import AutoModel


def _mlx_memory_mb(name):
    fn = getattr(mx, name, None)
    if fn is None:
        return None
    try:
        return fn() / 1024 / 1024
    except Exception:
        return None


def _copy_array(value):
    if value is None:
        return None
    # Force a distinct evaluated buffer. A real copy matters for KVCache because
    # its update path writes into preallocated arrays in place.
    out = value + mx.zeros((), dtype=value.dtype)
    mx.eval(out)
    return out


def _clone_cache_entry(cache):
    if isinstance(cache, ArraysCache):
        cloned = ArraysCache(len(cache.cache))
        cloned.cache = [_copy_array(value) for value in cache.cache]
        cloned.left_padding = _copy_array(cache.left_padding)
        cloned.lengths = _copy_array(cache.lengths)
        return cloned

    if isinstance(cache, KVCache):
        cloned = KVCache()
        cloned.offset = cache.offset
        if cache.keys is not None:
            # Copy only the live prefix. KVCache will regrow its allocation on the
            # next update, which keeps speculative snapshots small.
            cloned.keys = _copy_array(cache.keys[..., : cache.offset, :])
            cloned.values = _copy_array(cache.values[..., : cache.offset, :])
        return cloned

    raise TypeError(
        f"Unsupported speculative cache type: {type(cache).__name__}. "
        "This experiment currently expects Qwen3.5-family ArraysCache/KVCache only."
    )


def _clone_caches(caches):
    started = time.perf_counter()
    cloned = [_clone_cache_entry(cache) for cache in caches]
    mx.eval([cache.state for cache in cloned])
    return cloned, time.perf_counter() - started


def _target_forward(model, token_ids, caches, *, return_logits):
    """Advance the streamed target over a token block, optionally returning every position's logits."""
    if not token_ids:
        return None

    tokens = mx.array(token_ids, dtype=mx.int32)[None, :]

    embedding = model._load_embedding()
    hidden = embedding(tokens)
    mx.eval(hidden)
    del embedding
    model._cleanup()

    hidden = model._run_layers(hidden, caches)

    if not return_logits:
        del hidden
        model._cleanup()
        return None

    norm = model._load_norm()
    hidden = norm(hidden)
    mx.eval(hidden)
    del norm
    model._cleanup()

    # Unlike normal AirLLM decode, keep logits for every position in the block.
    # Position i predicts the token after token_ids[i].
    logits = model._project_logits(hidden)
    mx.eval(logits)
    return logits[0]


def _draft_forward(model, token_ids, caches, *, return_logits):
    if not token_ids:
        return None
    tokens = mx.array(token_ids, dtype=mx.uint32)[None, :]
    logits = model(tokens, cache=caches)
    mx.eval(logits, [cache.state for cache in caches])
    return logits[0] if return_logits else None


def _draft_generate(model, caches, current_token, count):
    drafted = []
    y = int(current_token)
    for _ in range(count):
        logits = _draft_forward(model, [y], caches, return_logits=True)
        y = int(mx.argmax(logits[-1], axis=-1).item())
        drafted.append(y)
    return drafted


def _format_piece(tokenizer, token_id):
    return tokenizer.decode([int(token_id)], skip_special_tokens=False)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark snapshot/restore speculative decoding for streamed Qwen3.8."
    )
    parser.add_argument(
        "--model",
        default="orcarouter/Qwen3.8-27B-Uncensored-FP8",
        help="Streamed AirLLM target model.",
    )
    parser.add_argument(
        "--draft-model",
        default="Qwen/Qwen3.5-0.8B",
        help="Small resident MLX-LM draft model.",
    )
    parser.add_argument("--num-draft-tokens", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument(
        "--resident-gib",
        type=float,
        default=0.0,
        help=(
            "Target resident-weight budget. Keep this at 0 for the first speculative "
            "benchmark so the BF16 0.8B draft has plenty of unified-memory headroom."
        ),
    )
    parser.add_argument(
        "--prompt",
        default="Reply with exactly: Qwen3.8 AirLLM MLX is running.",
    )
    args = parser.parse_args()

    if args.num_draft_tokens < 0:
        raise ValueError("--num-draft-tokens must be >= 0")

    print("loading streamed target...")
    target = AutoModel.from_pretrained(
        args.model,
        compression="4bit",
        max_seq_len=args.max_seq_len,
        mlx_resident_gib=args.resident_gib,
    )

    print("loading resident draft model...")
    draft, draft_tokenizer = mlx_load(args.draft_model, lazy=False)

    prompt = target.tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    target_ids = target.tokenizer.encode(prompt, add_special_tokens=False)
    draft_ids = draft_tokenizer.encode(prompt, add_special_tokens=False)
    if list(target_ids) != list(draft_ids):
        raise RuntimeError(
            "Target and draft tokenizers produced different prompt IDs. "
            "Speculative decoding requires an identical token space."
        )
    if len(target_ids) < 2:
        raise ValueError("Prompt must contain at least two tokens for this experiment.")

    print(f"target:       {args.model}")
    print(f"draft:        {args.draft_model}")
    print(f"draft tokens: {args.num_draft_tokens}")
    print(f"prompt tokens:{len(target_ids):>5}")
    print(f"target resident budget: {args.resident_gib:.2f} GiB")

    # Invariant: canonical caches contain every token before current_token; current_token
    # itself has not yet been consumed by either model.
    prefix = list(map(int, target_ids[:-1]))
    current_token = int(target_ids[-1])

    target_caches = [target._new_cache(i) for i in range(target.model_args.num_hidden_layers)]
    draft_caches = draft.make_cache()

    started = time.perf_counter()
    print("prefilling target cache...")
    target_prefill_started = time.perf_counter()
    _target_forward(target, prefix, target_caches, return_logits=False)
    target_prefill_s = time.perf_counter() - target_prefill_started

    print("prefilling draft cache...")
    draft_prefill_started = time.perf_counter()
    _draft_forward(draft, prefix, draft_caches, return_logits=False)
    draft_prefill_s = time.perf_counter() - draft_prefill_started

    generated = []
    proposed = 0
    accepted = 0
    target_passes = 1  # target prefill
    verify_passes = 0
    replay_passes = 0
    clone_total_s = 0.0
    draft_total_s = draft_prefill_s
    verify_total_s = 0.0
    replay_total_s = 0.0
    iteration = 0

    eos = target.tokenizer.eos_token_id
    eos_ids = set()
    if isinstance(eos, int):
        eos_ids.add(eos)
    elif eos is not None:
        eos_ids.update(eos)

    generation_started = time.perf_counter()
    stopped = False

    while len(generated) < args.max_new_tokens and not stopped:
        iteration += 1
        remaining = args.max_new_tokens - len(generated)
        # A fully accepted speculative block emits N drafts plus one target bonus token.
        n_draft = min(args.num_draft_tokens, max(remaining - 1, 0))

        draft_work, draft_clone_s = _clone_caches(draft_caches)
        clone_total_s += draft_clone_s

        draft_started = time.perf_counter()
        draft_tokens = _draft_generate(draft, draft_work, current_token, n_draft)
        draft_s = time.perf_counter() - draft_started
        draft_total_s += draft_s
        proposed += n_draft

        verify_caches, target_clone_s = _clone_caches(target_caches)
        clone_total_s += target_clone_s

        block = [current_token] + draft_tokens
        verify_started = time.perf_counter()
        logits = _target_forward(target, block, verify_caches, return_logits=True)
        target_predictions = [
            int(x)
            for x in mx.argmax(logits, axis=-1).tolist()
        ]
        verify_s = time.perf_counter() - verify_started
        verify_total_s += verify_s
        verify_passes += 1
        target_passes += 1

        n_accept = 0
        while n_accept < n_draft and target_predictions[n_accept] == draft_tokens[n_accept]:
            n_accept += 1
        accepted += n_accept

        if n_accept == n_draft:
            # The target cache already contains current + every draft token. The next
            # canonical current token is the target's bonus prediction after the block.
            emitted = draft_tokens + [target_predictions[n_draft]]
            sources = ["draft"] * n_draft + ["target"]
            target_caches = verify_caches

            # Draft generation has consumed current through draft[-2]. Advance it through
            # the final draft token (or current itself when N=0), then promote the cache.
            draft_tail = [draft_tokens[-1]] if n_draft else [current_token]
            align_started = time.perf_counter()
            _draft_forward(draft, draft_tail, draft_work, return_logits=False)
            draft_total_s += time.perf_counter() - align_started
            draft_caches = draft_work
            current_token = target_predictions[n_draft]
            replay_s = 0.0
        else:
            # The speculative target cache includes a rejected suffix and cannot be trimmed
            # because Qwen's ArraysCache is recurrent state. Discard it and advance the
            # canonical cache only through current + the accepted draft prefix.
            correction = target_predictions[n_accept]
            emitted = draft_tokens[:n_accept] + [correction]
            sources = ["draft"] * n_accept + ["target"]
            commit = [current_token] + draft_tokens[:n_accept]

            replay_started = time.perf_counter()
            _target_forward(target, commit, target_caches, return_logits=False)
            replay_s = time.perf_counter() - replay_started
            replay_total_s += replay_s
            replay_passes += 1
            target_passes += 1

            draft_align_started = time.perf_counter()
            _draft_forward(draft, commit, draft_caches, return_logits=False)
            draft_total_s += time.perf_counter() - draft_align_started
            current_token = correction

        print(
            f"[spec {iteration}] proposed={n_draft} accepted={n_accept} "
            f"verify={verify_s:.3f}s replay={replay_s:.3f}s "
            f"draft={draft_s:.3f}s"
        )

        for token_id, source in zip(emitted, sources):
            if len(generated) >= args.max_new_tokens:
                break
            generated.append(int(token_id))
            piece = _format_piece(target.tokenizer, token_id)
            print(
                f"  [{len(generated) - 1:02d}] {source:6s} "
                f"id={int(token_id)} piece={piece!r}"
            )
            if int(token_id) in eos_ids:
                stopped = True
                break

    generation_s = time.perf_counter() - generation_started
    total_s = time.perf_counter() - started
    decoded = target.tokenizer.decode(generated, skip_special_tokens=True)

    print("\ndecoded:", decoded)
    print("\n=== AirLLM Qwen3.8 speculative benchmark ===")
    print(f"generated tokens:          {len(generated)}")
    print(f"draft tokens proposed:     {proposed}")
    print(f"draft tokens accepted:     {accepted}")
    acceptance = (accepted / proposed) if proposed else 0.0
    print(f"draft acceptance rate:     {acceptance * 100:.1f}%")
    print(f"target verify passes:      {verify_passes}")
    print(f"target replay passes:      {replay_passes}")
    print(f"target passes incl prefill:{target_passes:>6}")
    if generated:
        print(f"verify passes/output tok:  {verify_passes / len(generated):.3f}")
        print(f"all target passes/out tok: {target_passes / len(generated):.3f}")
        print(f"generation sec/output tok: {generation_s / len(generated):.3f}")
        print(f"generation throughput:     {len(generated) / generation_s:.3f} tok/s")
    print(f"target prefill:            {target_prefill_s:.3f} s")
    print(f"draft prefill:             {draft_prefill_s:.3f} s")
    print(f"draft total:               {draft_total_s:.3f} s")
    print(f"cache cloning total:       {clone_total_s:.3f} s")
    print(f"target verify total:       {verify_total_s:.3f} s")
    print(f"target replay total:       {replay_total_s:.3f} s")
    print(f"generation total:          {generation_s:.3f} s")
    print(f"wall total incl prefill:   {total_s:.3f} s")

    active_mb = _mlx_memory_mb("get_active_memory")
    peak_mb = _mlx_memory_mb("get_peak_memory")
    if active_mb is not None:
        print(f"MLX active memory:         {active_mb:.2f} MiB")
    if peak_mb is not None:
        print(f"MLX peak memory:           {peak_mb:.2f} MiB")


if __name__ == "__main__":
    main()
