"""Experimental no-replay speculative decoding for streamed Qwen3.8 on macOS.

This benchmark keeps the working speculative prototype intact and tests a different
verification strategy for Qwen3.5/3.8 hybrid caches:

* the small draft model proposes N greedy tokens;
* the streamed target verifies current_token + drafts in one layer-stream traversal;
* recurrent (ArraysCache) layers are evaluated one token at a time while each layer's
  weights are resident, retaining the recurrent state after every candidate position;
* full-attention (KVCache) layers still evaluate the whole candidate block at once;
* after the target logits reveal the accepted prefix, recurrent caches select the
  already-computed prefix state and KV caches rewind only their logical offset.

This removes the expensive second target traversal after a draft rejection. The cost is
some extra compute inside recurrent layers during verification.
"""

import argparse
import time

import mlx.core as mx
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
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
            cloned.keys = _copy_array(cache.keys[..., : cache.offset, :])
            cloned.values = _copy_array(cache.values[..., : cache.offset, :])
        return cloned

    raise TypeError(
        f"Unsupported speculative cache type: {type(cache).__name__}. "
        "This experiment expects Qwen3.5-family ArraysCache/KVCache only."
    )


def _clone_caches(caches):
    started = time.perf_counter()
    cloned = [_clone_cache_entry(cache) for cache in caches]
    mx.eval([cache.state for cache in cloned])
    return cloned, time.perf_counter() - started


def _target_forward(model, token_ids, caches, *, return_logits):
    """Normal block target forward; used for prefill only."""
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

    logits = model._project_logits(hidden)
    mx.eval(logits)
    return logits[0]


def _snapshot_arrays_cache(cache):
    # Qwen's GatedDeltaNet replaces cache[0]/cache[1] with newly produced arrays
    # on each step, so retaining these evaluated references gives us a cheap
    # prefix-state checkpoint without copying the multi-megabyte recurrent state.
    mx.eval(cache.state)
    return (
        tuple(cache.cache),
        cache.left_padding,
        cache.lengths,
    )


def _restore_arrays_cache(cache, snapshot):
    state, left_padding, lengths = snapshot
    cache.cache = list(state)
    cache.left_padding = left_padding
    cache.lengths = lengths


def _target_verify_checkpointed(model, token_ids, caches):
    """Verify a candidate block and retain a target-cache checkpoint per position.

    Linear-attention layers are stepped token-by-token while their layer weights remain
    resident. This retains recurrent states for every prefix. Full-attention layers keep
    the efficient block call because KVCache can rewind by offset.
    """
    if not token_ids:
        raise ValueError("checkpointed verification requires at least one token")

    tokens = mx.array(token_ids, dtype=mx.int32)[None, :]

    embedding = model._load_embedding()
    hidden = embedding(tokens)
    mx.eval(hidden)
    del embedding
    model._cleanup()

    histories = []

    for index in range(model.model_args.num_hidden_layers):
        layer = model._load_layer(index)
        cache = caches[index]

        if layer.is_linear:
            outputs = []
            checkpoints = []
            for position in range(len(token_ids)):
                token_hidden = hidden[:, position : position + 1, :]
                mask = create_ssm_mask(token_hidden, cache)
                token_hidden = layer(token_hidden, mask=mask, cache=cache)
                mx.eval([token_hidden, cache.state])
                outputs.append(token_hidden)
                checkpoints.append(_snapshot_arrays_cache(cache))

            hidden = mx.concatenate(outputs, axis=1)
            mx.eval(hidden)
            histories.append(("arrays", checkpoints))
        else:
            base_offset = cache.offset
            mask = create_attention_mask(hidden, cache)
            hidden = layer(hidden, mask=mask, cache=cache)
            mx.eval([hidden, cache.state])
            histories.append(
                ("kv", [base_offset + step for step in range(1, len(token_ids) + 1)])
            )

        del layer
        model._cleanup()

    norm = model._load_norm()
    hidden = norm(hidden)
    mx.eval(hidden)
    del norm
    model._cleanup()

    logits = model._project_logits(hidden)
    mx.eval(logits)
    return logits[0], histories


def _commit_verified_prefix(caches, histories, keep_tokens):
    """Keep exactly the first keep_tokens inputs from a speculative verify block."""
    if keep_tokens <= 0:
        raise ValueError("keep_tokens must be >= 1")

    for cache, history in zip(caches, histories):
        kind, checkpoints = history
        checkpoint = checkpoints[keep_tokens - 1]

        if kind == "arrays":
            if not isinstance(cache, ArraysCache):
                raise TypeError("ArraysCache history paired with non-ArraysCache")
            _restore_arrays_cache(cache, checkpoint)
        elif kind == "kv":
            if not isinstance(cache, KVCache):
                raise TypeError("KV history paired with non-KVCache")
            # The rejected suffix can remain allocated. The next KV update starts
            # from this logical offset and overwrites it.
            cache.offset = checkpoint
        else:
            raise ValueError(f"Unknown cache history kind: {kind}")

    mx.eval([cache.state for cache in caches])


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
        description="Benchmark no-replay speculative decoding for streamed Qwen3.8."
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
        help="Target resident-weight budget.",
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
    print("verification: checkpointed recurrent state (no target replay)")

    prefix = list(map(int, target_ids[:-1]))
    current_token = int(target_ids[-1])

    target_caches = [
        target._new_cache(i) for i in range(target.model_args.num_hidden_layers)
    ]
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
    target_passes = 1
    verify_passes = 0
    clone_total_s = 0.0
    draft_total_s = draft_prefill_s
    verify_total_s = 0.0
    commit_total_s = 0.0
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
        logits, histories = _target_verify_checkpointed(target, block, verify_caches)
        target_predictions = [int(x) for x in mx.argmax(logits, axis=-1).tolist()]
        verify_s = time.perf_counter() - verify_started
        verify_total_s += verify_s
        verify_passes += 1
        target_passes += 1

        n_accept = 0
        while (
            n_accept < n_draft
            and target_predictions[n_accept] == draft_tokens[n_accept]
        ):
            n_accept += 1
        accepted += n_accept

        if n_accept == n_draft:
            emitted = draft_tokens + [target_predictions[n_draft]]
            sources = ["draft"] * n_draft + ["target"]
            target_caches = verify_caches

            draft_tail = [draft_tokens[-1]] if n_draft else [current_token]
            align_started = time.perf_counter()
            _draft_forward(draft, draft_tail, draft_work, return_logits=False)
            draft_total_s += time.perf_counter() - align_started
            draft_caches = draft_work
            current_token = target_predictions[n_draft]
            commit_s = 0.0
        else:
            correction = target_predictions[n_accept]
            emitted = draft_tokens[:n_accept] + [correction]
            sources = ["draft"] * n_accept + ["target"]

            # The target has already computed the valid prefix. Select that state
            # instead of replaying the 27B model over current + accepted drafts.
            commit_started = time.perf_counter()
            _commit_verified_prefix(
                verify_caches,
                histories,
                keep_tokens=1 + n_accept,
            )
            commit_s = time.perf_counter() - commit_started
            commit_total_s += commit_s
            target_caches = verify_caches

            commit = [current_token] + draft_tokens[:n_accept]
            draft_align_started = time.perf_counter()
            _draft_forward(draft, commit, draft_caches, return_logits=False)
            draft_total_s += time.perf_counter() - draft_align_started
            current_token = correction

        print(
            f"[spec {iteration}] proposed={n_draft} accepted={n_accept} "
            f"verify={verify_s:.3f}s commit={commit_s:.3f}s "
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
    print("\n=== AirLLM Qwen3.8 checkpointed speculative benchmark ===")
    print(f"generated tokens:          {len(generated)}")
    print(f"draft tokens proposed:     {proposed}")
    print(f"draft tokens accepted:     {accepted}")
    acceptance = (accepted / proposed) if proposed else 0.0
    print(f"draft acceptance rate:     {acceptance * 100:.1f}%")
    print(f"target verify passes:      {verify_passes}")
    print("target replay passes:      0")
    print(f"target passes incl prefill:{target_passes:>6}")
    if generated:
        print(f"verify passes/output tok:  {verify_passes / len(generated):.3f}")
        print(f"all target passes/out tok: {target_passes / len(generated):.3f}")
        print(f"generation sec/output tok: {generation_s / len(generated):.3f}")
        print(
            f"generation throughput:     {len(generated) / generation_s:.3f} tok/s"
        )
    print(f"target prefill:            {target_prefill_s:.3f} s")
    print(f"draft prefill:             {draft_prefill_s:.3f} s")
    print(f"draft total:               {draft_total_s:.3f} s")
    print(f"cache cloning total:       {clone_total_s:.3f} s")
    print(f"checkpoint commit total:   {commit_total_s:.3f} s")
    print(f"target verify total:       {verify_total_s:.3f} s")
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
