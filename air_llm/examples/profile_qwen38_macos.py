"""Break down streamed Qwen3.8 MLX latency into load, compute, and cleanup costs."""

import argparse
import gc
import time

import mlx.core as mx

from airllm import AutoModel
from mlx_lm.models.base import create_attention_mask, create_ssm_mask


def _pct(part, total):
    return 100.0 * part / total if total else 0.0


def _maybe_sync(barrier_mode):
    # mx.eval() is already blocking. "safe" mirrors the current backend's extra barriers;
    # "eval-only" mirrors current mlx-lm's eval + clear-cache pattern for an A/B measurement.
    if barrier_mode == "safe":
        mx.synchronize()


def _profile_cleanup(model, barrier_mode):
    if barrier_mode == "safe":
        model._cleanup()
        return

    gc.collect()
    clear_cache = getattr(mx, "clear_cache", None)
    if clear_cache is not None:
        clear_cache()


def _profile_embedding(model, token_ids, barrier_mode):
    started = time.perf_counter()
    module = model._load_embedding()
    loaded = time.perf_counter()
    hidden = module(token_ids)
    mx.eval(hidden)
    _maybe_sync(barrier_mode)
    computed = time.perf_counter()
    del module
    _profile_cleanup(model, barrier_mode)
    finished = time.perf_counter()
    return hidden, {
        "load": loaded - started,
        "compute": computed - loaded,
        "cleanup": finished - computed,
        "total": finished - started,
    }


def _profile_norm(model, hidden, barrier_mode):
    started = time.perf_counter()
    module = model._load_norm()
    loaded = time.perf_counter()
    hidden = module(hidden)
    mx.eval(hidden)
    _maybe_sync(barrier_mode)
    computed = time.perf_counter()
    del module
    _profile_cleanup(model, barrier_mode)
    finished = time.perf_counter()
    return hidden, {
        "load": loaded - started,
        "compute": computed - loaded,
        "cleanup": finished - computed,
        "total": finished - started,
    }


def _profile_layers(model, hidden, caches, barrier_mode):
    rows = []
    totals = {"load": 0.0, "mask": 0.0, "compute": 0.0, "cleanup": 0.0, "total": 0.0}

    for index in range(model.model_args.num_hidden_layers):
        started = time.perf_counter()
        layer = model._load_layer(index)
        loaded = time.perf_counter()

        cache = caches[index]
        mask = create_ssm_mask(hidden, cache) if layer.is_linear else create_attention_mask(hidden, cache)
        masked = time.perf_counter()

        hidden = layer(hidden, mask=mask, cache=cache)
        # Materializing the cache state is non-negotiable for Qwen's hybrid decoder: otherwise an
        # unevaluated ArraysCache/KVCache graph can retain streamed layer weights. Only the extra
        # synchronize barriers are varied by this profiler.
        mx.eval([hidden, cache.state])
        _maybe_sync(barrier_mode)
        computed = time.perf_counter()

        layer_kind = "delta" if layer.is_linear else "attention"
        del layer
        _profile_cleanup(model, barrier_mode)
        finished = time.perf_counter()

        row = {
            "index": index,
            "kind": layer_kind,
            "load": loaded - started,
            "mask": masked - loaded,
            "compute": computed - masked,
            "cleanup": finished - computed,
            "total": finished - started,
        }
        rows.append(row)
        for key in totals:
            totals[key] += row[key]

    return hidden, totals, rows


def _profile_logits(model, hidden):
    # Keep the backend's known-good lm_head path for this experiment. The decoder accounts for >90%
    # of pass time, so isolating its barriers is sufficient before changing production behavior.
    started = time.perf_counter()
    logits = model._project_logits(hidden[:, -1, :])
    finished = time.perf_counter()
    return logits, {"total": finished - started}


def _profile_pass(model, token_ids, caches, label, barrier_mode):
    pass_started = time.perf_counter()

    hidden, embedding = _profile_embedding(model, token_ids, barrier_mode)
    hidden, layers, rows = _profile_layers(model, hidden, caches, barrier_mode)
    hidden, norm = _profile_norm(model, hidden, barrier_mode)
    logits, head = _profile_logits(model, hidden)

    sample_started = time.perf_counter()
    token = model._sample(logits, 0.0)
    mx.eval(token)
    _maybe_sync(barrier_mode)
    sample_total = time.perf_counter() - sample_started

    total = time.perf_counter() - pass_started
    token_id = int(token.item())
    piece = model.tokenizer.decode([token_id], skip_special_tokens=False)

    return token, {
        "label": label,
        "barrier_mode": barrier_mode,
        "token_id": token_id,
        "piece": piece,
        "embedding": embedding,
        "layers": layers,
        "norm": norm,
        "head": head,
        "sample": sample_total,
        "total": total,
        "rows": rows,
    }


def _print_pass(result):
    total = result["total"]
    embedding = result["embedding"]
    layers = result["layers"]
    norm = result["norm"]
    head = result["head"]

    print(f"\n=== {result['label']} ===")
    print(f"barrier mode:            {result['barrier_mode']}")
    print(f"token:                   id={result['token_id']} piece={result['piece']!r}")
    print(f"pass total:              {total:8.3f} s")
    print(f"embedding total:         {embedding['total']:8.3f} s  ({_pct(embedding['total'], total):5.1f}%)")
    print(f"  embedding load:        {embedding['load']:8.3f} s")
    print(f"  embedding compute:     {embedding['compute']:8.3f} s")
    print(f"  embedding cleanup:     {embedding['cleanup']:8.3f} s")
    print(f"decoder layers total:    {layers['total']:8.3f} s  ({_pct(layers['total'], total):5.1f}%)")
    print(f"  layer load/construct:  {layers['load']:8.3f} s  ({_pct(layers['load'], total):5.1f}%)")
    print(f"  mask creation:         {layers['mask']:8.3f} s  ({_pct(layers['mask'], total):5.1f}%)")
    print(f"  forward + eval/sync:   {layers['compute']:8.3f} s  ({_pct(layers['compute'], total):5.1f}%)")
    print(f"  layer cleanup/sync:    {layers['cleanup']:8.3f} s  ({_pct(layers['cleanup'], total):5.1f}%)")
    print(f"final norm:              {norm['total']:8.3f} s  ({_pct(norm['total'], total):5.1f}%)")
    print(f"lm_head total:           {head['total']:8.3f} s  ({_pct(head['total'], total):5.1f}%)")
    print(f"sampling:                {result['sample']:8.3f} s  ({_pct(result['sample'], total):5.1f}%)")

    rows = result["rows"]
    delta = [r for r in rows if r["kind"] == "delta"]
    attention = [r for r in rows if r["kind"] == "attention"]
    for name, group in (("DeltaNet", delta), ("full attention", attention)):
        if not group:
            continue
        group_total = sum(r["total"] for r in group)
        group_load = sum(r["load"] for r in group)
        group_compute = sum(r["compute"] for r in group)
        print(
            f"{name:24s}{group_total:8.3f} s across {len(group):2d} layers "
            f"(load={group_load:.3f}s compute={group_compute:.3f}s)"
        )

    print("\nslowest layers by total:")
    for row in sorted(rows, key=lambda r: r["total"], reverse=True)[:8]:
        print(
            f"  layer {row['index']:2d} {row['kind']:9s} total={row['total']:.3f}s "
            f"load={row['load']:.3f}s compute={row['compute']:.3f}s cleanup={row['cleanup']:.3f}s"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--compression", default="4bit", choices=["none", "4bit"])
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--layer-path", default=None)
    parser.add_argument(
        "--barrier-mode",
        default="safe",
        choices=["safe", "eval-only"],
        help=(
            "safe mirrors the current backend's extra mx.synchronize() barriers; eval-only keeps "
            "blocking mx.eval() cache materialization and mx.clear_cache() but removes redundant "
            "explicit synchronization for this profiling run."
        ),
    )
    args = parser.parse_args()

    compression = None if args.compression == "none" else args.compression
    model = AutoModel.from_pretrained(
        args.model,
        max_seq_len=args.max_seq_len,
        layer_shards_saving_path=args.layer_path,
        compression=compression,
    )

    messages = [
        {"role": "user", "content": "Reply with exactly: Qwen3.8 AirLLM MLX is running."},
    ]
    prompt = model.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = model.tokenizer(
        [prompt],
        return_tensors="np",
        return_attention_mask=False,
        truncation=True,
        max_length=args.max_seq_len,
        padding=False,
    )
    input_ids = mx.array(inputs["input_ids"])
    caches = [model._new_cache(i) for i in range(model.model_args.num_hidden_layers)]

    first_token, prefill = _profile_pass(
        model, input_ids, caches, "prefill / first token", args.barrier_mode
    )
    second_input = first_token[:, None]
    _, decode = _profile_pass(
        model, second_input, caches, "single-token decode", args.barrier_mode
    )

    print(f"weights: {'MLX affine 4-bit' if getattr(model, 'mlx_quantized', False) else 'FP16'}")
    print(f"prompt tokens: {int(input_ids.shape[1])}")
    _print_pass(prefill)
    _print_pass(decode)

    active = getattr(mx, "get_active_memory", lambda: 0)() / 1024 / 1024
    peak = getattr(mx, "get_peak_memory", lambda: 0)() / 1024 / 1024
    print(f"\nMLX active memory: {active:.2f} MiB")
    print(f"MLX peak memory:   {peak:.2f} MiB")


if __name__ == "__main__":
    main()
