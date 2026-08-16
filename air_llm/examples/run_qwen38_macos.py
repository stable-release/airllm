"""Smoke-test and benchmark Qwen3.8-27B through AirLLM's streamed MLX backend."""

import argparse
import time
from pathlib import Path

import mlx.core as mx

from airllm import AutoModel


def _format_bytes(value):
    value = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0


def _logical_stream_bytes(model):
    """Return bytes AirLLM logically reloads for one complete model pass.

    This is intentionally a logical file-size metric. macOS may satisfy some reads from the file
    cache, so it is not a claim about physical SSD traffic.
    """
    checkpoint_path = Path(model.checkpoint_path)
    total = 0
    files = []
    for layer_name in model.layer_names:
        path = checkpoint_path / f"{layer_name}.mlx.npz"
        if path.exists():
            size = path.stat().st_size
            total += size
            files.append((path, size))
    return total, files


def _mlx_memory_mb(name):
    fn = getattr(mx, name, None)
    if fn is None:
        return None
    try:
        return fn() / 1024 / 1024
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--layer-path", default=None)
    parser.add_argument("--show-memory", action="store_true")
    parser.add_argument(
        "--compression",
        choices=["4bit"],
        default=None,
        help=(
            "Use MLX-native affine 4-bit streamed weights on macOS. The first run prepares a "
            "separate resumable quantized sidecar; the existing FP16 split is retained."
        ),
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help=(
            "Prepare/reuse the selected streamed-weight format and exit before inference. Useful "
            "for doing the one-time MLX 4-bit conversion in a separate Python process."
        ),
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable Qwen3.8 thinking mode. The smoke test defaults to direct/non-thinking mode.",
    )
    parser.add_argument(
        "--debug-tokens",
        action="store_true",
        help="Print each generated token id and raw decoded piece.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Measure first-token latency, decode latency, logical streamed bytes, and MLX memory.",
    )
    args = parser.parse_args()

    model = AutoModel.from_pretrained(
        args.model,
        max_seq_len=args.max_seq_len,
        layer_shards_saving_path=args.layer_path,
        show_memory_util=args.show_memory,
        compression=args.compression,
    )

    if args.prepare_only:
        logical_bytes, shard_files = _logical_stream_bytes(model)
        print("preparation complete")
        print(f"weights: {'MLX affine 4-bit' if getattr(model, 'mlx_quantized', False) else 'FP16'}")
        print(f"checkpoint path: {model.checkpoint_path}")
        print(f"streamed shard files/pass: {len(shard_files)}")
        print(f"logical weight bytes/pass: {_format_bytes(logical_bytes)}")
        return

    messages = [
        {"role": "user", "content": "Reply with exactly: Qwen3.8 AirLLM MLX is running."},
    ]
    prompt = model.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=args.thinking,
    )
    print(f"chat mode: {'thinking' if args.thinking else 'non-thinking'}")
    print(f"weights: {'MLX affine 4-bit' if getattr(model, 'mlx_quantized', False) else 'FP16'}")
    print(f"prompt tail: {prompt[-200:]!r}")

    inputs = model.tokenizer(
        [prompt],
        return_tensors="np",
        return_attention_mask=False,
        truncation=True,
        max_length=args.max_seq_len,
        padding=False,
    )
    input_ids = mx.array(inputs["input_ids"])
    prompt_tokens = int(input_ids.shape[1])

    # Keep the simple path available for normal smoke tests. Benchmark/debug mode consumes the
    # underlying token generator directly so we can timestamp each completed model pass.
    if not args.debug_tokens and not args.benchmark:
        output = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
        )
        print(output)
        return

    logical_bytes, shard_files = _logical_stream_bytes(model)
    if args.benchmark:
        print(f"prompt tokens: {prompt_tokens}")
        print(f"streamed shard files/pass: {len(shard_files)}")
        print(f"logical weight bytes/pass: {_format_bytes(logical_bytes)}")

    token_ids = []
    token_times = []
    eos_token_id = model.tokenizer.eos_token_id
    eos_ids = set()
    if isinstance(eos_token_id, int):
        eos_ids.add(eos_token_id)
    elif eos_token_id is not None:
        eos_ids.update(eos_token_id)

    generator = model.model_generate(input_ids, temperature=0.0)
    started = time.perf_counter()

    for index, token in enumerate(generator):
        now = time.perf_counter()
        elapsed = now - started if index == 0 else now - token_times[-1]
        token_times.append(now)

        token_id = int(token.item())
        token_ids.append(token_id)
        piece = model.tokenizer.decode([token_id], skip_special_tokens=False)

        if args.debug_tokens:
            print(f"[token {index}] id={token_id} piece={piece!r} pass={elapsed:.3f}s")
        elif args.benchmark:
            label = "first-token" if index == 0 else "decode"
            print(f"[{label} {index}] {elapsed:.3f}s piece={piece!r}")

        if token_id in eos_ids or len(token_ids) >= args.max_new_tokens:
            break

    finished = time.perf_counter()
    decoded = model.tokenizer.decode(token_ids, skip_special_tokens=True)
    print("decoded:", decoded)

    if not args.benchmark or not token_times:
        return

    first_token_s = token_times[0] - started
    total_s = finished - started
    decode_intervals = [token_times[i] - token_times[i - 1] for i in range(1, len(token_times))]

    print("\n=== AirLLM Qwen3.8 benchmark ===")
    print(f"weights:                 {'MLX affine 4-bit' if getattr(model, 'mlx_quantized', False) else 'FP16'}")
    print(f"prompt tokens:           {prompt_tokens}")
    print(f"generated tokens:        {len(token_ids)}")
    print(f"first-token latency:     {first_token_s:.3f} s")
    print(f"prefill prompt rate*:    {prompt_tokens / first_token_s:.3f} prompt tok/s")

    if decode_intervals:
        decode_total = sum(decode_intervals)
        avg_decode = decode_total / len(decode_intervals)
        print(f"decode tokens measured:  {len(decode_intervals)}")
        print(f"decode total:            {decode_total:.3f} s")
        print(f"average decode/token:    {avg_decode:.3f} s")
        print(f"decode throughput:       {1.0 / avg_decode:.3f} tok/s")
        if logical_bytes:
            print(f"logical bytes/token:     {_format_bytes(logical_bytes)}")
            print(f"logical stream rate:     {_format_bytes(logical_bytes / avg_decode)}/s")
    else:
        print("decode metrics:          need at least 2 generated tokens")

    print(f"total generation time:   {total_s:.3f} s")
    print(f"logical bytes/pass:      {_format_bytes(logical_bytes)}")

    active_mb = _mlx_memory_mb("get_active_memory")
    cache_mb = _mlx_memory_mb("get_cache_memory")
    peak_mb = _mlx_memory_mb("get_peak_memory")
    if active_mb is not None:
        print(f"MLX active memory:       {active_mb:.2f} MiB")
    if cache_mb is not None:
        print(f"MLX cache memory:        {cache_mb:.2f} MiB")
    if peak_mb is not None:
        print(f"MLX peak memory:         {peak_mb:.2f} MiB")

    print("* First-token time includes embedding, the full streamed prefill pass, norm, lm_head, and sampling.")
    print("  'Logical bytes' are shard file sizes traversed per pass; macOS filesystem caching can reduce physical SSD reads.")


if __name__ == "__main__":
    main()
