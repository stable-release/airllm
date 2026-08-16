"""Smoke-test Qwen3.8-27B through AirLLM's streamed MLX backend on Apple Silicon."""

import argparse

import mlx.core as mx

from airllm import AutoModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--layer-path", default=None)
    parser.add_argument("--show-memory", action="store_true")
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
    args = parser.parse_args()

    model = AutoModel.from_pretrained(
        args.model,
        max_seq_len=args.max_seq_len,
        layer_shards_saving_path=args.layer_path,
        show_memory_util=args.show_memory,
    )

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

    if not args.debug_tokens:
        output = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
        )
        print(output)
        return

    token_ids = []
    eos_token_id = model.tokenizer.eos_token_id
    eos_ids = set()
    if isinstance(eos_token_id, int):
        eos_ids.add(eos_token_id)
    elif eos_token_id is not None:
        eos_ids.update(eos_token_id)

    for index, token in enumerate(model.model_generate(input_ids, temperature=0.0)):
        token_id = int(token.item())
        token_ids.append(token_id)
        piece = model.tokenizer.decode([token_id], skip_special_tokens=False)
        print(f"[token {index}] id={token_id} piece={piece!r}")
        if token_id in eos_ids or len(token_ids) >= args.max_new_tokens:
            break

    print("decoded:", model.tokenizer.decode(token_ids, skip_special_tokens=True))


if __name__ == "__main__":
    main()
