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
    )
    inputs = model.tokenizer(
        [prompt],
        return_tensors="np",
        return_attention_mask=False,
        truncation=True,
        max_length=args.max_seq_len,
        padding=False,
    )

    output = model.generate(
        mx.array(inputs["input_ids"]),
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
    )
    print(output)


if __name__ == "__main__":
    main()
