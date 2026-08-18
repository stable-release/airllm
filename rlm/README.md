# `rlmd`

`rlmd` is a small, offline-first Recursive Language Model control plane for a
local inference service. It is deliberately split from code and media
processing:

```text
client -> rlmd (Rust) -> mlx_lm.server (loopback only)
                    \-> explicitly configured remote worker
```

The model never receives credentials, chooses endpoints, or connects to the
internet. Ordinary inference and model subcalls use only the loopback MLX
server. A context read or Python execution is sent to one preconfigured remote
worker only when the run grants the corresponding capability. Internet search
is reserved in the protocol but rejected by this version.

## Security and memory defaults

- MLX must be on `127.0.0.1`; redirects, proxies, retries, and model overrides
  are disabled.
- Model generations are serialized through one permit. Remote work can remain
  concurrent while the memory-constrained host processes one generation at a
  time.
- The API is loopback-only. Remote clients must use Tailscale Serve, an SSH
  tunnel, or a trusted TLS reverse proxy; v0 refuses plaintext LAN binding.
- The remote worker is absent by default. Remote workers require HTTPS except
  for loopback development.
- Context and artifacts are capability references. SQLite contains bounded
  run metadata and event previews, not original documents or media.
- Runs have hard iteration, admission, token, and output limits. Wall-time and
  cancellation are enforced at safe action boundaries; an already-dispatched
  generation is drained because the local inference API has no reliable
  per-request cancel operation.
- Generated Python is never executed by `rlmd`.

## Start the local model

Activate the Python environment, point `MODEL_PATH` at a complete local
text-only package, and start the inference service with memory-conscious
settings:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m mlx_lm server \
  --model "$MODEL_PATH" \
  --host 127.0.0.1 \
  --port 8080 \
  --decode-concurrency 1 \
  --prompt-concurrency 1 \
  --prefill-step-size 512 \
  --prompt-cache-size 0 \
  --max-tokens 1024 \
  --chat-template-args '{"enable_thinking":false}'
```

Do not use a remote registry identifier here: `MODEL_PATH` must resolve to a
complete local package so model startup itself is offline.

`rlmd` always sends MLX the special `default_model` alias; it is not configurable
through the API or command line. This prevents `mlx_lm.server` from interpreting
a request value as another local path or a Hugging Face repository. For a
defense-in-depth appliance deployment, also deny outbound traffic from the MLX
process at the OS/network layer.

## Run the control plane

```bash
cargo run --release -p rlmd -- \
  --model-url http://127.0.0.1:8080 \
  --database ./rlmd.db
```

The API provides health/readiness checks, run creation/status/cancellation,
and server-sent run events. See [`docs/protocol.md`](docs/protocol.md) for the
request contract and capability behavior.

## v0 scope

The first engine supports final answers, bounded model subcalls, notes, remote
context reads, and remote Python tasks. It intentionally does not include a
browser, generic HTTP tool, MCP client, shell, local file reader, or local code
executor. Those are capabilities to add deliberately, rather than ambient
abilities of the model process.
