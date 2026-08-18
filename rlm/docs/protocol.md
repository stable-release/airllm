# v0 protocol

## Run lifecycle

`POST /v1/runs` accepts a prompt, optional remote `ContextRef` values, explicit
capabilities, and limits. It returns the latest accepted snapshot; a very fast
run may already have advanced beyond `queued`. Use:

- `GET /v1/runs/{run_id}` for the latest snapshot;
- `GET /v1/runs/{run_id}/events` for a full SSE replay, then reconnect with
  `?after=<last-seen-sequence>`; the cursor is exclusive;
- `POST /v1/runs/{run_id}/cancel` for idempotent cancellation.

If `RLMD_API_TOKEN` is configured, every `/v1/*` request must contain
`Authorization: Bearer <token>`. Health endpoints do not expose run data.

## Capability model

Capabilities default to false and are intersected with server configuration.

| Capability | Effect |
| --- | --- |
| `remote_context` | Allows bounded reads through the configured worker. |
| `remote_exec` | Allows a Python task through the configured worker. Never local. |
| `internet_search` | Reserved and rejected by v0. |

The model emits typed actions. It cannot supply a URL, credential, executable,
model name, or network destination. `rlmd` validates actions and constructs any
worker request from trusted configuration.

If a local generation reaches its client timeout, `rlmd` quarantines the
generation queue rather than risking overlapping accelerator work on a
memory-constrained host. Restart both the inference service and `rlmd` after
confirming the old generation has stopped.

## Remote worker

The optional worker endpoint is versioned and capability-oriented. Requests
carry a run ID, request ID, deadline, bounded code/read request, and content-addressed
context handles. Responses contain a bounded textual observation plus artifact
references. Original media remains in remote object storage.

The worker is a separate trust boundary. Production workers should place every
Python session in a rootless container or stronger sandbox with a read-only
root, no ambient credentials, no general network, and strict CPU, memory, PID,
time, and output limits.
