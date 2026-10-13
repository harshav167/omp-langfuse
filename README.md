# Langfuse Observability Plugin for OMP

This extension sends oh-my-pi (`omp`) sessions to
[Langfuse](https://langfuse.com). It records the user prompts, the agent turns,
the model generations with their tokens and cost, and the tool calls.

The implementation uses the Langfuse tracing SDK and one isolated OpenTelemetry
provider. Local raw capture works independently of Langfuse credentials.

> [!WARNING]
> This is an experimental release, so future versions can bring breaking changes.

## What can this integration trace?

The plugin listens to OMP's lifecycle events and sends each user prompt to
Langfuse as its own trace:

- **Agent turns**: one trace per user prompt, grouped by session ID. Turn
  numbering survives a restart.
- **Runs without a prompt**: runs OMP starts on its own, such as live
  delegation, background job results and MCP notices, get their own trace with
  `turn_trigger: continuation` in the turn metadata.
- **Steering and follow-ups**: a prompt OMP delivers into a running turn, by
  queueing it or by aborting and restarting the loop, stays inside that turn's
  trace as a `steer` event; `steer_count` and `interrupted` on the turn record
  what happened. The turn ends when its last loop ends.
- **Side requests**: requests OMP sends while a reply is still pending (prewalk,
  memory recall) are paired by the time OMP stamped on each assistant message,
  so the reply lands on its own generation; an unanswered request closes as
  `unmatched_request` when the turn ends.
- **Session title**: OMP's session title is the trace tag `title:<title>` (first
  120 characters) and the turn metadata `session_title`.
- **Agent-loop generations**: provider-visible inputs, assistant outputs,
  cost, time to first token, and cache-read/cache-write/reasoning token usage.
  Time to first token is also recorded for generations that stream only
  reasoning or tool calls and no text.
- **Prompt-cache accounting**: each generation states how many prompt tokens
  could have been read from cache, how many were, and why the rest were not.
  See [Prompt-cache accounting](#prompt-cache-accounting).
- **System prompt**: the system prompt the provider actually receives, after
  other extensions' overrides. It is in the turn metadata as `system_prompt` and
  is the first `system` message of each generation's input.
- **Conversation history**: each generation has all messages of that call.
- **Tool definitions**: the tools offered to the model are attached to each
  generation's input.
- **Model parameters**: the request parameters on the wire, such as
  `max_tokens`, sampling parameters, thinking level and budget, `tool_choice`,
  `service_tier`, and prompt cache retention.
- **Reasoning**: each thinking block is on the step that made it and on the
  generation output.
- **Tool calls**: each tool OMP invokes, with input, output, and an `ERROR` level
  when the call fails.
  render inside the trace.
- **Retries and interventions**: automatic retries, model fallbacks, tool
  approval requests and decisions, and triggered TTSR rules are events on the
  turn.
- **Model and account switches**: `switch-model` and `switch-account` events on
  the turn when the model or the provider credential changes between requests.
- **Session-title generation**: OMP's title fork, a side request that shares
  the session's provider hooks, is its own `generate-title` generation on the
  turn; it never displaces the reply in flight.
- **Subagents**: in-process agents and child `omp` processes nest under their
  parent trace when the corresponding parent context survives.
- **Compaction accounting**: real summarizer calls are observed through the
  registered OMP session's public SDK telemetry callbacks. Each call has its
  own usage, reported cost, duration, and captured parent. The committed
  summary is a separate result event, not another billed generation.

Message content is sent in full. The plugin does not truncate prompts, outputs,
or tool results.

Native compaction accounting respects `telemetry.otlpExportEnabled`,
`OTEL_SDK_DISABLED`, and `OTEL_TRACES_EXPORTER=none`, as well as the Langfuse
kill switch and credentials. It does not register a global exporter or change
the SDK's content-capture policy. When a host does not expose a registered
session or declines native telemetry, summary observations explicitly mark
usage as missing instead of inventing accounting.

## Prerequisites

Use the compatible `@oh-my-pi/pi-coding-agent` host declared in `package.json`.
Development commands and historical backfill run with Bun.

## Install

```bash
omp install /absolute/path/to/pi-observability-plugin
```

To try it for a single run, without installing:

```bash
omp -e /absolute/path/to/pi-observability-plugin/src/index.ts
```

## Add your Langfuse credentials

Create `langfuse.json` inside the active agent directory, for example
`~/.omp/profiles/debug/agent/langfuse.json` for the debug profile:

```json
{
  "publicKey": "pk-lf-...",
  "secretKey": "sk-lf-...",
  "baseUrl": "https://cloud.langfuse.com",
  "userId": "your-user-id",
  "environment": "development"
}
```

Only `publicKey` and `secretKey` are required. If `baseUrl` is omitted, the
plugin uses `https://cloud.langfuse.com` (EU region). `userId`, `environment`
and `release` are optional labels that let you segment traces by teammate, stage
or version. Keep the file private, because it holds a secret key.

Alternatively, set your credentials with environment variables:

```bash
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."
export LANGFUSE_BASE_URL="https://cloud.langfuse.com"
export LANGFUSE_TRACING_ENVIRONMENT="development"
export LANGFUSE_USER_ID="your-user-id"
```

Environment variables take precedence over the config file, so you can override
a single value without editing the file.

## Environment variables

| Variable                       | Description                                                                                                                                       | Required            |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- |
| `LANGFUSE_PUBLIC_KEY`          | Your Langfuse public key (`pk-lf-...`)                                                                                                            | Yes                 |
| `LANGFUSE_SECRET_KEY`          | Your Langfuse secret key (`sk-lf-...`)                                                                                                            | Yes                 |
| `LANGFUSE_BASE_URL`            | Langfuse host. EU: `https://cloud.langfuse.com`, US: `https://us.cloud.langfuse.com`, Japan: `https://jp.cloud.langfuse.com`, HIPAA: `https://hipaa.cloud.langfuse.com` | No (defaults to EU) |
| `LANGFUSE_TRACING_ENVIRONMENT` | Environment label for the traces (e.g. `production`)                                                                                              | No                  |
| `LANGFUSE_USER_ID`             | User ID attached to all traces                                                                                                                    | No                  |
| `OTEL_SERVICE_NAME`            | OpenTelemetry service name; the plugin fallback is `omp`                                                                                        | No                  |
| `OTEL_RESOURCE_ATTRIBUTES`     | Extra OpenTelemetry resource attributes as a comma-separated `key=value` list (e.g. `service.namespace=engineering,team.name=platform`). Langfuse shows them under `resourceAttributes` in the trace and observation metadata | No                  |
| `LANGFUSE_PI_TRACEPARENT`      | W3C `traceparent` of an existing span. OMP turns nest under it; see [Attach to an existing trace](#attach-to-an-existing-trace)                    | No                  |

## Attach to an existing trace

To put OMP turns inside a trace that another program owns, such as a CI job or
an evaluation harness, set `LANGFUSE_PI_TRACEPARENT` to that span's W3C
`traceparent` value:

```bash
LANGFUSE_PI_TRACEPARENT="00-<32-hex trace id>-<16-hex span id>-01" omp
```

Each turn becomes a child of that span, and child `omp` processes stay in the
same trace. The plugin does not set the trace name, session, user, or tags, so
the external owner keeps them. Turns carry `attached_to_external_parent: true`
in their metadata. A malformed value is ignored with a warning on stderr, and
tracing continues with a normal trace.

## Enable and disable tracing

| Scope                   | How                                                         |
| ----------------------- | ----------------------------------------------------------- |
| One run                 | `LANGFUSE_TRACING_ENABLED=false omp`                        |
| The current shell       | `export LANGFUSE_TRACING_ENABLED=false` (undo with `unset`) |
| Profile or project      | Disable the extension in that scope's OMP configuration   |
| Remove the plugin       | `omp remove npm:@langfuse/pi-observability-plugin`          |

The kill switch has priority over environment keys and the config file. When
tracing is off, the status line shows `langfuse: off (no keys)` and nothing is
sent. This message reads the same whether the kill switch is set or the keys are
genuinely missing. To remove stored keys, delete the active agent directory's
`langfuse.json`.

`omp remove` needs the same source you installed from. For a local checkout, see
[Development](#development).

## Local raw capture

Independently of Langfuse, the plugin writes every provider request, response,
stream frame and assistant message to disk. Capture is on even when Langfuse is
off or has no keys.

- Location: `<profile root>/logs/raw-requests/<provider>/<UTC day>/`, next to
  omp's own logs (for example `~/.omp/profiles/debug/logs/raw-requests/`).
- Contents: full, unredacted payloads, including prompts, tool output and
  base64 images. Only secret-looking response headers are redacted.
- Retention: files older than 72 hours are deleted by file name, once at
  session start and every 5 minutes after.
- Analysis: request records note a model change (`analysis.modelSwitch`).
  Message records carry the credential ID (`analysis.credentialId`) and the
  cache assessment (`analysis.cache`). Lifecycle records mark `model_switch`,
  `account_switch` and `cache_loss`. Request, response, message, lifecycle and
  snapshot records carry the session title (`sessionTitle`).
- Unanswered requests: when a newer request starts with the full history of a
  request that got no response, the old request gets a gap record with reason
  `attempt-superseded-without-response`.

| Variable                 | Effect                                    |
| ------------------------ | ----------------------------------------- |
| `OMP_RAW_REQUESTS_DIR`   | Write captures to this directory instead  |
| `OMP_RAW_REQUESTS_SSE=0` | Skip stream frames; keep the other records |

## Prompt-cache accounting

Each successful reply is compared with the previous successful reply of the same
model in the same session. The previous prompt is what this request could have
read from cache, and whatever it did not read is a loss. There is no
session-wide cache share, because one full re-read of a large prefix disappears
inside hundreds of cached turns.

Each generation carries this metadata. The raw-capture message record carries
the full assessment under `analysis.cache`.

| Metadata                         | Meaning                                                                  |
| -------------------------------- | ------------------------------------------------------------------------ |
| `cache_reusable_tokens`          | Prompt tokens of the previous successful reply of this model             |
| `cache_missed_reusable_tokens`   | Reusable tokens this request did not read from cache (0 up to 1,024)     |
| `cache_new_tokens`               | Prompt tokens beyond the reusable prefix                                 |
| `cache_class`                    | The cause, from the table below                                          |
| `cache_cold_start`               | No earlier reply of this model in the session                            |
| `cache_idle_ms`                  | Time from the previous reply of this model to this request               |
| `cache_append_only`              | The request kept the previous request's history, tools and system prompt |
| `cache_after_disruption`         | An attempt failed, was aborted or was re-sent since the previous reply   |
| `session_cache_losses`           | Losses so far in this session                                            |
| `session_missed_reusable_tokens` | Reusable tokens missed so far in this session                            |

A loss of more than 1,024 reusable tokens sets the generation's level to
`WARNING` and writes a `cache_loss` lifecycle record. The first matching cause
wins:

| `cache_class`      | Cause                                                                         |
| ------------------ | ----------------------------------------------------------------------------- |
| `model-switch`     | The model changed since the previous reply; each model has its own cache      |
| `account-switch`   | The reply came from a different credential of the same provider               |
| `compaction`       | The session compacted since the previous reply                                |
| `prefix-rewrite`   | Earlier history, the tools or the system prompt changed                       |
| `ttl-sized-idle`   | At least 5 minutes, the shortest common cache lifetime, passed in between     |
| `after-interrupt`  | An attempt failed, was aborted or was re-sent in between                      |
| `stalled-tail`     | The cache returned what it held before but did not extend it                  |
| `unexplained-loss` | None of the above: the provider did not serve the cached prefix               |

Replies without a loss are `append-only-progression`, or `prefix-rewrite-no-loss`
when earlier history changed but the provider still served the prefix from
cache. The first reply of a model in the session is `first-observed`, or
`model-switch` when it follows a reply from another model. Aborted and failed
replies are not assessed, because their usage is partial; they mark the next
reply as `cache_after_disruption`.

## What the plugin cannot capture

OMP's extension hooks limit what any plugin can record:

- **Request headers.** `before_provider_request` carries only the request body.
  OMP adds authentication, session and provider headers in its provider code,
  some of them after the hook runs.
- **Failed responses.** OMP throws on a non-2xx response before it calls
  `after_provider_response`, so a 429 or 5xx response, its headers (such as
  `retry-after`) and its body are not recorded. The attempt appears as a request
  without a response. When a later request starts with its full history, raw
  capture writes a gap record with reason `attempt-superseded-without-response`.
- **Transport retries.** OMP's HTTP layer retries 408, 429, 5xx and network
  failures inside one provider call (up to six attempts on OpenAI-wire
  providers) without calling the request hook again. Only their combined
  latency is visible.
- **Account rotation.** When OMP retries on another credential of the same
  provider, no event names the new account. The plugin infers the switch from
  the `credentialId` of the next successful reply. Session-level retries do
  appear, as `retry-request` events.
- **Session names.** Langfuse has no session-name field, so the title is a tag
  and metadata. Traces sent before OMP sets or changes the title keep the old
  value.
- **What the provider processed.** Token counts come from the provider's usage
  report. Content a provider drops without an error, such as a developer-role
  message an OpenAI-compatible endpoint ignores, changes the captured request
  but not the counts.

## Historical raw-capture backfill

Run from this checkout, with the same profile and credentials as live tracing:

```bash
OMP_PROFILE=debug bun run backfill --dry-run
OMP_PROFILE=debug bun run backfill --limit 3
OMP_PROFILE=debug bun run backfill
```

The command replays retained schema-2 records into the **same tracing consumer
and Langfuse SDK** used by the extension. It does not create a second uploader,
start an agent, rewrite capture contents, or change the capture retention policy.
It understands OpenAI
Chat Completions, Responses/Codex, Anthropic Messages, Google GenerateContent/
Vertex, and Cloud Code Assist/Antigravity payloads.

Options:

| Option | Behavior |
| --- | --- |
| `--root <path>` | Read a different archive root; overrides `OMP_RAW_REQUESTS_DIR`, then the active profile's `logs/raw-requests` default. |
| `--session <id>` | Select a session; repeatable. |
| `--from <ISO time>` / `--until <ISO time>` | Select UTC-timestamped records. The default upper cutoff is command startup. |
| `--limit <number>` | Stop after at most this many provider request records and the available following output. |
| `--dry-run` | Analyze without constructing the exporter or requiring credentials. |
| `--keep-snapshot` | Retain the private input snapshot after SDK export for backend verification. Remove it when verification is complete. |
| `--only-observations <file>` | Export only exact SDK span IDs listed in a JSON string array. Full source context is replayed to preserve IDs and parent relationships; normal export gates still apply. |

Export honors `LANGFUSE_TRACING_ENABLED`, credential precedence, SDK media
settings, and Langfuse-key redaction. Disabled export and missing credentials
exit nonzero; dry-run remains available. The offline producer is drained in
small windows and remaining actors are closed/drained incrementally, avoiding
an archive-sized end-of-run burst into the SDK exporter.
For large histories, use its existing batching and timeout controls, without
changing the live profile:

```bash
OMP_PROFILE=debug LANGFUSE_FLUSH_AT=10 LANGFUSE_TIMEOUT=60 \
  OTEL_BSP_EXPORT_TIMEOUT=90000 bun run backfill
```

`LANGFUSE_TIMEOUT` is seconds; `OTEL_BSP_EXPORT_TIMEOUT` is milliseconds.
SDK drain failures terminate the command rather than being swallowed by the
live tracing exit path. The JSON report counts replayed source records, not
backend-verified observations. `observationIds`, `rootIds`, and each actual
`rootObservationId` come from the SDK, including inherited subagent trace IDs.
Dry-run IDs are anticipated only. Use exact observation IDs to verify backend
receipt; a successful SDK drain is not an exhaustive backend census.

If backend verification identifies missing IDs, recover only those IDs from the
retained snapshot using the same SDK path:

```bash
OMP_PROFILE=debug LANGFUSE_FLUSH_AT=10 LANGFUSE_TIMEOUT=60 \
  OTEL_BSP_EXPORT_TIMEOUT=90000 bun run backfill \
  --root /path/to/retained-snapshot --only-observations /path/to/missing-ids.json
```

The ID file must be a non-empty array of exact lowercase 16-hex observation IDs.
The selector narrows `shouldExportSpan`; it does not bypass the tracing kill
switch, credential requirements, redaction, media policy, or SDK batch limits.

### Fidelity and limits

- Historical observation IDs are deterministic for the same retained inputs
  and selection. Retrying that selection updates the same observations.
  Changing the selection or losing files to retention can change prompt roots.
- Transport mirrors and SSE frames are not imported as duplicate generations.
- Request/output pairs require an unambiguous archived capture ID. An orphan
  assistant message keeps its actual output and usage in an independent trace,
  with no invented input or guessed request parent.
- Delta-only requests retain their known context and flag missing anchors.
  Tool results are associated by archived call IDs; names are recovered only
  when the matching calls agree on one actual name. No execution span is fabricated for an orphan
  result.
- Prompt boundaries, tool execution timing, and missing prompt-level end events
  are explicitly reconstructed. Orphan generation intervals use archived
  duration when available. Metadata distinguishes `archived_facts`,
  `inferred_facts`, and `missing_data`.
- Generations without an archived assistant message have `usage_missing=true`.
  Backend token estimates, if present, are not archived billing evidence.
- Unknown provider fields, unsupported records, missing files, and ambiguous
  correlations are counted rather than represented as complete data.
- Before exporting, the command preserves the selected regular JSON records
  in a private temporary hard-link snapshot. This protects the bytes from
  the active logger's pruning without copying large payloads or rewriting the
  originals. Link counts and inode change-times can change. Snapshot creation
  requires `TMPDIR` on the capture filesystem and declines rather than silently
  copying across filesystems.
- A successful export removes its snapshot unless `--keep-snapshot` is set.
  A failed export prints the preserved snapshot path for a retry with `--root`.
  These snapshots contain private raw data; remove retained snapshots after
  verification. Already-deleted captures cannot be recovered by snapshotting.

### Other historical sources

Session JSONL files retain the transcript tree (`id`/`parentId`), model/thinking
changes, assistant usage and latency, tool results and details, custom tool-start
records, and some side-call `model_usage` entries. They can fill transcript,
tool-timing, and side-call-accounting gaps. They cannot recreate every final
provider payload, cache directive, provider-side transformation, or missing
in-memory subagent session. They are not imported by this raw-only command.

`history.db` stores unique normalized prompt text with the **latest** submission's
session, directory, timestamp, and a reuse counter, plus session titles/recaps.
It is not a complete chronological transcript or a source of assistant usage.

In the inspected OMP schema, `agent.db` stores last-used model keys, decaying
performance averages, settings/cache, and account-quota history. Its optional
client-usage table has no session IDs. These are not substitutes for per-turn
usage or tool output. Authentication data must not be sent to Langfuse.

## Development

```bash
bun install
bun run typecheck
bun test --timeout 60000 test/*.test.ts
```

The package ships raw TypeScript and OMP reads `src/index.ts` directly, so there
is no build step. Therefore `files` in `package.json` holds only `src`,
`README.md` and `LICENSE`.

To run a local checkout against OMP:

```bash
omp install /absolute/path/to/pi-observability-plugin
```

Remove it again with `omp remove /absolute/path/to/pi-observability-plugin`.

## Release

1. Update the version in `package.json` and `EXTENSION_VERSION` in
   `src/index.ts`, then merge the version bump into `main`.
2. Tag that commit with the matching `v`-prefixed version and push the tag:

   ```bash
   git tag v0.2.0
   git push origin v0.2.0
   ```

   The tag must exactly match the version in `package.json` or the release
   workflow will fail.
3. The [release workflow](./.github/workflows/release.yml) installs
   dependencies, typechecks the package, stages it on npm with provenance,
   and creates a draft GitHub release with generated release notes.
   Prerelease versions are staged with the `next` npm tag; stable versions
   use `latest`.
4. Review the staged package and approve it to publish, or reject it if
   anything is wrong. This can be done through npm's staged packages UI or
   with the npm CLI:

   ```bash
   npm stage list @langfuse/pi-observability-plugin
   npm stage view <stage-id>
   npm stage approve <stage-id>
   # Or: npm stage reject <stage-id>
   ```

   Approval and rejection require npm two-factor authentication.
5. Publish the draft GitHub release after the staged npm package is approved.

## License

[MIT](./LICENSE)
