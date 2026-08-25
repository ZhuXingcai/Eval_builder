# ADR 0019: Admit Bounded Trace Sources Through Private CAS Authority

- Status: Accepted
- Date: 2026-08-23
- Spec: `specs/003-evaluation-agent-harness/spec.md`
- Trellis: `.trellis/tasks/08-17-conversation-first-agent-shell-v1/`
- Beads: `env_mock_agent-ujc.19`

## Context

The Agent Shell needs browser-provided Trace inputs before Stage 5.3 can start
Factory Graph work. Browser paths are not trustworthy authority, and raw Trace
bytes cannot be copied into session events, public API values, logs, or
Artifact Envelopes. Admission must also survive retries and process restarts
without making a staging file or an orphaned CAS object authoritative.

V1 supports only the built-in `generic-agent-trace` profile:

```text
one manifest.csv
+ the exact referenced raw_traj_v1 JSONL files
```

General document, image, video, embodied, and dynamic Pack source admission
remains a V2 concern.

## Decision

Add a dedicated `HarnessSourceAdmissionStore` and
`HarnessSourceAdmissionService`. The multipart HTTP command contains exactly:

```text
command   strict JSON metadata and idempotency claims
manifest  one file named manifest.csv
traces    one or more repeated .jsonl file parts
```

The server streams uploads into a private staging directory, verifies declared
size and SHA-256 while writing, validates the exact manifest header and member
inventory, probes each source with `RawTrajV1Adapter`, and checks `sid`,
`p_date`, and `business` against every outer JSONL record.

Validated bytes are hard-linked atomically into private content-addressed
storage:

```text
cas/manifest/sha256/<prefix>/<sha256>.csv
cas/trace/sha256/<prefix>/<sha256>.jsonl
```

Only after CAS publication does one `BEGIN IMMEDIATE` transaction commit the
immutable admission, member and Artifact Envelope records, plus:

```text
scope + idempotency_key + request_sha256 -> admission_object_id
```

One session has one immutable admission. The same Trace content and source
identity may be reused by different sessions; CAS and source refs are
content-addressed, while admission ownership remains session-scoped.

## Authority And Privacy

`HarnessSourceAdmissionStore` owns uploaded bytes, immutable admission
metadata, member/envelope linkage, and idempotency. It does not own Harness
session state, Trace parsing authority, Factory execution, or Graph state.

The public `AgentShellSourceAdmissionV1` exposes only:

- admission/session refs and versions;
- safe relative names, media types, hashes, sizes, and counts;
- `trace-source/v2` refs;
- `artifact-envelope/v1` refs;
- creation time.

It never exposes physical paths, staging paths, private content refs, raw
manifest/Trace bytes, request/response/extra bodies, or full Artifact
Envelopes.

## Failure And Recovery

- Unsafe, aliased, duplicate, unsupported, missing, extra, malformed,
  metadata-mismatched, hash-mismatched, or over-budget inputs fail before
  database authority exists.
- Multipart parser failures map to a closed
  `SOURCE_ADMISSION_MULTIPART_INVALID` response.
- Exact same-key replay returns the first admission; changed same-key input
  conflicts.
- A second key returns the existing authority only when behavior and principal
  match; otherwise it conflicts.
- CAS-before-database failure may leave an unreachable orphan. Exact retry
  verifies and reuses it.
- Staging is removed on both success and failure. Abandoned staging cleanup is
  explicit.
- Reads verify canonical SQLite JSON, materialized columns, member/envelope
  closure, and physical CAS hashes. They never repair drift.

## Consequences

- Stage 5.3 can bind safe source and Artifact Envelope refs without accepting
  browser paths or reading arbitrary files.
- Identical content is physically deduplicated and can be admitted by multiple
  sessions without conflating session authority.
- Source admission remains additive and does not change
  `TraceSourceRegistry` or `raw_traj_v1`.
- Graph execution is intentionally not started by the upload endpoint.
- `python-multipart` is a direct runtime dependency of the product API.

## Validation

Acceptance requires:

- strict contract and public privacy tests;
- success, exact replay, second-key replay, cross-session content reuse,
  changed-input and principal conflict;
- manifest/header/inventory/metadata/hash/size/name/alias/limit failures;
- staging, CAS, SQLite drift, and all three injected fault boundaries;
- multipart HTTP/OpenAPI closed-error and private-field assertions;
- `TraceSourceRegistry` and Generic Pack compatibility;
- both configured hash seeds;
- at least 80% branch coverage per new deep module;
- full generated, Python, Node, Web, privacy, wheel/import, and continuity
  gates.
