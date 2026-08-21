# ADR 0009: Use Deterministic TraceIR With Audited Partial Recovery

- Status: Accepted
- Date: 2026-07-19
- Beads: `env_mock_agent-ujc.1.3`
- Spec: `specs/002-eval-dataset-factory/spec.md`

## Context

All factory stages need facts from the same trace, but they need different projections. Passing complete
raw traces to each stage would duplicate parsing, exhaust model context, expose quarantined content, and
make decisions difficult to reproduce.

The current `raw_traj` corpus contains 91 JSONL traces plus a manifest. Each trace is an outer JSON
record whose `request`, `response`, and `extra` values are nested JSON strings. A 2026-07-19 full scan
with the target Python standard `json` decoder found that all 91 outer records parse strictly, while
72/91 nested `request` values and 1/91 nested `response` values fail strict decoding; all nested `extra`
values parse. Individual trace files are approximately 0.86 MB to 4.18 MB. These are corpus
observations, not protocol guarantees. More permissive parsers do not redefine strict acceptance.

The system therefore needs one deterministic, auditable intermediate representation that preserves raw
location, admits useful partial traces conservatively, and can be queried without unrestricted raw-text
model access.

## Decision

Introduce a versioned TraceAdapter boundary and an immutable TraceIR fact model.

```python
class TraceAdapter(Protocol):
    name: str
    version: str

    def probe(self, source: TraceSource) -> TraceProbeResult: ...
    def parse(self, source: TraceSource, sink: TraceEventSink) -> TraceEnvelope: ...
```

Adapters:

- convert one source format into TraceIR facts;
- never call an LLM;
- never label, reconstruct tasks, infer attachments, or make release decisions;
- never modify the source;
- report strict, repaired, partial, or unparseable outcomes explicitly.

The first adapter is `raw_traj_v1`.

## Authority And Immutability

The authority order is:

```text
immutable raw blob
  > append-only TraceIR JSONL
  > SQLite query projection and FTS index
  > Parquet analysis export
```

SQLite and Parquet are rebuildable projections. They never become the source of trace facts.

An adapter version, TraceIR schema version, repair-policy version, segmentation-policy version, and raw
SHA-256 identify one deterministic parse. Changing any fact-shaping value produces a new TraceIR
version; it does not rewrite prior facts.

## Stable Identity

Source identity and parse-version identity are separate:

- `source_trace_id`: immutable raw SHA-256 and source namespace; stable across re-parses;
- `trace_ir_version_id`: source trace ID, adapter name/version, TraceIR schema version, repair-policy
  version, segmentation-policy version, and any other fact-shaping policy version.

All derived IDs are rooted in `trace_ir_version_id`:

- `event_id`: TraceIR version ID, event sequence, source span, and normalized event type;
- `call_id`: source call ID when valid, otherwise a deterministic derived ID marked synthetic;
- `observation_id`: TraceIR version ID, normalized logical path, operation, sequence, and source span;
- `segment_id`: TraceIR version ID, boundary method, sequence range, and ordered member IDs;
- `span_id`: TraceIR version ID, raw SHA-256, outer record index, field, and raw byte range.

IDs MUST NOT depend on database row IDs, wall-clock time, worker identity, model output, or parse order
outside the source sequence.

## Core TraceIR Objects

### TraceEnvelope

Records trace identity, source URI, raw hash, adapter and policy versions, parse quality, capability
matrix, repair references, diagnostics, session metadata, privacy class, and creation metadata.

### TraceEvent

Represents ordered facts with source spans. Initial event types are:

```text
user_text
assistant_text
system_context
tool_call
tool_result
runtime_error
continuation_marker
attachment_reference
```

TraceEvent contains no semantic label, reconstructed task requirement, or release judgment.

### ToolCallRecord

Normalizes tool name, tool family, arguments reference, call/result membership, status, sequence, and
error signature. Tool families are versioned and include file read/write/edit/list, search, fetch,
shell, code execution, user interaction, subagent, task management, and unknown.

### FileObservation

Records logical and raw path, operation, sequence, observed range, completeness, truncation, content
reference and hash, file-version identity, and source events. A long result is not presumed complete.

### InteractionSegment

R1 creates deterministic base segments from user turns, continuation markers, closed tool-call/result
windows, and file-operation windows. LLMs cannot change base membership or IDs. R4 may create separate
TaskEpisode annotations that reference one or more base segments.

## Source Coordinates

Every derived fact must resolve to one or more SourceSpans.

SourceSpan records:

```text
source_uri
outer_record_index
field
raw_byte_start
raw_byte_end
decoded_char_start
decoded_char_end
repair_map_ref
approximate
raw_sha256
```

Raw byte offsets always address the immutable outer file. Decoded character offsets address the nested
decoded value. Byte and character coordinates are not interchangeable.

## RepairMap

Any transformation between raw and decoded text records ordered mapping segments:

```text
raw_byte_start
raw_byte_end
decoded_char_start
decoded_char_end
transform
rule_id
exact
```

If a transform cannot preserve one-to-one coordinates, affected spans are marked `approximate=true`.
An approximate span cannot alone support a high-risk safety, provenance override, or release decision.

## Three-Level Parse Policy

### Level A: Strict

Parse the outer record and nested fields with standards-compliant JSON decoding and full schema
validation. Success yields `parse_quality=strict`.

### Level B: Audited Local Repair

Apply only allowlisted, local, deterministic rules for known encoding defects. Each rule:

- runs at most once per target string;
- records original and replacement ranges;
- cannot silently change message role, tool name, call ID, path, or content semantics;
- must be followed by strict decoding and schema validation.

Success yields `parse_quality=repaired` and immutable ParseRepair/RepairMap records. A repair touching a
semantically critical field forces downstream review even if parsing succeeds.

### Level C: Streaming Recovery

When the complete nested object remains invalid, a state machine may recover complete message and
content-block objects from recognized structural boundaries. It:

- never uses regular expressions as a general nested-JSON parser;
- records recovered and unrecoverable raw ranges;
- records orphan and ambiguous call/result relationships;
- emits only facts whose boundaries can be justified;
- never invents missing content.

Success yields `parse_quality=partial`.

If no useful facts can be recovered, the result is `unparseable`; the failure remains an auditable
TraceEnvelope rather than disappearing from the batch.

## Capability Matrix

Admission is capability-based, not a single confidence score. Every envelope reports:

```text
conversation_completeness
tool_pairing_completeness
file_observation_completeness
final_response_completeness
```

Each capability is `complete`, `partial`, or `unavailable`, with supporting diagnostics and spans.
Every downstream stage declares `required_trace_capabilities`.

| Parse state | Structured label | Semantic label | Task reconstruction | Attachment reconstruction | Automatic release |
|---|---|---|---|---|---|
| strict | allowed | allowed | allowed | allowed | requires all other gates |
| repaired | allowed within unaffected capabilities | allowed within unaffected capabilities | allowed with repair policy | allowed with repair policy | requires repair policy approval |
| partial, required evidence complete | local decision only | bounded spans only | needs review | evidence-complete artifacts only | forbidden |
| partial, critical evidence missing | observed positive facts and capability-missing diagnostics only | forbidden | forbidden | forbidden | forbidden |
| unparseable | parse-failure record only | forbidden | forbidden | forbidden | forbidden |

Critical evidence includes target-task user goal and order, every event used by a label or attachment,
the first relevant write/edit boundary, final response material needed for leakage analysis, and source
spans for all such facts.

Absence is not evidence in an incomplete range. Any negative predicate based on an event, tool call,
error, file operation, or content not occurring requires the corresponding capability and sequence
range to be `complete`. Partial traces may support explicitly observed positive facts; they may not
support inferred non-occurrence.

## File Completeness

A file observation is complete only when one condition holds:

1. the tool explicitly returns the whole file and proves EOF;
2. ordered observed ranges cover the file without gaps and the final range proves EOF;
3. an approved input workspace exposes the immutable original and its hash is verified.

Length alone never proves completeness. Conflicting observations create separate versions or an
explicit ambiguity; they are not merged silently.

## Query Boundary

TraceQueryService is the only downstream trace-content access path. It provides manifest, segment,
event search, event, span, tool-call, error, file-timeline, file-version, and EvidenceBundle queries.

Authorization comes from a trusted StageRun principal or a dedicated audit principal issued by the
control plane. The service resolves the principal to allowed stages, purposes, privacy profiles, trace
scopes, taint classes, and maximum budgets. Request fields can only narrow those grants; they cannot
claim or expand identity or permission.

Every stage request includes:

```text
purpose
allow_tainted=false
max_characters
```

The service:

- derives `consumer_stage`, `consumer_agent`, and maximum `privacy_profile` from the authenticated
  principal rather than trusting request text;
- rejects unauthorized consumer-purpose combinations and trace scopes;
- excludes tainted and quarantined content by default;
- enforces event and character limits and pagination;
- logs consumer, purpose, returned span IDs, and size;
- treats returned text as untrusted data;
- applies the same least-disclosure policy to internal models;
- never returns unrestricted raw trace text to a stage agent.

Stage runtimes have no filesystem or database credentials for raw blobs, canonical JSONL, SQLite/FTS,
CAS, or Parquet stores. Those stores are accessible only to trusted ingestion/index services and
TraceQueryService. Tool guards and isolated workspaces prevent path-based bypass.

Privileged audit access uses a separate principal, credential, endpoint policy, and audit event. It is
not enabled by setting `allow_tainted=true` on a normal stage request.

## Storage And Indexing

The first release uses:

- read-only files or content-addressed storage for raw blobs;
- append-only JSONL for canonical TraceIR facts;
- content-addressed blobs for large derived text;
- SQLite for metadata, relationships, SourceSpan projection, and state;
- SQLite FTS5 for lexical search;
- Parquet for non-authoritative analysis export.

Elasticsearch and a vector database are not first-release dependencies. Optional embedding indexes may
be introduced only as rebuildable projections under a later decision.

## Alternatives Considered

### Give Each Agent The Complete Raw Trace

Rejected because it duplicates parsing, expands context, weakens access control, and makes evidence
references unstable.

### LLM-First Trace Summarization

Rejected as the fact layer. Summaries may annotate TraceIR later, but cannot replace source spans,
tool-call pairing, file versions, or repair audit.

### Strict Parsing Only

Rejected because useful traces with local serialization defects or truncated tails would be discarded.

### Regex-Based Nested JSON Recovery

Rejected because arbitrary nesting, strings, and escapes make general regex recovery unsound.

### One Global Parse Confidence Score

Rejected because stages require different evidence. Capability-level admission prevents a complete
conversation from masking missing file or final-response evidence.

## Consequences

Positive:

- each trace is parsed once and reused by every stage;
- deterministic queries replace repeated raw-text scans;
- partial traces remain useful without being presented as complete;
- every fact and repair is auditable to raw bytes;
- downstream token and privacy exposure are bounded.

Negative:

- byte/character offset mapping and streaming recovery are complex;
- adapter and repair-policy version changes require re-indexing;
- canonical JSONL plus SQLite projections require consistency checks;
- conservative partial admission will reject some potentially useful tasks.

## Validation

R1 must provide:

- golden tests for strict, repaired, partial, and unparseable cases;
- property tests that repairs do not change valid inputs;
- deterministic ID and repeat-parse tests;
- raw-byte/decoded-character RepairMap tests;
- call/result pairing, orphan, duplicate-ID, and truncation tests;
- file completeness and version-timeline tests;
- authorization and query-budget contract tests;
- principal forgery and direct-storage bypass tests;
- negative-predicate rejection when the relevant capability is incomplete;
- source identity versus TraceIR-version identity tests across policy upgrades;
- JSONL-to-SQLite rebuild and query-equivalence tests;
- interrupted parse/index resume tests.
