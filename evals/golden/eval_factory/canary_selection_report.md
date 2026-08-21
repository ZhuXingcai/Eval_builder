# Eval Dataset Factory Development Canary Selection

- Schema: `eval-factory-canary-manifest/v1`
- Selector: `1.0.0`
- Source traces: 91
- Selected traces: 24
- Raw SID, account, prompt, response, and file content are not emitted.
- Malformed-request behavioral signals are heuristic and require R0-08 annotation.

## Coverage

| Trait | Required | Observed |
|---|---:|---:|
| `flow.completion_claim_signal` | 3 | 24 |
| `flow.continuation_signal` | 3 | 11 |
| `flow.pre_mutation_read` | 3 | 15 |
| `parse.invalid_escape` | 6 | 6 |
| `parse.invalid_response` | 1 | 1 |
| `parse.invalid_unicode_escape` | 3 | 3 |
| `parse.strict_request` | 10 | 15 |
| `size.large` | 3 | 8 |
| `size.small` | 2 | 6 |
| `tool.error_result` | 5 | 23 |
| `tool.fetch` | 2 | 9 |
| `tool.file_edit` | 6 | 24 |
| `tool.file_read` | 6 | 24 |
| `tool.file_write` | 6 | 24 |
| `tool.powershell` | 1 | 4 |
| `tool.search` | 3 | 11 |
| `tool.truncation_signal` | 3 | 14 |

## Selected Items

| Instance | Request parse | Signal quality | Annotation required | Traits |
|---|---|---|---:|---|
| `LH_003` | `invalid_escape` | `heuristic_requires_annotation` | true | `flow.completion_claim_signal`, `flow.continuation_signal`, `parse.invalid_escape`, `parse.request_invalid_escape`, `size.large`, `tool.error_result`, `tool.fetch`, `tool.file_edit`, `tool.file_read`, `tool.file_write`, `tool.powershell`, `tool.search`, `tool.truncation_signal` |
| `LH_006` | `strict` | `strict_events` | false | `flow.completion_claim_signal`, `flow.pre_mutation_read`, `parse.request_strict`, `parse.strict_request`, `size.small`, `tool.file_edit`, `tool.file_read`, `tool.file_write`, `tool.truncation_signal` |
| `LH_011` | `invalid_escape` | `heuristic_requires_annotation` | true | `flow.completion_claim_signal`, `flow.continuation_signal`, `parse.invalid_escape`, `parse.invalid_response`, `parse.request_invalid_escape`, `tool.error_result`, `tool.file_edit`, `tool.file_read`, `tool.file_write`, `tool.truncation_signal` |
| `LH_015` | `strict` | `strict_events` | false | `flow.completion_claim_signal`, `flow.pre_mutation_read`, `parse.request_strict`, `parse.strict_request`, `size.small`, `tool.error_result`, `tool.fetch`, `tool.file_edit`, `tool.file_read`, `tool.file_write`, `tool.search` |
| `LH_019` | `strict` | `strict_events` | false | `flow.completion_claim_signal`, `flow.continuation_signal`, `flow.pre_mutation_read`, `parse.request_strict`, `parse.strict_request`, `size.large`, `tool.error_result`, `tool.file_edit`, `tool.file_read`, `tool.file_write` |
| `LH_030` | `strict` | `strict_events` | false | `flow.completion_claim_signal`, `flow.pre_mutation_read`, `parse.request_strict`, `parse.strict_request`, `size.small`, `tool.error_result`, `tool.file_edit`, `tool.file_read`, `tool.file_write`, `tool.truncation_signal` |
| `LH_032` | `invalid_escape` | `heuristic_requires_annotation` | true | `flow.completion_claim_signal`, `flow.continuation_signal`, `parse.invalid_escape`, `parse.request_invalid_escape`, `size.large`, `tool.error_result`, `tool.fetch`, `tool.file_edit`, `tool.file_read`, `tool.file_write`, `tool.search`, `tool.truncation_signal` |
| `LH_035` | `strict` | `strict_events` | false | `flow.completion_claim_signal`, `flow.pre_mutation_read`, `parse.request_strict`, `parse.strict_request`, `tool.error_result`, `tool.file_edit`, `tool.file_read`, `tool.file_write`, `tool.truncation_signal` |
| `LH_040` | `strict` | `strict_events` | false | `flow.completion_claim_signal`, `flow.pre_mutation_read`, `parse.request_strict`, `parse.strict_request`, `tool.error_result`, `tool.file_edit`, `tool.file_read`, `tool.file_write`, `tool.search`, `tool.truncation_signal` |
| `LH_041` | `strict` | `strict_events` | false | `flow.completion_claim_signal`, `flow.pre_mutation_read`, `parse.request_strict`, `parse.strict_request`, `size.large`, `tool.error_result`, `tool.file_edit`, `tool.file_read`, `tool.file_write` |
| `LH_046` | `strict` | `strict_events` | false | `flow.completion_claim_signal`, `flow.pre_mutation_read`, `parse.request_strict`, `parse.strict_request`, `size.small`, `tool.error_result`, `tool.file_edit`, `tool.file_read`, `tool.file_write` |
| `LH_057` | `strict` | `strict_events` | false | `flow.completion_claim_signal`, `flow.continuation_signal`, `flow.pre_mutation_read`, `parse.request_strict`, `parse.strict_request`, `tool.error_result`, `tool.file_edit`, `tool.file_read`, `tool.file_write`, `tool.truncation_signal` |
| `LH_058` | `strict` | `strict_events` | false | `flow.completion_claim_signal`, `flow.pre_mutation_read`, `parse.request_strict`, `parse.strict_request`, `tool.error_result`, `tool.file_edit`, `tool.file_read`, `tool.file_write`, `tool.powershell`, `tool.truncation_signal` |
| `LH_065` | `strict` | `strict_events` | false | `flow.completion_claim_signal`, `flow.pre_mutation_read`, `parse.request_strict`, `parse.strict_request`, `tool.error_result`, `tool.fetch`, `tool.file_edit`, `tool.file_read`, `tool.file_write`, `tool.search` |
| `LH_066` | `strict` | `strict_events` | false | `flow.completion_claim_signal`, `flow.pre_mutation_read`, `parse.request_strict`, `parse.strict_request`, `tool.error_result`, `tool.file_edit`, `tool.file_read`, `tool.file_write` |
| `LH_067` | `invalid_unicode_escape` | `heuristic_requires_annotation` | true | `flow.completion_claim_signal`, `flow.continuation_signal`, `parse.invalid_unicode_escape`, `parse.request_invalid_unicode_escape`, `size.small`, `tool.error_result`, `tool.fetch`, `tool.file_edit`, `tool.file_read`, `tool.file_write`, `tool.search` |
| `LH_075` | `strict` | `strict_events` | false | `flow.completion_claim_signal`, `flow.continuation_signal`, `flow.pre_mutation_read`, `parse.request_strict`, `parse.strict_request`, `tool.error_result`, `tool.fetch`, `tool.file_edit`, `tool.file_read`, `tool.file_write`, `tool.search` |
| `LH_077` | `strict` | `strict_events` | false | `flow.completion_claim_signal`, `flow.pre_mutation_read`, `parse.request_strict`, `parse.strict_request`, `tool.error_result`, `tool.fetch`, `tool.file_edit`, `tool.file_read`, `tool.file_write`, `tool.search` |
| `LH_079` | `invalid_escape` | `heuristic_requires_annotation` | true | `flow.completion_claim_signal`, `flow.continuation_signal`, `parse.invalid_escape`, `parse.request_invalid_escape`, `size.large`, `tool.error_result`, `tool.file_edit`, `tool.file_read`, `tool.file_write`, `tool.truncation_signal` |
| `LH_093` | `invalid_unicode_escape` | `heuristic_requires_annotation` | true | `flow.completion_claim_signal`, `flow.continuation_signal`, `parse.invalid_unicode_escape`, `parse.request_invalid_unicode_escape`, `size.large`, `tool.error_result`, `tool.file_edit`, `tool.file_read`, `tool.file_write`, `tool.powershell`, `tool.truncation_signal` |
| `LH_098` | `invalid_escape` | `heuristic_requires_annotation` | true | `flow.completion_claim_signal`, `parse.invalid_escape`, `parse.request_invalid_escape`, `size.small`, `tool.error_result`, `tool.fetch`, `tool.file_edit`, `tool.file_read`, `tool.file_write`, `tool.powershell`, `tool.search`, `tool.truncation_signal` |
| `LH_100` | `invalid_escape` | `heuristic_requires_annotation` | true | `flow.completion_claim_signal`, `flow.continuation_signal`, `parse.invalid_escape`, `parse.request_invalid_escape`, `size.large`, `tool.error_result`, `tool.fetch`, `tool.file_edit`, `tool.file_read`, `tool.file_write`, `tool.search`, `tool.truncation_signal` |
| `LH_101` | `invalid_unicode_escape` | `heuristic_requires_annotation` | true | `flow.completion_claim_signal`, `flow.continuation_signal`, `parse.invalid_unicode_escape`, `parse.request_invalid_unicode_escape`, `size.large`, `tool.error_result`, `tool.file_edit`, `tool.file_read`, `tool.file_write`, `tool.search`, `tool.truncation_signal` |
| `LH_102` | `strict` | `strict_events` | false | `flow.completion_claim_signal`, `flow.pre_mutation_read`, `parse.request_strict`, `parse.strict_request`, `tool.error_result`, `tool.file_edit`, `tool.file_read`, `tool.file_write` |

## Interpretation

This is a development-canary manifest, not a quality benchmark or gold label set. R0-08 must adjudicate behavioral traits and add annotation truth. R8 uses separate frozen independent datasets for statistical claims.
