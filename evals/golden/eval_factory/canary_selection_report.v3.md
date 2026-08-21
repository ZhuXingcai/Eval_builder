# Eval Dataset Factory Development Canary Selection

- Schema: `eval-factory-canary-manifest/v3`
- Selector: `3.0.0`
- Source traces: 91
- Selected traces: 24
- Raw SID, account, prompt, response, and file content are not emitted.
- Every selected item requires R0-08 annotation.
- Only verified traits count as deterministic coverage.
- Semantic and malformed-request signals are annotation candidates, not truth.

## Verified Coverage

| Trait | Required | Observed |
|---|---:|---:|
| `flow.post_mutation_read` | 3 | 15 |
| `flow.pre_mutation_read` | 3 | 15 |
| `parse.invalid_escape` | 6 | 6 |
| `parse.invalid_response` | 1 | 1 |
| `parse.invalid_unicode_escape` | 3 | 3 |
| `parse.strict_request` | 10 | 15 |
| `size.large` | 3 | 9 |
| `size.small` | 2 | 5 |
| `tool.error_result` | 5 | 14 |
| `tool.fetch` | 2 | 4 |
| `tool.file_edit` | 6 | 15 |
| `tool.file_read` | 6 | 15 |
| `tool.file_write` | 6 | 15 |
| `tool.powershell` | 1 | 1 |
| `tool.search` | 3 | 5 |

## Annotation Candidate Coverage

| Candidate | Required | Observed |
|---|---:|---:|
| `candidate.continuation_signal` | 3 | 12 |
| `candidate.final_output_claim` | 2 | 11 |
| `candidate.no_attachment_read` | 3 | 9 |
| `candidate.truncation_signal` | 3 | 13 |

## Selected Items

| Instance | Request parse | Signal quality | Verified traits | Annotation candidates |
|---|---|---|---|---|
| `LH_005` | `strict` | `strict_events` | `flow.post_mutation_read`, `flow.pre_mutation_read`, `parse.request_strict`, `parse.strict_request`, `size.large`, `tool.error_result`, `tool.file_edit`, `tool.file_read`, `tool.file_write` | `candidate.continuation_signal` |
| `LH_006` | `strict` | `strict_events` | `flow.post_mutation_read`, `flow.pre_mutation_read`, `parse.request_strict`, `parse.strict_request`, `size.small`, `tool.file_edit`, `tool.file_read`, `tool.file_write` | `candidate.truncation_signal` |
| `LH_011` | `invalid_escape` | `heuristic_requires_annotation` | `parse.invalid_escape`, `parse.invalid_response`, `parse.request_invalid_escape` | `candidate.continuation_signal`, `candidate.final_output_claim`, `candidate.no_attachment_read`, `candidate.truncation_signal` |
| `LH_015` | `strict` | `strict_events` | `flow.post_mutation_read`, `flow.pre_mutation_read`, `parse.request_strict`, `parse.strict_request`, `size.small`, `tool.error_result`, `tool.fetch`, `tool.file_edit`, `tool.file_read`, `tool.file_write`, `tool.search` |  |
| `LH_019` | `strict` | `strict_events` | `flow.post_mutation_read`, `flow.pre_mutation_read`, `parse.request_strict`, `parse.strict_request`, `size.large`, `tool.error_result`, `tool.file_edit`, `tool.file_read`, `tool.file_write` | `candidate.continuation_signal` |
| `LH_032` | `invalid_escape` | `heuristic_requires_annotation` | `parse.invalid_escape`, `parse.request_invalid_escape`, `size.large` | `candidate.continuation_signal`, `candidate.final_output_claim`, `candidate.no_attachment_read`, `candidate.truncation_signal` |
| `LH_035` | `strict` | `strict_events` | `flow.post_mutation_read`, `flow.pre_mutation_read`, `parse.request_strict`, `parse.strict_request`, `tool.error_result`, `tool.file_edit`, `tool.file_read`, `tool.file_write` | `candidate.truncation_signal` |
| `LH_040` | `strict` | `strict_events` | `flow.post_mutation_read`, `flow.pre_mutation_read`, `parse.request_strict`, `parse.strict_request`, `tool.error_result`, `tool.file_edit`, `tool.file_read`, `tool.file_write`, `tool.search` | `candidate.truncation_signal` |
| `LH_041` | `strict` | `strict_events` | `flow.post_mutation_read`, `flow.pre_mutation_read`, `parse.request_strict`, `parse.strict_request`, `size.large`, `tool.error_result`, `tool.file_edit`, `tool.file_read`, `tool.file_write` |  |
| `LH_046` | `strict` | `strict_events` | `flow.post_mutation_read`, `flow.pre_mutation_read`, `parse.request_strict`, `parse.strict_request`, `size.small`, `tool.error_result`, `tool.file_edit`, `tool.file_read`, `tool.file_write` | `candidate.final_output_claim` |
| `LH_057` | `strict` | `strict_events` | `flow.post_mutation_read`, `flow.pre_mutation_read`, `parse.request_strict`, `parse.strict_request`, `tool.error_result`, `tool.file_edit`, `tool.file_read`, `tool.file_write` | `candidate.continuation_signal`, `candidate.final_output_claim`, `candidate.truncation_signal` |
| `LH_058` | `strict` | `strict_events` | `flow.post_mutation_read`, `flow.pre_mutation_read`, `parse.request_strict`, `parse.strict_request`, `tool.error_result`, `tool.file_edit`, `tool.file_read`, `tool.file_write`, `tool.powershell` | `candidate.truncation_signal` |
| `LH_065` | `strict` | `strict_events` | `flow.post_mutation_read`, `flow.pre_mutation_read`, `parse.request_strict`, `parse.strict_request`, `tool.error_result`, `tool.fetch`, `tool.file_edit`, `tool.file_read`, `tool.file_write`, `tool.search` |  |
| `LH_066` | `strict` | `strict_events` | `flow.post_mutation_read`, `flow.pre_mutation_read`, `parse.request_strict`, `parse.strict_request`, `tool.error_result`, `tool.file_edit`, `tool.file_read`, `tool.file_write` |  |
| `LH_067` | `invalid_unicode_escape` | `heuristic_requires_annotation` | `parse.invalid_unicode_escape`, `parse.request_invalid_unicode_escape`, `size.small` | `candidate.continuation_signal`, `candidate.final_output_claim`, `candidate.no_attachment_read` |
| `LH_075` | `strict` | `strict_events` | `flow.post_mutation_read`, `flow.pre_mutation_read`, `parse.request_strict`, `parse.strict_request`, `tool.error_result`, `tool.fetch`, `tool.file_edit`, `tool.file_read`, `tool.file_write`, `tool.search` | `candidate.continuation_signal` |
| `LH_077` | `strict` | `strict_events` | `flow.post_mutation_read`, `flow.pre_mutation_read`, `parse.request_strict`, `parse.strict_request`, `tool.error_result`, `tool.fetch`, `tool.file_edit`, `tool.file_read`, `tool.file_write`, `tool.search` |  |
| `LH_079` | `invalid_escape` | `heuristic_requires_annotation` | `parse.invalid_escape`, `parse.request_invalid_escape`, `size.large` | `candidate.continuation_signal`, `candidate.final_output_claim`, `candidate.no_attachment_read`, `candidate.truncation_signal` |
| `LH_089` | `invalid_escape` | `heuristic_requires_annotation` | `parse.invalid_escape`, `parse.request_invalid_escape`, `size.large` | `candidate.continuation_signal`, `candidate.final_output_claim`, `candidate.no_attachment_read`, `candidate.truncation_signal` |
| `LH_093` | `invalid_unicode_escape` | `heuristic_requires_annotation` | `parse.invalid_unicode_escape`, `parse.request_invalid_unicode_escape`, `size.large` | `candidate.continuation_signal`, `candidate.final_output_claim`, `candidate.no_attachment_read`, `candidate.truncation_signal` |
| `LH_098` | `invalid_escape` | `heuristic_requires_annotation` | `parse.invalid_escape`, `parse.request_invalid_escape`, `size.small` | `candidate.final_output_claim`, `candidate.no_attachment_read`, `candidate.truncation_signal` |
| `LH_100` | `invalid_escape` | `heuristic_requires_annotation` | `parse.invalid_escape`, `parse.request_invalid_escape`, `size.large` | `candidate.continuation_signal`, `candidate.final_output_claim`, `candidate.no_attachment_read`, `candidate.truncation_signal` |
| `LH_101` | `invalid_unicode_escape` | `heuristic_requires_annotation` | `parse.invalid_unicode_escape`, `parse.request_invalid_unicode_escape`, `size.large` | `candidate.continuation_signal`, `candidate.final_output_claim`, `candidate.no_attachment_read`, `candidate.truncation_signal` |
| `LH_102` | `strict` | `strict_events` | `flow.post_mutation_read`, `flow.pre_mutation_read`, `parse.request_strict`, `parse.strict_request`, `tool.error_result`, `tool.file_edit`, `tool.file_read`, `tool.file_write` |  |

## Business Category Coverage

| Category | Source | Selected |
|---|---:|---:|
| 人力资源 | 2 | 1 |
| 内容创作与创意设计 | 11 | 3 |
| 医疗 | 1 | 1 |
| 客服与销售 | 3 | 1 |
| 教育 | 11 | 1 |
| 数据处理与分析 | 10 | 2 |
| 日常办公 | 15 | 3 |
| 日常生活 | 1 | 1 |
| 科学研究 | 16 | 3 |
| 软件开发 | 9 | 2 |
| 金融与财务 | 12 | 6 |

## Interpretation

This is a development-canary manifest, not a quality benchmark or gold label set. R0-08 must adjudicate every item and add annotation truth. In particular, `candidate.no_attachment_read` means no normalized file-read tool event was observed; it does not prove the task had no attachment read requirement. R8 uses separate frozen independent datasets for statistical claims.
