---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: 0.1.0
kind: added
summary: Added compound-aware word scoring with visible boundary variants
status: accepted
audience: null
scopes: []
source_refs:
  - tl:task-0002
paths:
  - speechscope/metrics.py
  - speechscope/api.py
  - speechscope/evaluation.py
  - speechscope/cli.py
issues: []
prs: []
sources: []
contributors: []
breaking: false
internal: false
order: 1
---

Retained conventional whitespace WER as a diagnostic while adding bounded split/join matching, explicit boundary spans, max_compound_wer thresholds, and exact-content reporting.
