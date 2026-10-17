---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0002
release_version: 0.1.1
kind: added
summary:
  Added catalog-backed model discovery and verified lifecycle management for
  whisper.cpp model artifacts
status: accepted
audience: null
scopes: []
source_refs:
  - git:377e6579844852f8e612d111646514058fa2c526
  - git:78795206a6e8e82f98b2b324c9d51faef32c46c7
paths:
  - speechscope/models.py
  - speechscope/cli.py
  - speechscope/data/model-catalog-v1.json
  - docs/model-management.md
  - docs/whispercpp-integration.md
  - tests/test_models.py
  - tests/test_whispercpp_models.py
issues: []
prs: []
sources:
  - git:377e6579844852f8e612d111646514058fa2c526
  - git:78795206a6e8e82f98b2b324c9d51faef32c46c7
contributors:
  - "@holgern"
breaking: false
internal: false
order: 2
---
