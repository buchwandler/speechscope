---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: 0.1.1
kind: added
summary:
  Added a local whisper.cpp transcription backend with canonical audio conversion
  and backend-input provenance reporting
status: accepted
audience: null
scopes: []
source_refs:
  - git:76ea52c80929640e5b2b1384958230541514cad7
paths:
  - speechscope/backends/whispercpp.py
  - speechscope/backend_audio.py
  - docs/whispercpp-integration.md
  - docs/backends.md
  - tests/test_whispercpp.py
  - tests/test_backend_audio.py
issues: []
prs: []
sources:
  - git:76ea52c80929640e5b2b1384958230541514cad7
contributors:
  - "@holgern"
breaking: false
internal: false
order: 1
---
