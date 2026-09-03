# Repository agent instructions

Repository files are always the authoritative source of truth. Cached Project Memory results are discovery pointers only; open and verify the referenced source before making claims or edits.

## Project Memory v2.4

Use the portable repository-root entrypoint and this repository's isolated `gaussian-splatting-webviewers:project-memory:v2` namespace:

```bash
python3 project-memory.py status
python3 project-memory.py index --incremental
python3 project-memory.py validate --deep
python3 project-memory.py search "QUERY" --limit 5
python3 project-memory.py symbols "qualified.name" --limit 20
python3 project-memory.py impact "symbol_name" --limit 20
python3 project-memory.py path "source.qualified.name" "target.qualified.name" --edge-kind calls
python3 project-memory.py evaluate --limit 10
```

- Run `status` before querying. Exit `0` means fresh, `2` means missing or stale and should be followed by an incremental index, and `1` means an operational error that must not be treated as stale state.
- Redis defaults to `redis://127.0.0.1:6379/0`. Override it with `--url`, `PROJECT_MEMORY_URL`, or `GAUSSIAN_SPLATTING_WEBVIEWERS_PROJECT_MEMORY_URL`.
- After changing an indexed file, re-index incrementally and confirm that `status` exits `0`.
- If Redis is unavailable, continue from repository files and disclose `cache_consulted: false`.
- Never use `FLUSHDB`, `FLUSHALL`, wildcard deletion, raw Redis keys, or another project's namespace.
- Never index `.env`, credentials, tokens, model binaries, vendor dependencies, generated captures, test reports, or output artifacts. The source-oriented corpus and exact exclusions are defined in `.project-memory.json`.

Bundle documentation and schema details: [project_memory/README.md](project_memory/README.md).
