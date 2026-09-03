# Portable Project Memory v2.4.0 bundle

Copy these paths to another repository while preserving their relative locations:

```text
project-memory.py
project_memory/
.project-memory.json
```

The root entrypoint discovers the repository from the copied package location. The bundle uses
only Python 3.11+ standard-library modules. Redis remains optional and defaults to
`redis://127.0.0.1:6379/0`; set `PROJECT_MEMORY_URL` or pass `--url` to override it.
Release details and schema compatibility are recorded in [CHANGELOG.md](CHANGELOG.md).

Start with a minimal configuration:

```json
{
  "project_slug": "my-repository",
  "redis_url_envs": ["PROJECT_MEMORY_URL"],
  "exclude_directories": ["generated", "vendor"]
}
```

Add `include_patterns` only when the default multi-language corpus is unsuitable. Configured
patterns replace the defaults. `exclude_directories` only adds repository-specific exclusions;
mandatory secret, runtime-data, dependency, build, symlink, and size exclusions remain enforced.
`redis_url_envs` lists the portable and any legacy repository-specific Redis URL variables.

The privacy boundary is not configurable: sensitive credential/key/token path names and key
container suffixes are always rejected, even when a custom include glob matches them. Only known
text-source types that pass an initial UTF-8/binary probe are admitted; binary fixtures under broad
directories such as `tests/` are skipped rather than aborting an index build. These filters reduce
accidental admission, but repositories must still keep real secrets outside source-oriented paths.
This installation also admits `.mjs` and `.cjs` JavaScript modules so the repository's ESM test
sources participate in discovery.

```bash
python3 project-memory.py status
python3 project-memory.py --version
python3 project-memory.py index --incremental
python3 project-memory.py index --incremental --repair-deep
python3 project-memory.py validate
python3 project-memory.py validate --deep
python3 project-memory.py search "authentication boundary" --limit 5
python3 project-memory.py symbols "qualified.name" --limit 20
python3 project-memory.py impact "symbol_name" --limit 20
python3 project-memory.py path "source.qualified.name" "target.qualified.name" --edge-kind calls
python3 project-memory.py evaluate --limit 10
```

v2.1 records a `file_chunks` map beside `file_hashes`. Incremental indexing hashes the admitted
corpus, reparses and embeds only new, changed, or cache-repair files, and reuses unchanged files'
chunk keys. v2.1.2 bounds batched `MGET` calls, gives every build a unique staged generation, and
uses a namespaced renewable Redis lock with token-fenced activation. Existing registry entries are
carried into staging, including leftovers from interrupted attempts, and are collected before the
registry is reduced to the active generation. A final repository fingerprint check prevents
activating a mixed source snapshot. Operational timing/reuse metrics are returned beside, not stored
inside, the immutable generation manifest.

`status` checks compact manifest metadata and repository freshness without loading chunk or graph
payloads. Use
`validate --deep` to load every active chunk and recompute its identity, ownership,
tokens, and vector from the stored text. If it reports semantic corruption, use
`index --incremental --repair-deep` to regenerate only affected owner files.

v2.2 adds per-file symbol and edge maps. Python syntax records come from the standard-library AST
with authoritative extraction confidence, module-qualified names, decorator-call extraction, and
scope-aware absolute and relative import bindings. Link resolution has separate status and
confidence fields; Rust records come
from a deterministic lightweight syntax extractor and are
marked `heuristic` with an explicit diagnostic that it is not an AST parser. `symbols` returns exact
definition pointers, `impact` returns distinct incoming edges with resolved-versus-name-match
provenance, and normal search reports cosine, lexical, and symbol-boost contributions. Optional
`evaluation_queries` in `.project-memory.json`
provide a repeatable recall-at-limit benchmark. Graph results remain discovery pointers: open the
current source before relying on them.

v2.3 moves extraction graphs out of the active manifest into path-owned, content-addressed per-file
Redis records. Identity includes extractor schema, record schema, relative path, and file hash;
identical bytes at different paths remain distinct because extracted names are path-dependent.
Manifests retain graph references and aggregate counts, making `status` metadata-only again.
`search` and `symbols` load graph records on demand; `impact` additionally derives cross-file
resolution in memory. Deep validation verifies record identity, ownership, schema, current-source
extraction, and resolved summary counts. Incremental indexing reuses unchanged graph records,
transactionally stages exact graph keys, and safely collects obsolete records after activation.
`index --incremental --repair-deep` semantically repairs only corrupted graph owners without
regenerating chunks. Embedded v2.2 graphs migrate automatically without re-embedding unchanged
chunks.

v2.4.0 adds a read-only `path` query over the resolved code graph. It uses deterministic
breadth-first search because graph edges are categorical and unweighted, traverses only strong
resolutions by default, supports forward or reverse direction, repeated edge-kind filters, and a
bounded depth. `--include-probable` explicitly admits heuristic unique-short-name resolutions.
Ambiguous and unresolved edges are never traversed. Path lookup derives adjacency in memory and
does not change Redis records or the graph schema.

Copy the Project Memory operating contract from the source repository's `AGENTS.md` into the
target repository instructions. Repository files always remain authoritative; Redis results are
candidate pointers only.
