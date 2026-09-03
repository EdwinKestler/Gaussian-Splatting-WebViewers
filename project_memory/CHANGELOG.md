# Project Memory changelog

All notable portable-bundle changes are recorded here. Bundle versions describe executable and
operator-facing behavior; cache and graph schemas are versioned independently.

## 2.4.0 - 2026-08-17

### Added

- Read-only `path SOURCE TARGET` queries using deterministic breadth-first search over the resolved
  dependency graph.
- Forward and reverse traversal, repeatable edge-kind filters, bounded depth, and explicit opt-in
  for probable unique-short-name resolutions.
- Ordered symbol paths, hop counts, resolution metadata, and source pointers in path results.
- `project-memory.py --version` and a package-level `project_memory.__version__` value.
- Exact repository-relative `exclude_paths` configuration and boundary-safe strict UTF-8 probing.

### Compatibility

- Cache schema: `project-memory:v2` (unchanged).
- Code-graph schema: `project-memory:code-graph:v3` (unchanged).
- Graph-record schema: `project-memory:graph-record:v1` (unchanged).
- Redis namespace and mandatory privacy boundaries are unchanged.
- Upgrades require an incremental index to activate a 2.4.0 manifest; unchanged chunks and graph
  records remain reusable.

## 2.3.0

- Moved per-file extraction graphs into path-owned, content-addressed Redis records.
- Kept compact graph references and aggregate counts in the active manifest.
- Added on-demand graph loading, cross-file resolution, deep graph validation, incremental record
  reuse and repair, and migration from embedded 2.2 graphs.
