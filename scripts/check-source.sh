#!/usr/bin/env bash
set -euo pipefail

mapfile -d '' javascript_files < <(rg --files -0 -g '*.js' -g '*.mjs' -g '!vendor/**' -g '!node_modules/**' -g '!_site/**')
for path in "${javascript_files[@]}"; do
  node --check "$path"
done

mapfile -d '' json_files < <(rg --files -0 -g '*.json' -g '!vendor/**' -g '!node_modules/**' -g '!_site/**' -g '!artifacts/**')
for path in "${json_files[@]}"; do
  node -e 'JSON.parse(require("node:fs").readFileSync(process.argv[1], "utf8"))' "$path"
done

python3 - <<'PY'
import ast
from pathlib import Path

for path in [Path("semantic_sidecar/server.py"), Path("project-memory.py")]:
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
PY

echo "Checked ${#javascript_files[@]} JavaScript modules, ${#json_files[@]} JSON files, and Python entrypoints"
