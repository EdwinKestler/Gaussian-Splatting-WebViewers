"""Portable, project-scoped Redis retrieval cache for source discovery."""

from __future__ import annotations

import argparse
import ast
import base64
import codecs
import contextlib
import hashlib
import json
import math
import os
import re
import secrets
import socket
import struct
import sys
import time
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

SCHEMA = "project-memory:v2"
BUNDLE_VERSION = "2.4.0"
GRAPH_SCHEMA = "project-memory:code-graph:v3"
GRAPH_RECORD_SCHEMA = "project-memory:graph-record:v1"
CONTENT_ADDRESSED_GRAPH_BUNDLE_SERIES = ("2.3", "2.4")
EMBEDDING_ID = "feature-hash-sha256-unigram-bigram-v1"
DIMENSIONS = 384
DEFAULT_URL = "redis://127.0.0.1:6379/0"
URL_ENV = "PROJECT_MEMORY_URL"
DEPENDENCY_EDGE_KINDS = ("calls", "decorated_by", "imports", "inherits")
PATH_DIRECTIONS = ("forward", "reverse")
DEFAULT_PATH_MAX_DEPTH = 8
MAX_PATH_DEPTH = 100
CHUNK_LINES = 80
CHUNK_OVERLAP = 16
ROOT = Path(__file__).resolve().parents[1]
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+", re.UNICODE)
CONFIG_FILE = ".project-memory.json"
DEFAULT_PATTERNS = (
    "docs/**/*.md",
    ".github/workflows/*.yml",
    ".github/workflows/*.yaml",
    "src/**/*.py",
    "src/**/*.pyi",
    "src/**/*.js",
    "src/**/*.jsx",
    "src/**/*.ts",
    "src/**/*.tsx",
    "src/**/*.go",
    "src/**/*.rs",
    "src/**/*.java",
    "src/**/*.kt",
    "src/**/*.c",
    "src/**/*.h",
    "src/**/*.cpp",
    "src/**/*.hpp",
    "src/**/*.sql",
    "src/**/*.proto",
    "src/**/*.graphql",
    "tests/**/*",
    "schemas/**/*",
    "*.json",
    "*.yaml",
    "*.yml",
    "Dockerfile*",
    "docker-compose*.yml",
    "docker-compose*.yaml",
    "Makefile",
    "*.mk",
    "*.tf",
    ".claude/skills/*.md",
    "agent-skill/**/*.md",
    "agent-skill/**/Dockerfile",
    "agent-skill/**/*.sh",
    "agent-skill/**/.env.example",
    "agent-skill/**/.gitignore",
)
EXCLUDED_PARTS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "reports",
    "fixtures",
    "data",
    "secrets",
    "images",
    "screenshots",
    "dist",
    "build",
    "coverage",
    "htmlcov",
    "__pycache__",
    "project_memory",
}
ENV_NAME_RE = re.compile(r"[A-Z_][A-Z0-9_]*")
ALLOWED_DOT_DIRS = {".github", ".claude", ".agents", ".codex"}
SENSITIVE_NAME_PATTERNS = (
    re.compile(r"(^|[._/-])credentials?([._/-]|$)", re.IGNORECASE),
    re.compile(r"(^|[._/-])secrets?([._/-]|$)", re.IGNORECASE),
    re.compile(r"(^|[._/-])private[_-]?keys?([._/-]|$)", re.IGNORECASE),
    re.compile(r"(^|[._/-])api[_-]?keys?([._/-]|$)", re.IGNORECASE),
    re.compile(r"(^|[._/-])access[_-]?tokens?([._/-]|$)", re.IGNORECASE),
    re.compile(r"(^|[._/-])service[_-]?accounts?([._/-]|$)", re.IGNORECASE),
)
SENSITIVE_SUFFIXES = {
    ".pem",
    ".key",
    ".p12",
    ".pfx",
    ".jks",
    ".keystore",
}
SENSITIVE_EXACT_NAMES = {
    "authorized_keys",
    "credentials.json",
    "id_ed25519",
    "id_rsa",
    "known_hosts",
    "service-account.json",
}
TEXT_SUFFIXES = {
    ".avsc",
    ".bash",
    ".c",
    ".cjs",
    ".cfg",
    ".conf",
    ".cpp",
    ".css",
    ".graphql",
    ".h",
    ".hpp",
    ".htm",
    ".html",
    ".ini",
    ".java",
    ".js",
    ".json",
    ".jsonl",
    ".jsx",
    ".kt",
    ".kts",
    ".less",
    ".md",
    ".mjs",
    ".mk",
    ".proto",
    ".py",
    ".pyi",
    ".rs",
    ".rst",
    ".scss",
    ".sh",
    ".sql",
    ".svelte",
    ".tf",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".vue",
    ".xml",
    ".yaml",
    ".yml",
    ".zsh",
}
TEXT_EXACT_NAMES = {
    ".env.example",
    ".gitignore",
    "dockerfile",
    "license",
    "makefile",
    "readme",
}
TEXT_PROBE_BYTES = 8192
MGET_BATCH_SIZE = 1_000
INDEX_LOCK_MS = 60_000
SENSITIVE_DATA_SUFFIXES = {
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
    ".txt",
}
SENSITIVE_TOKEN_PATTERNS = (
    re.compile(
        r"(^|[._/-])(?:access|refresh|auth|bearer|oauth)?[_-]?tokens?([._/-]|$)",
        re.IGNORECASE,
    ),
    re.compile(r"(^|[._/-])client[_-]?secrets?([._/-]|$)", re.IGNORECASE),
)
RUST_DEFINITION_RE = re.compile(
    r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+|unsafe\s+)?"
    r"(fn|struct|enum|trait|mod|type|const|static)\s+([A-Za-z_][A-Za-z0-9_]*)"
)
RUST_USE_RE = re.compile(r"^\s*(?:pub\s+)?use\s+([^;]+);")
RUST_CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)*)\s*\(")
RUST_CALL_EXCLUSIONS = {"fn", "if", "while", "for", "loop", "match", "return", "Some", "Ok", "Err"}


def project_config(root: Path = ROOT) -> dict[str, Any]:
    path = root / CONFIG_FILE
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{CONFIG_FILE} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{CONFIG_FILE} must contain a JSON object")
    return value


def project_slug(root: Path = ROOT) -> str:
    configured = project_config(root).get("project_slug")
    if configured is not None and not isinstance(configured, str):
        raise ValueError(f"{CONFIG_FILE} project_slug must be a string")
    source = configured or root.resolve().name
    slug = re.sub(r"[^a-z0-9]+", "-", source.lower()).strip("-")
    if not slug:
        raise ValueError("project directory does not produce a valid slug")
    return slug


def redis_url_envs(root: Path = ROOT) -> tuple[str, ...]:
    configured = project_config(root).get("redis_url_envs")
    if configured is None:
        return (URL_ENV,)
    if not isinstance(configured, list) or not configured or not all(
        isinstance(item, str) and ENV_NAME_RE.fullmatch(item) for item in configured
    ):
        raise ValueError(f"{CONFIG_FILE} redis_url_envs must be an environment-variable list")
    return tuple(configured)


def configured_excluded_parts(root: Path = ROOT) -> set[str]:
    configured = project_config(root).get("exclude_directories", [])
    if not isinstance(configured, list) or not all(
        isinstance(item, str) and item and "/" not in item and "\\" not in item
        for item in configured
    ):
        raise ValueError(f"{CONFIG_FILE} exclude_directories must contain directory names")
    return EXCLUDED_PARTS | set(configured)


def configured_excluded_paths(root: Path = ROOT) -> set[str]:
    configured = project_config(root).get("exclude_paths", [])
    if not isinstance(configured, list) or not all(
        isinstance(item, str)
        and item
        and not item.startswith(("/", "\\"))
        and ".." not in Path(item).parts
        for item in configured
    ):
        raise ValueError(
            f"{CONFIG_FILE} exclude_paths must contain repository-relative paths"
        )
    return {Path(item).as_posix() for item in configured}


def namespace(root: Path = ROOT) -> str:
    return f"{project_slug(root)}:{SCHEMA}"


def is_sensitive_path(path: Path, relative: str) -> bool:
    """Reject secret-bearing path names independently of repository configuration."""
    name = path.name.lower()
    normalized = relative.replace("\\", "/").lower()
    sensitive_data_name = path.suffix.lower() in SENSITIVE_DATA_SUFFIXES and any(
        pattern.search(name) or pattern.search(normalized) for pattern in SENSITIVE_TOKEN_PATTERNS
    )
    return (
        name in SENSITIVE_EXACT_NAMES
        or path.suffix.lower() in SENSITIVE_SUFFIXES
        or any(pattern.search(name) or pattern.search(normalized) for pattern in SENSITIVE_NAME_PATTERNS)
        or sensitive_data_name
    )


def is_text_source_path(path: Path) -> bool:
    """Admit known text source types and reject binary/non-UTF-8 payloads."""
    name = path.name.lower()
    type_allowed = (
        path.suffix.lower() in TEXT_SUFFIXES
        or name in TEXT_EXACT_NAMES
        or name.startswith(("dockerfile.", "makefile."))
    )
    if not type_allowed:
        return False
    try:
        sample = path.read_bytes()[:TEXT_PROBE_BYTES]
        if b"\x00" in sample:
            return False
        # A bounded sample can end in the middle of a valid multibyte code
        # point. Incremental strict decoding validates the available bytes
        # without rejecting that valid boundary split.
        codecs.getincrementaldecoder("utf-8")(errors="strict").decode(
            sample, final=False
        )
    except (OSError, UnicodeDecodeError):
        return False
    return True


def included_files(root: Path = ROOT) -> list[Path]:
    """Return deterministic, privacy-conscious source corpus paths."""
    exact = [
        "README.md",
        "AGENTS.md",
        "agent_instructions.md",
        "instructions.md",
        "pyproject.toml",
        ".gitignore",
        ".env.example",
    ]
    paths = [root / item for item in exact if (root / item).is_file()]
    patterns = DEFAULT_PATTERNS
    configured = project_config(root).get("include_patterns")
    if configured is not None:
        if not isinstance(configured, list) or not all(
            isinstance(item, str) and item for item in configured
        ):
            raise ValueError(f"{CONFIG_FILE} include_patterns must be a string list")
        patterns = tuple(configured)
    excluded_parts = configured_excluded_parts(root)
    excluded_paths = configured_excluded_paths(root)
    for pattern in patterns:
        paths.extend(path for path in root.glob(pattern) if path.is_file())
    result: list[Path] = []
    for path in sorted(set(paths), key=lambda p: p.relative_to(root).as_posix()):
        rel = path.relative_to(root).as_posix()
        if rel in excluded_paths:
            continue
        if any(part in excluded_parts for part in Path(rel).parts):
            continue
        if any(
            part.startswith(".") and part not in ALLOWED_DOT_DIRS
            for part in Path(rel).parts[:-1]
        ):
            continue
        if (
            path.name in {"project_memory.py", "project-memory.py"}
            or path.name.startswith(".env") and path.name != ".env.example"
            or path.is_symlink()
            or path.stat().st_size > 1_000_000
            or is_sensitive_path(path, rel)
            or not is_text_source_path(path)
        ):
            continue
        result.append(path)
    return result


def corpus_fingerprint(files: Iterable[Path], root: Path = ROOT) -> str:
    digest = hashlib.sha256()
    for path in files:
        rel = path.relative_to(root).as_posix().encode("utf-8")
        raw = path.read_bytes()
        digest.update(struct.pack(">I", len(rel)))
        digest.update(rel)
        digest.update(struct.pack(">Q", len(raw)))
        digest.update(raw)
    return digest.hexdigest()


def file_fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def embedding(text: str, dimensions: int = DIMENSIONS) -> tuple[float, ...]:
    tokens = tokenize(text)
    features = tokens + [f"{a}\x1f{b}" for a, b in pairwise(tokens)]
    vector = [0.0] * dimensions
    for feature in features:
        hashed = hashlib.sha256(feature.encode("utf-8")).digest()
        index = int.from_bytes(hashed[:8], "big") % dimensions
        vector[index] += 1.0 if hashed[8] & 1 else -1.0
    norm = math.sqrt(sum(value * value for value in vector))
    if norm:
        vector = [value / norm for value in vector]
    return tuple(vector)


def encode_vector(vector: Iterable[float]) -> str:
    values = tuple(vector)
    return base64.b64encode(struct.pack(f">{len(values)}f", *values)).decode("ascii")


def decode_vector(value: str, dimensions: int = DIMENSIONS) -> tuple[float, ...]:
    raw = base64.b64decode(value.encode("ascii"), validate=True)
    if len(raw) != dimensions * 4:
        raise ValueError("invalid vector dimensions")
    return struct.unpack(f">{dimensions}f", raw)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    path: str
    start_line: int
    end_line: int
    text: str


def split_chunks(
    path: str, text: str, size: int = CHUNK_LINES, overlap: int = CHUNK_OVERLAP
) -> list[Chunk]:
    if size < 1 or overlap < 0 or overlap >= size:
        raise ValueError("chunk size must be positive and overlap smaller than size")
    lines = text.splitlines()
    if not lines:
        return []
    chunks = []
    step = size - overlap
    for start in range(0, len(lines), step):
        selected = lines[start : start + size]
        if not selected:
            break
        chunk_text = "\n".join(selected)
        identity = hashlib.sha256(
            f"{path}\0{start + 1}\0{start + len(selected)}\0".encode() + chunk_text.encode()
        ).hexdigest()[:24]
        chunks.append(Chunk(identity, path, start + 1, start + len(selected), chunk_text))
        if start + size >= len(lines):
            break
    return chunks


def _symbol_id(path: str, qualified_name: str, kind: str, start_line: int) -> str:
    digest = hashlib.sha256(
        f"{path}\0{qualified_name}\0{kind}\0{start_line}".encode()
    ).hexdigest()[:20]
    return f"symbol:{digest}"


def _dotted_python_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Call):
        return _dotted_python_name(node.func)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_python_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


class _PythonGraphVisitor(ast.NodeVisitor):
    def __init__(self, path: str):
        self.path = path
        self.module_name = path.removesuffix(".py").replace("/", ".")
        self.scope: list[str] = []
        self.scope_kinds: list[str] = []
        self.current_symbols: list[str] = []
        self.symbols: list[dict[str, Any]] = []
        self.edges: list[dict[str, Any]] = []
        self.bindings: dict[str, str] = {}
        self.module_bindings = self.bindings
        self.module_id = self._add_symbol(
            self.module_name, self.module_name, "module", 1, 1
        )

    def _qualified_name(self, name: str) -> str:
        return ".".join((self.module_name, *self.scope, name))

    def _canonical_target(self, target: str) -> tuple[str, bool]:
        head, separator, tail = target.partition(".")
        canonical = self.bindings.get(head)
        if canonical is None:
            return target, False
        return canonical + (separator + tail if separator else ""), True

    def _relative_import_module(self, module: str | None, level: int) -> str | None:
        if level == 0:
            return module or ""
        path_parts = self.path.removesuffix(".py").split("/")
        package_parts = path_parts[:-1]
        ascend = level - 1
        if ascend > len(package_parts):
            return None
        base = package_parts[: len(package_parts) - ascend]
        if module:
            base.extend(module.split("."))
        return ".".join(base)

    def _add_symbol(
        self, name: str, qualified_name: str, kind: str, start_line: int, end_line: int
    ) -> str:
        symbol_id = _symbol_id(self.path, qualified_name, kind, start_line)
        self.symbols.append(
            {
                "id": symbol_id,
                "name": name,
                "qualified_name": qualified_name,
                "kind": kind,
                "path": self.path,
                "start_line": start_line,
                "end_line": end_line,
                "parser": "python-ast",
                "confidence": "authoritative",
            }
        )
        return symbol_id

    def _source(self) -> str:
        return self.current_symbols[-1] if self.current_symbols else self.module_id

    def _edge(self, kind: str, target: str, line: int, *, source_id: str | None = None) -> None:
        canonical_target, binding_applied = self._canonical_target(target)
        self.edges.append(
            {
                "source_id": source_id or self._source(),
                "target": canonical_target,
                "kind": kind,
                "path": self.path,
                "line": line,
                "parser": "python-ast",
                "extraction_confidence": "authoritative",
                "binding_applied": binding_applied,
                "resolution": {
                    "status": "unresolved",
                    "confidence": "unknown",
                    "target_id": None,
                },
            }
        )

    def _visit_definition(self, node: ast.AST, name: str, kind: str) -> None:
        qualified_name = self._qualified_name(name)
        symbol_id = self._add_symbol(
            name,
            qualified_name,
            kind,
            getattr(node, "lineno", 1),
            getattr(node, "end_lineno", getattr(node, "lineno", 1)),
        )
        decorators = getattr(node, "decorator_list", [])
        for decorator in decorators:
            target = _dotted_python_name(decorator)
            if target:
                self._edge(
                    "decorated_by",
                    target,
                    getattr(decorator, "lineno", getattr(node, "lineno", 1)),
                    source_id=symbol_id,
                )
        self.scope.append(name)
        self.scope_kinds.append(kind)
        self.current_symbols.append(symbol_id)
        previous_bindings = self.bindings
        self.bindings = dict(
            self.module_bindings if kind == "method" else previous_bindings
        )
        self.generic_visit(node)
        self.bindings = previous_bindings
        self.current_symbols.pop()
        self.scope_kinds.pop()
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        kind = "method" if self.scope_kinds and self.scope_kinds[-1] == "class" else "function"
        self._visit_definition(node, node.name, kind)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        kind = "method" if self.scope_kinds and self.scope_kinds[-1] == "class" else "function"
        self._visit_definition(node, node.name, kind)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        qualified_name = self._qualified_name(node.name)
        symbol_id = self._add_symbol(
            node.name,
            qualified_name,
            "class",
            node.lineno,
            getattr(node, "end_lineno", node.lineno),
        )
        for base in node.bases:
            target = _dotted_python_name(base)
            if target:
                self._edge(
                    "inherits",
                    target,
                    getattr(base, "lineno", node.lineno),
                    source_id=symbol_id,
                )
        self.scope.append(node.name)
        self.scope_kinds.append("class")
        self.current_symbols.append(symbol_id)
        previous_bindings = self.bindings
        self.bindings = dict(previous_bindings)
        self.generic_visit(node)
        self.bindings = previous_bindings
        self.current_symbols.pop()
        self.scope_kinds.pop()
        self.scope.pop()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.asname:
                self.bindings[alias.asname] = alias.name
            else:
                top_level = alias.name.split(".", 1)[0]
                self.bindings[top_level] = top_level
            self._edge("imports", alias.name, node.lineno)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = self._relative_import_module(node.module, node.level)
        for alias in node.names:
            canonical = f"{module}.{alias.name}" if module else alias.name
            self.bindings[alias.asname or alias.name] = canonical
            self._edge("imports", canonical, node.lineno)

    def visit_Call(self, node: ast.Call) -> None:
        target = _dotted_python_name(node.func)
        if target:
            self._edge("calls", target, node.lineno)
        self.generic_visit(node)


def _python_file_graph(path: str, text: str) -> dict[str, Any]:
    try:
        tree = ast.parse(text, filename=path)
    except SyntaxError as exc:
        return {
            "schema": GRAPH_SCHEMA,
            "language": "python",
            "parser": "python-ast",
            "symbols": [],
            "edges": [],
            "diagnostics": [f"syntax error at line {exc.lineno or 0}"],
        }
    visitor = _PythonGraphVisitor(path)
    visitor.visit(tree)
    return {
        "schema": GRAPH_SCHEMA,
        "language": "python",
        "parser": "python-ast",
        "symbols": visitor.symbols,
        "edges": visitor.edges,
        "diagnostics": [],
    }


def _rust_file_graph(path: str, text: str) -> dict[str, Any]:
    symbols: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    module_name = path.removesuffix(".rs").replace("/", "::")
    module_id = _symbol_id(path, module_name, "module", 1)
    symbols.append(
        {
            "id": module_id,
            "name": module_name.rsplit("::", 1)[-1],
            "qualified_name": module_name,
            "kind": "module",
            "path": path,
            "start_line": 1,
            "end_line": max(1, len(text.splitlines())),
            "parser": "rust-syntax-v1",
            "confidence": "heuristic",
        }
    )
    current_source = module_id
    for line_number, line in enumerate(text.splitlines(), 1):
        definition = RUST_DEFINITION_RE.search(line)
        if definition:
            rust_kind, name = definition.groups()
            kind = "function" if rust_kind == "fn" else rust_kind
            qualified_name = f"{module_name}::{name}"
            current_source = _symbol_id(path, qualified_name, kind, line_number)
            symbols.append(
                {
                    "id": current_source,
                    "name": name,
                    "qualified_name": qualified_name,
                    "kind": kind,
                    "path": path,
                    "start_line": line_number,
                    "end_line": line_number,
                    "parser": "rust-syntax-v1",
                    "confidence": "heuristic",
                }
            )
        imported = RUST_USE_RE.search(line)
        if imported:
            edges.append(
                {
                    "source_id": module_id,
                    "target": imported.group(1).strip(),
                    "kind": "imports",
                    "path": path,
                    "line": line_number,
                    "parser": "rust-syntax-v1",
                    "extraction_confidence": "heuristic",
                    "binding_applied": False,
                    "resolution": {
                        "status": "unresolved",
                        "confidence": "unknown",
                        "target_id": None,
                    },
                }
            )
        for target in RUST_CALL_RE.findall(line):
            if target not in RUST_CALL_EXCLUSIONS and not (
                definition and target == definition.group(2)
            ):
                edges.append(
                    {
                        "source_id": current_source,
                        "target": target,
                        "kind": "calls",
                        "path": path,
                        "line": line_number,
                        "parser": "rust-syntax-v1",
                        "extraction_confidence": "heuristic",
                        "binding_applied": False,
                        "resolution": {
                            "status": "unresolved",
                            "confidence": "unknown",
                            "target_id": None,
                        },
                    }
                )
    return {
        "schema": GRAPH_SCHEMA,
        "language": "rust",
        "parser": "rust-syntax-v1",
        "symbols": symbols,
        "edges": edges,
        "diagnostics": ["Rust records use deterministic syntax heuristics, not an AST parser."],
    }


def extract_file_graph(path: str, text: str) -> dict[str, Any]:
    if path.endswith(".py"):
        return _python_file_graph(path, text)
    if path.endswith(".rs"):
        return _rust_file_graph(path, text)
    return {
        "schema": GRAPH_SCHEMA,
        "language": "text",
        "parser": "none",
        "symbols": [],
        "edges": [],
        "diagnostics": [],
    }


def _resolve_graph(
    file_graphs: dict[str, dict[str, Any]],
) -> tuple[int, int, dict[str, int]]:
    symbols = [symbol for graph in file_graphs.values() for symbol in graph["symbols"]]
    symbols_by_id = {symbol["id"]: symbol for symbol in symbols}
    by_qualified: dict[str, list[str]] = {}
    by_short: dict[str, list[str]] = {}
    for symbol in symbols:
        by_qualified.setdefault(symbol["qualified_name"], []).append(symbol["id"])
        by_short.setdefault(symbol["name"], []).append(symbol["id"])
    resolution_counts = {
        status: 0
        for status in (
            "exact_qualified",
            "lexical_scope",
            "import_binding",
            "heuristic_unique_short_name",
            "ambiguous",
            "unresolved",
        )
    }
    edge_count = 0
    for graph in file_graphs.values():
        for edge in graph["edges"]:
            target = edge["target"]
            resolution = {
                "status": "unresolved",
                "confidence": "unknown",
                "target_id": None,
            }
            exact_candidates = by_qualified.get(target, [])
            if len(exact_candidates) == 1:
                resolution = {
                    "status": (
                        "import_binding" if edge.get("binding_applied") else "exact_qualified"
                    ),
                    "confidence": "strong",
                    "target_id": exact_candidates[0],
                }
            elif len(exact_candidates) > 1:
                resolution["status"] = "ambiguous"
                resolution["confidence"] = "ambiguous"
            elif graph.get("language") == "python" and "." not in target:
                source = symbols_by_id.get(edge["source_id"])
                source_qualified = source["qualified_name"] if source else ""
                parts = source_qualified.split(".")[:-1]
                lexical_candidates = []
                for end in range(len(parts), 0, -1):
                    candidate = ".".join((*parts[:end], target))
                    matches = by_qualified.get(candidate, [])
                    if matches:
                        lexical_candidates = matches
                        break
                if len(lexical_candidates) == 1:
                    resolution = {
                        "status": "lexical_scope",
                        "confidence": "strong",
                        "target_id": lexical_candidates[0],
                    }
                elif len(lexical_candidates) > 1:
                    resolution["status"] = "ambiguous"
                    resolution["confidence"] = "ambiguous"
            allow_short_heuristic = graph.get("language") != "python" or "." not in target
            if resolution["status"] == "unresolved" and allow_short_heuristic:
                short_target = re.split(r"[.:]+", target)[-1]
                short_candidates = by_short.get(short_target, [])
                if len(short_candidates) == 1:
                    resolution = {
                        "status": "heuristic_unique_short_name",
                        "confidence": "probable",
                        "target_id": short_candidates[0],
                    }
                elif len(short_candidates) > 1:
                    resolution["status"] = "ambiguous"
                    resolution["confidence"] = "ambiguous"
            edge["resolution"] = resolution
            resolution_counts[resolution["status"]] += 1
            edge_count += 1
    return len(symbols), edge_count, resolution_counts


def _reset_graph_resolution(file_graphs: dict[str, dict[str, Any]]) -> None:
    for graph in file_graphs.values():
        for edge in graph["edges"]:
            edge["resolution"] = {
                "status": "unresolved",
                "confidence": "unknown",
                "target_id": None,
            }


def _valid_graphs(
    graphs: dict[str, dict[str, Any]],
    files: list[str],
    *,
    symbol_count: int | None = None,
    expected_edge_count: int | None = None,
    resolution_metrics: dict[str, int] | None = None,
) -> bool:
    if not isinstance(graphs, dict) or set(graphs) != set(files):
        return False
    symbols = []
    observed_edge_count = 0
    resolution_counts = {
        status: 0
        for status in (
            "exact_qualified",
            "lexical_scope",
            "import_binding",
            "heuristic_unique_short_name",
            "ambiguous",
            "unresolved",
        )
    }
    for path, graph in graphs.items():
        if (
            not isinstance(graph, dict)
            or graph.get("schema") != GRAPH_SCHEMA
            or not isinstance(graph.get("symbols"), list)
            or not isinstance(graph.get("edges"), list)
            or not isinstance(graph.get("diagnostics"), list)
        ):
            return False
        for symbol in graph["symbols"]:
            if (
                not isinstance(symbol, dict)
                or symbol.get("path") != path
                or not isinstance(symbol.get("id"), str)
                or not isinstance(symbol.get("name"), str)
                or not isinstance(symbol.get("qualified_name"), str)
                or not isinstance(symbol.get("start_line"), int)
                or not isinstance(symbol.get("end_line"), int)
            ):
                return False
            symbols.append(symbol)
        for edge in graph["edges"]:
            resolution = edge.get("resolution")
            if (
                not isinstance(edge, dict)
                or edge.get("path") != path
                or not isinstance(edge.get("source_id"), str)
                or not isinstance(edge.get("target"), str)
                or not isinstance(edge.get("kind"), str)
                or not isinstance(edge.get("line"), int)
                or not isinstance(edge.get("extraction_confidence"), str)
                or not isinstance(resolution, dict)
                or resolution.get("status") not in {
                    "exact_qualified",
                    "lexical_scope",
                    "import_binding",
                    "heuristic_unique_short_name",
                    "ambiguous",
                    "unresolved",
                }
                or resolution.get("confidence")
                not in {"strong", "probable", "ambiguous", "unknown"}
                or (
                    resolution.get("target_id") is not None
                    and not isinstance(resolution.get("target_id"), str)
                )
            ):
                return False
            status = resolution["status"]
            resolution_counts[status] = resolution_counts.get(status, 0) + 1
            observed_edge_count += 1
    symbol_ids = [symbol["id"] for symbol in symbols]
    return (
        len(set(symbol_ids)) == len(symbol_ids)
        and (symbol_count is None or symbol_count == len(symbols))
        and (expected_edge_count is None or expected_edge_count == observed_edge_count)
        and (resolution_metrics is None or resolution_metrics == resolution_counts)
    )


class RedisError(RuntimeError):
    pass


class RedisClient:
    """Tiny RESP2 client with pipelined commands over a single connection."""

    def __init__(self, url: str, timeout: float = 3.0):
        parsed = urlparse(url)
        if parsed.scheme != "redis" or parsed.username or parsed.password:
            raise RedisError("URL must be redis://host:port/db with no authentication")
        if parsed.query or parsed.fragment or parsed.path.count("/") > 1:
            raise RedisError("unsupported Redis URL")
        try:
            self.db = int(parsed.path.lstrip("/") or "0")
        except ValueError as exc:
            raise RedisError("Redis database must be an integer") from exc
        if self.db < 0:
            raise RedisError("Redis database must be non-negative")
        self.host = unquote(parsed.hostname or "localhost")
        self.port = parsed.port or 6379
        self.timeout = timeout

    @staticmethod
    def _request(parts: tuple[bytes, ...]) -> bytes:
        body = b"".join(b"$%d\r\n%s\r\n" % (len(part), part) for part in parts)
        return b"*%d\r\n" % len(parts) + body

    @staticmethod
    def _read(stream: Any) -> Any:
        marker = stream.read(1)
        if not marker:
            raise RedisError("Redis closed the connection")
        line = stream.readline()
        if not line.endswith(b"\r\n"):
            raise RedisError("malformed RESP2 response")
        payload = line[:-2]
        if marker == b"+":
            return payload.decode("utf-8")
        if marker == b"-":
            raise RedisError(f"Redis error: {payload.decode('utf-8', 'replace')}")
        if marker == b":":
            return int(payload)
        if marker == b"$":
            length = int(payload)
            if length == -1:
                return None
            value = stream.read(length)
            if len(value) != length or stream.read(2) != b"\r\n":
                raise RedisError("truncated RESP2 bulk string")
            return value
        if marker == b"*":
            length = int(payload)
            return None if length == -1 else [RedisClient._read(stream) for _ in range(length)]
        raise RedisError("unsupported RESP2 response type")

    def execute(self, command: str, *args: str | bytes) -> Any:
        return self.execute_many([(command, *args)])[0]

    def execute_many(self, commands: list[tuple[str | bytes, ...]]) -> list[Any]:
        """Execute a pipeline over one connection and return ordered responses."""
        encoded = []
        for command in commands:
            encoded.append(
                tuple(part if isinstance(part, bytes) else part.encode("utf-8") for part in command)
            )
        try:
            with socket.create_connection((self.host, self.port), self.timeout) as conn:
                conn.settimeout(self.timeout)
                stream = conn.makefile("rb")
                if self.db:
                    conn.sendall(self._request((b"SELECT", str(self.db).encode())))
                    if self._read(stream) != "OK":
                        raise RedisError("Redis SELECT failed")
                conn.sendall(b"".join(self._request(parts) for parts in encoded))
                return [self._read(stream) for _ in encoded]
        except (OSError, ValueError) as exc:
            raise RedisError(f"Redis connection/protocol failure: {exc}") from exc


def _json_load(raw: bytes | None) -> Any:
    if raw is None:
        return None
    return json.loads(raw.decode("utf-8"))


def _valid_chunk_value(raw: bytes | None) -> bool:
    try:
        chunk = _json_load(raw)
        if not isinstance(chunk, dict):
            return False
        vector = chunk.get("vector")
        if not isinstance(vector, str):
            return False
        decode_vector(vector)
        return (
            isinstance(chunk.get("id"), str)
            and isinstance(chunk.get("path"), str)
            and isinstance(chunk.get("start_line"), int)
            and isinstance(chunk.get("end_line"), int)
            and isinstance(chunk.get("text"), str)
            and isinstance(chunk.get("tokens"), list)
            and all(isinstance(token, str) for token in chunk["tokens"])
        )
    except (KeyError, TypeError, ValueError, UnicodeError, json.JSONDecodeError):
        return False


def _validate_chunk(key: str, raw: bytes | None, *, owner_path: str) -> bool:
    if not _valid_chunk_value(raw):
        return False
    try:
        chunk = _json_load(raw)
        start_line = chunk["start_line"]
        end_line = chunk["end_line"]
        text = chunk["text"]
        expected_id = hashlib.sha256(
            f"{owner_path}\0{start_line}\0{end_line}\0".encode() + text.encode()
        ).hexdigest()[:24]
        stored_vector = decode_vector(chunk["vector"])
        expected_vector = embedding(text)
        return (
            chunk["path"] == owner_path
            and start_line >= 1
            and end_line >= start_line
            and chunk["id"] == expected_id
            and key.split(":chunk:", 1)[-1].split(":", 1)[0] == expected_id
            and chunk["tokens"] == sorted(set(tokenize(text)))
            and all(
                math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-7)
                for a, b in zip(stored_vector, expected_vector, strict=True)
            )
        )
    except (KeyError, TypeError, ValueError, UnicodeError, json.JSONDecodeError):
        return False


def _graph_identity(path: str, file_hash: str) -> str:
    return hashlib.sha256(
        f"{GRAPH_SCHEMA}\0{GRAPH_RECORD_SCHEMA}\0{path}\0{file_hash}".encode()
    ).hexdigest()


def _graph_key(prefix: str, path: str, file_hash: str) -> str:
    return f"{prefix}:graph:{_graph_identity(path, file_hash)}"


def _graph_payload(path: str, file_hash: str, graph: dict[str, Any]) -> str:
    return json.dumps(
        {
            "path": path,
            "file_hash": file_hash,
            "graph_schema": GRAPH_SCHEMA,
            "record_schema": GRAPH_RECORD_SCHEMA,
            "graph": graph,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _load_graph_record(
    key: str,
    raw: bytes | None,
    *,
    prefix: str,
    path: str,
    file_hash: str,
) -> dict[str, Any] | None:
    try:
        payload = _json_load(raw)
        if (
            not isinstance(payload, dict)
            or key != _graph_key(prefix, path, file_hash)
            or payload.get("path") != path
            or payload.get("file_hash") != file_hash
            or payload.get("graph_schema") != GRAPH_SCHEMA
            or payload.get("record_schema") != GRAPH_RECORD_SCHEMA
            or not isinstance(payload.get("graph"), dict)
            or not _valid_graphs({path: payload["graph"]}, [path])
        ):
            return None
        return payload["graph"]
    except (KeyError, TypeError, ValueError, UnicodeError, json.JSONDecodeError):
        return None


def _load_manifest_graphs(
    client: Any,
    manifest: dict[str, Any],
    root: Path,
    *,
    resolve: bool = False,
) -> dict[str, dict[str, Any]]:
    references = manifest.get("file_graphs")
    files = manifest.get("files")
    hashes = manifest.get("file_hashes")
    if (
        not isinstance(references, dict)
        or not isinstance(files, list)
        or not isinstance(hashes, dict)
    ):
        raise TypeError("manifest contains invalid graph references")
    keys = [references[path] for path in files]
    values = _mget_batched(client, keys)
    graphs = {}
    for path, key, raw in zip(files, keys, values, strict=True):
        graph = _load_graph_record(
            key,
            raw,
            prefix=namespace(root),
            path=path,
            file_hash=hashes[path],
        )
        if graph is None:
            raise ValueError("index contains malformed graph data; re-index required")
        graphs[path] = graph
    if resolve:
        _resolve_graph(graphs)
    return graphs


def _mget_batched(client: Any, keys: list[str]) -> list[Any]:
    values = []
    for start in range(0, len(keys), MGET_BATCH_SIZE):
        batch = keys[start : start + MGET_BATCH_SIZE]
        response = client.execute("MGET", *batch)
        if not isinstance(response, list) or len(response) != len(batch):
            raise ValueError("Redis returned an invalid MGET response")
        values.extend(response)
    return values


@contextlib.contextmanager
def _index_lock(client: Any, root: Path):
    lock_key = f"{namespace(root)}:index-lock"
    owner = secrets.token_hex(16)
    acquired = client.execute("SET", lock_key, owner, "NX", "PX", str(INDEX_LOCK_MS))
    if acquired != "OK":
        raise ValueError("another Project Memory indexer is active")
    try:
        yield owner
    finally:
        active_exception = sys.exc_info()[0] is not None
        try:
            client.execute(
                "EVAL",
                "if redis.call('get',KEYS[1]) == ARGV[1] then return redis.call('del',KEYS[1]) else return 0 end",
                "1",
                lock_key,
                owner,
            )
        except Exception:
            if not active_exception:
                raise


def _refresh_index_lock(client: Any, root: Path, owner: str) -> None:
    refreshed = client.execute(
        "EVAL",
        "if redis.call('get',KEYS[1]) == ARGV[1] then return redis.call('pexpire',KEYS[1],ARGV[2]) else return 0 end",
        "1",
        f"{namespace(root)}:index-lock",
        owner,
        str(INDEX_LOCK_MS),
    )
    if refreshed != 1:
        raise ValueError("Project Memory index lock ownership was lost")


def _activate_generation(client: Any, root: Path, owner: str, manifest_key: str) -> None:
    activated = client.execute(
        "EVAL",
        "if redis.call('get',KEYS[1]) == ARGV[1] then redis.call('set',KEYS[2],ARGV[2]); return 1 else return 0 end",
        "2",
        f"{namespace(root)}:index-lock",
        f"{namespace(root)}:active-generation",
        owner,
        manifest_key,
    )
    if activated != 1:
        raise ValueError("Project Memory index lock ownership was lost before activation")


def _registered_keys(client: Any, prefix: str) -> tuple[set[str], set[str], set[str]]:
    try:
        value = _json_load(client.execute("GET", f"{prefix}:chunk-registry"))
    except (ValueError, UnicodeError, json.JSONDecodeError):
        return set(), set(), set()
    if not isinstance(value, dict):
        return set(), set(), set()
    chunks = {
        key
        for key in value.get("chunk_keys", [])
        if isinstance(key, str) and key.startswith(prefix + ":chunk:")
    }
    manifests = {
        key
        for key in value.get("manifest_keys", [])
        if isinstance(key, str) and key.startswith(prefix + ":generation:")
    }
    graphs = {
        key
        for key in value.get("graph_keys", [])
        if isinstance(key, str) and key.startswith(prefix + ":graph:")
    }
    return chunks, manifests, graphs


def _execute_many(client: Any, commands: list[tuple[str | bytes, ...]]) -> list[Any]:
    pipeline = getattr(client, "execute_many", None)
    if pipeline is not None:
        return pipeline(commands)
    return [client.execute(command[0], *command[1:]) for command in commands]


def _active_manifest(client: Any, root: Path) -> dict[str, Any] | None:
    manifest_key = client.execute("GET", f"{namespace(root)}:active-generation")
    if manifest_key is None:
        return None
    if isinstance(manifest_key, bytes):
        manifest_key = manifest_key.decode("utf-8")
    if not isinstance(manifest_key, str) or not manifest_key.startswith(
        f"{namespace(root)}:generation:"
    ):
        return None
    return _json_load(client.execute("GET", manifest_key))


def _manifest_state(
    client: RedisClient, root: Path = ROOT, *, deep: bool = False
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    files = included_files(root)
    current = {
        "files": [path.relative_to(root).as_posix() for path in files],
        "fingerprint": corpus_fingerprint(files, root),
    }
    try:
        manifest = _active_manifest(client, root)
        if not isinstance(manifest, dict):
            return current, None
        required = {
            "schema",
            "namespace",
            "embedding_id",
            "dimensions",
            "generation",
            "files",
            "file_hashes",
            "chunk_count",
            "fingerprint",
            "chunk_keys",
        }
        if not required.issubset(manifest) or manifest["schema"] != SCHEMA:
            return current, None
        if (
            manifest["namespace"] != namespace(root)
            or manifest["embedding_id"] != EMBEDDING_ID
            or manifest["dimensions"] != DIMENSIONS
        ):
            return current, None
        generation = manifest["generation"]
        if (
            not isinstance(generation, str)
            or not generation.startswith(manifest["fingerprint"])
            or not isinstance(manifest["file_hashes"], dict)
            or set(manifest["file_hashes"]) != set(manifest["files"])
        ):
            return current, None
        keys = manifest["chunk_keys"]
        if (
            not isinstance(keys, list)
            or len(keys) != manifest["chunk_count"]
            or len(set(keys)) != len(keys)
            or any(
                not isinstance(k, str) or not k.startswith(namespace(root) + ":chunk:")
                for k in keys
            )
        ):
            return current, None
        file_chunks = manifest.get("file_chunks")
        if file_chunks is not None and (
            not isinstance(file_chunks, dict)
            or set(file_chunks) != set(manifest["files"])
            or any(not isinstance(value, list) for value in file_chunks.values())
            or [key for path in manifest["files"] for key in file_chunks[path]] != keys
        ):
            return current, None
        bundle_version = manifest.get("bundle_version", "2.0.0")
        if isinstance(bundle_version, str) and bundle_version.startswith("2.2"):
            embedded_graphs = manifest.get("file_graphs")
            if not isinstance(embedded_graphs, dict) or not _valid_graphs(
                embedded_graphs,
                manifest["files"],
                symbol_count=manifest.get("symbol_count"),
                expected_edge_count=manifest.get("edge_count"),
                resolution_metrics=manifest.get("resolution_metrics"),
            ):
                return current, None
            if deep:
                expected_graphs = {
                    path.relative_to(root).as_posix(): extract_file_graph(
                        path.relative_to(root).as_posix(), path.read_text(encoding="utf-8")
                    )
                    for path in files
                }
                _resolve_graph(expected_graphs)
                if expected_graphs != embedded_graphs:
                    return current, None
        elif isinstance(bundle_version, str) and bundle_version.startswith(
            CONTENT_ADDRESSED_GRAPH_BUNDLE_SERIES
        ):
            graph_references = manifest.get("file_graphs")
            if (
                not isinstance(graph_references, dict)
                or set(graph_references) != set(manifest["files"])
                or any(
                    not isinstance(key, str)
                    or key
                    != _graph_key(namespace(root), path, manifest["file_hashes"][path])
                    for path, key in graph_references.items()
                )
                or not isinstance(manifest.get("symbol_count"), int)
                or not isinstance(manifest.get("edge_count"), int)
                or not isinstance(manifest.get("resolution_metrics"), dict)
            ):
                return current, None
            if deep:
                stored_graphs = _load_manifest_graphs(client, manifest, root)
                expected_graphs = {
                    path.relative_to(root).as_posix(): extract_file_graph(
                        path.relative_to(root).as_posix(), path.read_text(encoding="utf-8")
                    )
                    for path in files
                }
                if expected_graphs != stored_graphs:
                    return current, None
                _resolve_graph(stored_graphs)
                if not _valid_graphs(
                    stored_graphs,
                    manifest["files"],
                    symbol_count=manifest["symbol_count"],
                    expected_edge_count=manifest["edge_count"],
                    resolution_metrics=manifest["resolution_metrics"],
                ):
                    return current, None
        if deep and keys:
            values = _mget_batched(client, keys)
            if any(value is None for value in values):
                return current, None
            owner_by_key = (
                {
                    key: owner
                    for owner, owned_keys in file_chunks.items()
                    for key in owned_keys
                }
                if isinstance(file_chunks, dict)
                else {}
            )
            for key, value in zip(keys, values, strict=True):
                owner = owner_by_key.get(key)
                if owner is None or not _validate_chunk(key, value, owner_path=owner):
                    return current, None
        return current, manifest
    except (KeyError, TypeError, ValueError, UnicodeError, json.JSONDecodeError):
        return current, None


def status(client: RedisClient, root: Path = ROOT) -> tuple[dict[str, Any], bool]:
    current, manifest = _manifest_state(client, root, deep=False)
    fresh = bool(
        manifest
        and manifest["fingerprint"] == current["fingerprint"]
        and manifest["files"] == current["files"]
    )
    return {
        "status": "fresh" if fresh else ("stale" if manifest else "missing_or_invalid"),
        "fresh": fresh,
        "namespace": namespace(root),
        "schema": SCHEMA,
        "current_fingerprint": current["fingerprint"],
        "manifest": manifest,
    }, fresh


def validate(client: RedisClient, root: Path = ROOT, *, deep: bool = False) -> tuple[dict[str, Any], bool]:
    current, manifest = _manifest_state(client, root, deep=deep)
    fresh = bool(
        manifest
        and manifest["fingerprint"] == current["fingerprint"]
        and manifest["files"] == current["files"]
    )
    return {
        "status": "fresh" if fresh else ("stale" if manifest else "missing_or_invalid"),
        "fresh": fresh,
        "namespace": namespace(root),
        "schema": SCHEMA,
        "current_fingerprint": current["fingerprint"],
        "manifest": manifest,
        "validation": "passed" if fresh else "failed",
        "deep": deep,
    }, fresh


def _build_index_locked(
    client: RedisClient, root: Path, *, build_id: str, repair_deep: bool
) -> dict[str, Any]:
    started = time.monotonic()
    files = included_files(root)
    prefix = namespace(root)
    fingerprint = corpus_fingerprint(files, root)
    generation = f"{fingerprint}:{BUNDLE_VERSION}:{build_id[:16]}"
    manifest_key = f"{prefix}:generation:{generation}:manifest"
    old_manifest: dict[str, Any] | None = None
    with contextlib.suppress(ValueError, UnicodeError, json.JSONDecodeError):
        old_manifest = _active_manifest(client, root)
    registered_chunks, registered_manifests, registered_graphs = _registered_keys(client, prefix)
    old_hashes = old_manifest.get("file_hashes", {}) if isinstance(old_manifest, dict) else {}
    old_file_chunks = old_manifest.get("file_chunks", {}) if isinstance(old_manifest, dict) else {}
    old_file_graph_refs = old_manifest.get("file_graphs", {}) if isinstance(old_manifest, dict) else {}
    old_file_graphs: dict[str, dict[str, Any]] = {}
    if isinstance(old_file_graph_refs, dict):
        embedded = {
            path: graph
            for path, graph in old_file_graph_refs.items()
            if isinstance(path, str) and isinstance(graph, dict)
        }
        old_file_graphs.update(embedded)
        referenced = [
            (path, key)
            for path, key in old_file_graph_refs.items()
            if isinstance(path, str) and isinstance(key, str)
        ]
        referenced_values = _mget_batched(client, [key for _, key in referenced])
        for (path, key), raw in zip(referenced, referenced_values, strict=True):
            file_hash = old_hashes.get(path)
            if not isinstance(file_hash, str):
                continue
            graph = _load_graph_record(
                key, raw, prefix=prefix, path=path, file_hash=file_hash
            )
            if graph is not None:
                old_file_graphs[path] = graph
    can_reuse_files = (
        isinstance(old_hashes, dict)
        and isinstance(old_file_chunks, dict)
        and set(old_hashes) == set(old_file_chunks)
    )
    file_hashes = {
        path.relative_to(root).as_posix(): file_fingerprint(path) for path in files
    }
    current_names = set(file_hashes)
    old_names = set(old_hashes) if isinstance(old_hashes, dict) else set()
    unchanged = {
        name
        for name, digest in file_hashes.items()
        if can_reuse_files and old_hashes.get(name) == digest
    }
    new = current_names - old_names
    changed = current_names & old_names - unchanged
    deleted = old_names - current_names

    reusable_keys = [key for name in sorted(unchanged) for key in old_file_chunks[name]]
    reusable_values = _mget_batched(client, reusable_keys)
    reusable_owners = {
        key: owner for owner in unchanged for key in old_file_chunks[owner]
    }
    invalid_reusable = {
        key
        for key, value in zip(reusable_keys, reusable_values, strict=True)
        if not (
            _validate_chunk(key, value, owner_path=reusable_owners[key])
            if repair_deep
            else _valid_chunk_value(value)
        )
    }
    repair = {
        name for name in unchanged if invalid_reusable.intersection(old_file_chunks[name])
    }
    unchanged -= repair
    changed |= repair
    _refresh_index_lock(client, root, build_id)

    file_chunks: dict[str, list[str]] = {}
    file_graphs: dict[str, dict[str, Any]] = {}
    generated_payloads: dict[str, str] = {}
    generated_count = 0
    graph_generated_files = 0
    graph_repaired_files = 0
    graph_force_write_paths: set[str] = set()
    for path in files:
        relative = path.relative_to(root).as_posix()
        text: str | None = None
        expected_graph: dict[str, Any] | None = None
        chunks_reusable = relative in unchanged
        graph_reusable = (
            old_hashes.get(relative) == file_hashes[relative]
            and isinstance(old_file_graphs.get(relative), dict)
            and old_file_graphs[relative].get("schema") == GRAPH_SCHEMA
        )
        if graph_reusable and repair_deep:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(f"indexed file is not UTF-8: {relative}") from exc
            expected_graph = extract_file_graph(relative, text)
            _reset_graph_resolution({relative: expected_graph})
            if old_file_graphs[relative] != expected_graph:
                graph_reusable = False
                graph_repaired_files += 1
                graph_force_write_paths.add(relative)
        if chunks_reusable:
            file_chunks[relative] = list(old_file_chunks[relative])
        if graph_reusable:
            file_graphs[relative] = old_file_graphs[relative]
        if chunks_reusable and graph_reusable:
            continue
        if text is None:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(f"indexed file is not UTF-8: {relative}") from exc
        if not chunks_reusable:
            keys = []
            for chunk in split_chunks(relative, text):
                key = f"{prefix}:chunk:{chunk.chunk_id}:{build_id[:16]}"
                payload = {
                    "id": chunk.chunk_id,
                    "path": chunk.path,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "text": chunk.text,
                    "tokens": sorted(set(tokenize(chunk.text))),
                    "vector": encode_vector(embedding(chunk.text)),
                }
                generated_payloads[key] = json.dumps(
                    payload, sort_keys=True, separators=(",", ":")
                )
                keys.append(key)
                generated_count += 1
            file_chunks[relative] = keys
        if not graph_reusable:
            file_graphs[relative] = expected_graph or extract_file_graph(relative, text)
            graph_generated_files += 1

    _reset_graph_resolution(file_graphs)
    resolved_graphs = json.loads(json.dumps(file_graphs))
    symbol_count, edge_count, resolution_metrics = _resolve_graph(resolved_graphs)

    file_graph_references = {
        path: _graph_key(prefix, path, file_hashes[path]) for path in file_hashes
    }
    graph_payloads = {
        file_graph_references[path]: _graph_payload(path, file_hashes[path], file_graphs[path])
        for path in file_hashes
    }

    chunk_keys = [key for path in file_hashes for key in file_chunks[path]]
    writes: list[tuple[str | bytes, ...]] = []
    generated_keys = list(generated_payloads)
    existing = _mget_batched(client, generated_keys)
    for key, value in zip(generated_keys, existing, strict=True):
        if not _valid_chunk_value(value):
            writes.append(("SET", key, generated_payloads[key]))
    graph_keys = list(file_graph_references.values())
    existing_graph_values = _mget_batched(client, graph_keys)
    graph_write_count = 0
    for path, key, value in zip(file_hashes, graph_keys, existing_graph_values, strict=True):
        if path in graph_force_write_paths or (
            _load_graph_record(
                key,
                value,
                prefix=prefix,
                path=path,
                file_hash=file_hashes[path],
            )
            is None
        ):
            writes.append(("SET", key, graph_payloads[key]))
            graph_write_count += 1
    old_chunk_keys = set(old_manifest.get("chunk_keys", [])) if old_manifest else set()
    deleted_chunk_keys = (old_chunk_keys | registered_chunks) - set(chunk_keys)
    old_graph_keys = {
        key for key in old_file_graph_refs.values() if isinstance(key, str)
    } if isinstance(old_file_graph_refs, dict) else set()
    deleted_graph_keys = (old_graph_keys | registered_graphs) - set(graph_keys)
    reused_chunk_count = len(chunk_keys) - generated_count
    metrics = {
        "files": {
            "total": len(file_hashes),
            "new": len(new),
            "changed": len(changed),
            "unchanged": len(unchanged),
            "deleted": len(deleted),
        },
        "chunks": {
            "total": len(chunk_keys),
            "generated": generated_count,
            "reused": reused_chunk_count,
            "deleted": len(deleted_chunk_keys),
        },
        "graph": {
            "symbols": symbol_count,
            "edges": edge_count,
            "resolution": resolution_metrics,
            "generated_files": graph_generated_files,
            "repaired_files": graph_repaired_files,
            "reused_files": len(file_graphs) - graph_generated_files,
            "written_records": graph_write_count,
            "deleted_records": len(deleted_graph_keys),
        },
    }
    manifest = {
        "schema": SCHEMA,
        "bundle_version": BUNDLE_VERSION,
        "namespace": prefix,
        "embedding_id": EMBEDDING_ID,
        "dimensions": DIMENSIONS,
        "generation": generation,
        "files": [path.relative_to(root).as_posix() for path in files],
        "file_hashes": file_hashes,
        "file_chunks": file_chunks,
        "file_graphs": file_graph_references,
        "symbol_count": symbol_count,
        "edge_count": edge_count,
        "resolution_metrics": resolution_metrics,
        "chunk_count": len(chunk_keys),
        "fingerprint": fingerprint,
        "chunk_keys": chunk_keys,
    }
    writes.append(
        ("SET", manifest_key, json.dumps(manifest, sort_keys=True, separators=(",", ":")))
    )
    # Register every old and staged key before any staged write can create an orphan.
    staged_chunk_keys = registered_chunks | set(chunk_keys)
    staged_graph_keys = registered_graphs | set(graph_keys)
    staged_manifest_keys = registered_manifests | {manifest_key}
    if old_manifest:
        staged_chunk_keys.update(old_manifest.get("chunk_keys", []))
        if isinstance(old_file_graph_refs, dict):
            staged_graph_keys.update(
                key for key in old_file_graph_refs.values() if isinstance(key, str)
            )
        old_generation = old_manifest.get("generation")
        if isinstance(old_generation, str):
            staged_manifest_keys.add(f"{prefix}:generation:{old_generation}:manifest")
    client.execute(
        "SET",
        f"{prefix}:chunk-registry",
        json.dumps(
            {
                "chunk_keys": sorted(staged_chunk_keys),
                "graph_keys": sorted(staged_graph_keys),
                "manifest_keys": sorted(staged_manifest_keys),
            },
            separators=(",", ":"),
        ),
    )
    _refresh_index_lock(client, root, build_id)
    _execute_many(client, writes)
    staged_values = _mget_batched(client, chunk_keys)
    if any(
        not _valid_chunk_value(value) for value in staged_values
    ):
        raise ValueError("staged generation failed chunk validation")
    staged_graph_values = _mget_batched(client, graph_keys)
    for path, key, raw in zip(file_hashes, graph_keys, staged_graph_values, strict=True):
        if (
            _load_graph_record(
                key,
                raw,
                prefix=prefix,
                path=path,
                file_hash=file_hashes[path],
            )
            is None
        ):
            raise ValueError("staged generation failed graph validation")
    write_finished = time.monotonic()

    final_files = included_files(root)
    final_names = [path.relative_to(root).as_posix() for path in final_files]
    final_fingerprint = corpus_fingerprint(final_files, root)
    if final_names != manifest["files"] or final_fingerprint != fingerprint:
        raise ValueError("repository changed during indexing; staged generation was not activated")
    _refresh_index_lock(client, root, build_id)

    # Commit point: readers continue using the old complete generation until this SET succeeds.
    activation_started = time.monotonic()
    _activate_generation(client, root, build_id, manifest_key)
    activation_finished = time.monotonic()

    # Garbage collection happens only after the new generation is active.
    garbage_collection_started = time.monotonic()
    obsolete = sorted(deleted_chunk_keys)
    obsolete_graphs = sorted(deleted_graph_keys)
    obsolete_manifests = sorted(staged_manifest_keys - {manifest_key})
    if obsolete:
        client.execute("DEL", *obsolete)
    if obsolete_graphs:
        client.execute("DEL", *obsolete_graphs)
    if obsolete_manifests:
        client.execute("DEL", *obsolete_manifests)
    garbage_collection_finished = time.monotonic()

    # Shrink the registry only after obsolete keys have actually been removed.
    client.execute(
        "SET",
        f"{prefix}:chunk-registry",
        json.dumps(
            {
                "chunk_keys": chunk_keys,
                "graph_keys": graph_keys,
                "manifest_keys": [manifest_key],
            },
            separators=(",", ":"),
        ),
    )
    finished = time.monotonic()
    metrics.update(
        {
            "indexing_ms": round((write_finished - started) * 1000),
            "activation_ms": round((activation_finished - activation_started) * 1000),
            "garbage_collection_ms": round(
                (garbage_collection_finished - garbage_collection_started) * 1000
            ),
            "total_ms": round((finished - started) * 1000),
        }
    )
    return {"manifest": manifest, "metrics": metrics}


def build_index(
    client: RedisClient, root: Path = ROOT, *, repair_deep: bool = False
) -> dict[str, Any]:
    with _index_lock(client, root) as build_id:
        return _build_index_locked(
            client, root, build_id=build_id, repair_deep=repair_deep
        )


def search(client: RedisClient, query: str, limit: int, root: Path = ROOT) -> dict[str, Any]:
    state, fresh = status(client, root)
    if not fresh:
        raise ValueError(f"index is {state['status']}; run index before search")
    manifest = state["manifest"]
    values = _mget_batched(client, manifest["chunk_keys"])
    qvec = embedding(query)
    qtokens = set(tokenize(query))
    query_normalized = query.strip().lower()
    symbols_by_path: dict[str, list[dict[str, Any]]] = {}
    for path, graph in _load_manifest_graphs(client, manifest, root).items():
        symbols_by_path[path] = graph.get("symbols", [])
    results = []
    for raw in values:
        try:
            chunk = _json_load(raw)
            vector = decode_vector(chunk["vector"])
            cosine = sum(a * b for a, b in zip(qvec, vector, strict=True))
            tokens = set(chunk["tokens"])
            lexical = len(qtokens & tokens) / len(qtokens) if qtokens else 0.0
            matching_symbols = [
                symbol
                for symbol in symbols_by_path.get(chunk["path"], [])
                if symbol["start_line"] <= chunk["end_line"]
                and symbol["end_line"] >= chunk["start_line"]
                and (
                    symbol["name"].lower() == query_normalized
                    or symbol["qualified_name"].lower() == query_normalized
                    or symbol["name"].lower() in qtokens
                )
            ]
            symbol_boost = 0.2 if matching_symbols else 0.0
            score = 0.72 * cosine + 0.18 * lexical + symbol_boost
            results.append(
                {
                    "path": chunk["path"],
                    "start_line": chunk["start_line"],
                    "end_line": chunk["end_line"],
                    "score": round(score, 6),
                    "pointer": f"{chunk['path']}:{chunk['start_line']}-{chunk['end_line']}",
                    "text": chunk["text"],
                    "matches": [
                        {
                            "type": "symbol",
                            "name": symbol["qualified_name"],
                            "kind": symbol["kind"],
                            "parser": symbol["parser"],
                            "confidence": symbol["confidence"],
                        }
                        for symbol in matching_symbols
                    ],
                    "explanation": {
                        "cosine": round(cosine, 6),
                        "lexical": round(lexical, 6),
                        "symbol_boost": symbol_boost,
                    },
                }
            )
        except (KeyError, TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("index contains malformed chunk data; re-index required") from exc
    results.sort(key=lambda item: (-item["score"], item["path"], item["start_line"]))
    return {"query": query, "fresh": True, "results": results[:limit]}


def symbols(client: RedisClient, query: str, limit: int, root: Path = ROOT) -> dict[str, Any]:
    state, fresh = status(client, root)
    if not fresh:
        raise ValueError(f"index is {state['status']}; run index before symbol lookup")
    normalized = query.strip().lower()
    results = []
    graphs = _load_manifest_graphs(client, state["manifest"], root)
    for graph in graphs.values():
        for symbol in graph.get("symbols", []):
            name = symbol["name"].lower()
            qualified = symbol["qualified_name"].lower()
            rank = 0 if normalized in {name, qualified} else (1 if normalized in qualified else 2)
            if normalized in name or normalized in qualified:
                results.append(
                    {
                        **symbol,
                        "pointer": f"{symbol['path']}:{symbol['start_line']}-{symbol['end_line']}",
                        "explanation": "exact symbol" if rank == 0 else "symbol substring",
                        "_rank": rank,
                    }
                )
    results.sort(key=lambda item: (item["_rank"], item["qualified_name"], item["path"]))
    for result in results:
        result.pop("_rank")
    return {"query": query, "fresh": True, "results": results[:limit]}


def impact(client: RedisClient, query: str, limit: int, root: Path = ROOT) -> dict[str, Any]:
    state, fresh = status(client, root)
    if not fresh:
        raise ValueError(f"index is {state['status']}; run index before impact lookup")
    manifest = state["manifest"]
    graphs = _load_manifest_graphs(client, manifest, root, resolve=True)
    all_symbols = [
        symbol
        for graph in graphs.values()
        for symbol in graph.get("symbols", [])
    ]
    normalized = query.strip().lower()
    targets = {
        symbol["id"]
        for symbol in all_symbols
        if normalized in {symbol["name"].lower(), symbol["qualified_name"].lower()}
    }
    symbols_by_id = {symbol["id"]: symbol for symbol in all_symbols}
    results = []
    for graph in graphs.values():
        for edge in graph.get("edges", []):
            edge_target = edge.get("target", "").lower()
            short_target = re.split(r"[.:]+", edge_target)[-1]
            resolution = edge["resolution"]
            resolved_match = resolution.get("target_id") in targets
            name_match = normalized in {edge_target, short_target}
            if not resolved_match and not name_match:
                continue
            source = symbols_by_id.get(edge["source_id"])
            if source:
                results.append(
                    {
                        "source": source,
                        "edge": edge,
                        "pointer": f"{edge['path']}:{edge['line']}",
                        "explanation": (
                            f"{edge['kind']} edge via {edge['parser']} "
                            f"(extraction={edge['extraction_confidence']}; "
                            f"resolution={resolution['status']}/{resolution['confidence']}; "
                            f"{'target id match' if resolved_match else 'exact target name'})"
                        ),
                    }
                )
    results.sort(key=lambda item: (item["edge"]["kind"], item["pointer"]))
    distinct_results = []
    seen_sources = set()
    for result in results:
        source_id = result["source"]["id"]
        if source_id in seen_sources:
            continue
        seen_sources.add(source_id)
        distinct_results.append(result)
    return {
        "query": query,
        "fresh": True,
        "target_ids": sorted(targets),
        "results": distinct_results[:limit],
    }


def _path_symbol(symbols: list[dict[str, Any]], query: str, role: str) -> dict[str, Any]:
    normalized = query.strip().lower()
    if not normalized:
        raise ValueError(f"{role} symbol query must not be empty")

    qualified_matches = [
        symbol
        for symbol in symbols
        if normalized in {symbol["id"].lower(), symbol["qualified_name"].lower()}
    ]
    if len(qualified_matches) == 1:
        return qualified_matches[0]
    matches = qualified_matches or [
        symbol for symbol in symbols if symbol["name"].lower() == normalized
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"{role} symbol not found: {query!r}")

    candidates = sorted(
        f"{symbol['qualified_name']} ({symbol['path']}:{symbol['start_line']})"
        for symbol in matches
    )
    preview = ", ".join(candidates[:8])
    if len(candidates) > 8:
        preview += f", ... ({len(candidates)} matches)"
    raise ValueError(
        f"{role} symbol is ambiguous: {query!r}; use a qualified name or symbol id; "
        f"candidates: {preview}"
    )


def _symbol_result(symbol: dict[str, Any]) -> dict[str, Any]:
    return {
        **symbol,
        "pointer": f"{symbol['path']}:{symbol['start_line']}-{symbol['end_line']}",
    }


def shortest_dependency_path(
    graphs: dict[str, dict[str, Any]],
    source_query: str,
    target_query: str,
    *,
    direction: str = "forward",
    edge_kinds: Iterable[str] | None = None,
    max_depth: int = DEFAULT_PATH_MAX_DEPTH,
    include_probable: bool = False,
) -> dict[str, Any]:
    """Find a deterministic minimum-hop path over resolved dependency edges."""
    if direction not in PATH_DIRECTIONS:
        raise ValueError(f"direction must be one of: {', '.join(PATH_DIRECTIONS)}")
    if type(max_depth) is not int or not 0 <= max_depth <= MAX_PATH_DEPTH:
        raise ValueError(f"max_depth must be between 0 and {MAX_PATH_DEPTH}")
    if edge_kinds is None:
        selected_edge_kinds = set(DEPENDENCY_EDGE_KINDS)
    else:
        if isinstance(edge_kinds, str):
            raise ValueError("edge_kinds must be an iterable of edge kind names")
        selected_edge_kinds = set(edge_kinds)
        if not selected_edge_kinds:
            raise ValueError("edge_kinds must contain at least one edge kind")
        unknown_edge_kinds = selected_edge_kinds - set(DEPENDENCY_EDGE_KINDS)
        if unknown_edge_kinds:
            raise ValueError(
                "unsupported edge kinds: " + ", ".join(sorted(unknown_edge_kinds))
            )

    _resolve_graph(graphs)
    all_symbols = [
        symbol
        for graph in graphs.values()
        for symbol in graph.get("symbols", [])
    ]
    symbols_by_id = {symbol["id"]: symbol for symbol in all_symbols}
    source = _path_symbol(all_symbols, source_query, "source")
    target = _path_symbol(all_symbols, target_query, "target")
    source_id = source["id"]
    target_id = target["id"]

    adjacency: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    eligible_edges = 0
    for graph in graphs.values():
        for edge in graph.get("edges", []):
            if edge.get("kind") not in selected_edge_kinds:
                continue
            resolution = edge.get("resolution", {})
            resolved_target_id = resolution.get("target_id")
            if not isinstance(resolved_target_id, str):
                continue
            confidence = resolution.get("confidence")
            if confidence != "strong" and not (
                include_probable and confidence == "probable"
            ):
                continue
            resolved_source_id = edge.get("source_id")
            if (
                resolved_source_id not in symbols_by_id
                or resolved_target_id not in symbols_by_id
            ):
                continue
            if direction == "forward":
                start_id, end_id = resolved_source_id, resolved_target_id
            else:
                start_id, end_id = resolved_target_id, resolved_source_id
            adjacency.setdefault(start_id, []).append((end_id, edge))
            eligible_edges += 1

    def adjacency_key(item: tuple[str, dict[str, Any]]) -> tuple[Any, ...]:
        neighbor_id, edge = item
        neighbor = symbols_by_id[neighbor_id]
        return (
            neighbor["qualified_name"].lower(),
            neighbor["path"],
            neighbor["start_line"],
            edge["kind"],
            edge["path"],
            edge["line"],
        )

    for neighbors in adjacency.values():
        neighbors.sort(key=adjacency_key)

    depths = {source_id: 0}
    previous: dict[str, tuple[str, dict[str, Any]]] = {}
    frontier = deque([source_id])
    while frontier:
        current_id = frontier.popleft()
        if current_id == target_id:
            break
        if depths[current_id] >= max_depth:
            continue
        for neighbor_id, edge in adjacency.get(current_id, []):
            if neighbor_id in depths:
                continue
            depths[neighbor_id] = depths[current_id] + 1
            previous[neighbor_id] = (current_id, edge)
            frontier.append(neighbor_id)

    result = {
        "source_query": source_query,
        "target_query": target_query,
        "source": _symbol_result(source),
        "target": _symbol_result(target),
        "direction": direction,
        "edge_kinds": sorted(selected_edge_kinds),
        "include_probable": include_probable,
        "max_depth": max_depth,
        "algorithm": "breadth-first-search",
        "eligible_edges": eligible_edges,
        "visited_symbols": len(depths),
    }
    if target_id not in depths:
        return {
            **result,
            "found": False,
            "hop_count": None,
            "path": [],
            "steps": [],
        }

    path_ids = [target_id]
    path_edges = []
    while path_ids[-1] != source_id:
        parent_id, edge = previous[path_ids[-1]]
        path_ids.append(parent_id)
        path_edges.append(edge)
    path_ids.reverse()
    path_edges.reverse()

    steps = []
    for start_id, end_id, edge in zip(
        path_ids[:-1], path_ids[1:], path_edges, strict=True
    ):
        steps.append(
            {
                "from": _symbol_result(symbols_by_id[start_id]),
                "to": _symbol_result(symbols_by_id[end_id]),
                "edge": {**edge, "resolution": dict(edge["resolution"])},
                "pointer": f"{edge['path']}:{edge['line']}",
            }
        )
    return {
        **result,
        "found": True,
        "hop_count": len(path_edges),
        "path": [_symbol_result(symbols_by_id[symbol_id]) for symbol_id in path_ids],
        "steps": steps,
    }


def dependency_path(
    client: RedisClient,
    source_query: str,
    target_query: str,
    *,
    direction: str = "forward",
    edge_kinds: Iterable[str] | None = None,
    max_depth: int = DEFAULT_PATH_MAX_DEPTH,
    include_probable: bool = False,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Load the current graph and return a read-only shortest dependency path."""
    state, fresh = status(client, root)
    if not fresh:
        raise ValueError(f"index is {state['status']}; run index before path lookup")
    graphs = _load_manifest_graphs(client, state["manifest"], root)
    return {
        "fresh": True,
        **shortest_dependency_path(
            graphs,
            source_query,
            target_query,
            direction=direction,
            edge_kinds=edge_kinds,
            max_depth=max_depth,
            include_probable=include_probable,
        ),
    }


def evaluate(client: RedisClient, limit: int, root: Path = ROOT) -> dict[str, Any]:
    cases = project_config(root).get("evaluation_queries", [])
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{CONFIG_FILE} evaluation_queries must be a non-empty list")
    outcomes = []
    for case in cases:
        if (
            not isinstance(case, dict)
            or case.get("mode") not in {"search", "symbols", "impact"}
            or not isinstance(case.get("query"), str)
            or not isinstance(case.get("expected_paths"), list)
            or not all(isinstance(path, str) for path in case["expected_paths"])
        ):
            raise ValueError(f"{CONFIG_FILE} contains an invalid evaluation query")
        mode = case["mode"]
        response = {
            "search": search,
            "symbols": symbols,
            "impact": impact,
        }[mode](client, case["query"], limit, root)
        if mode in {"search", "symbols"}:
            returned_paths = {item["path"] for item in response["results"]}
        else:
            returned_paths = {item["source"]["path"] for item in response["results"]}
        missing = sorted(set(case["expected_paths"]) - returned_paths)
        outcomes.append(
            {
                "mode": mode,
                "query": case["query"],
                "expected_paths": case["expected_paths"],
                "returned_paths": sorted(returned_paths),
                "missing_paths": missing,
                "passed": not missing,
            }
        )
    passed = sum(outcome["passed"] for outcome in outcomes)
    return {
        "status": "passed" if passed == len(outcomes) else "failed",
        "cases": len(outcomes),
        "passed": passed,
        "recall_at_limit": passed / len(outcomes),
        "limit": limit,
        "outcomes": outcomes,
    }


def _clear_locked(client: RedisClient, root: Path) -> dict[str, Any]:
    prefix = namespace(root)
    registry_key = f"{prefix}:chunk-registry"
    active_key = f"{prefix}:active-generation"
    keys = [registry_key, active_key]
    for key in (registry_key, active_key):
        try:
            raw = client.execute("GET", key)
            if key == active_key and raw is not None:
                manifest_key = raw.decode() if isinstance(raw, bytes) else raw
                value = _json_load(client.execute("GET", manifest_key))
                keys.append(manifest_key)
            else:
                value = _json_load(raw)
            candidates = value.get("chunk_keys", []) if isinstance(value, dict) else []
            keys.extend(
                candidate
                for candidate in candidates
                if isinstance(candidate, str) and candidate.startswith(prefix + ":chunk:")
            )
            if isinstance(value, dict):
                graph_candidates = list(value.get("graph_keys", []))
                file_graphs = value.get("file_graphs", {})
                if isinstance(file_graphs, dict):
                    graph_candidates.extend(file_graphs.values())
                keys.extend(
                    candidate
                    for candidate in graph_candidates
                    if isinstance(candidate, str) and candidate.startswith(prefix + ":graph:")
                )
                keys.extend(
                    candidate
                    for candidate in value.get("manifest_keys", [])
                    if isinstance(candidate, str)
                    and candidate.startswith(prefix + ":generation:")
                )
        except (ValueError, UnicodeError, json.JSONDecodeError):
            continue
    unique = sorted(set(keys))
    deleted = client.execute("DEL", *unique)
    return {"status": "cleared", "namespace": prefix, "deleted_keys": deleted}


def clear(client: RedisClient, root: Path = ROOT) -> dict[str, Any]:
    with _index_lock(client, root):
        return _clear_locked(client, root)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--version", action="version", version=BUNDLE_VERSION)
    configured_url = next(
        (os.environ[name] for name in redis_url_envs() if os.environ.get(name)), DEFAULT_URL
    )
    result.add_argument(
        "--url",
        default=configured_url,
        help=f"Redis URL (env: {', '.join(redis_url_envs())})",
    )
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("status")
    index_parser = commands.add_parser("index")
    index_parser.add_argument(
        "--incremental",
        action="store_true",
        help="reuse unchanged content-addressed chunks (the v2 default)",
    )
    index_parser.add_argument(
        "--repair-deep",
        action="store_true",
        help="semantically validate reused chunks and regenerate affected files",
    )
    validate_parser = commands.add_parser("validate")
    validate_parser.add_argument(
        "--deep",
        action="store_true",
        help="verify every active chunk and encoded vector",
    )
    search_parser = commands.add_parser("search")
    search_parser.add_argument("query")
    search_parser.add_argument("--limit", type=int, default=5)
    symbols_parser = commands.add_parser("symbols")
    symbols_parser.add_argument("query")
    symbols_parser.add_argument("--limit", type=int, default=20)
    impact_parser = commands.add_parser("impact")
    impact_parser.add_argument("query")
    impact_parser.add_argument("--limit", type=int, default=20)
    path_parser = commands.add_parser(
        "path", help="find a minimum-hop path over resolved dependency edges"
    )
    path_parser.add_argument("source")
    path_parser.add_argument("target")
    path_parser.add_argument(
        "--direction", choices=PATH_DIRECTIONS, default="forward"
    )
    path_parser.add_argument(
        "--edge-kind",
        dest="edge_kinds",
        action="append",
        choices=DEPENDENCY_EDGE_KINDS,
        help="restrict traversal to an edge kind; repeat to select multiple kinds",
    )
    path_parser.add_argument(
        "--max-depth", type=int, default=DEFAULT_PATH_MAX_DEPTH
    )
    path_parser.add_argument(
        "--include-probable",
        action="store_true",
        help="include heuristic unique-name resolutions in addition to strong links",
    )
    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("--limit", type=int, default=10)
    commands.add_parser("clear")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        client = RedisClient(args.url)
        if args.command in {"status", "validate"}:
            output, fresh = (
                validate(client, deep=args.deep) if args.command == "validate" else status(client)
            )
            print(json.dumps(output, sort_keys=True))
            return 0 if fresh else 2
        if args.command == "index":
            print(
                json.dumps(
                    {
                        "status": "indexed",
                        **build_index(client, repair_deep=args.repair_deep),
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.command in {"search", "symbols", "impact"}:
            if args.limit < 1 or args.limit > 100:
                raise ValueError("--limit must be between 1 and 100")
            lookup = {"search": search, "symbols": symbols, "impact": impact}[args.command]
            print(json.dumps(lookup(client, args.query, args.limit), sort_keys=True))
            return 0
        if args.command == "path":
            print(
                json.dumps(
                    dependency_path(
                        client,
                        args.source,
                        args.target,
                        direction=args.direction,
                        edge_kinds=args.edge_kinds,
                        max_depth=args.max_depth,
                        include_probable=args.include_probable,
                    ),
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "evaluate":
            if args.limit < 1 or args.limit > 100:
                raise ValueError("--limit must be between 1 and 100")
            output = evaluate(client, args.limit)
            print(json.dumps(output, sort_keys=True))
            return 0 if output["status"] == "passed" else 2
        print(json.dumps(clear(client), sort_keys=True))
        return 0
    except (RedisError, TypeError, ValueError) as exc:
        print(
            json.dumps({"status": "error", "error": str(exc), "cache_consulted": False}),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
