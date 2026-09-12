"""Rust manifest dependency usage detector."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from desloppify.base.discovery.file_paths import rel, resolve_path
from desloppify.languages.rust.support import (
    find_rust_files,
    iter_use_specs,
    normalize_crate_name,
    read_text_or_none,
    strip_rust_comments,
)

from ._shared import (
    _FEATURE_REF_RE,
    _entry,
    _group_files_by_manifest,
    _line_number,
)

_EXTERN_CRATE_RE = re.compile(r"\bextern\s+crate\s+([A-Za-z_]\w*)")
_DERIVE_ATTR_RE = re.compile(r"#\s*\[\s*derive\s*\(([^)]*)\)")
_ATTR_PATH_RE = re.compile(r"#\s*\[\s*([A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)")
_QUALIFIED_PATH_RE = re.compile(r"\b([A-Za-z_]\w*)\s*::")
_TEST_DIR_NAMES = frozenset({"tests", "examples", "benches"})
_DEPENDENCY_SECTIONS = ("dependencies", "dev-dependencies", "build-dependencies")


@dataclass(frozen=True)
class _ManifestDependency:
    """One dependency declaration from a Cargo.toml manifest."""

    name: str
    crate: str
    section: str
    target: str | None
    optional: bool
    proc_macro: bool
    line: int


@dataclass
class _FileEvidence:
    """Per-source-file usage evidence, bucketed by zone."""

    zone: str
    use_roots: set[str] = field(default_factory=set)
    extern_crates: set[str] = field(default_factory=set)
    attr_tokens: set[str] = field(default_factory=set)
    path_roots: set[str] = field(default_factory=set)
    feature_refs: set[str] = field(default_factory=set)


def _squash(name: str) -> str:
    return name.replace("-", "").replace("_", "").lower()


def detect_unused_dependencies(path: Path) -> tuple[list[dict], int]:
    """Flag Cargo.toml dependencies that no crate source file references.

    Evidence for usage: `use` roots, `extern crate` names, qualified
    `<crate>::` paths, `#[derive(...)]` / attribute path segments, and
    `cfg(feature = "...")` references for optional dependencies.

    Limitations: workspace-inherited (`workspace = true`) dependencies are
    skipped entirely; `#[cfg(test)]` detection is a file-level textual
    heuristic; deeper target-table forms beyond `[target.'cfg(...)'.<sec>]`
    are not modelled.
    """
    entries: list[dict] = []
    by_manifest = _group_files_by_manifest(path)
    for manifest_dir, files in by_manifest.items():
        manifest_path = manifest_dir / "Cargo.toml"
        manifest_text = read_text_or_none(manifest_path)
        if manifest_text is None:
            continue
        dependencies = _manifest_dependencies(manifest_path, manifest_text)
        if not dependencies:
            continue
        evidence = [_collect_file_evidence(manifest_dir, filepath) for filepath in files]
        evidence = [item for item in evidence if item is not None]
        for dep in dependencies:
            scoped = [item for item in evidence if _zone_allows(dep.section, item.zone)]
            if not _dependency_used(dep, scoped):
                entries.append(_unused_entry(manifest_path, dep))
    return entries, len(find_rust_files(path))


def _manifest_dependencies(
    manifest_path: Path, manifest_text: str
) -> list[_ManifestDependency]:
    try:
        data = tomllib.loads(manifest_text)
    except tomllib.TOMLDecodeError:
        return []
    dependencies: list[_ManifestDependency] = []
    for section in _DEPENDENCY_SECTIONS:
        _extend_dependencies(
            dependencies, data.get(section), section, None, manifest_path, manifest_text
        )
    target = data.get("target")
    if isinstance(target, dict):
        for target_name, target_data in target.items():
            if not isinstance(target_data, dict):
                continue
            for section in _DEPENDENCY_SECTIONS:
                _extend_dependencies(
                    dependencies,
                    target_data.get(section),
                    section,
                    str(target_name),
                    manifest_path,
                    manifest_text,
                )
    return dependencies


def _extend_dependencies(
    dependencies: list[_ManifestDependency],
    section_data: Any,
    section: str,
    target: str | None,
    manifest_path: Path,
    manifest_text: str,
) -> None:
    if not isinstance(section_data, dict):
        return
    for name, value in section_data.items():
        dep_name = str(name)
        if isinstance(value, dict) and value.get("workspace") is True:
            continue
        crate = normalize_crate_name(dep_name) or dep_name
        dependencies.append(
            _ManifestDependency(
                name=dep_name,
                crate=crate,
                section=section,
                target=target,
                optional=isinstance(value, dict) and value.get("optional") is True,
                proc_macro=_is_proc_macro_dependency(value, manifest_path.parent),
                line=_dependency_line(manifest_text, dep_name),
            )
        )


def _is_proc_macro_dependency(value: Any, manifest_dir: Path) -> bool:
    if not isinstance(value, dict):
        return False
    path_value = value.get("path")
    if not isinstance(path_value, str) or not path_value.strip():
        return False
    dep_manifest = read_text_or_none(manifest_dir / path_value / "Cargo.toml")
    if dep_manifest is None:
        return False
    try:
        data = tomllib.loads(dep_manifest)
    except tomllib.TOMLDecodeError:
        return False
    lib = data.get("lib")
    return isinstance(lib, dict) and lib.get("proc-macro") is True


def _dependency_line(manifest_text: str, name: str) -> int:
    match = re.search(rf"(?m)^\s*{re.escape(name)}\s*=", manifest_text)
    if match is None:
        return 1
    return _line_number(manifest_text, match.start())


def _collect_file_evidence(manifest_dir: Path, filepath: str) -> _FileEvidence | None:
    absolute = Path(resolve_path(filepath))
    content = read_text_or_none(absolute)
    if content is None:
        return None
    try:
        relative = absolute.relative_to(manifest_dir)
    except ValueError:
        relative = Path(absolute.name)
    stripped = strip_rust_comments(content)
    evidence = _FileEvidence(zone=_file_zone(relative, stripped))
    for spec in iter_use_specs(content):
        root = spec.split("::", 1)[0].strip()
        if root:
            evidence.use_roots.add(_squash(root))
    evidence.extern_crates = {_squash(name) for name in _EXTERN_CRATE_RE.findall(stripped)}
    evidence.attr_tokens = _attribute_tokens(stripped)
    evidence.path_roots = {_squash(name) for name in _QUALIFIED_PATH_RE.findall(stripped)}
    evidence.feature_refs = {name.strip() for name in _FEATURE_REF_RE.findall(stripped)}
    return evidence


def _file_zone(relative: Path, stripped: str) -> str:
    if relative == Path("build.rs"):
        return "build"
    if relative.parts and relative.parts[0] in _TEST_DIR_NAMES:
        return "test"
    if "#[cfg(test)]" in stripped:
        return "test"
    return "runtime"


def _attribute_tokens(stripped: str) -> set[str]:
    tokens: set[str] = set()
    for match in _DERIVE_ATTR_RE.finditer(stripped):
        for argument in match.group(1).split(","):
            segments = argument.split("::")
            head = segments[0].strip()
            tail = segments[-1].strip()
            if head:
                tokens.add(_squash(head))
            if tail:
                tokens.add(_squash(tail))
    for match in _ATTR_PATH_RE.finditer(stripped):
        segments = match.group(1).split("::")
        tokens.add(_squash(segments[0]))
        tokens.add(_squash(segments[-1]))
    return tokens


def _zone_allows(section: str, zone: str) -> bool:
    if section == "build-dependencies":
        return zone == "build"
    if section == "dev-dependencies":
        return zone == "test"
    return True


def _dependency_used(dep: _ManifestDependency, evidence: list[_FileEvidence]) -> bool:
    crate = _squash(dep.crate)
    dep_feature = _squash(dep.name)
    for item in evidence:
        if (
            crate in item.use_roots
            or crate in item.extern_crates
            or crate in item.attr_tokens
            or crate in item.path_roots
        ):
            return True
        if dep.optional and any(dep_feature == _squash(ref) for ref in item.feature_refs):
            return True
    return False


def _unused_entry(manifest_path: Path, dep: _ManifestDependency) -> dict:
    if dep.proc_macro:
        summary = (
            f"Dependency `{dep.name}` in {rel(manifest_path)} has zero references "
            "in crate sources; proc-macro crates may be used via attributes that "
            "textual scans cannot see"
        )
        confidence = "low"
    else:
        summary = (
            f"Dependency `{dep.name}` in {rel(manifest_path)} has zero references "
            "in crate sources"
        )
        confidence = "medium"
    return _entry(
        manifest_path,
        line=dep.line,
        name=dep.name,
        summary=summary,
        tier=3,
        confidence=confidence,
        detail=dict(
            dependency=dep.name,
            section=dep.section,
            target=dep.target,
            manifest=rel(manifest_path),
        ),
    )


__all__ = ["detect_unused_dependencies"]
