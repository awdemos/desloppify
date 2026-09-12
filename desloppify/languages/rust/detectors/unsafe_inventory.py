"""Rust unsafe-code inventory and footprint detectors."""

from __future__ import annotations

import re
from pathlib import Path

from desloppify.base.discovery.file_paths import resolve_path
from desloppify.languages.rust.support import (
    describe_rust_file,
    find_rust_files,
    read_text_or_none,
    strip_rust_comments,
)

from ._shared import (
    _entry,
    _find_block_start,
    _is_runtime_source_file,
    _line_number,
    _preceding_comment_block,
    _preceding_metadata,
)

_UNSAFE_FN_RE = re.compile(
    r"(?m)^\s*(?P<vis>pub(?:\([^)]*\))?\s+)?(?:(?:async|const)\s+)*unsafe\s+"
    r"(?:(?:async|const)\s+)*fn\s+(?P<name>[A-Za-z_]\w*)\b"
)
_UNSAFE_IMPL_RE = re.compile(
    r"(?m)^\s*unsafe\s+impl(?:\s*<[^>{}]*>)?\s+(?P<header>[^;{\n]+)"
)
_SAFETY_DOC_HEADING_RE = re.compile(r"(?m)^\s*#{1,6}\s+Safety\b")
_FOR_TYPE_RE = re.compile(r"\bfor\s+([A-Za-z_]\w*)")
_UNSAFE_SITE_RE = re.compile(r"\bunsafe\s*\{|\bunsafe\s+(?:fn|impl|trait|extern)\b")
_UNSAFE_FOOTPRINT_THRESHOLD = 5


def detect_unsafe_inventory(path: Path) -> tuple[list[dict], int]:
    """Inventory `unsafe fn`/`unsafe impl` declarations and unsafe-heavy files."""
    entries: list[dict] = []
    files = find_rust_files(path)
    for filepath in files:
        absolute = Path(resolve_path(filepath))
        content = read_text_or_none(absolute)
        if content is None:
            continue
        context = describe_rust_file(absolute)
        if not _is_runtime_source_file(context):
            continue

        for match in _UNSAFE_FN_RE.finditer(content):
            token_offset = _token_offset(match)
            metadata = _preceding_metadata(content, token_offset)
            if _has_safety_doc_section(metadata):
                continue
            visibility = "pub unsafe fn" if match.group("vis") else "unsafe fn"
            entries.append(
                _entry(
                    absolute,
                    line=_line_number(content, token_offset),
                    name=f"unsafe_fn_contract::{match.group('name')}",
                    summary=(
                        f"`{visibility} {match.group('name')}` does not document its safety contract in a `# Safety` doc section"
                    ),
                    tier=2 if match.group("vis") else 3,
                    confidence="high",
                )
            )

        for match in _UNSAFE_IMPL_RE.finditer(content):
            token_offset = _token_offset(match)
            line_start = content.rfind("\n", 0, token_offset) + 1
            if _preceding_comment_block(content[:line_start]):
                continue
            if _impl_body_starts_with_comment(content, match.end()):
                continue
            target = _impl_target(match.group("header"))
            entries.append(
                _entry(
                    absolute,
                    line=_line_number(content, token_offset),
                    name=f"unsafe_impl_justification::{target}",
                    summary=(
                        f"`unsafe impl` for `{target}` has no justification comment on the impl header or inside the block"
                    ),
                    tier=3,
                    confidence="medium",
                )
            )

        stripped = strip_rust_comments(content, preserve_lines=True)
        unsafe_sites = list(_UNSAFE_SITE_RE.finditer(stripped))
        if len(unsafe_sites) > _UNSAFE_FOOTPRINT_THRESHOLD:
            entries.append(
                _entry(
                    absolute,
                    line=_line_number(stripped, unsafe_sites[0].start()),
                    name=f"unsafe_footprint::{absolute.stem}",
                    summary=(
                        f"File contains {len(unsafe_sites)} unsafe blocks/fns/impls; elevated unsafe footprint worth auditing"
                    ),
                    tier=3,
                    confidence="low",
                    detail={"unsafe_sites": len(unsafe_sites)},
                )
            )
    return entries, len(files)


def _has_safety_doc_section(metadata: str) -> bool:
    doc_lines = [
        line.strip()[3:]
        for line in metadata.splitlines()
        if line.strip().startswith("///")
    ]
    return bool(_SAFETY_DOC_HEADING_RE.search("\n".join(doc_lines)))


def _token_offset(match: re.Match[str]) -> int:
    return match.start() + len(match.group(0)) - len(match.group(0).lstrip())


def _impl_body_starts_with_comment(content: str, match_end: int) -> bool:
    body_start = _find_block_start(content, match_end)
    if body_start is None:
        return False
    for raw_line in content[body_start + 1 :].splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        return stripped.startswith("//")
    return False


def _impl_target(header: str) -> str:
    for_match = _FOR_TYPE_RE.search(header)
    if for_match:
        return for_match.group(1)
    return re.split(r"[\s:<(,]+", header.strip(), maxsplit=1)[0]


__all__ = ["detect_unsafe_inventory"]
