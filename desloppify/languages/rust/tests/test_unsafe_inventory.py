"""Tests for the Rust unsafe-code inventory detector."""

from __future__ import annotations

from pathlib import Path

from desloppify.base.runtime_state import RuntimeContext, runtime_scope
from desloppify.languages.rust.detectors.unsafe_inventory import detect_unsafe_inventory


def _write(path: Path, rel_path: str, content: str) -> Path:
    target = path / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return target


def _entries(tmp_path, files: dict[str, str]) -> list[dict]:
    _write(tmp_path, "Cargo.toml", '[package]\nname = "demo"\nversion = "0.1.0"\nedition = "2021"\n')
    for rel_path, content in files.items():
        _write(tmp_path, rel_path, content)
    with runtime_scope(RuntimeContext(project_root=tmp_path)):
        entries, _ = detect_unsafe_inventory(tmp_path)
    return entries


def test_detect_unsafe_inventory_ignores_documented_pub_unsafe_fn(tmp_path):
    entries = _entries(
        tmp_path,
        {
            "src/lib.rs": (
                "/// Reads a byte.\n"
                "///\n"
                "/// # Safety\n"
                "///\n"
                "/// `ptr` must be valid for reads.\n"
                "pub unsafe fn read(ptr: *const u8) -> u8 {\n"
                "    unsafe { *ptr }\n"
                "}\n"
            )
        },
    )

    assert entries == []


def test_detect_unsafe_inventory_flags_undocumented_pub_unsafe_fn(tmp_path):
    entries = _entries(
        tmp_path,
        {
            "src/lib.rs": (
                "/// Reads a byte.\n"
                "pub unsafe fn read(ptr: *const u8) -> u8 {\n"
                "    unsafe { *ptr }\n"
                "}\n"
            )
        },
    )

    assert len(entries) == 1
    entry = entries[0]
    assert entry["name"] == "unsafe_fn_contract::read"
    assert entry["line"] == 2
    assert entry["tier"] == 2
    assert entry["confidence"] == "high"
    assert "`pub unsafe fn read`" in entry["summary"]
    assert "# Safety" in entry["summary"]


def test_detect_unsafe_inventory_flags_pub_crate_unsafe_fn_at_tier_2(tmp_path):
    entries = _entries(
        tmp_path,
        {
            "src/lib.rs": (
                "pub(crate) unsafe fn refresh(ptr: *const u8) -> u8 {\n"
                "    unsafe { *ptr }\n"
                "}\n"
            )
        },
    )

    assert len(entries) == 1
    assert entries[0]["name"] == "unsafe_fn_contract::refresh"
    assert entries[0]["tier"] == 2


def test_detect_unsafe_inventory_flags_private_unsafe_fn_at_tier_3(tmp_path):
    entries = _entries(
        tmp_path,
        {
            "src/lib.rs": (
                "unsafe fn read(ptr: *const u8) -> u8 {\n"
                "    unsafe { *ptr }\n"
                "}\n"
            )
        },
    )

    assert len(entries) == 1
    entry = entries[0]
    assert entry["name"] == "unsafe_fn_contract::read"
    assert entry["tier"] == 3
    assert entry["confidence"] == "high"


def test_detect_unsafe_inventory_flags_unsafe_impl_without_comment(tmp_path):
    entries = _entries(
        tmp_path,
        {
            "src/lib.rs": (
                "pub struct Wrapper(*mut u8);\n"
                "\n"
                "unsafe impl Send for Wrapper {}\n"
            )
        },
    )

    assert len(entries) == 1
    entry = entries[0]
    assert entry["name"] == "unsafe_impl_justification::Wrapper"
    assert entry["line"] == 3
    assert entry["tier"] == 3
    assert entry["confidence"] == "medium"
    assert "justification" in entry["summary"]


def test_detect_unsafe_inventory_ignores_unsafe_impl_with_header_comment(tmp_path):
    entries = _entries(
        tmp_path,
        {
            "src/lib.rs": (
                "pub struct Wrapper(*mut u8);\n"
                "\n"
                "// justified: raw pointer is owned and never shared across threads.\n"
                "unsafe impl Send for Wrapper {}\n"
            )
        },
    )

    assert entries == []


def test_detect_unsafe_inventory_ignores_unsafe_impl_with_body_comment(tmp_path):
    entries = _entries(
        tmp_path,
        {
            "src/lib.rs": (
                "pub struct Wrapper(*mut u8);\n"
                "\n"
                "unsafe impl Sync for Wrapper {\n"
                "    // justified: all access happens under the crate-global lock.\n"
                "}\n"
            )
        },
    )

    assert entries == []


def test_detect_unsafe_inventory_reports_elevated_unsafe_footprint(tmp_path):
    body = "\n".join(
        f"        let value{i} = unsafe {{ *ptr.add({i}) }};"
        for i in range(6)
    )
    entries = _entries(
        tmp_path,
        {
            "src/lib.rs": (
                "pub fn read_all(ptr: *const u8) -> [u8; 6] {\n"
                f"{body}\n"
                "    [value0, value1, value2, value3, value4, value5]\n"
                "}\n"
            )
        },
    )

    assert len(entries) == 1
    entry = entries[0]
    assert entry["name"] == "unsafe_footprint::lib"
    assert entry["tier"] == 3
    assert entry["confidence"] == "low"
    assert "6 unsafe blocks/fns/impls" in entry["summary"]
    assert entry["detail"]["unsafe_sites"] == 6


def test_detect_unsafe_inventory_ignores_low_unsafe_footprint(tmp_path):
    entries = _entries(
        tmp_path,
        {
            "src/lib.rs": (
                "pub unsafe fn read(ptr: *const u8) -> u8 {\n"
                "    unsafe { *ptr }\n"
                "}\n"
            )
        },
    )

    assert [entry["name"] for entry in entries] == ["unsafe_fn_contract::read"]


def test_detect_unsafe_inventory_ignores_unsafe_blocks_without_fn_or_impl_issues(tmp_path):
    entries = _entries(
        tmp_path,
        {
            "src/lib.rs": (
                "pub fn read(ptr: *const u8) -> u8 {\n"
                "    unsafe { *ptr }\n"
                "}\n"
            )
        },
    )

    assert entries == []


def test_detect_unsafe_inventory_only_scans_runtime_source_files(tmp_path):
    entries = _entries(
        tmp_path,
        {
            "src/lib.rs": "pub fn keep() {}\n",
            "tests/ffi.rs": "unsafe fn helper() {}\n",
            "examples/demo.rs": "pub unsafe fn demo() {}\n",
        },
    )

    assert entries == []
