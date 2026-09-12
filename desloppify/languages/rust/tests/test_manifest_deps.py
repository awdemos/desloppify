"""Tests for the Rust unused-manifest-dependency detector."""

from __future__ import annotations

from pathlib import Path

from desloppify.base.runtime_state import RuntimeContext, runtime_scope
from desloppify.languages.rust.detectors.manifest_deps import detect_unused_dependencies


def _write(tmp_path: Path, relpath: str, content: str) -> None:
    path = tmp_path / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _detect(tmp_path: Path) -> list[dict]:
    with runtime_scope(RuntimeContext(project_root=tmp_path)):
        entries, total = detect_unused_dependencies(tmp_path)
    assert total >= 1
    return entries


def test_dependency_used_via_use_statement_has_no_finding(tmp_path):
    _write(
        tmp_path,
        "Cargo.toml",
        '[package]\nname = "demo-app"\nversion = "0.1.0"\n\n[dependencies]\nserde = "1"\n',
    )
    _write(tmp_path, "src/lib.rs", "use serde::Deserialize;\n")

    assert _detect(tmp_path) == []


def test_unreferenced_dependency_is_flagged(tmp_path):
    _write(
        tmp_path,
        "Cargo.toml",
        """
[package]
name = "demo-app"
version = "0.1.0"

[dependencies]
leftpad = "1.0"
""",
    )
    _write(tmp_path, "src/lib.rs", "pub fn run() {}\n")

    entries = _detect(tmp_path)

    assert [entry["name"] for entry in entries] == ["leftpad"]
    assert entries[0]["file"] == "Cargo.toml"
    assert entries[0]["tier"] == 3
    assert entries[0]["confidence"] == "medium"
    assert "leftpad" in entries[0]["summary"]
    assert "zero references" in entries[0]["summary"]


def test_workspace_inherited_dependency_is_skipped(tmp_path):
    _write(
        tmp_path,
        "Cargo.toml",
        """
[workspace]
members = ["app"]

[workspace.dependencies]
shared = "1.0"
""",
    )
    _write(
        tmp_path,
        "app/Cargo.toml",
        """
[package]
name = "app"
version = "0.1.0"

[dependencies]
shared = { workspace = true }
""",
    )
    _write(tmp_path, "app/src/lib.rs", "pub fn run() {}\n")

    assert _detect(tmp_path) == []


def test_build_dependency_used_in_build_rs_has_no_finding(tmp_path):
    _write(
        tmp_path,
        "Cargo.toml",
        """
[package]
name = "demo-app"
version = "0.1.0"

[build-dependencies]
cc = "1.0"
""",
    )
    _write(tmp_path, "src/lib.rs", "pub fn run() {}\n")
    _write(tmp_path, "build.rs", "fn main() {\n    cc::Build::new();\n}\n")

    assert _detect(tmp_path) == []


def test_build_dependency_unused_is_flagged(tmp_path):
    _write(
        tmp_path,
        "Cargo.toml",
        """
[package]
name = "demo-app"
version = "0.1.0"

[build-dependencies]
cc = "1.0"
""",
    )
    _write(tmp_path, "src/lib.rs", "pub fn run() {}\n")
    _write(tmp_path, "build.rs", "fn main() {}\n")

    assert [entry["name"] for entry in _detect(tmp_path)] == ["cc"]


def test_dev_dependency_used_in_tests_dir_has_no_finding(tmp_path):
    _write(
        tmp_path,
        "Cargo.toml",
        """
[package]
name = "demo-app"
version = "0.1.0"

[dev-dependencies]
pretty_assertions = "1.0"
""",
    )
    _write(tmp_path, "src/lib.rs", "pub fn run() {}\n")
    _write(tmp_path, "tests/integration.rs", "use pretty_assertions::assert_eq;\n")

    assert _detect(tmp_path) == []


def test_dev_dependency_used_in_cfg_test_module_has_no_finding(tmp_path):
    _write(
        tmp_path,
        "Cargo.toml",
        """
[package]
name = "demo-app"
version = "0.1.0"

[dev-dependencies]
pretty_assertions = "1.0"
""",
    )
    _write(
        tmp_path,
        "src/lib.rs",
        """
#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
}
""",
    )

    assert _detect(tmp_path) == []


def test_unused_dev_dependency_is_flagged(tmp_path):
    _write(
        tmp_path,
        "Cargo.toml",
        """
[package]
name = "demo-app"
version = "0.1.0"

[dev-dependencies]
test-utils = "0.1.0"
""",
    )
    _write(tmp_path, "src/lib.rs", "pub fn run() {}\n")

    entries = _detect(tmp_path)

    assert [entry["name"] for entry in entries] == ["test-utils"]
    assert entries[0]["detail"]["section"] == "dev-dependencies"


def test_proc_macro_referenced_only_via_derive_has_no_finding(tmp_path):
    _write(
        tmp_path,
        "Cargo.toml",
        """
[package]
name = "demo-app"
version = "0.1.0"

[dependencies]
widget-macros = { path = "crates/widget-macros" }
""",
    )
    _write(
        tmp_path,
        "crates/widget-macros/Cargo.toml",
        """
[package]
name = "widget-macros"
version = "0.1.0"

[lib]
proc-macro = true
""",
    )
    _write(tmp_path, "crates/widget-macros/src/lib.rs", "")
    _write(tmp_path, "src/lib.rs", "#[derive(WidgetMacros)]\npub struct Widget;\n")

    assert _detect(tmp_path) == []


def test_proc_macro_with_zero_evidence_is_flagged_with_low_confidence(tmp_path):
    _write(
        tmp_path,
        "Cargo.toml",
        """
[package]
name = "demo-app"
version = "0.1.0"

[dependencies]
widget-macros = { path = "crates/widget-macros" }
""",
    )
    _write(
        tmp_path,
        "crates/widget-macros/Cargo.toml",
        """
[package]
name = "widget-macros"
version = "0.1.0"

[lib]
proc-macro = true
""",
    )
    _write(tmp_path, "crates/widget-macros/src/lib.rs", "")
    _write(tmp_path, "src/lib.rs", "pub struct Widget;\n")

    entries = _detect(tmp_path)

    assert [entry["name"] for entry in entries] == ["widget-macros"]
    assert entries[0]["confidence"] == "low"
    assert "proc-macro" in entries[0]["summary"]


def test_hyphenated_crate_name_used_via_underscore_has_no_finding(tmp_path):
    _write(
        tmp_path,
        "Cargo.toml",
        """
[package]
name = "demo-app"
version = "0.1.0"

[dependencies]
some-crate = "1.0"
""",
    )
    _write(tmp_path, "src/lib.rs", "use some_crate::thing;\n")

    assert _detect(tmp_path) == []


def test_optional_dependency_referenced_via_feature_cfg_has_no_finding(tmp_path):
    _write(
        tmp_path,
        "Cargo.toml",
        """
[package]
name = "demo-app"
version = "0.1.0"

[dependencies]
extra = { version = "1.0", optional = true }
""",
    )
    _write(
        tmp_path,
        "src/lib.rs",
        '#[cfg(feature = "extra")]\npub fn extra() {}\n',
    )

    assert _detect(tmp_path) == []


def test_target_specific_dependency_usage_is_checked(tmp_path):
    _write(
        tmp_path,
        "Cargo.toml",
        """
[package]
name = "demo-app"
version = "0.1.0"

[target.'cfg(unix)'.dependencies]
libc = "0.2"
""",
    )
    _write(tmp_path, "src/lib.rs", "use libc::c_int;\n")

    assert _detect(tmp_path) == []


def test_unused_target_specific_dependency_is_flagged(tmp_path):
    _write(
        tmp_path,
        "Cargo.toml",
        """
[package]
name = "demo-app"
version = "0.1.0"

[target.'cfg(unix)'.dependencies]
libc = "0.2"
""",
    )
    _write(tmp_path, "src/lib.rs", "pub fn run() {}\n")

    entries = _detect(tmp_path)

    assert [entry["name"] for entry in entries] == ["libc"]
    assert entries[0]["detail"]["section"] == "dependencies"
    assert entries[0]["detail"]["target"] == "cfg(unix)"
