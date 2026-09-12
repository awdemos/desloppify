"""Tests for Rust cargo diagnostic parsing helpers."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from desloppify.languages._framework.generic_parts.parsers import ToolParserError
from desloppify.languages.rust.tools import (
    AUDIT_CMD,
    build_rustdoc_bin_cmd,
    build_rustdoc_warning_cmd,
    parse_audit_messages,
    parse_cargo_errors,
    parse_cargo_unused_imports,
    parse_clippy_messages,
    parse_rustdoc_messages,
    run_audit_result,
    run_rustdoc_result,
)


def test_parse_clippy_messages_ignores_non_json_noise():
    message = {
        "reason": "compiler-message",
        "message": {
            "level": "warning",
            "message": "unused variable: `name`",
            "spans": [
                {
                    "is_primary": True,
                    "file_name": "src/lib.rs",
                    "line_start": 7,
                }
            ],
        },
    }
    output = "\n".join(
        [
            "Compiling demo v0.1.0",
            "[]",
            json.dumps(message),
        ]
    )

    entries = parse_clippy_messages(output, Path("."))

    assert entries == [
        {
            "file": "src/lib.rs",
            "line": 7,
            "message": "unused variable: `name`",
        }
    ]


def test_parse_clippy_messages_skips_inline_cfg_test_module_diagnostics(tmp_path):
    source = tmp_path / "src" / "lib.rs"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        """
pub fn runtime_value() -> usize {
    1
}

#[cfg(test)]
mod tests {
    #[test]
    fn inline_test_uses_unwrap() {
        let _ = Some(1usize).unwrap();
    }
}
""".strip()
        + "\n"
    )

    message = {
        "reason": "compiler-message",
        "message": {
            "level": "warning",
            "message": "used `unwrap()` on an `Option` value",
            "code": {"code": "clippy::unwrap_used"},
            "spans": [
                {
                    "is_primary": True,
                    "file_name": "src/lib.rs",
                    "line_start": 9,
                }
            ],
        },
    }

    entries = parse_clippy_messages(json.dumps(message), tmp_path)

    assert entries == []


def test_parse_clippy_messages_keeps_non_test_diagnostics_in_same_file(tmp_path):
    source = tmp_path / "src" / "lib.rs"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        """
pub fn runtime_value() -> usize {
    Some(1usize).unwrap()
}

#[cfg(test)]
mod tests {
    #[test]
    fn inline_test_uses_unwrap() {
        let _ = Some(1usize).unwrap();
    }
}
""".strip()
        + "\n"
    )

    message = {
        "reason": "compiler-message",
        "message": {
            "level": "warning",
            "message": "used `unwrap()` on an `Option` value",
            "code": {"code": "clippy::unwrap_used"},
            "spans": [
                {
                    "is_primary": True,
                    "file_name": "src/lib.rs",
                    "line_start": 2,
                }
            ],
        },
    }

    entries = parse_clippy_messages(json.dumps(message), tmp_path)

    assert entries == [
        {
            "file": "src/lib.rs",
            "line": 2,
            "message": "[clippy::unwrap_used] used `unwrap()` on an `Option` value",
        }
    ]


def test_parse_clippy_messages_keeps_cfg_not_test_inline_module(tmp_path):
    source = tmp_path / "src" / "lib.rs"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        """
#[cfg(not(test))]
mod production_only {
    pub fn value() -> usize {
        Some(1usize).unwrap()
    }
}
""".strip()
        + "\n"
    )

    message = {
        "reason": "compiler-message",
        "message": {
            "level": "warning",
            "message": "used `unwrap()` on an `Option` value",
            "code": {"code": "clippy::unwrap_used"},
            "spans": [
                {
                    "is_primary": True,
                    "file_name": "src/lib.rs",
                    "line_start": 4,
                }
            ],
        },
    }

    entries = parse_clippy_messages(json.dumps(message), tmp_path)

    assert entries == [
        {
            "file": "src/lib.rs",
            "line": 4,
            "message": "[clippy::unwrap_used] used `unwrap()` on an `Option` value",
        }
    ]


def test_parse_clippy_messages_skips_cfg_all_test_inline_module(tmp_path):
    source = tmp_path / "src" / "lib.rs"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        """
#[cfg(all(test, feature = "unstable"))]
mod tests {
    pub fn helper() {
        let _ = Some(1usize).unwrap();
    }
}
""".strip()
        + "\n"
    )

    message = {
        "reason": "compiler-message",
        "message": {
            "level": "warning",
            "message": "used `unwrap()` on an `Option` value",
            "code": {"code": "clippy::unwrap_used"},
            "spans": [
                {
                    "is_primary": True,
                    "file_name": "src/lib.rs",
                    "line_start": 4,
                }
            ],
        },
    }

    entries = parse_clippy_messages(json.dumps(message), tmp_path)

    assert entries == []


def test_parse_clippy_messages_keeps_cfg_any_test_or_other_inline_module(tmp_path):
    source = tmp_path / "src" / "lib.rs"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        """
#[cfg(any(test, feature = "bench"))]
mod maybe_test {
    pub fn helper() {
        let _ = Some(1usize).unwrap();
    }
}
""".strip()
        + "\n"
    )

    message = {
        "reason": "compiler-message",
        "message": {
            "level": "warning",
            "message": "used `unwrap()` on an `Option` value",
            "code": {"code": "clippy::unwrap_used"},
            "spans": [
                {
                    "is_primary": True,
                    "file_name": "src/lib.rs",
                    "line_start": 4,
                }
            ],
        },
    }

    entries = parse_clippy_messages(json.dumps(message), tmp_path)

    assert entries == [
        {
            "file": "src/lib.rs",
            "line": 4,
            "message": "[clippy::unwrap_used] used `unwrap()` on an `Option` value",
        }
    ]


def test_parse_clippy_messages_skips_inline_cfg_test_with_comment_between_attr_and_mod(
    tmp_path,
):
    source = tmp_path / "src" / "lib.rs"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        """
#[cfg(test)]
// inline tests live below
mod tests {
    pub fn helper() {
        let _ = Some(1usize).unwrap();
    }
}
""".strip()
        + "\n"
    )

    message = {
        "reason": "compiler-message",
        "message": {
            "level": "warning",
            "message": "used `unwrap()` on an `Option` value",
            "code": {"code": "clippy::unwrap_used"},
            "spans": [
                {
                    "is_primary": True,
                    "file_name": "src/lib.rs",
                    "line_start": 5,
                }
            ],
        },
    }

    entries = parse_clippy_messages(json.dumps(message), tmp_path)

    assert entries == []


def test_parse_clippy_messages_ignores_commented_out_cfg_test_marker(tmp_path):
    source = tmp_path / "src" / "lib.rs"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        """
// #[cfg(test)]
mod production {
    pub fn helper() {
        let _ = Some(1usize).unwrap();
    }
}
""".strip()
        + "\n"
    )

    message = {
        "reason": "compiler-message",
        "message": {
            "level": "warning",
            "message": "used `unwrap()` on an `Option` value",
            "code": {"code": "clippy::unwrap_used"},
            "spans": [
                {
                    "is_primary": True,
                    "file_name": "src/lib.rs",
                    "line_start": 4,
                }
            ],
        },
    }

    entries = parse_clippy_messages(json.dumps(message), tmp_path)

    assert entries == [
        {
            "file": "src/lib.rs",
            "line": 4,
            "message": "[clippy::unwrap_used] used `unwrap()` on an `Option` value",
        }
    ]


def test_parse_clippy_messages_skips_full_inline_module_when_strings_contain_closing_braces(
    tmp_path,
):
    source = tmp_path / "src" / "lib.rs"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        """
#[cfg(test)]
mod tests {
    pub fn first() {
        println!("brace in string: }");
    }

    pub fn second() {
        let _ = Some(1usize).unwrap();
    }
}
""".strip()
        + "\n"
    )

    message = {
        "reason": "compiler-message",
        "message": {
            "level": "warning",
            "message": "used `unwrap()` on an `Option` value",
            "code": {"code": "clippy::unwrap_used"},
            "spans": [
                {
                    "is_primary": True,
                    "file_name": "src/lib.rs",
                    "line_start": 8,
                }
            ],
        },
    }

    entries = parse_clippy_messages(json.dumps(message), tmp_path)

    assert entries == []


def test_parse_clippy_messages_skips_inline_module_when_test_contains_url_string(
    tmp_path,
):
    source = tmp_path / "src" / "lib.rs"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        """
#[cfg(test)]
mod tests {
    pub fn first() {
        let _url = "http://example.com";
    }

    pub fn second() {
        let _ = Some(1usize).unwrap();
    }
}
""".strip()
        + "\n"
    )

    message = {
        "reason": "compiler-message",
        "message": {
            "level": "warning",
            "message": "used `unwrap()` on an `Option` value",
            "code": {"code": "clippy::unwrap_used"},
            "spans": [
                {
                    "is_primary": True,
                    "file_name": "src/lib.rs",
                    "line_start": 8,
                }
            ],
        },
    }

    entries = parse_clippy_messages(json.dumps(message), tmp_path)

    assert entries == []


def test_parse_clippy_messages_skips_inline_module_when_test_contains_lifetime(
    tmp_path,
):
    source = tmp_path / "src" / "lib.rs"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        """
#[cfg(test)]
mod tests {
    pub fn with_lifetime(input: &'static str) -> &'static str {
        let _ = Some(input).unwrap();
        input
    }
}
""".strip()
        + "\n"
    )

    message = {
        "reason": "compiler-message",
        "message": {
            "level": "warning",
            "message": "used `unwrap()` on an `Option` value",
            "code": {"code": "clippy::unwrap_used"},
            "spans": [
                {
                    "is_primary": True,
                    "file_name": "src/lib.rs",
                    "line_start": 4,
                }
            ],
        },
    }

    entries = parse_clippy_messages(json.dumps(message), tmp_path)

    assert entries == []


def test_parse_clippy_messages_skips_inline_module_with_raw_identifier_name(tmp_path):
    source = tmp_path / "src" / "lib.rs"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        """
#[cfg(test)]
mod r#tests {
    pub fn helper() {
        let _ = Some(1usize).unwrap();
    }
}
""".strip()
        + "\n"
    )

    message = {
        "reason": "compiler-message",
        "message": {
            "level": "warning",
            "message": "used `unwrap()` on an `Option` value",
            "code": {"code": "clippy::unwrap_used"},
            "spans": [
                {
                    "is_primary": True,
                    "file_name": "src/lib.rs",
                    "line_start": 4,
                }
            ],
        },
    }

    entries = parse_clippy_messages(json.dumps(message), tmp_path)

    assert entries == []


def test_parse_clippy_messages_skips_inline_module_with_doc_attr_containing_bracket(
    tmp_path,
):
    source = tmp_path / "src" / "lib.rs"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        """
#[cfg(test)]
#[doc = "]"]
mod tests {
    pub fn helper() {
        let _ = Some(1usize).unwrap();
    }
}
""".strip()
        + "\n"
    )

    message = {
        "reason": "compiler-message",
        "message": {
            "level": "warning",
            "message": "used `unwrap()` on an `Option` value",
            "code": {"code": "clippy::unwrap_used"},
            "spans": [
                {
                    "is_primary": True,
                    "file_name": "src/lib.rs",
                    "line_start": 5,
                }
            ],
        },
    }

    entries = parse_clippy_messages(json.dumps(message), tmp_path)

    assert entries == []


def test_parse_clippy_messages_skips_inline_module_when_block_comment_contains_double_slash(
    tmp_path,
):
    source = tmp_path / "src" / "lib.rs"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        """
#[cfg(test)]
/* this comment includes // text that should not terminate scanning */
mod tests {
    pub fn helper() {
        let _ = Some(1usize).unwrap();
    }
}
""".strip()
        + "\n"
    )

    message = {
        "reason": "compiler-message",
        "message": {
            "level": "warning",
            "message": "used `unwrap()` on an `Option` value",
            "code": {"code": "clippy::unwrap_used"},
            "spans": [
                {
                    "is_primary": True,
                    "file_name": "src/lib.rs",
                    "line_start": 5,
                }
            ],
        },
    }

    entries = parse_clippy_messages(json.dumps(message), tmp_path)

    assert entries == []


def test_parse_cargo_errors_prefers_primary_span_and_includes_error_code():
    message = {
        "reason": "compiler-message",
        "message": {
            "level": "error",
            "message": "cannot find value `answer` in this scope",
            "code": {"code": "E0425"},
            "spans": [
                {
                    "is_primary": False,
                    "file_name": "src/other.rs",
                    "line_start": 3,
                },
                {
                    "is_primary": True,
                    "file_name": "src/lib.rs",
                    "line_start": 11,
                },
            ],
        },
    }

    entries = parse_cargo_errors(json.dumps(message), Path("."))

    assert entries == [
        {
            "file": "src/lib.rs",
            "line": 11,
            "message": "[E0425] cannot find value `answer` in this scope",
        }
    ]


def _cargo_warning(code: str, message: str, file_name: str = "src/lib.rs") -> dict:
    return {
        "reason": "compiler-message",
        "message": {
            "level": "warning",
            "message": message,
            "code": {"code": code},
            "spans": [
                {
                    "is_primary": True,
                    "file_name": file_name,
                    "line_start": 4,
                },
            ],
        },
    }


def test_parse_cargo_unused_imports_keeps_only_unused_import_codes():
    lines = "\n".join(
        json.dumps(m)
        for m in [
            _cargo_warning("unused_imports", "unused import: `std::fmt::Write`"),
            _cargo_warning("dead_code", "struct `Old` is never constructed"),
            _cargo_warning("unused_variables", "unused variable: `x`"),
        ]
    )

    entries = parse_cargo_unused_imports(lines, Path("."))

    assert entries == [
        {
            "file": "src/lib.rs",
            "line": 4,
            "message": "[unused_imports] unused import: `std::fmt::Write`",
        }
    ]


def test_parse_cargo_unused_imports_skips_errors_and_codeless_warnings():
    lines = "\n".join(
        json.dumps(m)
        for m in [
            _cargo_warning("unused_imports", "unused import: `x`"),
            {
                "reason": "compiler-message",
                "message": {
                    "level": "error",
                    "message": "mismatched types",
                    "code": {"code": "E0308"},
                    "spans": [
                        {"is_primary": True, "file_name": "src/lib.rs", "line_start": 1},
                    ],
                },
            },
            {
                "reason": "compiler-message",
                "message": {
                    "level": "warning",
                    "message": "unknown lint",
                    "spans": [],
                },
            },
        ]
    )

    entries = parse_cargo_unused_imports(lines, Path("."))

    assert len(entries) == 1
    assert entries[0]["message"] == "[unused_imports] unused import: `x`"


def test_parse_rustdoc_messages_includes_lint_code():
    message = {
        "reason": "compiler-message",
        "message": {
            "level": "warning",
            "message": "no documentation found for this crate's top-level module",
            "code": {"code": "rustdoc::missing_crate_level_docs"},
            "spans": [
                {
                    "is_primary": True,
                    "file_name": "crates/lib/src/lib.rs",
                    "line_start": 1,
                }
            ],
        },
    }

    entries = parse_rustdoc_messages(json.dumps(message), Path("."))

    assert entries == [
        {
            "file": "crates/lib/src/lib.rs",
            "line": 1,
            "message": (
                "[rustdoc::missing_crate_level_docs] "
                "no documentation found for this crate's top-level module"
            ),
        }
    ]


def test_build_rustdoc_warning_cmd_targets_one_package():
    command = build_rustdoc_warning_cmd("demo-crate")

    assert "cargo rustdoc" in command
    assert "--package demo-crate" in command
    assert "--workspace" not in command
    assert "--lib" in command
    assert "-W missing_docs" in command


def test_build_rustdoc_bin_cmd_targets_one_binary():
    command = build_rustdoc_bin_cmd("demo-crate", "cli")

    assert "cargo rustdoc" in command
    assert "--package demo-crate" in command
    assert "--bin cli" in command
    assert "--lib" not in command
    assert "--workspace" not in command
    assert "-D rustdoc::broken_intra_doc_links" in command
    assert "-D rustdoc::private_intra_doc_links" in command
    assert "-W rustdoc::missing_crate_level_docs" in command
    assert "-W missing_docs" not in command


def test_run_rustdoc_result_scans_each_workspace_library_package(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "pkg-a" / "src").mkdir(parents=True)
    (workspace / "pkg-a" / "src" / "lib.rs").write_text("pub fn a() {}\n")
    (workspace / "pkg-c" / "src").mkdir(parents=True)
    (workspace / "pkg-c" / "src" / "lib.rs").write_text("pub fn c() {}\n")
    commands: list[str] = []

    metadata = {
        "workspace_members": ["pkg-a 0.1.0 (path+file:///workspace/pkg-a)", "pkg-b 0.1.0 (path+file:///workspace/pkg-b)", "pkg-c 0.1.0 (path+file:///workspace/pkg-c)"],
        "packages": [
            {
                "id": "pkg-a 0.1.0 (path+file:///workspace/pkg-a)",
                "name": "pkg-a",
                "targets": [{"kind": ["lib"], "crate_types": ["lib"]}],
            },
            {
                "id": "pkg-b 0.1.0 (path+file:///workspace/pkg-b)",
                "name": "pkg-b",
                "targets": [{"kind": ["bin"], "crate_types": ["bin"]}],
            },
            {
                "id": "pkg-c 0.1.0 (path+file:///workspace/pkg-c)",
                "name": "pkg-c",
                "targets": [{"kind": ["proc-macro"], "crate_types": ["proc-macro"]}],
            },
        ],
    }
    def rustdoc_message(file_name: str, line_no: int) -> str:
        return json.dumps(
            {
                "reason": "compiler-message",
                "message": {
                    "level": "warning",
                    "message": "missing docs",
                    "spans": [
                        {
                            "is_primary": True,
                            "file_name": file_name,
                            "line_start": line_no,
                        }
                    ],
                },
            }
        )

    def runner(args, **kwargs):
        command = args[2] if args[:2] == ["/bin/sh", "-lc"] else " ".join(args)
        commands.append(command)
        if command == "cargo metadata --format-version=1 --no-deps":
            return subprocess.CompletedProcess(args=args, returncode=0, stdout=json.dumps(metadata), stderr="")
        if "--package pkg-a" in command:
            return subprocess.CompletedProcess(
                args=args,
                returncode=1,
                stdout=rustdoc_message("pkg-a/src/lib.rs", 3),
                stderr="",
            )
        if "--package pkg-c" in command:
            return subprocess.CompletedProcess(
                args=args,
                returncode=1,
                stdout=rustdoc_message("pkg-c/src/lib.rs", 8),
                stderr="",
            )
        raise AssertionError(f"unexpected command: {command}")

    result = run_rustdoc_result(workspace, run_subprocess=runner)

    assert result.status == "ok"
    assert result.entries == [
        {"file": "pkg-a/src/lib.rs", "line": 3, "message": "missing docs"},
        {"file": "pkg-c/src/lib.rs", "line": 8, "message": "missing docs"},
    ]
    assert commands[0] == "cargo metadata --format-version=1 --no-deps"
    assert any("--package pkg-a" in command for command in commands)
    assert not any("--package pkg-b" in command for command in commands)
    assert any("--package pkg-c" in command for command in commands)
    pkg_a_commands = [
        command for command in commands if "--package pkg-a" in command
    ]
    assert len(pkg_a_commands) == 1
    assert "--lib" in pkg_a_commands[0]
    assert "-W missing_docs" in pkg_a_commands[0]


def test_run_rustdoc_result_filters_missing_primary_span_files(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "pkg-a" / "src").mkdir(parents=True)
    (workspace / "pkg-a" / "src" / "lib.rs").write_text("pub fn ok() {}\n")

    metadata = {
        "workspace_members": ["pkg-a 0.1.0 (path+file:///workspace/pkg-a)"],
        "packages": [
            {
                "id": "pkg-a 0.1.0 (path+file:///workspace/pkg-a)",
                "name": "pkg-a",
                "targets": [{"kind": ["lib"], "crate_types": ["lib"]}],
            }
        ],
    }

    def rustdoc_messages() -> str:
        existing = {
            "reason": "compiler-message",
            "message": {
                "level": "warning",
                "message": "missing docs",
                "spans": [{"is_primary": True, "file_name": "pkg-a/src/lib.rs", "line_start": 1}],
            },
        }
        missing = {
            "reason": "compiler-message",
            "message": {
                "level": "warning",
                "message": "stale docs warning",
                "spans": [{"is_primary": True, "file_name": "src/lib.rs", "line_start": 1}],
            },
        }
        return "\n".join(json.dumps(message) for message in (existing, missing))

    def runner(args, **kwargs):
        command = args[2] if args[:2] == ["/bin/sh", "-lc"] else " ".join(args)
        if command == "cargo metadata --format-version=1 --no-deps":
            return subprocess.CompletedProcess(args=args, returncode=0, stdout=json.dumps(metadata), stderr="")
        if "--package pkg-a" in command:
            return subprocess.CompletedProcess(args=args, returncode=1, stdout=rustdoc_messages(), stderr="")
        raise AssertionError(f"unexpected command: {command}")

    result = run_rustdoc_result(workspace, run_subprocess=runner)

    assert result.status == "ok"
    assert result.entries == [
        {"file": "pkg-a/src/lib.rs", "line": 1, "message": "missing docs"}
    ]


def test_run_rustdoc_result_scans_binary_only_packages(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    commands: list[str] = []
    metadata = {
        "workspace_members": ["pkg-bin 0.1.0 (path+file:///workspace/pkg-bin)"],
        "packages": [
            {
                "id": "pkg-bin 0.1.0 (path+file:///workspace/pkg-bin)",
                "name": "pkg-bin",
                "targets": [
                    {"name": "pkg-bin", "kind": ["bin"], "crate_types": ["bin"]}
                ],
            }
        ],
    }

    def runner(args, **kwargs):
        command = args[2] if args[:2] == ["/bin/sh", "-lc"] else " ".join(args)
        commands.append(command)
        if command == "cargo metadata --format-version=1 --no-deps":
            return subprocess.CompletedProcess(args=args, returncode=0, stdout=json.dumps(metadata), stderr="")
        if "--package pkg-bin" in command:
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")
        raise AssertionError(f"unexpected command: {command}")

    result = run_rustdoc_result(workspace, run_subprocess=runner)

    assert result.status == "empty"
    assert commands[0] == "cargo metadata --format-version=1 --no-deps"
    bin_commands = [command for command in commands if "--package pkg-bin" in command]
    assert len(bin_commands) == 1
    assert "--bin pkg-bin" in bin_commands[0]
    assert "--lib" not in bin_commands[0]
    assert "-D rustdoc::broken_intra_doc_links" in bin_commands[0]
    assert "-W rustdoc::missing_crate_level_docs" in bin_commands[0]
    assert "-W missing_docs" not in bin_commands[0]


def test_run_rustdoc_result_scans_libs_and_bins_with_target_lint_sets(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "pkg-a" / "src" / "bin").mkdir(parents=True)
    (workspace / "pkg-a" / "src" / "lib.rs").write_text("pub fn a() {}\n")
    (workspace / "pkg-a" / "src" / "bin" / "cli.rs").write_text("fn main() {}\n")
    commands: list[str] = []

    metadata = {
        "workspace_members": ["pkg-a 0.1.0 (path+file:///workspace/pkg-a)"],
        "packages": [
            {
                "id": "pkg-a 0.1.0 (path+file:///workspace/pkg-a)",
                "name": "pkg-a",
                "targets": [
                    {"name": "pkg-a", "kind": ["lib"], "crate_types": ["lib"]},
                    {"name": "cli", "kind": ["bin"], "crate_types": ["bin"]},
                    {"name": "daemon", "kind": ["bin"], "crate_types": ["bin"]},
                    {"kind": ["bin"], "crate_types": ["bin"]},
                ],
            }
        ],
    }

    def rustdoc_message(file_name: str, line_no: int) -> str:
        return json.dumps(
            {
                "reason": "compiler-message",
                "message": {
                    "level": "warning",
                    "message": "broken doc link",
                    "spans": [
                        {
                            "is_primary": True,
                            "file_name": file_name,
                            "line_start": line_no,
                        }
                    ],
                },
            }
        )

    def runner(args, **kwargs):
        command = args[2] if args[:2] == ["/bin/sh", "-lc"] else " ".join(args)
        commands.append(command)
        if command == "cargo metadata --format-version=1 --no-deps":
            return subprocess.CompletedProcess(args=args, returncode=0, stdout=json.dumps(metadata), stderr="")
        if "--package pkg-a" in command and "--lib" in command:
            return subprocess.CompletedProcess(
                args=args,
                returncode=1,
                stdout=rustdoc_message("pkg-a/src/lib.rs", 3),
                stderr="",
            )
        if "--package pkg-a" in command and "--bin cli" in command:
            return subprocess.CompletedProcess(
                args=args,
                returncode=1,
                stdout=rustdoc_message("pkg-a/src/bin/cli.rs", 5),
                stderr="",
            )
        if "--package pkg-a" in command and "--bin daemon" in command:
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")
        raise AssertionError(f"unexpected command: {command}")

    result = run_rustdoc_result(workspace, run_subprocess=runner)

    assert result.status == "ok"
    assert result.entries == [
        {"file": "pkg-a/src/lib.rs", "line": 3, "message": "broken doc link"},
        {"file": "pkg-a/src/bin/cli.rs", "line": 5, "message": "broken doc link"},
    ]
    lib_commands = [
        command
        for command in commands
        if "--package pkg-a" in command and "--lib" in command
    ]
    bin_commands = [
        command
        for command in commands
        if "--package pkg-a" in command and "--bin" in command
    ]
    assert len(lib_commands) == 1
    assert "-W missing_docs" in lib_commands[0]
    assert [command for command in bin_commands if "--bin cli" in command]
    assert [command for command in bin_commands if "--bin daemon" in command]
    assert len(bin_commands) == 2
    for command in bin_commands:
        assert "-W missing_docs" not in command


def test_run_rustdoc_result_returns_error_for_unparsed_package_failure(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    metadata = {
        "workspace_members": ["pkg-a 0.1.0 (path+file:///workspace/pkg-a)"],
        "packages": [
            {
                "id": "pkg-a 0.1.0 (path+file:///workspace/pkg-a)",
                "name": "pkg-a",
                "targets": [{"kind": ["lib"], "crate_types": ["lib"]}],
            }
        ],
    }

    def runner(args, **kwargs):
        command = args[2] if args[:2] == ["/bin/sh", "-lc"] else " ".join(args)
        if command == "cargo metadata --format-version=1 --no-deps":
            return subprocess.CompletedProcess(args=args, returncode=0, stdout=json.dumps(metadata), stderr="")
        if "--package pkg-a" in command:
            return subprocess.CompletedProcess(args=args, returncode=2, stdout="not json", stderr="")
        raise AssertionError(f"unexpected command: {command}")

    result = run_rustdoc_result(workspace, run_subprocess=runner)

    assert result.status == "error"
    assert result.error_kind == "tool_failed_unparsed_output"
    assert result.message is not None
    assert result.message.startswith("pkg-a:")


def _audit_payload(entries: list[dict]) -> str:
    return json.dumps(
        {
            "vulnerabilities": {
                "summary": {"found": len(entries)},
                "list": entries,
            }
        }
    )


def test_parse_audit_messages_extracts_vulnerability_entries():
    raw = _audit_payload(
        [
            {
                "advisory": {
                    "id": "RUSTSEC-2024-0001",
                    "title": "Buffer overflow in format parsing",
                    "severity": "high",
                },
                "package": {"name": "serde", "version": "1.0.0"},
            },
            {
                "advisory": {
                    "id": "RUSTSEC-2024-0002",
                    "title": "Integer overflow",
                    "severity": None,
                },
                "package": {"name": "time", "version": "0.3.0"},
            },
        ]
    )

    entries = parse_audit_messages(raw, Path("."))

    assert entries == [
        {
            "file": "Cargo.lock",
            "line": 1,
            "message": (
                "[RUSTSEC-2024-0001] serde 1.0.0: "
                "Buffer overflow in format parsing (severity: high)"
            ),
        },
        {
            "file": "Cargo.lock",
            "line": 1,
            "message": "[RUSTSEC-2024-0002] time 0.3.0: Integer overflow (severity: unknown)",
        },
    ]


def test_parse_audit_messages_returns_empty_for_empty_vulnerability_list():
    entries = parse_audit_messages(_audit_payload([]), Path("."))

    assert entries == []


def test_parse_audit_messages_returns_empty_for_summary_only_payload():
    raw = json.dumps({"vulnerabilities": {"summary": {"found": 0}}})

    entries = parse_audit_messages(raw, Path("."))

    assert entries == []


def test_parse_audit_messages_raises_for_malformed_json():
    with pytest.raises(ToolParserError):
        parse_audit_messages("not json", Path("."))


def test_run_audit_result_uses_audit_json_command(tmp_path):
    commands: list[str] = []

    def runner(args, **kwargs):
        command = args[2] if args[:2] == ["/bin/sh", "-lc"] else " ".join(args)
        commands.append(command)
        return subprocess.CompletedProcess(
            args=args,
            returncode=1,
            stdout=_audit_payload(
                [
                    {
                        "advisory": {
                            "id": "RUSTSEC-2024-0001",
                            "title": "Buffer overflow",
                            "severity": "medium",
                        },
                        "package": {"name": "serde", "version": "1.0.0"},
                    }
                ]
            ),
            stderr="",
        )

    result = run_audit_result(tmp_path, run_subprocess=runner)

    assert result.status == "ok"
    assert commands == [AUDIT_CMD]
    assert commands[0] == "cargo audit --json"
    assert result.entries == [
        {
            "file": "Cargo.lock",
            "line": 1,
            "message": "[RUSTSEC-2024-0001] serde 1.0.0: Buffer overflow (severity: medium)",
        }
    ]


def test_run_audit_result_reports_reduced_coverage_on_failure(tmp_path):
    def runner(args, **kwargs):
        raise FileNotFoundError("cargo audit not installed")

    result = run_audit_result(tmp_path, run_subprocess=runner)

    assert result.status == "error"
    assert result.error_kind == "tool_not_found"
