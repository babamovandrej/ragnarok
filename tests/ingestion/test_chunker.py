from pathlib import Path

import pytest

from src.ingestion.chunker import chunk_file, chunk_source, collect_chunks, is_excluded_path

FIXTURES = Path(__file__).parent.parent / "fixtures" / "chunker"


def test_python_chunks_match_expected_line_ranges():
    source = (FIXTURES / "sample.py").read_text()

    chunks = chunk_source(source, "python", "sample.py", "sha1")

    actual = [(c.start_line, c.end_line, c.symbol_type, c.symbol_name) for c in chunks]
    assert actual == [
        (6, 9, "function", "top_level_function"),
        (12, 20, "class", "Example"),
        (15, 16, "method", "method_one"),
        (18, 20, "method", "method_two"),
        (23, 25, "function", "decorated_function"),
    ]


def test_python_qualified_names_disambiguate_same_named_methods():
    source = (FIXTURES / "sample.py").read_text()

    chunks = chunk_source(source, "python", "sample.py", "sha1")

    by_name = {c.symbol_name: c.qualified_name for c in chunks}
    assert by_name["method_one"] == "Example.method_one"
    assert by_name["method_two"] == "Example.method_two"
    assert by_name["top_level_function"] == "top_level_function"


def test_python_leading_comment_is_included_in_chunk_text():
    source = (FIXTURES / "sample.py").read_text()

    chunks = chunk_source(source, "python", "sample.py", "sha1")

    top_level = next(c for c in chunks if c.symbol_name == "top_level_function")
    assert top_level.text.startswith("# Leading comment for top_level_function")


def test_typescript_chunks_match_expected_line_ranges():
    source = (FIXTURES / "sample.ts").read_text()

    chunks = chunk_source(source, "typescript", "sample.ts", "sha1")

    actual = [(c.start_line, c.end_line, c.symbol_type, c.symbol_name) for c in chunks]
    assert actual == [
        (1, 4, "function", "topLevelFunction"),
        (6, 15, "class", "Example"),
        (7, 10, "method", "methodOne"),
        (12, 14, "method", "methodTwo"),
    ]


def test_go_chunks_match_expected_line_ranges():
    source = (FIXTURES / "sample.go").read_text()

    chunks = chunk_source(source, "go", "sample.go", "sha1")

    actual = [(c.start_line, c.end_line, c.symbol_type, c.symbol_name) for c in chunks]
    assert actual == [
        (3, 6, "function", "Foo"),
        (8, 10, "struct", "Bar"),
        (12, 14, "method", "Method"),
    ]


def test_go_method_qualified_name_uses_receiver_type():
    source = (FIXTURES / "sample.go").read_text()

    chunks = chunk_source(source, "go", "sample.go", "sha1")

    method = next(c for c in chunks if c.symbol_name == "Method")
    assert method.qualified_name == "Bar.Method"


def test_go_type_declaration_distinguishes_struct_interface_and_alias():
    source = (
        "package main\n\n"
        "type Bar struct {\n\tX int\n}\n\n"
        "type Greeter interface {\n\tGreet() string\n}\n\n"
        "type ID int\n"
    )

    chunks = chunk_source(source, "go", "sample.go", "sha1")

    by_name = {c.symbol_name: c.symbol_type for c in chunks}
    assert by_name["Bar"] == "struct"
    assert by_name["Greeter"] == "interface"
    assert by_name["ID"] == "type"


def test_rust_chunks_match_expected_line_ranges():
    source = (FIXTURES / "sample.rs").read_text()

    chunks = chunk_source(source, "rust", "sample.rs", "sha1")

    actual = [(c.start_line, c.end_line, c.symbol_type, c.symbol_name) for c in chunks]
    assert actual == [
        (1, 4, "function", "foo"),
        (6, 9, "struct", "Bar"),
        (12, 14, "method", "method"),
    ]


def test_rust_method_qualified_name_uses_impl_type():
    source = (FIXTURES / "sample.rs").read_text()

    chunks = chunk_source(source, "rust", "sample.rs", "sha1")

    method = next(c for c in chunks if c.symbol_name == "method")
    assert method.qualified_name == "Bar.method"


def test_java_chunks_match_expected_line_ranges():
    source = (FIXTURES / "Sample.java").read_text()

    chunks = chunk_source(source, "java", "Sample.java", "sha1")

    actual = [(c.start_line, c.end_line, c.symbol_type, c.symbol_name) for c in chunks]
    assert actual == [
        (1, 6, "class", "Bar"),
        (2, 5, "method", "method"),
        (8, 10, "interface", "Baz"),
        (9, 9, "method", "other"),
    ]


def test_c_chunks_match_expected_line_ranges():
    source = (FIXTURES / "sample.c").read_text()

    chunks = chunk_source(source, "c", "sample.c", "sha1")

    actual = [(c.start_line, c.end_line, c.symbol_type, c.symbol_name) for c in chunks]
    assert actual == [
        (1, 4, "function", "foo"),
        (6, 8, "struct", "Bar"),
    ]


def test_cpp_chunks_match_expected_line_ranges():
    source = (FIXTURES / "sample.cpp").read_text()

    chunks = chunk_source(source, "cpp", "sample.cpp", "sha1")

    actual = [(c.start_line, c.end_line, c.symbol_type, c.symbol_name) for c in chunks]
    assert actual == [
        (1, 3, "function", "foo"),
        (5, 10, "class", "Bar"),
        (7, 9, "method", "method"),
        (12, 14, "struct", "Baz"),
        (13, 13, "method", "structMethod"),
    ]


def test_cpp_template_function_includes_template_line():
    source = "template<typename T>\nT foo(T x) {\n    return x;\n}\n"

    chunks = chunk_source(source, "cpp", "sample.cpp", "sha1")

    assert len(chunks) == 1
    assert (chunks[0].start_line, chunks[0].end_line) == (1, 4)
    assert chunks[0].text.startswith("template<typename T>")


def test_cpp_constructor_destructor_and_operator_names():
    source = (
        "class Bar {\n"
        "public:\n"
        "    Bar() {}\n"
        "    ~Bar() {}\n"
        "    int operator+(int x) { return x; }\n"
        "};\n"
    )

    chunks = chunk_source(source, "cpp", "sample.cpp", "sha1")

    names = [c.symbol_name for c in chunks if c.symbol_type == "method"]
    assert names == ["Bar", "~Bar", "operator+"]


def test_cpp_overloaded_methods_get_unique_disambiguated_ids():
    source = "class Bar {\npublic:\n    void foo(int x) {}\n    void foo(double x) {}\n};\n"

    chunks = chunk_source(source, "cpp", "overload.cpp", "sha1")
    overloads = [c for c in chunks if c.symbol_name == "foo"]

    assert [c.qualified_name for c in overloads] == ["Bar.foo", "Bar.foo#2"]
    assert len({c.chunk_id for c in overloads}) == 2


def test_yaml_produces_one_whole_file_chunk():
    source = (FIXTURES / "config.yaml").read_text()

    chunks = chunk_source(source, "yaml", "config.yaml", "sha1")

    assert len(chunks) == 1
    chunk = chunks[0]
    assert (chunk.start_line, chunk.end_line, chunk.symbol_type) == (1, 2, "file")
    assert chunk.symbol_name == "config.yaml"
    assert chunk.text == source


def test_json_and_toml_also_produce_one_whole_file_chunk():
    json_chunks = chunk_source('{"a": 1}\n', "json", "data.json", "sha1")
    assert len(json_chunks) == 1
    assert json_chunks[0].symbol_type == "file"

    toml_chunks = chunk_source("a = 1\n", "toml", "data.toml", "sha1")
    assert len(toml_chunks) == 1
    assert toml_chunks[0].symbol_type == "file"


def test_empty_config_file_produces_no_chunks():
    assert chunk_source("", "yaml", "empty.yaml", "sha1") == []


def test_oversized_whole_file_is_split_into_bounded_parts():
    big_yaml = "\n".join(f"key{i}: {i}" for i in range(950))

    chunks = chunk_source(big_yaml, "yaml", "big.yaml", "sha1")

    assert [(c.start_line, c.end_line) for c in chunks] == [
        (1, 400),
        (401, 800),
        (801, 950),
    ]
    assert [c.qualified_name for c in chunks] == [
        "big.yaml#1",
        "big.yaml#2",
        "big.yaml#3",
    ]


def test_oversized_single_line_file_is_split_by_character_count(monkeypatch):
    monkeypatch.setattr("src.ingestion.chunker.MAX_WHOLE_FILE_CHARS", 100)
    minified = "x=1;" * 50

    chunks = chunk_source(minified, "json", "bundle.min.json", "sha1")

    assert len(chunks) == 2
    assert all(len(c.text) <= 100 for c in chunks)
    assert "".join(c.text for c in chunks) == minified
    assert [c.qualified_name for c in chunks] == ["bundle.min.json#1", "bundle.min.json#2"]


def test_oversized_single_line_file_chunks_still_report_the_files_line_range(monkeypatch):
    monkeypatch.setattr("src.ingestion.chunker.MAX_WHOLE_FILE_CHARS", 100)
    minified = "x=1;" * 50

    chunks = chunk_source(minified, "json", "bundle.min.json", "sha1")

    assert all((c.start_line, c.end_line) == (1, 1) for c in chunks)


def test_file_under_both_caps_is_not_split(monkeypatch):
    monkeypatch.setattr("src.ingestion.chunker.MAX_WHOLE_FILE_CHARS", 100)

    chunks = chunk_source("x=1;", "json", "small.json", "sha1")

    assert len(chunks) == 1


def test_file_with_no_definitions_falls_back_to_whole_file_chunk():
    source = 'import os\n\nCONFIG = {"a": 1}\n'

    chunks = chunk_source(source, "python", "constants.py", "sha1")

    assert len(chunks) == 1
    assert chunks[0].symbol_type == "file"
    assert chunks[0].text == source


def test_unsupported_language_raises_value_error():
    with pytest.raises(ValueError, match="Unsupported chunking language"):
        chunk_source("x", "markdown", "x.md", "sha1")


def test_leading_comment_attaches_only_when_adjacent():
    attached = chunk_source("# comment\ndef foo():\n    pass\n", "python", "x.py", "sha1")
    assert attached[0].start_line == 1

    not_attached = chunk_source(
        "# unrelated comment\n\ndef foo():\n    pass\n", "python", "x.py", "sha1"
    )
    assert not_attached[0].start_line == 3


def test_chunk_id_is_stable_against_unrelated_line_changes():
    original = "def foo():\n    return 1\n"
    with_extra_blank_lines_above = "\n\n\ndef foo():\n    return 1\n"

    chunk_a = chunk_source(original, "python", "sample.py", "sha1")[0]
    chunk_b = chunk_source(with_extra_blank_lines_above, "python", "sample.py", "sha1")[0]

    assert chunk_a.chunk_id == chunk_b.chunk_id


def test_chunk_file_reads_invalid_utf8_raises(tmp_path):
    bad_file = tmp_path / "bad.py"
    bad_file.write_bytes(b"x = 1\ny = \xff\xfe invalid utf8\n")

    with pytest.raises(UnicodeDecodeError):
        chunk_file(bad_file, tmp_path, "sha1")


def test_chunk_file_uses_posix_relative_paths(tmp_path):
    nested = tmp_path / "pkg" / "mod.py"
    nested.parent.mkdir()
    nested.write_text("def foo():\n    pass\n")

    chunks = chunk_file(nested, tmp_path, "sha1")

    assert chunks[0].file_path == "pkg/mod.py"


def test_collect_chunks_prunes_excluded_directories_and_reports_stats(tmp_path):
    (tmp_path / "good.py").write_text("def foo():\n    return 1\n")
    excluded = tmp_path / "node_modules"
    excluded.mkdir()
    (excluded / "lib.py").write_text("def should_not_be_chunked():\n    pass\n")

    chunks, stats = collect_chunks(tmp_path, "sha1")

    assert [c.symbol_name for c in chunks] == ["foo"]
    assert stats.files_scanned == 1
    assert stats.files_chunked == 1
    assert stats.chunks_created == 1


def test_collect_chunks_skips_invalid_utf8_and_continues(tmp_path):
    (tmp_path / "good.py").write_text("def foo():\n    return 1\n")
    (tmp_path / "broken.py").write_bytes(b"\xff\xfe garbage")

    chunks, stats = collect_chunks(tmp_path, "sha1")

    assert [c.symbol_name for c in chunks] == ["foo"]
    assert stats.decode_failures == 1
    assert stats.files_skipped == 1


def test_collect_chunks_flags_truncation_when_the_chunk_cap_is_hit(tmp_path, monkeypatch):
    monkeypatch.setattr("src.ingestion.chunker.MAX_INDEXABLE_CHUNKS", 1)
    (tmp_path / "a.py").write_text("def foo():\n    return 1\n")
    (tmp_path / "b.py").write_text("def bar():\n    return 2\n")

    chunks, stats = collect_chunks(tmp_path, "sha1")

    assert len(chunks) == 1
    assert stats.chunks_created == 1
    assert stats.truncated is True


def test_collect_chunks_does_not_flag_truncation_under_the_cap(tmp_path):
    (tmp_path / "a.py").write_text("def foo():\n    return 1\n")

    _, stats = collect_chunks(tmp_path, "sha1")

    assert stats.truncated is False


def test_collect_chunks_stamps_every_chunk_with_the_given_repo_sha(tmp_path):
    (tmp_path / "a.py").write_text("def foo():\n    return 1\n")

    chunks, _ = collect_chunks(tmp_path, "deadbeef")

    assert all(c.repo_sha == "deadbeef" for c in chunks)


def test_python_definition_chunks_carry_the_files_imports():
    source = (FIXTURES / "sample.py").read_text()

    chunks = chunk_source(source, "python", "sample.py", "sha1")

    assert all(c.imports == "import os" for c in chunks)


def test_imports_are_excluded_from_the_embedded_chunk_text():
    source = (FIXTURES / "sample.py").read_text()

    chunks = chunk_source(source, "python", "sample.py", "sha1")

    assert all("import os" not in c.text for c in chunks)


def test_multiple_import_statements_are_joined_in_document_order():
    source = "import os\nfrom sys import argv\n\n\ndef foo():\n    return os.getcwd()\n"

    chunks = chunk_source(source, "python", "multi_import.py", "sha1")

    assert chunks[0].imports == "import os\nfrom sys import argv"


def test_typescript_definition_chunks_carry_the_files_imports():
    source = "import { Foo } from './foo';\n\nfunction bar(): number {\n    return 1;\n}\n"

    chunks = chunk_source(source, "typescript", "bar.ts", "sha1")

    assert chunks[0].imports == "import { Foo } from './foo';"


def test_go_definition_chunks_carry_the_files_imports():
    source = 'package main\n\nimport "fmt"\n\nfunc Foo() {\n\tfmt.Println("hi")\n}\n'

    chunks = chunk_source(source, "go", "main.go", "sha1")

    assert chunks[0].imports == 'import "fmt"'


def test_rust_definition_chunks_carry_the_files_imports():
    source = "use std::io;\n\nfn foo() -> i32 {\n    return 1;\n}\n"

    chunks = chunk_source(source, "rust", "lib.rs", "sha1")

    assert chunks[0].imports == "use std::io;"


def test_file_with_no_imports_has_an_empty_imports_string():
    source = "def foo():\n    return 1\n"

    chunks = chunk_source(source, "python", "no_imports.py", "sha1")

    assert chunks[0].imports == ""


def test_whole_file_fallback_chunk_has_no_separate_imports_field():
    source = 'import os\n\nCONFIG = {"a": 1}\n'

    chunks = chunk_source(source, "python", "constants.py", "sha1")

    assert chunks[0].symbol_type == "file"
    assert chunks[0].imports == ""


@pytest.mark.parametrize(
    "relative_path",
    ["tests/test_app.py", "pkg/tests/conftest.py", "node_modules/x/index.js", "src/__tests__/a.ts"],
)
def test_paths_under_an_excluded_directory_are_excluded(relative_path):
    assert is_excluded_path(relative_path)


@pytest.mark.parametrize("relative_path", ["starlette/routing.py", "tests.py", "src/testing.py"])
def test_paths_outside_excluded_directories_are_not_excluded(relative_path):
    assert not is_excluded_path(relative_path)


def test_collect_chunks_skips_test_directories(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "core.py").write_text("def core():\n    return 1\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_core.py").write_text("def test_core():\n    assert True\n")

    chunks, _ = collect_chunks(tmp_path, "sha1")

    assert {chunk.file_path for chunk in chunks} == {"pkg/core.py"}
