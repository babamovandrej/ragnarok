import hashlib
import logging
import os
from collections.abc import Iterator, Mapping
from dataclasses import replace
from pathlib import Path

from tree_sitter import Node
from tree_sitter_language_pack import SupportedLanguage, get_parser

from src.config import (
    EXCLUDED_DIRECTORY_NAMES,
    LANGUAGE_BY_EXTENSION,
    LEADING_COMMENT_MAX_GAP_LINES,
    MAX_FILE_SIZE_BYTES,
    MAX_INDEXABLE_CHUNKS,
    MAX_SOURCE_BYTES,
    MAX_WHOLE_FILE_LINES,
    SOURCE_EXTENSIONS,
    WHOLE_FILE_LANGUAGES,
)
from src.ingestion.schemas import Chunk, IngestionStats, SymbolType

logger = logging.getLogger(__name__)

_LEADING_NODE_TYPES: frozenset[str] = frozenset(
    {"comment", "line_comment", "block_comment", "attribute_item"}
)

_WRAPPER_NODE_TYPES: frozenset[str] = frozenset({"decorated_definition", "template_declaration"})

_ENCLOSING_CONTAINER_TYPES: frozenset[str] = frozenset(
    {
        "class_definition",
        "class_declaration",
        "interface_declaration",
        "class_specifier",
        "struct_specifier",
        "impl_item",
        "trait_item",
    }
)

_CONTAINER_NAME_FIELD: dict[str, str] = {"impl_item": "type"}

_PYTHON_DEFINITION_TYPES: dict[str, SymbolType] = {
    "function_definition": "function",
    "class_definition": "class",
}

_TS_JS_DEFINITION_TYPES: dict[str, SymbolType] = {
    "function_declaration": "function",
    "method_definition": "method",
    "class_declaration": "class",
}

_GO_DEFINITION_TYPES: dict[str, SymbolType] = {
    "function_declaration": "function",
    "method_declaration": "method",
    "type_declaration": "type",
}

_RUST_DEFINITION_TYPES: dict[str, SymbolType] = {
    "function_item": "function",
    "struct_item": "struct",
}

_JAVA_DEFINITION_TYPES: dict[str, SymbolType] = {
    "method_declaration": "method",
    "class_declaration": "class",
    "interface_declaration": "interface",
}

_C_DEFINITION_TYPES: dict[str, SymbolType] = {
    "function_definition": "function",
    "struct_specifier": "struct",
}

_CPP_DEFINITION_TYPES: dict[str, SymbolType] = {
    "function_definition": "function",
    "class_specifier": "class",
    "struct_specifier": "struct",
}

_DEFINITION_TYPES_BY_LANGUAGE: dict[SupportedLanguage, dict[str, SymbolType]] = {
    "python": _PYTHON_DEFINITION_TYPES,
    "typescript": _TS_JS_DEFINITION_TYPES,
    "tsx": _TS_JS_DEFINITION_TYPES,
    "javascript": _TS_JS_DEFINITION_TYPES,
    "go": _GO_DEFINITION_TYPES,
    "rust": _RUST_DEFINITION_TYPES,
    "java": _JAVA_DEFINITION_TYPES,
    "c": _C_DEFINITION_TYPES,
    "cpp": _CPP_DEFINITION_TYPES,
}


def _dig_declarator_name(declarator: Node) -> Node:
    """Follow a declarator chain down to the actual identifier node."""
    node = declarator
    while True:
        inner = node.child_by_field_name("declarator")
        if inner is None:
            return node
        node = inner


def _node_name(node: Node, source: bytes) -> str:
    """Resolve a definition node's local name."""
    name_node = node.child_by_field_name("name")

    if name_node is None:
        declarator = node.child_by_field_name("declarator")
        if declarator is not None:
            name_node = _dig_declarator_name(declarator)

    if name_node is None:
        type_spec = next((c for c in node.children if c.type == "type_spec"), None)
        if type_spec is not None:
            name_node = type_spec.child_by_field_name("name")

    if name_node is None:
        return "<anonymous>"
    return source[name_node.start_byte : name_node.end_byte].decode("utf-8")


def _leading_start(node: Node, max_gap_lines: int = LEADING_COMMENT_MAX_GAP_LINES) -> Node:
    """Extend backwards over adjacent leading comments/attributes."""
    start = node
    sibling = node.prev_sibling
    while sibling is not None and sibling.type in _LEADING_NODE_TYPES:
        gap = start.start_point[0] - sibling.end_point[0]
        if gap > max_gap_lines:
            break
        start = sibling
        sibling = sibling.prev_sibling
    return start


def _container_name(container: Node, source: bytes) -> str | None:
    """Read an enclosing container node's name."""
    field_name = _CONTAINER_NAME_FIELD.get(container.type, "name")
    name_node = container.child_by_field_name(field_name)
    if name_node is None:
        return None
    return source[name_node.start_byte : name_node.end_byte].decode("utf-8")


def _go_receiver_type_name(method_node: Node, source: bytes) -> str | None:
    """Extract a Go method's receiver type name."""
    receiver = method_node.child_by_field_name("receiver")
    if receiver is None:
        return None
    receiver_param = next(
        (c for c in receiver.children if c.type == "parameter_declaration"), None
    )
    if receiver_param is None:
        return None
    type_node = receiver_param.child_by_field_name("type")
    if type_node is None:
        return None
    if type_node.type == "pointer_type":
        type_node = next(
            (c for c in type_node.children if c.type == "type_identifier"), type_node
        )
    return source[type_node.start_byte : type_node.end_byte].decode("utf-8")


def _qualified_name(
    def_node: Node, name: str, language: SupportedLanguage, source: bytes
) -> str:
    """Build a dotted name including enclosing scope."""
    if language == "go" and def_node.type == "method_declaration":
        receiver_type = _go_receiver_type_name(def_node, source)
        return f"{receiver_type}.{name}" if receiver_type is not None else name

    parts = [name]
    current = def_node.parent
    while current is not None:
        if current.type in _ENCLOSING_CONTAINER_TYPES:
            container_name = _container_name(current, source)
            if container_name is not None:
                parts.append(container_name)
        current = current.parent
    return ".".join(reversed(parts))


def _python_symbol_type(def_node: Node) -> SymbolType:
    """Distinguish a Python method from a plain function."""
    outer = def_node.parent
    if outer is not None and outer.type == "decorated_definition":
        outer = outer.parent
    if outer is not None and outer.type == "block":
        enclosing = outer.parent
        if enclosing is not None and enclosing.type == "decorated_definition":
            enclosing = enclosing.parent
        if enclosing is not None and enclosing.type == "class_definition":
            return "method"
    return "function"


def _rust_symbol_type(def_node: Node) -> SymbolType:
    """Distinguish a Rust method from a plain function."""
    outer = def_node.parent
    if outer is not None and outer.type == "declaration_list":
        enclosing = outer.parent
        if enclosing is not None and enclosing.type in ("impl_item", "trait_item"):
            return "method"
    return "function"


def _cpp_symbol_type(def_node: Node) -> SymbolType:
    """Distinguish a C++ method from a plain function."""
    outer = def_node.parent
    if outer is not None and outer.type == "field_declaration_list":
        enclosing = outer.parent
        if enclosing is not None and enclosing.type in ("class_specifier", "struct_specifier"):
            return "method"
    return "function"


def _go_type_symbol_type(def_node: Node) -> SymbolType:
    """Classify a Go type declaration as struct/interface/alias."""
    type_spec = next((c for c in def_node.children if c.type == "type_spec"), None)
    if type_spec is not None:
        value = type_spec.child_by_field_name("type")
        if value is not None:
            if value.type == "struct_type":
                return "struct"
            if value.type == "interface_type":
                return "interface"
    return "type"


def _symbol_type(
    def_node: Node, definition_types: Mapping[str, SymbolType], language: SupportedLanguage
) -> SymbolType:
    """Resolve a matched node's SymbolType."""
    if language == "go" and def_node.type == "type_declaration":
        return _go_type_symbol_type(def_node)

    base_type = definition_types[def_node.type]
    if base_type != "function":
        return base_type
    if language == "python":
        return _python_symbol_type(def_node)
    if language == "rust":
        return _rust_symbol_type(def_node)
    if language == "cpp":
        return _cpp_symbol_type(def_node)
    return base_type


def _iter_definitions(
    node: Node, definition_types: Mapping[str, SymbolType]
) -> Iterator[tuple[Node, Node]]:
    """Yield (definition_node, outer_node) pairs in document order; class
    chunks and their methods are both emitted (intentional overlap, not a
    bug — it lets retrieval match at either granularity)."""
    if node.type in _WRAPPER_NODE_TYPES:
        inner = next((c for c in node.children if c.type in definition_types), None)
        if inner is not None:
            yield inner, node
    elif node.type in definition_types and (
        node.parent is None or node.parent.type not in _WRAPPER_NODE_TYPES
    ):
        yield node, node

    for child in node.children:
        yield from _iter_definitions(child, definition_types)


def _chunk_id(file_path: str, symbol_type: SymbolType, qualified_name: str) -> str:
    """Hash-based chunk id, stable against unrelated line changes."""
    digest_input = f"{file_path}:{symbol_type}:{qualified_name}"
    return hashlib.sha256(digest_input.encode("utf-8")).hexdigest()


def _build_chunk(
    def_node: Node,
    outer_node: Node,
    symbol_type: SymbolType,
    source_bytes: bytes,
    lines: list[str],
    language: SupportedLanguage,
    file_path: str,
) -> Chunk:
    """Build a Chunk from a matched definition node."""
    start_node = _leading_start(outer_node)
    start_line = start_node.start_point[0] + 1
    end_line = def_node.end_point[0] + 1
    name = _node_name(def_node, source_bytes)
    qualified_name = _qualified_name(def_node, name, language, source_bytes)
    text = "\n".join(lines[start_line - 1 : end_line])
    return Chunk(
        chunk_id=_chunk_id(file_path, symbol_type, qualified_name),
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        symbol_name=name,
        qualified_name=qualified_name,
        symbol_type=symbol_type,
        language=language,
        text=text,
    )


def _disambiguate_qualified_names(chunks: list[Chunk]) -> list[Chunk]:
    """Disambiguate chunks that share the same qualified name (e.g. overloads)."""
    occurrences_by_key: dict[tuple[str, str, str], int] = {}
    disambiguated: list[Chunk] = []
    for chunk in chunks:
        key = (chunk.file_path, chunk.symbol_type, chunk.qualified_name)
        occurrences_by_key[key] = occurrences_by_key.get(key, 0) + 1
        occurrence = occurrences_by_key[key]
        if occurrence == 1:
            disambiguated.append(chunk)
            continue
        qualified_name = f"{chunk.qualified_name}#{occurrence}"
        disambiguated.append(
            replace(
                chunk,
                qualified_name=qualified_name,
                chunk_id=_chunk_id(chunk.file_path, chunk.symbol_type, qualified_name),
            )
        )
    return disambiguated


def _whole_file_chunk(
    source_text: str, language: SupportedLanguage, file_path: str
) -> list[Chunk]:
    """Chunk a whole file, splitting into bounded parts if too long."""
    lines = source_text.splitlines()
    if not lines:
        return []

    name = Path(file_path).name

    if len(lines) <= MAX_WHOLE_FILE_LINES:
        return [
            Chunk(
                chunk_id=_chunk_id(file_path, "file", file_path),
                file_path=file_path,
                start_line=1,
                end_line=len(lines),
                symbol_name=name,
                qualified_name=file_path,
                symbol_type="file",
                language=language,
                text=source_text,
            )
        ]

    line_groups = [
        lines[i : i + MAX_WHOLE_FILE_LINES] for i in range(0, len(lines), MAX_WHOLE_FILE_LINES)
    ]
    parts: list[Chunk] = []
    for index, part_lines in enumerate(line_groups, start=1):
        start_line = (index - 1) * MAX_WHOLE_FILE_LINES + 1
        end_line = start_line + len(part_lines) - 1
        qualified_name = f"{file_path}#{index}"
        parts.append(
            Chunk(
                chunk_id=_chunk_id(file_path, "file", qualified_name),
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                symbol_name=name,
                qualified_name=qualified_name,
                symbol_type="file",
                language=language,
                text="\n".join(part_lines),
            )
        )
    return parts


def chunk_source(
    source_text: str, language: SupportedLanguage, file_path: str
) -> list[Chunk]:
    """Chunk one file's source text into a list of Chunks."""
    if language in WHOLE_FILE_LANGUAGES:
        return _whole_file_chunk(source_text, language, file_path)

    definition_types = _DEFINITION_TYPES_BY_LANGUAGE.get(language)
    if definition_types is None:
        raise ValueError(f"Unsupported chunking language: {language}")

    source_bytes = source_text.encode("utf-8")
    tree = get_parser(language).parse(source_bytes)
    if tree.root_node.has_error:
        logger.warning("Parse errors in %s (language=%s)", file_path, language)
    lines = source_text.splitlines()

    chunks = [
        _build_chunk(
            def_node,
            outer_node,
            _symbol_type(def_node, definition_types, language),
            source_bytes,
            lines,
            language,
            file_path,
        )
        for def_node, outer_node in _iter_definitions(tree.root_node, definition_types)
    ]

    if not chunks:
        return _whole_file_chunk(source_text, language, file_path)

    return _disambiguate_qualified_names(chunks)


def chunk_file(path: Path, repo_root: Path) -> list[Chunk]:
    """Read a file from disk and chunk it based on its extension."""
    language = LANGUAGE_BY_EXTENSION.get(path.suffix)
    if language is None:
        return []
    source_text = path.read_text(encoding="utf-8")
    relative_path = path.relative_to(repo_root).as_posix()
    return chunk_source(source_text, language, relative_path)


def collect_chunks(repo_root: Path) -> tuple[list[Chunk], IngestionStats]:
    """Walk a repo and chunk every recognized source file."""
    stats = IngestionStats()
    chunks: list[Chunk] = []
    total_bytes = 0

    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDED_DIRECTORY_NAMES)

        for filename in sorted(filenames):
            path = Path(dirpath) / filename
            if path.suffix not in SOURCE_EXTENSIONS:
                continue

            stats.files_scanned += 1

            try:
                file_size = path.stat().st_size
            except OSError:
                stats.files_skipped += 1
                continue

            if file_size > MAX_FILE_SIZE_BYTES or total_bytes + file_size > MAX_SOURCE_BYTES:
                stats.files_skipped += 1
                continue

            try:
                file_chunks = chunk_file(path, repo_root)
            except UnicodeDecodeError:
                logger.warning("Skipping %s: invalid UTF-8", path)
                stats.decode_failures += 1
                stats.files_skipped += 1
                continue
            except Exception:
                logger.warning("Skipping %s: unexpected chunking error", path, exc_info=True)
                stats.parse_failures += 1
                stats.files_skipped += 1
                continue

            if not file_chunks:
                continue

            total_bytes += file_size
            stats.files_chunked += 1
            chunks.extend(file_chunks)

            if len(chunks) >= MAX_INDEXABLE_CHUNKS:
                chunks = chunks[:MAX_INDEXABLE_CHUNKS]
                stats.chunks_created = len(chunks)
                return chunks, stats

    stats.chunks_created = len(chunks)
    return chunks, stats
