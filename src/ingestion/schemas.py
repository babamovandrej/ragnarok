from dataclasses import dataclass
from typing import Literal

from tree_sitter_language_pack import SupportedLanguage

SymbolType = Literal["function", "method", "class", "interface", "struct", "type", "file"]


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    file_path: str
    start_line: int
    end_line: int
    symbol_name: str
    qualified_name: str
    symbol_type: SymbolType
    language: SupportedLanguage
    text: str


@dataclass
class IngestionStats:
    files_scanned: int = 0
    files_chunked: int = 0
    files_skipped: int = 0
    decode_failures: int = 0
    parse_failures: int = 0
    chunks_created: int = 0
