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
    repo_sha: str
    text: str
    imports: str = ""

    @property
    def search_text(self) -> str:
        """The text embedded, keyword-indexed and reranked: a location header, then the code.

        Code cut out of its file loses where it lives; the header puts the file path and
        symbol name back so they count when matching a question against the chunk.
        """
        location = (
            self.file_path
            if self.qualified_name == self.file_path
            else f"{self.file_path} · {self.qualified_name}"
        )
        return f"{location}\n{self.text}"


@dataclass
class IngestionStats:
    files_scanned: int = 0
    files_chunked: int = 0
    files_skipped: int = 0
    decode_failures: int = 0
    parse_failures: int = 0
    chunks_created: int = 0
    truncated: bool = False
