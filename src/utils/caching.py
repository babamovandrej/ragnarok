import logging
from collections import OrderedDict
from typing import Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class LRUCache(Generic[T]):
    """A capped, least-recently-used cache; inserting past capacity evicts the oldest entry."""

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be at least 1, got {capacity}")
        self._capacity = capacity
        self._entries: OrderedDict[str, T] = OrderedDict()

    def get(self, key: str) -> T | None:
        """Return the cached value for key, marking it most-recently-used, or None if absent."""
        if key not in self._entries:
            return None
        self._entries.move_to_end(key)
        return self._entries[key]

    def put(self, key: str, value: T) -> None:
        """Insert/replace key as most-recently-used, evicting the oldest entry if over capacity."""
        if key in self._entries:
            self._entries.move_to_end(key)
        self._entries[key] = value
        if len(self._entries) > self._capacity:
            evicted_key, _ = self._entries.popitem(last=False)
            logger.info("Evicted %r from LRU cache (capacity=%d)", evicted_key, self._capacity)

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: str) -> bool:
        return key in self._entries
