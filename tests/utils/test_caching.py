import pytest

from src.utils.caching import LRUCache


class TestLRUCache:
    def test_get_returns_none_for_a_missing_key(self):
        cache: LRUCache[str] = LRUCache(capacity=2)

        assert cache.get("missing") is None

    def test_put_then_get_returns_the_stored_value(self):
        cache: LRUCache[str] = LRUCache(capacity=2)

        cache.put("a", "value-a")

        assert cache.get("a") == "value-a"
        assert len(cache) == 1
        assert "a" in cache

    def test_evicts_the_least_recently_used_entry_past_capacity(self):
        cache: LRUCache[str] = LRUCache(capacity=2)

        cache.put("a", "1")
        cache.put("b", "2")
        cache.put("c", "3")

        assert "a" not in cache
        assert cache.get("b") == "2"
        assert cache.get("c") == "3"
        assert len(cache) == 2

    def test_getting_an_entry_marks_it_most_recently_used(self):
        cache: LRUCache[str] = LRUCache(capacity=2)
        cache.put("a", "1")
        cache.put("b", "2")

        cache.get("a")
        cache.put("c", "3")

        assert "a" in cache
        assert "b" not in cache

    def test_re_putting_an_existing_key_updates_its_value_and_recency(self):
        cache: LRUCache[str] = LRUCache(capacity=2)
        cache.put("a", "1")
        cache.put("b", "2")

        cache.put("a", "1-updated")
        cache.put("c", "3")

        assert cache.get("a") == "1-updated"
        assert "b" not in cache

    def test_rejects_a_non_positive_capacity(self):
        with pytest.raises(ValueError):
            LRUCache(capacity=0)
