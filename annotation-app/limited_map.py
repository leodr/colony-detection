import time
from typing import Generic, TypeVar

T = TypeVar("T")


class LimitedMap(Generic[T]):
    """A map that has a limit on the number of items it can hold. When the
    limit is reached, the oldest item is removed to make room for the new
    item.
    """

    def __init__(self, item_limit: int):
        self.item_limit = item_limit
        self.map = {}
        self.time_map = {}

    def __getitem__(self, key: str) -> T:
        return self.map[key]

    def __setitem__(self, key: str, value: T) -> None:
        if len(self.map) >= self.item_limit:
            oldest_key = min(self.time_map, key=self.time_map.get)
            del self.map[oldest_key]
            del self.time_map[oldest_key]

        self.map[key] = value
        self.time_map[key] = time.time()

    def __delitem__(self, key: str) -> None:
        del self.map[key]

    def __contains__(self, key: str) -> bool:
        return key in self.map
