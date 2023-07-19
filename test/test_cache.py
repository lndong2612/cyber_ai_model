import os
import sys
from cachetools import LRUCache

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(WORKING_DIR, "../"))


class CacheTest():
    def __init__(self, cache):
        self.cache = cache

    def get_cache(self, id):
        if id not in self.cache:
            self.cache[id] = id
        return self.cache[id]

    def get_total(self):
        for id in self.cache:
            print(id)
        return len(self.cache)

    def remove_item(self, id):
        if id in self.cache:
            self.cache.pop(id)


if __name__ == "__main__":

    model_repository = CacheTest(
        LRUCache(maxsize=2)
    )
    model_repository.get_cache(1)
    model_repository.get_cache(2)
    model_repository.remove_item(2)

    print(model_repository.get_total())
