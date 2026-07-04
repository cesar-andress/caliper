"""Result persistence and structured storage."""

from caliper.storage.base import ResultStore
from caliper.storage.formats import read_results, write_results

__all__ = ["ResultStore", "read_results", "write_results"]
