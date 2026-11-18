"""The visual span of one image: its feature maps and its newline tokens. Pure; no model access."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Map:
    offset: int  # span index of the map's first token
    rows: int
    cols: int

    @property
    def size(self) -> int:
        return self.rows * self.cols


@dataclass(frozen=True)
class Layout:
    """Maps in span order; a map's tokens are row-major, with any newlines between them excluded."""

    maps: tuple[Map, ...]
    newlines: tuple[int, ...] = ()

    def __post_init__(self):
        if list(self.newlines) != sorted(set(self.newlines)) or not all(0 <= n < self.total for n in self.newlines):
            raise ValueError("newlines must be distinct, sorted positions inside the span")
        features, before = self.features, 0
        for map_ in self.maps:
            if map_.size and features[before] != map_.offset:
                raise ValueError(f"a map starts at {map_.offset}, but its first feature token is at {features[before]}")
            before += map_.size

    @property
    def total(self) -> int:
        return sum(map_.size for map_ in self.maps) + len(self.newlines)

    @property
    def features(self) -> np.ndarray:
        """Span indices of every feature token: all but the newlines."""
        return np.setdiff1d(np.arange(self.total), np.array(self.newlines, dtype=np.int64))

    def positions(self, index: int) -> np.ndarray:
        """Span indices of map ``index`` as a (rows, cols) array."""
        before = sum(map_.size for map_ in self.maps[:index])
        map_ = self.maps[index]
        return self.features[before : before + map_.size].reshape(map_.rows, map_.cols)
