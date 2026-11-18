"""NeXT's budgets: a share of each image's feature tokens, every share's kept set from one probe."""

from __future__ import annotations

import numpy as np

from kvsel.models.layout import Layout
from kvsel.selectors import Budget, kept_set

SHARES = (0.125, 0.25, 0.5)


def kept_sets_per_share(layout: Layout, attention: np.ndarray) -> dict[float, np.ndarray]:
    """Every share's PW-attn kept set from one probe: ``attention`` is the engine's (suffix tokens, visual) array."""
    return {share: kept_set(layout, "pw", Budget(share=share), attention=attention) for share in SHARES}
