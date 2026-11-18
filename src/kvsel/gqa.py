"""Old location of kvsel.items.gqa; ticket 14 removes it."""

import sys

from .items import gqa

sys.modules[__name__] = gqa
