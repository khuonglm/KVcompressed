"""Old location of kvsel.items.images; ticket 14 removes it."""

import sys

from .items import images

sys.modules[__name__] = images
