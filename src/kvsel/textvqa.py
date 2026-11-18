"""Old location of kvsel.items.textvqa; ticket 14 removes it."""

import sys

from .items import textvqa

sys.modules[__name__] = textvqa
