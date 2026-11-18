"""Old location of kvsel.verification; ticket 14 removes it."""

import sys

from . import verification

sys.modules[__name__] = verification
