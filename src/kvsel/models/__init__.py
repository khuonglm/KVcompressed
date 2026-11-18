"""Model families and the layouts of their visual spans. ``FAMILIES`` is the registry, by family name."""

from .family import Family
from .llava15 import LLAVA
from .llava_next import NEXT

FAMILIES: dict[str, Family] = {family.name: family for family in (LLAVA, NEXT)}
