import sys

from ._bindings import core

sys.modules[f"{__name__}.core"] = core