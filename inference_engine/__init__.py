import sys

from ._bindings import core, server

sys.modules[f"{__name__}.core"] = core
sys.modules[f"{__name__}.server"] = server