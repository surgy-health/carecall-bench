"""Let tau2 import on machines where pandas can't load.

tau2 imports pandas at package import time, but only its metrics/report code
(`Results.to_df`, `tau2.metrics`) uses it. Simulating, grading and saving results
do not. On machines where Windows Application Control blocks pandas' compiled
DLL, `install_if_blocked()` puts an inert stub in `sys.modules` so the rest of
tau2 still works. Don't call tau2's metrics code while the stub is installed.
"""

import sys
import types


class _Any:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return _Any()

    def __getattr__(self, name):
        return _Any()


def install_if_blocked() -> bool:
    """Install the stub if pandas can't be imported. Returns True if it did."""
    try:
        import pandas  # noqa: F401
    except ImportError:
        stub = types.ModuleType("pandas")
        stub.__dict__["CARECALL_STUB"] = True
        stub.__getattr__ = lambda name: _Any
        sys.modules["pandas"] = stub
        return True
    return False


def stub_installed() -> bool:
    """True if the pandas in sys.modules is our stub (set here or by an earlier caller)."""
    return getattr(sys.modules.get("pandas"), "__dict__", {}).get("CARECALL_STUB", False)
