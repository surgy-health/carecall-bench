"""Shared test setup.

If pandas can't be imported (e.g. Windows Application Control blocks its compiled DLL),
replace it with an inert stub; see scripts/pandas_compat.py.
"""

from scripts.pandas_compat import install_if_blocked

install_if_blocked()
