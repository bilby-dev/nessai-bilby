"""Interface and plugin for using nessai in bilby.

Includes support for standard nessai and inessai.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"
