from importlib.metadata import version


try:
    __version__ = version("cubed-xarray")
except Exception:
    # Local copy or not installed with setuptools.
    # Disable minimum version checks on downstream libraries.
    __version__ = "999"
