"""Single source of truth for the service version.

Previously the version was a literal repeated in careflow/main.py (`2.5.0`), the health
endpoint (`1.0.0` via a `hasattr` guard on a setting that never existed), and
pyproject.toml -- three values that had already drifted apart.
"""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    API_VERSION = _pkg_version("careflow-dual-rag")
except PackageNotFoundError:  # running from a source tree without an install
    API_VERSION = "0.0.0+local"
