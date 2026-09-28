"""Package-level checks: the public names import, and the version is stated once."""
import importlib.metadata
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import reliax_core  # noqa: E402


def test_public_names_import():
    for name in reliax_core.__all__:
        assert hasattr(reliax_core, name), name


def test_version_matches_metadata():
    try:
        installed = importlib.metadata.version("reliax-core")
    except importlib.metadata.PackageNotFoundError:
        return  # running from a checkout that is not installed
    assert installed == reliax_core.__version__
