from pathlib import Path

import toml

import penelope


def test_versions_are_in_sync():
    """Checks if the pyproject.toml and package.__init__.py __version__ are in sync."""

    with open(str(Path(__file__).resolve().parents[2] / "pyproject.toml")) as fp:
        pyproject: str = toml.loads(fp.read())

    pyproject_version: str = pyproject["tool"]["poetry"]["version"]

    assert penelope.__version__ == pyproject_version
