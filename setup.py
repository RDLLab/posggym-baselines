"""Setup for the posggym_baselines package."""
import pathlib

from setuptools import setup

CWD = pathlib.Path(__file__).absolute().parent


def get_version():
    """Gets the library version."""
    path = CWD / "posggym_baselines" / "__init__.py"
    content = path.read_text()

    for line in content.splitlines():
        if line.startswith("__version__"):
            return line.strip().split()[-1].strip().strip('"')
    raise RuntimeError("bad version data in __init__.py")


setup(name="posggym_baselines", version=get_version())
