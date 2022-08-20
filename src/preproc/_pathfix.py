"""
    Python really needs to fix its path game :/
    You don't need to do this IF you're running this via an IDE and the python paths have been set properly.

    What is happening here?: **Add source folder root to sys.path**
    1. Get os.getcwd()
        output: something like ~/Dev/research/src/new/src/ or /home/priyansh/Dev/research/src/new
    2. If the path is latter (repo root), not source folder root (i.e. <repo root>/src,
        append src to the path
    3. Add it to sys.path

    How to use it?
    - import _pathfix
        in all local files -> all files in the same directory
    - but but... pycharm gives module not used error.
        glad you asked. Invoke _pathfix.suppress_unused_import() somewhere. It does nothing.
"""
import os
import sys
from pathlib import Path

root_dir: Path = Path(os.getcwd())
if root_dir.name == "src":
    root_dir = root_dir.parent
if root_dir.name == 'preproc':
    root_dir = root_dir.parent.parent

sys.path.append(str(root_dir))
sys.path.append(str(root_dir / "src"))


def suppress_unused_import():
    pass


print(f"Fixing paths from {root_dir}")
