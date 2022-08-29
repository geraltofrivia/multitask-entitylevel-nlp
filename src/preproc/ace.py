"""
    We take already processed stuff from ACE2005-Toolkit repo and use those nice JSONs to create our docs
    Should be straightforward really.
"""
from pathlib import Path
from typing import Iterable

# Local imports
try:
    import _pathfix
except ImportError:
    from . import _pathfix
from config import LOCATIONS as LOC
from preproc.commons import GenericParser


class ACE2005Parser(GenericParser):
    """
        TODO: figure out how to control for certain folders not being counted as a valid split

    """

    def __init__(
            self,
            raw_dir: Path,
            suffixes: Iterable[str] = (),
            ignore_empty_documents: bool = False
    ):
        super().__init__(raw_dir=raw_dir, suffixes=suffixes, ignore_empty_documents=ignore_empty_documents)

        self.parsed: dict = {split_nm: [] for split_nm in suffixes}
        self.write_dir = LOC.parsed / "ace2005"
        self.write_dir.mkdir(parents=True, exist_ok=True)
