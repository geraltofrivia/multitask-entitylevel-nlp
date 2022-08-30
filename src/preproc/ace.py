"""
    We take already processed stuff from ACE2005-Toolkit repo and use those nice JSONs to create our docs
    Should be straightforward really.
"""
import json
from pathlib import Path
from typing import Iterable, Union, List

# Local imports
try:
    import _pathfix
except ImportError:
    from . import _pathfix
from config import LOCATIONS as LOC
from preproc.commons import GenericParser
from utils.data import Document

class ACE2005Parser(GenericParser):

    def __init__(
            self,
            raw_dir: Path,
            suffixes: Iterable[str] = (),
            ignore_empty_documents: bool = False
    ):
        cleared_suffixes = []
        for suffix in suffixes:
            if suffix not in ['train', 'test', 'dev']:
                continue
            cleared_suffixes.append(suffix)

        super().__init__(raw_dir=raw_dir, suffixes=cleared_suffixes, ignore_empty_documents=ignore_empty_documents)

        self.parsed: dict = {split_nm: [] for split_nm in cleared_suffixes}
        self.write_dir = LOC.parsed / "ace2005"
        self.write_dir.mkdir(parents=True, exist_ok=True)

    def parse(self, split_nm: Union[Path, str]) -> List[Document]:

        outputs: List[Document] = []
        filenames: Iterable[Path] = (self.dir / split_nm).glob('*.json')

        # TODO:test this stuff out
        for filename in filenames:
            document = self._parse_document_(filename)
            outputs.append(document)

        return outputs

    def _parse_document_(self, filename: Path):

        with filename.open('r') as f:
            raw = json.load(f)

        print(raw[:10])
        return ''


if __name__ == '__main__':
    parser = ACE2005Parser(LOC.ace, ['train'])
    parser.run()
