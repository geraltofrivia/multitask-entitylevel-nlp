"""
As always, we start with trying to parse ontonotes well.
Specifically, we want
    - coref clusters
    - noun phrases
    - named entities (if annotated)

    - and predicted version of these properties? maybe not.

The dataclass is going to be document based. That is, one instance is one document.
"""
from typing import Iterable
from pathlib import Path


class CoNLLOntoNotesParser:

    def __init__(self, ontonotes_dir: Path, splits: Iterable[str] = ('train',)):
        """
            :param ontonotes_dir: Path to the folder containing `development`, `train`, `test` subfolders.
            :param splits: a tuple of which subfolders should we process
        """
        self.dir: Path = ontonotes_dir
        self.parsed: dict = {split_nm: [] for split_nm in splits}

    def run(self):
        ...

    def parse(self, split_nm):
        """ Where the actual parsing happens. One split at a time. """
        folder_dir: Path = self.dir / split_nm
        assert folder_dir.exists(), f"The split {split_nm} does not exist in {self.dir}."

        folder_dir: Path = folder_dir / 'data' / 'english' / 'annotations'

