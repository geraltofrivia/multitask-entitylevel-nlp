import pickle
from pathlib import Path
from spacy import tokens
from abc import ABC, abstractmethod
from typing import Union, List, Dict, Tuple, Iterable

# Local imports
try:
    import _pathfix
except ImportError:
    from . import _pathfix
from utils.data import Document
from config import LOCATIONS as LOC


class GenericParser(ABC):

    def __init__(self, raw_dir: Path, splits: Iterable[str], ignore_empty_documents: bool = False):
        self.dir = raw_dir
        self.splits = splits
        self.ignore_empty_documents = ignore_empty_documents
        self.write_dir = LOC.parsed

    @abstractmethod
    def parse(self, split_nm: Union[Path, str]):
        ...

    def run(self):

        for split in self.splits:
            # First, clear out all the previously processed things from the disk
            self.delete_preprocessed_files(split)
            outputs = self.parse(split)

            # Dump them to disk
            self.write_to_disk(split, outputs)

    def get_span_heads(
            self, document: tokens.Doc, spans: List[List[int]]
    ) -> Dict[Tuple[int], List[int]]:
        """Return a dict where every original span has its corresponding head found out"""
        return {
            tuple(span): self._spacy_get_span_heads_(span, document) for span in spans
        }

    def _spacy_get_span_heads_(
            self, span: List[int], doc: tokens.Doc
    ) -> List[int]:
        """Get the spacy Span, get its root and return its word index (_, _+1)"""
        root = doc[span[0]: span[1]].root
        stop_list = ["det", "num", "adp", "DET", "NUM", "ADP"]
        if (
                root.pos_ in stop_list
        ):  # det can't be a root. So a hack to make sure that det can't be a det.
            if span[1] - span[0] == 1:
                # return the same thing
                assert root.i >= 0
                return [root.i, root.i + 1]
            elif span[1] - span[0] == 2:
                # return the other word
                root = doc[span[0] + 1]
                assert root.i >= 0
                return [root.i, root.i + 1]
            elif doc[span[0]].pos_ in stop_list:
                # remove the span[0] and call get_span_head
                return self._spacy_get_span_heads_([span[0] + 1, span[1]], doc)
            elif doc[span[1]].pos_ in stop_list:
                # remove the last one and call get_span_head again
                return self._spacy_get_span_heads_([span[0], span[1] - 1], doc)
            else:
                # root is somewhere in somewhere between span.
                assert root.i >= 0

                return [root.i, root.i + 1]
        else:
            return [root.i, root.i + 1]

    def delete_preprocessed_files(self, split: Union[str, Path]):
        """Since we are going to write things to the disk,
        it makes sense to delete all the processed gunk from this dir"""

        write_dir = self.write_dir / split

        if not write_dir.exists():
            return

        for f_name in write_dir.glob("*.pkl"):
            f_name.unlink()

    def write_to_disk(self, split: Union[str, Path], instances: List[Document]):
        """Write a (large) list of documents to disk"""

        # Assert that folder exists
        write_dir = self.write_dir / split
        write_dir.mkdir(parents=True, exist_ok=True)

        with (write_dir / "dump.pkl").open("wb+") as f:
            pickle.dump(instances, f)

        # with (write_dir / 'dump.jsonl').open('w+', encoding='utf8') as f:
        #     with jsonlines.Writer(f) as writer:
        #         writer.write_all([asdict(instance) for instance in instances])

        print(f"Successfully written {len(instances)} at {write_dir}.")
