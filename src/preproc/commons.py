import json
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, List, Dict, Tuple, Iterable, Optional

import spacy
from spacy import tokens

# Local imports
try:
    import _pathfix
except ImportError:
    from . import _pathfix
from utils.data import Document
from utils.nlp import PreTokenizedPreSentencizedTokenizer
from config import LOCATIONS as LOC


class GenericParser(ABC):

    def __init__(
            self,
            raw_dir: Path,
            suffixes: Iterable[str],
            write_dir: Path = LOC.parsed,
            ignore_empty_documents: bool = False,
            ignore_token_replacements: bool = False
    ):
        self.dir = raw_dir
        self.suffixes = suffixes
        self.write_dir = write_dir
        self.flag_ignore_empty_documents = ignore_empty_documents
        self.flag_ignore_token_replacements = ignore_token_replacements

        # Pull word replacements from the manually entered list
        with (LOC.manual / "replacements.json").open("r") as f:
            self.replacements = json.load(f)

        exclude = ["senter", "parser"]
        self.nlp = spacy.load("en_core_web_sm", exclude=exclude)
        self.nlp.tokenizer = PreTokenizedPreSentencizedTokenizer(self.nlp.vocab)

        self._speaker_vocab_ = {}

    @abstractmethod
    def parse(self, split_nm: Union[Path, str]):
        ...

    @staticmethod
    def get_pos_tags(doc: tokens.Doc) -> List[List[str]]:
        """ Get pos tags for each token, respecting the sentence boundaries, i.e. each sent is a list """
        return [[token.pos_ for token in sent] for sent in doc.sents]

    def run(self):

        for split in self.suffixes:
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

    def write_to_disk(self, suffix: Optional[Union[str, Path]], instances: List[Document]):
        """Write a (large) list of documents to disk"""

        # Assert that folder exists
        if suffix:
            write_dir = self.write_dir / suffix
        else:
            write_dir = self.write_dir
        write_dir.mkdir(parents=True, exist_ok=True)

        with (write_dir / "dump.pkl").open("wb+") as f:
            pickle.dump(instances, f)

        # with (write_dir / 'dump.jsonl').open('w+', encoding='utf8') as f:
        #     with jsonlines.Writer(f) as writer:
        #         writer.write_all([asdict(instance) for instance in instances])

        print(f"Successfully written {len(instances)} at {write_dir}.")

    def _finalise_instance_(self, document: Document, spacy_doc: tokens.Doc) -> Document:
        """ Find span heads... populate words in different annotations etc etc. """
        spans = document.get_all_spans()
        span_heads = self.get_span_heads(spacy_doc, spans=spans)

        # Allocate span heads to these different forms of annotations
        document.coref.allocate_span_heads(span_heads)
        document.ner.allocate_span_heads(span_heads)
        document.rel.allocate_span_heads(span_heads)
        document.bridging.allocate_span_heads(span_heads)

        # Assign words to coref (should be done AFTER assigning span heads)
        # NOTE: this automatically adds words_head and pos_head also.
        document.coref.add_words(document.document)
        document.ner.add_words(document.document)
        document.coref.add_pos(document.pos)
        document.ner.add_pos(document.pos)

        return document
