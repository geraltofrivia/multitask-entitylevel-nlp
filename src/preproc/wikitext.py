"""
    We use Huggingface datasets to get Wikitext and then do our magic to transform them into Documents to be dumped.
    [Link to raw](https://huggingface.co/datasets/DFKI-SLT/wikitext_linked_
"""

from pathlib import Path
from typing import Iterable, Union, List, Dict, Tuple

import click
from tqdm.auto import tqdm
from datasets import load_dataset

# Local imports
try
    import _pathfix
except ImportError:
    from . import _pathfix
from preproc.commons import GenericParser
from config import LOCATIONS as LOC
from dataiter import DocumentReader
from utils.data import Document, NamedEntities


class WikiTextParser(GenericParser):

    def __init__(
            self,
            suffixes: Iterable[str] = ()
    ):
        super().__init__(raw_dir=Path('.'), suffixes=suffixes, ignore_empty_documents=False)
        self.splits = ['train', 'validation', 'test']
        self.write_dir = LOC.parsed / "wikitext"

        # We don't need raw dir, or nlp
        self.hf_ds = load_dataset("wikitext")

    def parse(self, split_nm: Union[Path, str]):
        outputs = []
        for instance in self.hf_ds[split_nm]:
            document = self._parse_(instance)
            outputs.append(document)

    def _parse_(self, instance: dict) -> Document:
        """
            Dict keys:
                ['ent_domains', 'ent_ner', 'ent_span', 'ent_wikipedia_external_ref', 'original_id', 'text',
                 'tok_dephead', 'tok_deprel', 'tok_lemma', 'tok_ner', 'tok_span', 'tok_upos', 'tok_xpos']

            Stuff I'm interested in:
                tok_span to get text tokens
                tok_upos as the pos tokens
                ent_ner as ner tags
                ent_span to get tokens for the entity

        :param instance:
        :return: a full document
        """

        doc_toks = []
