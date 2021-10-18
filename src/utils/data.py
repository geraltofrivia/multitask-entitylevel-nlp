from dataclasses import dataclass, field
from typing import List, Iterable, Tuple, Dict

from utils.nlp import to_toks


@dataclass
class Document:

    # The text of the document, broken down as list of sentence. Each sentence is a list of words
    document: List[List[str]]

    # Pos tags, noun phrases in the document computed using the spacy doc
    pos: List[List[str]]

    # Docname corresponds to the ontonotes doc
    docname: str
    docpart: int    # In some cases, ontonotes documents are divided into parts.
    genre: str      # The ontonotes doc will belong to certain genre. Good to keep track.

    # Split (ontonotes split: train;test;development; conll-2012-test
    split: str

    # Actual coreference information: gold stuff
    clusters: List[List[Iterable[int]]]
    clusters_: List[List[List[str]]]
    # cluster_ids: List[List[int]] = field(default_factory=list)

    # Gold annotations for named entity spans
    named_entities_gold: List[list]     # Looks like [['PERSON', 112, 113] ... ]
    named_entities_gold_: List[list]    # Looks like [['PERSON', 'Michael', 'Jackson'] ... ]

    span_heads: Dict[Tuple[int], Tuple[int]] = field(default_factory=dict)
    span_heads_: Dict[Tuple[int], Tuple[str]] = field(default_factory=dict)

    # Entity mentions: all spans that could be coreferent (candidate list)
    # TODO: how do I find them?
    mentions: List[Iterable[int]] = field(default_factory=list)
    mentions_: List[List[str]] = field(default_factory=list)

    # TODO: also add entity linking stuff here

    def finalise(self):
        """
            Sanity checks:
                -> every span in clusters, named_entities_gold has their corresponding span head found out
                -> length of doc pos and doc words (flattened) is the same
        """

        if not (len(self.document) == len(self.pos) and len(to_toks(self.document)) == len(to_toks(self.pos))):
            raise AssertionError("Length mismatch between doc words and doc pos tags")

