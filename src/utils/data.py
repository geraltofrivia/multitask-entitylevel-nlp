from dataclasses import dataclass, field
from typing import List, Iterable


@dataclass
class CorefDocument:

    # The text of the document, broken down as list of sentence. Each sentence is a list of words
    document: List[List[str]]

    # Pos tags, noun phrases in the document computed using the spacy doc
    pos: List[List[str]]

    # Docname corresponds to the ontonotes doc
    docname: str
    docpart: int    # In some cases, ontonotes documents are divided into parts.

    # Split (ontonotes split: train;test;development; conll-2012-test
    split: str

    # Entity mentions: all spans that could be coreferent
    # TODO: how do I find them?
    mentions: List[Iterable[int]] = field(default_factory=list)
    mentions_: List[List[str]] = field(default_factory=list)

    # Actual coreference information: gold stuff
    clusters: List[List[Iterable[int]]] = field(default_factory=list)
    clusters_: List[List[List[str]]] = field(default_factory=list)
    # cluster_ids: List[List[int]] = field(default_factory=list)

    # TODO: also add entity linking stuff here
