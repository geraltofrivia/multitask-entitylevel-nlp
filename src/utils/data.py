"""
    Contain multiple dataclasses and types to control the chaos that is a large python codebase.
"""

from functools import cached_property
from dataclasses import dataclass, field
from typing import List

from utils.nlp import to_toks
from config import KNOWN_TASKS
from utils.exceptions import UnknownTaskException


@dataclass
class Clusters:
    """
    This dataclass is a collection of coreference clusters inside a document data class (below)
    It provides some functionalities that might come in handy like
        finding clusters based on a particular pos
        easy representation for training data prep etc etc

    Fields:
        spans: List of List of span boundaries like [ [ [2, 5], [10, 20] ], [ [1,2], [11, 16] ] ]
        words: Similar except instead of span boundary, each object itself is a list of tokens repr. that span.
    """

    """
        Look like:  [ 
                        [ [2, 5], [10,20], [17, 35] ... ]   # one cluster
                        [ ... ]                             # another cluster
                    ]
    """
    spans: list
    words: list = field(default_factory=list)
    pos: list = field(default_factory=list)
    spans_head: list = field(default_factory=list)
    words_head: list = field(default_factory=list)
    pos_head: list = field(default_factory=list)

    @property
    def isempty(self) -> bool:
        return len(self.spans) == 0

    def get_all_spans(self) -> list:
        """Return a set of all unique spans in a doc. Returns List[Tuple[int]]"""
        unique_spans = set()
        for cluster in self.spans:
            for span in cluster:
                unique_spans.add(tuple(span))

        return list(unique_spans)

    def allocate_span_heads(self, span_heads: dict):
        """Given a dict of {full span: span head}, allocate them based on the clusters in self.data"""
        cluster_heads = []
        for i, cluster in enumerate(self.spans):
            cluster_heads.append([])
            for j, span in enumerate(cluster):
                cluster_heads[i].append(list(span_heads[tuple(span)]))

        self.spans_head = cluster_heads

    def add_words(self, doc: List[List[str]]):
        """
        Based on self.clusters, and clusters_h , fill in clusters_ and clusters_h_
        That is, I already know the spans of mentions and their span heads.
        If I know the words in the document, I can also store the mention words in a list ...
        """
        clusters_, clusters_h_ = [], []
        f_doc = to_toks(doc)

        for i, (cluster, cluster_h) in enumerate(zip(self.spans, self.spans_head)):
            clusters_.append([])
            clusters_h_.append([])
            for span, span_h in zip(cluster, cluster_h):
                clusters_[-1].append(f_doc[span[0]: span[1]])
                clusters_h_[-1].append(f_doc[span_h[0]: span_h[1]])

        self.words = clusters_
        self.words_head = clusters_h_

    def add_pos(self, pos: List[List[str]]):
        """Same as self.add_words but for pos tags"""
        clusters_pos, clusters_h_pos = [], []
        f_pos = to_toks(pos)

        for i, (cluster, cluster_h) in enumerate(zip(self.spans, self.spans_head)):
            clusters_pos.append([])
            clusters_h_pos.append([])
            for span, span_h in zip(cluster, cluster_h):
                clusters_pos[-1].append(f_pos[span[0]: span[1]])
                clusters_h_pos[-1].append(f_pos[span_h[0]: span_h[1]])

        self.pos = clusters_pos
        self.pos_head = clusters_h_pos


@dataclass
class BinaryLinks:
    """
        Another data class obj containing all the things needed for smoothly using typed relations in a document
    """
    spans: List[List[List[int]]]  # Looks like [ [[112, 113], [155, 159]], [[2, 6], [11, 13]], ... ]
    words: List[List[List[str]]] = field(default_factory=list)  # Looks like [ [['the', 'neighbor'], ['a', 'boy']], ..]
    pos: List[List[List[str]]] = field(default_factory=list)  # Looks same as words
    spans_head: List[List[List[int]]] = field(default_factory=list)
    words_head: List[List[List[str]]] = field(default_factory=list)
    pos_heads: List[List[List[str]]] = field(default_factory=list)

    def __len__(self):
        return len(self.spans)

    @property
    def isempty(self):
        return len(self.spans) == 0

    def allocate_span_heads(self, span_heads: dict):
        """Given a dict of {full span: span head}, allocate them based on the clusters in self.data"""
        output = []
        for i, pair in enumerate(self.spans):
            output_pair = []
            for span in pair:
                output_pair.append(list(span_heads[tuple(span)]))
            output.append(output_pair)

        self.spans_head = output


@dataclass
class BridgingAnaphors(BinaryLinks):
    """ No change from above, just using a better name for the task. """


@dataclass
class TypedRelations(BinaryLinks):
    """
        You might notice that tags are also given a default value.
        Very unideal since they should get concrete values. We should never have NER spans w/o tags
        However, this can't be done w/o switching over to python 3.10 and I don't want to do that right now
    """

    tags: List[str] = field(default_factory=list)  # Looks like [ 'meronym',  'lives-in' ... ]

    def get_all(self, tag: str, src: str) -> list:
        if src not in ["spans", "words", "pos", "spans_head", "words_head", "pos_head"]:
            raise AssertionError(f"Data source {src} not understood.")
        return [getattr(self, src)[i] for i, _tag in self.tags if _tag == tag]


@dataclass
class NamedEntities:
    """
    Another data class obj containing all the things needed for smoothly using named entity information in a doc
    """

    # Gold annotations for named entity spans
    spans: List[List[int]]  # Looks like [[112, 113], [155, 159], ... ]
    tags: List[str]  # Looks like [ 'Person', 'Cardinal' ...
    words: List[List[str]] = field(
        default_factory=list
    )  # Looks like [['Michael', 'Jackson'] ... ]
    pos: List[List[str]] = field(default_factory=list)
    spans_head: List[List[int]] = field(default_factory=list)
    words_head: List[List[str]] = field(default_factory=list)
    pos_head: List[List[str]] = field(default_factory=list)

    def __len__(self):
        return len(self.spans)

    @property
    def isempty(self):
        return len(self.spans) == 0

    def get_tag_of(self, span, src: str = "spans") -> str:
        # Get span ID and based on it, fetch the tag
        if src not in ["spans", "words", "pos", "spans_head", "words_head", "pos_head"]:
            raise AssertionError(f"Data source {src} not understood.")
        index = getattr(self, src).index(span)
        return self.tags[index]

    def allocate_span_heads(self, span_heads: dict):
        """Given a dict of {full span: span head}, allocate them based on the clusters in self.data"""
        output = []
        for i, span in enumerate(self.spans):
            output.append(list(span_heads[tuple(span)]))

        self.spans_head = output

    def add_words(self, doc: List[List[str]]):
        """
        Based on self.clusters, and clusters_h , fill in clusters_ and clusters_h_
        That is, I already know the spans of mentions and their span heads.
        If I know the words in the document, I can also store the mention words in a list ...
        """
        words, words_head = [], []
        f_doc = to_toks(doc)

        for span, span_h in zip(self.spans, self.spans_head):
            words.append(f_doc[span[0]: span[1]])
            words_head.append(f_doc[span_h[0]: span_h[1]])

        self.words = words
        self.words_head = words_head

    def add_pos(self, pos: List[List[str]]):
        """Same as self.add_words() but for pos tags"""
        temp_pos, temp_pos_head = [], []
        f_pos = to_toks(pos)

        for span, span_h in zip(self.spans, self.spans_head):
            temp_pos.append(f_pos[span[0]: span[1]])
            temp_pos_head.append(f_pos[span_h[0]: span_h[1]])

        self.pos = temp_pos
        self.pos_head = temp_pos_head

    def get_all(self, tag: str, src: str) -> list:
        if src not in ["spans", "words", "pos", "spans_head", "words_head", "pos_head"]:
            raise AssertionError(f"Data source {src} not understood.")
        return [getattr(self, src)[i] for i, _tag in self.tags if _tag == tag]


@dataclass
class Document:
    # The text of the document, broken down as list of sentence. Each sentence is a list of words
    document: List[List[str]]

    # Pos tags, noun phrases in the document computed using the spacy doc
    pos: List[List[str]]

    # Docname corresponds to the ontonotes doc
    docname: str

    # A clusters object storing all things about coreference, empty is fine.
    coref: Clusters

    # Named Entity objects storing gold named entities (if found), empty is fine.
    ner: NamedEntities

    # Typed Relation object storing gold typed relations (if found), empty is fine.
    rel: TypedRelations

    # Bridging Anaphora object storing gold annotations (if found), empty is fine.
    bridging: BridgingAnaphors = field(default_factory=BridgingAnaphors)

    # Split (ontonotes split: train;test;development; conll-2012-test
    split: str = field(default_factory=str)

    # More Ontonotes specific stuff
    genre: str = field(default_factory=str)  # The ontonotes doc will belong to certain genre. Good to keep track.
    docpart: int = field(default_factory=int)  # In some cases, ontonotes documents are divided into parts.

    # TODO: also add entity linking stuff

    def finalise(self):
        """
        Sanity checks:
            -> every span in clusters, named_entities_gold has their corresponding span head found out
            -> length of doc pos and doc words (flattened) is the same
        """

        if not (
                len(self.document) == len(self.pos)
                and len(to_toks(self.document)) == len(to_toks(self.pos))
        ):
            raise AssertionError("Length mismatch between doc words and doc pos tags")

    @cached_property
    def sentence_map(self) -> List[int]:
        """
        A list of sentence ID corresponding to every token in to_toks(document).
        In other words, for any word index (in a flattened document),
            this can tell you the sentence ID of that word by:

            `sent_id = instance.sentence_id[4]` # prob 0 th sentence for 4th word
        """
        sentence_lengths = [len(sent) for sent in self.document]
        sentence_map = [
            [sentence_id] * sentence_length
            for sentence_id, sentence_length in enumerate(sentence_lengths)
        ]
        sentence_map = to_toks(
            sentence_map
        )  # len(to_toks(self.document)) == len(sentence_map)
        return sentence_map


class Tasks(list):
    """
        Custom list which has restricted values.
        Also, you can't delete things from it :]

        Can use it like
            Tasks('coref', 'ner')
        or
            Tasks(['coref', 'ner'])

        PS: Don't bother deeply understanding this code, I wrote this in a fit of pseudo productive stupour
            wherein I want to write something non trivial but also not important.
    """

    # noinspection PyUnusedLocal
    def __init__(self, *args, **kwargs):

        # First you want to allow for both forms of addressing (in comments above)
        if type(args[0]) in [list, tuple]:
            if len(args) != 1:
                raise ValueError("You both provide a list in args[0] and provide other args? Please don't.")
            args = args[0]

        # Check Values
        if len(set(args)) != len(args):
            raise ValueError("Duplicates were passed in args. Please don't.")

        for val in args:
            if not type(val) is str:
                raise TypeError("We only expected to deal with strings here !?!")

            if val not in KNOWN_TASKS:
                raise UnknownTaskException(f"{val} is not a known task.")

        args = sorted(args)

        # Init List
        super().__init__(args)

    def __setitem__(self, ii, val):
        raise ValueError("Es Ist Verboten !!")

    def insert(self, ii, val):
        raise ValueError("Es Ist Verboten !!")

    def pop(self, *args, **kwargs):
        raise ValueError("Es Ist Verboten !!")

    def append(self, val):
        raise ValueError("Es Ist Verboten !!")
