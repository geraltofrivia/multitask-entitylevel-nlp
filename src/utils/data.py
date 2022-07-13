"""
    Contain multiple dataclasses and types to control the chaos that is a large python codebase.
"""

import copy
from dataclasses import dataclass, field
from functools import cached_property
from typing import List, Optional, Tuple

from config import KNOWN_TASKS, LOSS_SCALES
from utils.exceptions import UnknownTaskException, BadParameters
from utils.misc import argsort
from utils.nlp import to_toks


@dataclass
class Clusters:
    """
    This dataclass is a collection of coreference clusters inside a document data class (below)
    It provides some functionalities that might come in handy like
        finding clusters based on a particular pos
        easy representation for training data prep etc etc

    Primarily, it consists of a list of spans (indices).
    These indices correspond to the Document.document field (see "I see ... fandango" snippet above).
    So, let's imagine our document looks something like:

    ```py
    # I saw a dog in a car. It was really cute. Its skin was brown with white spots !
    doc = [
        ["I", "saw", "a", "dog", "in", "a", "car", "."],                # 8 tokens
        ["It", "was", "really", "cute", "."],                           # 5 tokens
        ["Its", "skin", "was", "brown", "with", "white", "spots", "!"]  # 8 tokens
    ] # total: 21 tokens
    ```

    The clusters here would be <"I">, <"a dog", "it", and "its">, and <"a car">.
    That is, two singletons and one cluster with three spans. It would be represented by something like:

    ```py
    clusters = [
        [[0, 1]],                        # one cluster with one span (a span is a list of two ints)
        [[2, 4], [8, 9], [13, 14]]       # next cluster with three spans
        [
    ```

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
        Note that we do not make separate data fields for argument a and argument b.
        Currently it is stored as
            [ [rel1_arga, rel1_argb], [rel2_arga, rel2_argb] ... ]
    """
    spans: List[List[List[int]]]  # Looks like [ [[112, 113], [155, 159]], [[2, 6], [11, 13]], ... ]
    words: List[List[List[str]]] = field(default_factory=list)  # Looks like [ [['the', 'neighbor'], ['a', 'boy']], ..]
    pos: List[List[List[str]]] = field(default_factory=list)  # Looks same as words
    spans_head: List[List[List[int]]] = field(default_factory=list)
    words_head: List[List[List[str]]] = field(default_factory=list)
    pos_head: List[List[List[str]]] = field(default_factory=list)

    @property
    def isempty(self) -> bool:
        return len(self.spans) == 0

    @property
    def arga(self):
        """ return spans in the first argument position """
        return [pair[0] for pair in self.spans]

    @property
    def argb(self):
        """ return spans in the first argument position """
        return [pair[1] for pair in self.spans]

    @property
    def arga_h(self):
        """ return spans in the first argument position """
        return [pair[0] for pair in self.spans_head]

    @property
    def argb_h(self):
        """ return spans in the first argument position """
        return [pair[1] for pair in self.spans_head]

    @property
    def arga_w(self):
        """ return spans in the first argument position """
        return [pair[0] for pair in self.words]

    @property
    def argb_w(self):
        """ return spans in the first argument position """
        return [pair[1] for pair in self.words]

    @property
    def arga_wh(self):
        """ return spans in the first argument position """
        return [pair[0] for pair in self.words_head]

    @property
    def argb_wh(self):
        """ return spans in the first argument position """
        return [pair[1] for pair in self.words_head]

    def __len__(self):
        return len(self.spans)

    def allocate_span_heads(self, span_heads: dict):
        """Given a dict of {full span: span head}, allocate them based on the clusters in self.data"""
        output = []
        for i, pair in enumerate(self.spans):
            output_pair = []
            for span in pair:
                output_pair.append(list(span_heads[tuple(span)]))
            output.append(output_pair)

        self.spans_head = output

    def add_words(self, doc: List[List[str]]):
        """
        Based on self.spans, and spans_head , fill in words and words_head
        That is, I already know the spans of mentions and their span heads.
        If I know the words in the document, I can also store the mention words in a list ...
        """
        words, words_head = [], []
        f_doc = to_toks(doc)

        for pair, pair_h in zip(self.spans, self.spans_head):
            pair_words, pair_words_head = [], []
            for span, span_h in zip(pair, pair_h):
                pair_words.append(f_doc[span[0]: span[1]])
                pair_words_head.append(f_doc[span_h[0]: span_h[1]])

            words.append(pair_words)
            words_head.append(pair_words_head)

        self.words = words
        self.words_head = words_head

    def add_pos(self, doc: List[List[str]]):
        """
        Based on self.spans, and spans_head , fill in pos and pos_head
        That is, I already know the spans of mentions and their span heads.
        If I know the POS of words in the document, I can also store the mention tags in a list ...
        """
        pos, pos_head = [], []
        f_doc = to_toks(doc)

        for pair, pair_h in zip(self.spans, self.spans_head):
            pair_pos, pair_pos_head = [], []
            for span, span_h in zip(pair, pair_h):
                pair_pos.append(f_doc[span[0]: span[1]])
                pair_pos_head.append(f_doc[span_h[0]: span_h[1]])

            pos.append(pair_pos)
            pos_head.append(pair_pos_head)

        self.pos = pos
        self.pos_head = pos_head

    @classmethod
    def new(cls):
        return TypedRelations([], [])


@dataclass
class BridgingAnaphors(BinaryLinks):
    """ No change from above, just using a better name for the task. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.antecedents = self.arga
        self.anaphors = self.argb
        self.antecedents_head = self.arga_h
        self.anaphors_head = self.argb_h
        self.antecedents_words = self.arga_w
        self.anaphors_words = self.argb_w
        self.antecedents_words_head = self.argb_wh
        self.anaphors_words_head = self.argb_wh

    def get_anaphor_for(self, span, return_index=False):
        """ give antecedent span, get anaphor span. greedy: if there are multiple matches, returns the first one """
        for i, ante in enumerate(self.antecedents):
            if ante[0] == span[0] and ante[1] == span[1]:
                if return_index:
                    return i
                else:
                    return self.anaphors[i]

    def get_antecedent_for(self, span, return_index=False):
        """ give anaphor span, get antecedent span. greedy: if there are multiple matches, returns the first one """
        for i, ana in enumerate(self.anaphors):
            if ana[0] == span[0] and ana[1] == span[1]:
                if return_index:
                    return i
                else:
                    return self.antecedents[i]


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

    @classmethod
    def new(cls):
        return NamedEntities(spans=[], tags=[])

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
    """
        Each document has the following fields:

        **document**: `List[List[str]]`: A list of sentences where each sentence itself is a list of strings.
        For instance:

        ```py
        [
            ["I", "see", "a", "little", "silhouette", "of", "a", "man"],
            ["Scaramouche", "Scaramouche", "will", "you", "do", "the", "Fandango"]
        ]
        ```

        **pos**: `List[List[str]]`: The same as above except every string is replaced by its POS tag.
        Warning: this is not an optional field. So in case your document is not annotated with pos tags,
        you can pass fake pos tags (and choose to not exercise them down the line). You can do this simply by:

        ```py
        from pprint import pprint
        from utils.data import Document
        doc_text = [
            ["I", "see", "a", "little", "silhouette", "of", "a", "man"],
            ["Scaramouche", "Scaramouche", "will", "you", "do", "the", "Fandango"]
        ]
        fake_pos = Document.generate_pos_tags(doc_text)
        pprint(fake_pos)
        print("Corresponding to")
        pprint(doc_text)
        ```

        **docname**: str

        **genre**: str

        are both metadata fields that you can choose to use however you want.
        Ideally, docname should contain the docname. Genre can be left empty.

        **coref**: Cluster

        **ner**: NamedEntities

        **bridging**: BridgingAnaphor

        **rel**: TypedRelations

        are the fields which contain task specific annotations.
        All these four things are represented with their custom data classes (also found in `src/utils/data.py`)
    """
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
    bridging: BridgingAnaphors

    # Split (ontonotes split: train;test;development; conll-2012-test
    split: str = field(default_factory=str)

    # More Ontonotes specific stuff
    genre: str = field(default_factory=str)  # The ontonotes doc will belong to certain genre. Good to keep track.
    docpart: int = field(default_factory=int)  # In some cases, ontonotes documents are divided into parts.

    # TODO: also add entity linking stuff

    @classmethod
    def generate_pos_tags(cls, doc_text):
        return [['FAKE' for _ in sent] for sent in doc_text]

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

    def get_all_spans(self) -> List[List[int]]:
        """ Find all unique spans in coref, ner, re, ba and return them """
        spans = []

        if not self.coref.isempty:
            spans += [tuple(span) for cluster in self.coref.spans for span in cluster]

        if not self.ner.isempty:
            spans += [tuple(span) for span in self.ner.spans]

        if not self.rel.isempty:
            spans += [tuple(span) for pair in self.rel.spans for span in pair]

        if not self.bridging.isempty:
            spans += [tuple(span) for pair in self.bridging.spans for span in pair]

        return [list(span) for span in set(spans)]


@dataclass
class Tasks:
    names: List[str]
    loss_scales: List[float]
    use_class_weights: List[bool]
    dataset: str

    n_classes_ner: Optional[int] = field(default_factory=int)
    n_classes_pruner: Optional[int] = field(default_factory=int)

    @classmethod
    def parse(cls, datasrc: Optional[str], tuples: List[Tuple[str, float, bool]]):

        if not type(datasrc) in [type(None), str]:
            raise BadParameters(
                f"datasrc is not a string but {type(datasrc)}. Maybe you forgot to pass the data source?"
                f"Ensure that you're calling Tasks(datasource, tuples=task_tuples) and not"
                f"Tasks(*task_tuples) or Tasks(tuples=task_tuples.")

        _raw_ = copy.deepcopy(tuples)
        dataset = datasrc

        # Check if every element is a tuple or not
        for tupl in tuples:
            if type(tupl) is not tuple:
                raise TypeError(f"Expected a list of tuples as input. Got {type(tupl)}")

        names = [tupl[0] for tupl in tuples]

        # Check if every task name is known
        for task_nm in names:
            if task_nm not in KNOWN_TASKS:
                raise UnknownTaskException(f"An unrecognized task name sent: {task_nm}. "
                                           f"So far, we can work with {KNOWN_TASKS}.")

        # Check for duplicates
        if len(set(names)) != len(names):
            raise ValueError("Duplicates were passed in args. Please don't.")

        # Picking loss scales
        loss_scales = cls._parse_loss_scales_(names=names, scales=[arg[1] for arg in tuples])
        use_class_weights = [arg[2] for arg in tuples]

        n_classes_ner = -1
        n_classes_pruner = -1

        return cls(names=names, loss_scales=loss_scales, use_class_weights=use_class_weights,
                   dataset=dataset, n_classes_ner=n_classes_ner, n_classes_pruner=n_classes_pruner)

    def __post_init__(self, *args, **kwargs):
        self.sort()

    def sort(self):
        """ Rearranges all artefacts to sort them in the right order """

        if len(self) == 0:
            return

        sorted_ind = argsort(self)
        self.names = [self.names[i] for i in sorted_ind]
        self.loss_scales = [self.loss_scales[i] for i in sorted_ind]
        self.use_class_weights = [self.use_class_weights[i] for i in sorted_ind]

    def _task_unweighted_(self, task_nm: str) -> bool:
        if task_nm not in self:
            raise UnknownTaskException(f"Asked for {task_nm} but task does not exist.")

        task_index = self.names.index(task_nm)
        return not self.use_class_weights[task_index]

    def ner_unweighted(self) -> bool:
        return self._task_unweighted_('ner')

    def coref_unweighted(self) -> bool:
        return self._task_unweighted_('coref')

    def pruner_unweighted(self) -> bool:
        return self._task_unweighted_('pruner')

    @staticmethod
    def _parse_loss_scales_(names: List[str], scales: List[float]) -> List[float]:
        """
            If all scales are negative, use the predefined scale in LOSS_SCALES (for the task combination).
            If there is at least one positive value, replace the negative values with the defaults given in LOSS_SCALES
            If they're all positive, well, just return it as-is.

            If any value is zero, we consider it as positive.
        """

        if len(scales) == 0:
            return []

        all_neg = all([val < 0 for val in scales])
        if all_neg:

            key = '_'.join(sorted(names))
            return LOSS_SCALES[key]

        else:
            # There is at least one positive value
            for i, val in enumerate(scales):
                if val < 0:
                    scales[i] = LOSS_SCALES[names[i]][0]

            return scales

    def __len__(self):
        return len(self.names)

    def raw(self):
        return [(self.names[i], self.loss_scales[i], self.use_class_weights[i]) for i in range(len(self))]

    def __iter__(self):
        for task in self.names:
            yield task

    def __getitem__(self, i):
        return self.names[i]

    @classmethod
    def create(cls):
        return Tasks.parse(datasrc=None, tuples=[])

    def isempty(self):
        return self.dataset is None and len(self) == 0
