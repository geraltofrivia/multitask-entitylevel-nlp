from copy import deepcopy
from typing import List, Any
from spacy.tokens import Doc
try:
    from spacy.util import DummyTokenizer
except ImportError:
    # noinspection PyUnusedLocal,PyMethodMayBeStatic
    class DummyTokenizer:
        def __call__(self, text):
            raise NotImplementedError

        def pipe(self, texts, **kwargs):
            for text in texts:
                yield self(text)

        # add dummy methods for to_bytes, from_bytes, to_disk and from_disk to
        # allow serialization (see #1557)
        def to_bytes(self, **kwargs):
            return b""

        def from_bytes(self, _bytes_data, **kwargs):
            return self

        def to_disk(self, _path, **kwargs):
            return None

        def from_disk(self, _path, **kwargs):
            return self


SPAN_POS_BLACKLIST_PREFIX = ('DT', 'JJ')
SPAN_POS_BLACKLIST_SUFFIX = ('.', 'POS')
NCHUNK_POS_WHITELIST = ('NN', 'NNS', 'NNP', 'NNPS')


def to_toks(doc: List[List[Any]]) -> List[Any]:
    """ [ ['a', 'sent'], ['another' 'sent'] ] -> ['a', 'sent', 'another', 'sent'] """
    return [word for sent in doc for word in sent]


def to_str(raw: List[List[str]]) -> str:
    sents = [' '.join(sent) for sent in raw]
    return ' '.join(sents)


def is_nchunk(span: List[int], pos: List[List[str]]) -> bool:
    """ Check if every element in the span belongs to whitelist of noun pos tags. """
    span_pos = to_toks(pos)[span[0]: span[1]]
    for pos_tag in span_pos:
        if not pos_tag in NCHUNK_POS_WHITELIST:
            return False

    return True


def remove_pos(span: List[int], pos: List[List[str]], remove_all: bool = False,
               prefix: List[str] = SPAN_POS_BLACKLIST_PREFIX, suffix: List[str] = SPAN_POS_BLACKLIST_SUFFIX) -> List[int]:
    """
        We remove certain words from the given span (from the start or from the end) based on the pos tags defined.
        We may remove everything from the span if the span contains only these pos based things based on remove_all flag
    """

    pos = to_toks(pos)[span[0]: span[1]]
    span = deepcopy(span)

    while True:
        if (span[0] == span[1] and remove_all) or (span[0] + 1 == span[1] and not remove_all):
            """ If all words are gone or if one word remains but remove_all is turned off, return """
            break

        if pos[0] in prefix:
            pos.pop(0)
            span[0] = span[0] + 1
            continue

        if pos[-1] in suffix:
            pos.pop(-1)
            span[1] = span[1] - 1
            continue

        # If we're still here, then we changed nothing
        break

    return span


class NullTokenizer(DummyTokenizer):
    """
        Use it when the text is already tokenized but the doc's gotta go through spacy.
        Usage: `nlp.tokenizer = CustomTokenizer(nlp.vocab)`
    """
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, words):
        return Doc(self.vocab, words=words)