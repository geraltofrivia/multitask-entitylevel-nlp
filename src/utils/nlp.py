import torch
import unidecode
import transformers
from copy import deepcopy
from typing import List, Any, Dict
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

# Local Imports
from utils.misc import pop

SPAN_POS_BLACKLIST_PREFIX = ("DT", "JJ")
SPAN_POS_BLACKLIST_SUFFIX = (".", "POS")
NCHUNK_POS_WHITELIST = ("NN", "NNS", "NNP", "NNPS")


def to_toks(doc: List[List[Any]]) -> List[Any]:
    """[ ['a', 'sent'], ['another' 'sent'] ] -> ['a', 'sent', 'another', 'sent']"""
    return [word for sent in doc for word in sent]


def to_str(raw: List[List[str]]) -> str:
    sents = [" ".join(sent) for sent in raw]
    return " ".join(sents)


def is_nchunk(span: List[int], pos: List[List[str]]) -> bool:
    """Check if every element in the span belongs to whitelist of noun pos tags."""
    span_pos = to_toks(pos)[span[0]: span[1]]
    for pos_tag in span_pos:
        if pos_tag not in NCHUNK_POS_WHITELIST:
            return False

    return True


def remove_pos(
        span: List[int],
        pos: List[List[str]],
        remove_all: bool = False,
        prefix: List[str] = SPAN_POS_BLACKLIST_PREFIX,
        suffix: List[str] = SPAN_POS_BLACKLIST_SUFFIX,
) -> List[int]:
    """
    We remove certain words from the given span (from the start or from the end) based on the pos tags defined.
    We may remove everything from the span if the span contains only these pos based things based on remove_all flag
    """

    pos = to_toks(pos)[span[0]: span[1]]
    span = deepcopy(span)

    while True:
        if (span[0] == span[1] and remove_all) or (
                span[0] + 1 == span[1] and not remove_all
        ):
            """If all words are gone or if one word remains but remove_all is turned off, return"""
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


def match_subwords_to_words(
        tokens: List[str],
        input_ids: dict,
        tokenizer: transformers.BertTokenizer,
        ignore_cases: bool = True,
) -> Dict[int, int]:
    """
    Create a dictionary that matches subword indices to word indices
    Expects the subwords to be done by a BertTokenizer.
    """
    # DEBUG
    if len(tokens) == 1017 and tokens[1] == "Miracle":
        print("potato")

    sw2w = {}
    input_ids = torch.masked_select(input_ids.squeeze(0), input_ids.squeeze(0) != 0)
    sw_tokens = tokenizer.convert_ids_to_tokens(
        input_ids.tolist(), skip_special_tokens=False
    )[:]

    # We do have to remove the PAD tokens however
    # We also need to remove accents since the HF tokenizer removes them as well.
    # ### And if we try to match vis-à-vis with vis-a-vis, the program crashes.
    tokens = [token.lower() for token in tokens[:]] if ignore_cases else tokens[:]
    # tokens = [unidecode.unidecode(token) for token in tokens]
    curr_sw_index = 0
    curr_w_index = 0

    while True:

        # break if sw tokens are empty
        if not sw_tokens:
            break

        if sw_tokens[0] == tokens[0]:
            sw2w[curr_sw_index] = curr_w_index
            sw_tokens.pop(0)
            tokens.pop(0)
            curr_sw_index += 1
            curr_w_index += 1
        else:
            sw_phrase = ""
            sw_selected = -1

            # DEBUG
            if sw_phrase == ':' and tokens[0] == ':)':
                print('potato')

            for i, next_word in enumerate(sw_tokens):
                next_word = next_word[:]
                next_word = (
                    next_word if not next_word.startswith("##") else next_word[2:]
                )
                sw_phrase += next_word

                sw_selected = i

                # DEBUG: every time there are more than 8 sw in a word, figure out what's up!
                if sw_selected > 8 and not (
                        sw_phrase.startswith("http")
                        or "@yahoo" in sw_phrase
                        or "@hotmail" in sw_phrase
                        or "@gmail" in sw_phrase
                        or sw_phrase.endswith('.com')
                        or sw_phrase.startswith("~~")
                        or sw_phrase.startswith("--")
                        or sw_phrase.startswith("__")
                        or sw_phrase.startswith("..")
                        or sw_phrase.startswith("!!")
                        or sw_phrase.startswith("hahahaha")
                        or sw_phrase.startswith("==")
                ):
                    print("here")

                if sw_phrase == tokens[0]:
                    break

            for i in range(sw_selected + 1):
                sw2w[curr_sw_index + i] = curr_w_index

            curr_w_index += 1
            curr_sw_index += sw_selected + 1
            tokens.pop(0)
            pop(sw_tokens, list(range(sw_selected + 1)))

    return sw2w
