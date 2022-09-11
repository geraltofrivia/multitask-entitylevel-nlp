"""
    We take already processed stuff from ACE2005-Toolkit repo and use those nice JSONs to create our docs
    Should be straightforward really.
"""
import json
from pathlib import Path
from typing import Iterable, Union, List

# Local imports
try:
    import _pathfix
except ImportError:
    from . import _pathfix
from config import LOCATIONS as LOC
from preproc.commons import GenericParser
from utils.nlp import to_toks
from utils.data import Document, Clusters, TypedRelations, NamedEntities


class ACE2005Parser(GenericParser):

    def __init__(
            self,
            raw_dir: Path,
            suffixes: Iterable[str] = (),
            ignore_empty_documents: bool = False
    ):
        cleared_suffixes = []
        for suffix in suffixes:
            if suffix not in ['train', 'test', 'dev']:
                continue
            cleared_suffixes.append(suffix)

        super().__init__(raw_dir=raw_dir, suffixes=cleared_suffixes, ignore_empty_documents=ignore_empty_documents)

        self.parsed: dict = {split_nm: [] for split_nm in cleared_suffixes}
        self.write_dir = LOC.parsed / "ace2005"
        self.write_dir.mkdir(parents=True, exist_ok=True)

    def parse(self, split_nm: Union[Path, str]) -> List[Document]:

        outputs: List[Document] = []
        filenames: Iterable[Path] = list((self.dir / split_nm).glob('*v2.json'))

        # TODO:test this stuff out
        for filename in filenames:
            document = self._parse_document_(filename)
            outputs.append(document)

        return outputs

    def _parse_document_(self, filename: Path):

        with filename.open('r') as f:
            raw = json.load(f)

        doc_sents = [x['word'] for x in raw['sentences']]
        doc_spacy = self.nlp(doc_sents)
        # doc = self._get_spacy_doc_(doc_text)
        doc_pos = self.get_pos_tags(doc_spacy)
        doc_name = filename.name
        speakers = [0] * len(doc_sents)
        coref = Clusters([])
        ner = self.get_ner_obj(raw, doc_sents, doc_pos)
        rel = self.get_rel_obj(raw, doc_sents, doc_pos)

        print(raw[:10])
        return ''

    def get_ner_obj(self, raw: dict, doc: List[List[str]], pos: List[List[str]]) -> NamedEntities:
        """ We need ner_spans, and ner_tags"""

        sentlen_offsets = [to_toks(doc[:i]).__len__() for i, sent in enumerate(doc)]
        ner_spans, ner_spans_head = [], []
        ner_words_head = []  # We note down what the raw says and compare to what we infer to assure we infer correctly

        for ent in raw['entities']:
            # Find sentence offsets and add the positions after that
            sentlen_offset = sentlen_offsets[int(ent['sent_id']) - 1]
            ner_span = [sentlen_offset + ent['position'][0],
                        sentlen_offset + ent['position'][1] + 1]
            ner_span_head = [sentlen_offset + ent['head']['position'][0],
                             sentlen_offset + ent['head']['position'][1] + 1]
            ner_word_head = ent['head']['text']

            ner_spans.append(ner_span)
            ner_spans_head.append(ner_span_head)
            ner_words_head.append(ner_word_head)

        ner_tags = [x['entity-type'] for x in raw['entities']]

        ner_obj = NamedEntities(spans=ner_spans, tags=ner_tags, spans_head=ner_spans_head)
        ner_obj.add_words(doc)
        ner_obj.add_pos(pos)

        # Compare span heads
        for i, head in enumerate(ner_obj.words_head):
            raw_head = ner_words_head[i]
            if not (''.join(head) == raw_head or ' '.join(head) == raw_head):
                raise AssertionError(f'There was a problem mapping spans as the {i}th span head is originally:\n'
                                     f'\t`{raw_head}` \n but we inferred it to be \n\t`{head}`')
        return ner_obj

    def get_rel_obj(self, raw: dict, doc: List[List[str]], pos: List[List[str]]) -> TypedRelations:
        """ Same, we need rel_spans, and tags to create object and can add words and pos later """
        spans = []
        tags = []

        words = []
        sentlen_offsets = [to_toks(doc[:i]).__len__() for i, sent in enumerate(doc)]

        for rel in raw['relations']:
            tags.append(rel['relation-type'])
            spans_this_rel = []  # list of two span objs e.g. [ [ 2, 3], [10, 13] ]
            span_offset = sentlen_offsets[int(rel['sent_id']) - 1]
            words_this_rel = []
            for arg in rel['arguments']:
                spans_this_rel.append([arg['position'][0] + span_offset,
                                       arg['position'][1] + span_offset + 1])
                words_this_rel.append(arg['text'])

            spans.append(spans_this_rel)
            words.append(words_this_rel)

        rel_obj = TypedRelations(spans=spans, tags=tags)
        rel_obj.add_words(doc)
        rel_obj.add_pos(pos)

        # Verify the texts time
        for i, word_pairs in enumerate(rel_obj.words):
            if not (' '.join(word_pairs[0]) == words[i][0] or ''.join(word_pairs[0]) == words[i][0]) and \
                    (' '.join(word_pairs[1]) == words[i][1] or ''.join(word_pairs[1]) == words[i][1]):
                raise AssertionError(f"There is a problem with span {i}.\n"
                                     f"Expected text:\n"
                                     f"\t`{words[i][0]}` || `{words[i][1]}`"
                                     f"What we got:\n"
                                     f"\t`{word_pairs[0]}`, `{word_pairs[1]}`")

        return rel_obj


if __name__ == '__main__':
    parser = ACE2005Parser(LOC.ace, ['train'])
    parser.run()
