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

        ner_spans = [x['position'] for x in raw['entities']]
        for span in ner_spans:
            span[1] = span[1] + 1

        ner_tags = [x['entity-type'] for x in raw['entities']]

        ner_obj = NamedEntities(spans=ner_spans, tags=ner_tags)
        ner_obj.add_words(doc)
        ner_obj.add_pos(pos)

        return ner_obj

    def get_rel_obj(self, raw: dict, doc: List[List[str]], pos: List[List[str]]) -> TypedRelations:
        """ Same, we need rel_spans, and tags to create object and can add words and pos later """
        spans = []
        tags = []

        for rel in raw['relations']:
            tags.append(rel['relation-type'])
            spans_this_rel = []  # list of two span objs e.g. [ [ 2, 3], [10, 13] ]
            for arg in rel['arguments']:
                spans_this_rel.append(arg['position'])

            spans.append(spans_this_rel)

        rel_obj = TypedRelations(spans=spans, tags=tags)
        rel_obj.add_words(doc)
        rel_obj.add_pos(pos)

        return rel_obj

if __name__ == '__main__':
    parser = ACE2005Parser(LOC.ace, ['train'])
    parser.run()
