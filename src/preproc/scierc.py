"""
    Code to read data/raw/scierc stuff and turn it into a bunch of Document instances.

"""
import click
import spacy
import jsonlines
from spacy import tokens
from pathlib import Path
from typing import Iterable, Union, List

# Local imports
try:
    import _pathfix
except ImportError:
    from . import _pathfix
from config import LOCATIONS as LOC
from preproc.commons import GenericParser
from utils.nlp import to_toks, NullTokenizer
from utils.data import Document, NamedEntities, TypedRelations, Clusters


class SciERCParser(GenericParser):
    def __init__(self, raw_dir: Path, splits: Iterable[str] = (), ignore_empty_documents: bool = False):

        super().__init__(raw_dir=raw_dir, splits=splits, ignore_empty_documents=ignore_empty_documents)

        self.dir = raw_dir
        self.parsed: dict = {split_nm: [] for split_nm in splits}
        self.splits = splits

        self.flag_ignore_empty_documents: bool = ignore_empty_documents
        self.write_dir = LOC.parsed / "scierc"
        self.write_dir.mkdir(parents=True, exist_ok=True)

        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.tokenizer = NullTokenizer(self.nlp.vocab)

    @staticmethod
    def get_pos_tags(doc: tokens.Doc) -> List[List[str]]:
        """ Get pos tags for each token, respecting the sentence boundaries, i.e. each sent is a list """
        return [[token.pos_ for token in sent] for sent in doc.sents]

    @staticmethod
    def get_named_entity_objs(inst: dict) -> NamedEntities:

        doc = to_toks(inst['sentences'])
        ner_spans, ner_words, ner_tags = [], [], []
        for sent_id in range(len(inst['sentences'])):

            for ner in inst['ner'][sent_id]:
                ner_spans.append([ner[0], ner[1] + 1])
                ner_tags.append(ner[2])
                ner_words.append(doc[ner[0]: ner[1] + 1])

        return NamedEntities(spans=ner_spans, tags=ner_tags, words=ner_words)

    @staticmethod
    def get_typed_relations(inst: dict) -> TypedRelations:
        doct = to_toks(inst['sentences'])
        spans = []
        tags = []
        words = []
        for rel_in_this_sent in inst['relations']:
            for rel in rel_in_this_sent:
                words_a = doct[rel[0]: rel[1] + 1]
                words_b = doct[rel[2]: rel[3] + 1]

                spans.append([[rel[0], rel[1]], [rel[2], rel[3]]])
                tags.append(rel[4])
                words.append([words_a, words_b])

        return TypedRelations(spans=spans, tags=tags, words=words)

    def parse(self, split_nm: Union[Path, str]) -> List[Document]:
        """ where the actual preproc happens"""

        outputs: List[Document] = []
        filedir: Path = self.dir / (split_nm + '.json')

        assert filedir.exists()

        # Read JSONLines file
        with jsonlines.open(filedir) as reader:
            lines = list(iter(reader))

        for line in lines:
            doc_text = line['sentences']
            # noinspection PyTypeChecker
            doc = self.nlp(to_toks(doc_text))
            doc_pos = self.get_pos_tags(doc)
            doc_name = line['doc_key']
            split = split_nm

            # Parse out NER stuff
            ner = self.get_named_entity_objs(line)
            # ners

            # Parse out REL stuff
            rel = self.get_typed_relations(line)

            # TODO: Parse out Coref stuff
            coref = Clusters([])

            # Make the object
            document = Document(
                document=doc_text,
                pos=doc_pos,
                docname=doc_name,
                coref=coref,
                ner=ner,
                rel=rel
            )

            outputs.append(document)
        return outputs


@click.command()
@click.option("--split", "-s", type=str,
              help="The name of the dataset SPLIT e.g. train, test, development, conll-2012-test etc")
@click.option("--ignore-empty", "-i", is_flag=True,
              help="If True, we ignore the documents without any coref annotation")
def run(split: str, ignore_empty: bool):
    if split == 'all':
        splits = ['train', 'test', 'dev']
    else:
        splits = [split, ]
    parser = SciERCParser(LOC.scierc, splits=splits, ignore_empty_documents=ignore_empty)
    parser.run()


if __name__ == "__main__":
    run()
