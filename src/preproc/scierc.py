"""
    Code to read data/raw/scierc stuff and turn it into a bunch of Document instances.

"""
import json
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
from config import LOCATIONS as LOC, KNOWN_SPLITS
from preproc.commons import GenericParser
from dataiter import DocumentReader
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


def create_label_dict():
    """
        Check if two trainable splits of scierc - train, and dev are already processed or not.
        If not, return error.

        If yes, go through all of them and find all unique output labels to encode them in a particular fashion.
    :return: None
    """
    relevant_splits: List[str] = KNOWN_SPLITS.scierc[:-1]

    # Check if dump.json exists in all of these
    ner_labels = set()
    rel_labels = set()
    for split in relevant_splits:
        reader = DocumentReader('scierc', split=split)
        for doc in reader:
            ner_labels = ner_labels.union(doc.ner.tags)
            rel_labels = rel_labels.union(doc.rel.tags)

    # Turn them into dicts and dump them as json
    with (LOC.manual / 'ner_scierc_tag_dict.json').open('w+', encoding='utf8') as f:
        ner_labels = {tag: i for i, tag in enumerate(ner_labels)}
        json.dump(ner_labels, f)
        print(f"Wrote a dict of {len(ner_labels)} items to {(LOC.manual / 'ner_scierc_tag_dict.json')}")

    with (LOC.manual / 'rel_scierc_tag_dict.json').open('w+', encoding='utf8') as f:
        rel_labels = {tag: i for i, tag in enumerate(rel_labels)}
        json.dump(rel_labels, f)
        print(f"Wrote a dict of {len(rel_labels)} items to {(LOC.manual / 'rel_scierc_tag_dict.json')}")


@click.command()
@click.option("--split", "-s", type=str,
              help="The name of the dataset SPLIT e.g. train, test, dev")
@click.option("--ignore-empty", "-i", is_flag=True,
              help="If True, we ignore the documents without any coref annotation")
@click.option("--collect-labels", is_flag=True,
              help="If this flag is True, we ignore everything else, "
                   "just go through train, dev splits and collect unique labels and create a dict out of them.")
def run(split: str, ignore_empty: bool, collect_labels: bool):
    if collect_labels:
        create_label_dict()
    else:
        if split == 'all':
            splits = ['train', 'test', 'dev']
        else:
            splits = [split, ]
        parser = SciERCParser(LOC.scierc, splits=splits, ignore_empty_documents=ignore_empty)
        parser.run()

        if split == 'all':
            create_label_dict()


if __name__ == "__main__":
    run()
