"""
    The dataset is already preprocessed in a very nice format.
    The only thing is, we now need to convert it to _our_ format.
    For that, we utilise the tokenizer that they provide to generate spans for every token.
    Then, we cross those token span info with the span info provided for coref, ner and rel tasks
        and make our objects

"""
import json
from pathlib import Path
from typing import Iterable, Union, List, Dict

import numpy as np
import spacy

# Local imports
try:
    import _pathfix
except ImportError:
    from . import _pathfix
from preproc.commons import GenericParser
from utils.nlp import PreTokenziedTokenizer
from config import LOCATIONS as LOC, NPRSEED as SEED
from modules.dwie.src.dataset.utils.tokenizer import TokenizerCPN
from utils.data import Document, NamedEntities, Clusters, TypedRelations, BridgingAnaphors

np.random.seed(SEED)


class DWIEParser(GenericParser):

    def __init__(
            self,
            raw_dir: Path,
            suffixes: Iterable[str] = (),
    ):
        super().__init__(raw_dir=raw_dir, suffixes=suffixes, ignore_empty_documents=False)
        self.splits = ['train', 'dev', 'test']  # We do a 80, 10, 10 split
        self.raw_dir = raw_dir
        self.write_dir = LOC.parsed / "dwie"
        self._prebuilt_tokenizer_ = TokenizerCPN()

        exclude = ["parser"]
        self.nlp = spacy.load("en_core_web_sm", exclude=exclude)
        self.nlp.add_pipe('sentencizer')
        self.nlp.tokenizer = PreTokenziedTokenizer(self.nlp.vocab)

    def parse(self, split_nm: Union[Path, str]):
        # This is to shut up the AbstractMethodNotImplementedError
        ...

    def run(self):
        """
            Each file is an instance.
            Each file has a 'tags' column which indicates whether instance is train or test instance.

            Algo:
            1. Download entire dataset to memory
            2. Based on the 'tag', pass it either to train or to test set.
            3. Once done, split the train set into a proportional dev set as well.
            4. Write all three sets to disk

        :return:
        """

        self.delete_preprocessed_files('train')
        self.delete_preprocessed_files('test')
        self.delete_preprocessed_files('valid')

        # Load all the datasets in memory
        dataset = []
        for fname in self.raw_dir.glob('*'):
            with fname.open('r') as f:
                dataset.append(json.load(f))

        test_docs = []
        train_docs = []
        valid_docs = []
        for instance in dataset:
            if 'test' in instance['tags']:
                doc = self._parse_(instance, 'test')
                test_docs.append(doc)
            elif 'train' in instance['tags']:
                doc = self._parse_(instance, 'train')
                train_docs.append(doc)
            else:
                raise ValueError(f"Unexpected content of instance['tags']. Expected either 'train' or 'test' in there."
                                 f"Got {instance['tags']}")

        tv_len = len(train_docs)
        tv_indices = np.arange(tv_len)
        np.random.shuffle(tv_indices)

        print('potato')
        #
        # train_valid_split, valid_test_index = int(flen * 0.8), int(flen * 0.1)
        # train_fnames, dev_fnames, test_fnames = fnames[:train_valid_split], \
        #                                         fnames[train_valid_split: train_valid_split + valid_test_index], \
        #                                         fnames[train_valid_split + valid_test_index:]
        #
        # outputs = [self._parse_(fname, 'train') for fname in train_fnames]
        # self.write_to_disk('train', outputs)
        #
        # outputs = [self._parse_(fname, 'dev') for fname in dev_fnames]
        # self.write_to_disk('dev', outputs)
        #
        # outputs = [self._parse_(fname, 'test') for fname in test_fnames]
        # self.write_to_disk('test', outputs)

        # for split in self.suffixes:
        #     # First, clear out all the previously processed things from the disk
        #     self.delete_preprocessed_files(split)
        #     outputs = self.parse(split)
        #
        #     # Dump them to disk
        #     self.write_to_disk(split, outputs)

    @staticmethod
    def _get_tokenid_for_mention_(mention: dict, tokens: List[dict]) -> int:
        for i, token in enumerate(tokens):
            if mention['begin'] == token['offset'] and \
                    mention['end'] - mention['begin'] == token['length'] and \
                    mention['text'] == token['token']:
                return i

        raise ValueError(f"Tried to find mention {mention['text']} ({mention['begin']}:{mention['end']}) but couldn't.")

    @staticmethod
    def _convert_charspan_to_tokspan_(char_start: int, char_end: int, tokens: List[dict]) -> (int, int):
        """
            Given a span of characters in the document (pretokenized), return the tokens span.
            E.g. in `Potatoes grew in a farm', a char span of [15: 18] (in a) should be [2: 3]
        :param char_start: the int of beginning of span
        :param char_end: the int of end of span
        :param tokens: the list of tokenized stuff
        :return: (int, int) token start, end span
        """

        start_token = -1
        end_token = -1
        for token_id, token in enumerate(tokens):
            if char_start == token['offset']:
                start_token = token_id
            if char_end == token['offset'] + token['length']:
                end_token = token_id + 1
            if start_token >= 0 and end_token >= 0:
                break

        if start_token < 0 or end_token < 0:
            raise ValueError(f"Could not find span corresponding to [{start_token}: {end_token}]."
                             f"Our result was [{start_token}: {end_token}]")

        return start_token, end_token

    @staticmethod
    def _find_concept_index_(concept_id: int, concepts: List[dict]) -> int:
        for i, concept in enumerate(concepts):
            if concept['concept'] == concept_id:
                return i
        raise ValueError(f"Could not find concept ID: {concept_id}")

    def annotate_mention(self, mention: dict, tokens: List[dict], concepts: List[dict]):
        """ Add spans, words and tags to this token """
        token_start, token_end = self._convert_charspan_to_tokspan_(mention['begin'], mention['end'], tokens)
        mention_words = [token['token'] for token in tokens[token_start: token_end]]

        if (not ' '.join(mention_words) == mention['text']) and mention['text'].isalnum():
            raise AssertionError(f"Given was: `{mention['text']}`. Found was `{' '.join(mention_words)}`")

        mention['words'] = mention_words
        mention['span'] = [token_start, token_end]

        # Find the relevant concept
        concept_id = mention['concept']
        if not concepts[mention['concept']] == mention['concept']:
            concept_id = self._find_concept_index_(concept_id, concepts)

        mention['tags'] = concepts[concept_id]['tags']
        return mention

    @staticmethod
    def aggregate_mentions(mentions: List[dict]):
        """ Rearrange mentions based on concept IDs """
        clusters = {}
        for mention in mentions:
            clusters[mention['concept']] = clusters.get(mention['concept'], []) + [mention]

        return clusters

    def _parse_(self, content: dict, split_name: str) -> Document:

        tokenized = self._prebuilt_tokenizer_.tokenize(content['content'])

        """ 
            Step 1: NER
            - spans (correspond content['mentions'] with token in tokenized to get spans
            - words (same as above)
            - tags (get tag from content['concept']['tags']
        """
        mentions = [self.annotate_mention(mention, tokenized, content['concepts']) for mention in content['mentions']]
        ner_spans = [mention['span'] for mention in mentions]
        ner_words = [mention['words'] for mention in mentions]
        ner_tags = [mention['tags'] for mention in mentions]

        ner = NamedEntities(spans=ner_spans, tags=ner_tags, words=ner_words)

        """
            Step 2: Coref
                - aggregate the mentions based on the cluster key     
        """
        clusters = self.aggregate_mentions(mentions)
        cluster_spans = [[mention['span'] for mention in cluster] for cluster in clusters.values()]
        coref = Clusters(cluster_spans)

        """ 
            Step 3: REL
                - for each rel, get all mentions in subject pos; all mentions in object pos.
                - get rel label 
                - get words, similarly for both things
                
                (basically DWIE provides it on a cluster level but we want it on a relation level)
        """
        spans, tags = self.parse_relations(clusters, content['relations'])
        rel = TypedRelations(spans=spans, tags=tags)

        # Empty Bridging Instance
        bridging = BridgingAnaphors.new()

        """
            Document Time
        """
        document_flat = [tok['token'] for tok in tokenized]
        document_spacy = self.nlp(document_flat)
        document_sents = []
        for sent in document_spacy.sents:
            document_sents.append([tok.text for tok in sent])

        document_pos = self.get_pos_tags(document_spacy)
        document_name = content['id']
        document_speakers = [0] * len(document_sents)

        all_spans = []

        if not coref.isempty:
            all_spans += [tuple(span) for cluster in coref.spans for span in cluster]
        if not ner.isempty:
            all_spans += [tuple(span) for span in ner.spans]
        if rel.isempty:
            all_spans += [tuple(span) for pair in rel.spans for span in pair]
        if bridging.isempty:
            all_spans += [tuple(span) for pair in bridging.spans for span in pair]
        all_spans = [list(span) for span in set(all_spans)]
        all_spans_head = self.get_span_heads(document_spacy, spans=all_spans)

        coref.allocate_span_heads(span_heads=all_spans_head)
        ner.allocate_span_heads(span_heads=all_spans_head)
        rel.allocate_span_heads(span_heads=all_spans_head)
        bridging.allocate_span_heads(span_heads=all_spans_head)

        document = Document(
            document=document_sents,
            pos=document_pos,
            docname=document_name,
            speakers=document_speakers,
            coref=coref,
            ner=ner,
            rel=rel,
            bridging=bridging
        )

        return document

    @staticmethod
    def parse_relations(clusters: Dict[str, List[dict]], relations: List[dict]):
        """  """

        spans, tags = [], []

        for relation in relations:
            try:
                sub_mentions = clusters[relation['s']]
                obj_mentions = clusters[relation['o']]
            except KeyError as e:
                print(relation)
                raise e
            relation_tag = relation['p']

            for sub_mention in sub_mentions:
                for obj_mention in obj_mentions:
                    spans.append([sub_mention['span'], obj_mention['span']])
                    tags.append(relation_tag)

        return spans, tags


def run():
    parser = DWIEParser(LOC.dwie)
    parser.run()


if __name__ == '__main__':
    run()
