"""
    The dataset is already preprocessed in a very nice format.
    The only thing is, we now need to convert it to _our_ format.
    For that, we utilise the tokenizer that they provide to generate spans for every token.
    Then, we cross those token span info with the span info provided for coref, ner and rel tasks
        and make our objects

"""
import json
from copy import deepcopy
from pathlib import Path
from typing import Iterable, Union, List, Dict

import numpy as np
import spacy
from tqdm.auto import tqdm

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

        self.replacements = json.load((LOC.manual / 'replacements_dwie.json').open('r'))
        # for key, value in self.replacements.items():
        #     if not len(key) == len(value):
        #         raise AssertionError(f"Bad replacement dict. The length of key and value are unequal.\n"
        #                              f"\tlen({key}) = {len(key)}.\n"
        #                              f"\tlen({value}) = {len(value)}.")

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
        train_valid_docs = []
        for i, instance in enumerate(tqdm(dataset)):

            if 'test' in instance['tags']:
                doc = self._parse_(instance, 'test')
                test_docs.append(doc)
            elif 'train' in instance['tags']:
                doc = self._parse_(instance, 'train')
                train_valid_docs.append(doc)
            else:
                raise ValueError(f"Unexpected content of instance['tags']. Expected either 'train' or 'test' in there."
                                 f"Got {instance['tags']}")

        train_valid_len = len(train_valid_docs)
        test_len = len(test_docs)
        tv_indices = np.arange(train_valid_len)
        np.random.shuffle(tv_indices)

        # Get a test len sized portion out of tv_indices
        train_indices, valid_indices = tv_indices[:-test_len], tv_indices[-test_len:]

        valid_docs = [train_valid_docs[i] for i in valid_indices]
        train_docs = [train_valid_docs[i] for i in train_indices]

        self.write_to_disk('train', train_docs)
        self.write_to_disk('dev', valid_docs)
        self.write_to_disk('test', test_docs)
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

    def manually_fix(self, raw: dict) -> dict:
        """
            If the replacements are smaller than the instance, fix the entire document's char to reflect that.
            If they're the same size, just replace

        """

        """ Pull a replacement dict from manually written things. """
        for k, v in self.replacements.items():
            if len(k) == len(v):
                # simple replace
                raw['content'] = raw['content'].replace(k, v)

            elif k in raw['content']:
                raw = self._manually_fix_considerate_(raw, k, v)

        return raw

    def _manually_fix_considerate_(self, raw: dict, k: str, v: str) -> dict:

        while k in raw['content']:
            start_index = raw['content'].index(k)
            pre = raw['content'][:start_index]
            post = raw['content'][start_index + len(k):]

            # replace the text
            raw['content'] = pre + v + post
            raw['mentions'] = self._update_mentions_(mentions=raw['mentions'],
                                                     start_index=start_index,
                                                     end_index=start_index + len(k),
                                                     offset=len(v) - len(k), text=raw['content'])

            return raw

    def _update_mentions_(self, mentions: List[dict], start_index: int, end_index: int, offset: int, text: str) \
            -> List[dict]:
        """
            When your manual change causes the length of document to change, you have to adjust every char based
                span change.

            Start and End indices are old ones
        """

        for mention in mentions:
            backup = deepcopy(mention)
            # Case 1: mention lies before anything happens
            if mention['end'] < start_index:
                continue

            # if mention['text'] == '18':
            #     print('tomato')

            # Case 2: bad overlap:
            # the mention starts and ends within the change
            if start_index <= mention['begin'] < mention['end'] < end_index:

                _start_index = text[start_index: end_index].index(mention['text'])
                _new_text = text[start_index + _start_index: start_index + _start_index + len(mention['text'])]

                if not _new_text == mention['text']:
                    raise AssertionError(f"POTATO")

                mention['text'] = _new_text
                mention['begin'] = start_index + _start_index
                mention['end'] = start_index + _start_index + len(mention['text'])
                mention['changed'] = 1

                continue

            # Case 3: good overlap (mention is bigger than change on both sides)
            if mention['begin'] < start_index < end_index < mention['end']:
                # Simply update the margins and pull new text
                mention['end'] += offset
                mention['text'] = text[mention['begin']: mention['end']]
                mention['changed'] = 2

                continue

            # Case 4: left overlap:
            # the mention starts before the start of change, and ends before the end of it
            if mention['begin'] < start_index < mention['end'] < end_index:

                # Same, just check if something needs to be changed
                if not mention['text'] == text[mention['begin']: mention['end']]:
                    raise AssertionError(f"TODO: ideally you should update index and text in this case")

                continue

            # Case 5: right overlap:
            # the mention overlaps with the trailing end
            if start_index < mention['begin'] < end_index < mention['end']:
                mention['end'] += offset

                # It could be either that the "change" happens before the span beigins or that it happens after
                if mention['text'] == text[mention['begin']: mention['end']]:
                    ...
                else:
                    mention['begin'] += offset

                mention['text'] = text[mention['begin']: mention['end']]
                mention['changed'] = 3

                continue

            # Case 6: mention lies after the entire change happens
            if mention['begin'] >= end_index:
                mention['begin'] += offset
                mention['end'] += offset
                mention['changed'] = 4

                continue

            if not mention['text'] == text[mention['begin']: mention['end']]:
                raise AssertionError(f"Text Mismatch")
            raise AssertionError(f"New unknown condition")

        return mentions

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
            if (char_start == token['offset']) or \
                    (abs(char_start - token['offset']) == 1 and
                     token['token'][0] in ["'", '"', '-']):
                start_token = token_id
            if (char_end == token['offset'] + token['length']) or \
                    (abs(char_end - (token['offset'] + token['length'])) == 1 and
                     token['token'][-1] in ["'", '.', 's']):
                # i.e. if they match exactly or there is a one char difference and the token's last element is a s
                end_token = token_id + 1
            if start_token >= 0 and end_token >= 0:
                break

        if start_token < 0 or end_token < 0:
            raise ValueError(f"Could not find span corresponding to [{start_token}: {end_token}]. "
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

        if not (
                ' '.join(mention_words) == mention['text'] or
                ''.join(mention_words) == mention['text'] or
                (' '.join(mention_words)[:-1] == mention['text'] and ' '.join(mention_words)[-1] in ['-', '.', 's']) or
                (' '.join(mention_words)[1:] == mention['text'] and ' '.join(mention_words)[0] in ['-', '.'])
        ) and mention['text'].isalnum():
            # Exceptions time baby!
            if ' '.join(mention_words) not in ['"Gauck', ]:
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
    def get_empty_clusters(mention_based, content_based: list):
        """ find all cntent based clusters which dont exist in mention based. check if they have count zero. """
        empty_clusters = {}
        for content_based_v in content_based:
            content_based_k = content_based_v['concept']
            if not content_based_k in mention_based:
                """
                    Check if the cluster is actually empty.
                    Manually put in intercepts here, in case of annotation problems.
                """
                if content_based_v['count'] != 0 and not content_based_v['text'] in ['BSA']:
                    raise AssertionError(f"Expected this cluster to have some mentions but none were found: "
                                         f"{content_based_v}")
                empty_clusters[content_based_k] = content_based_v

        return empty_clusters

    @staticmethod
    def aggregate_mentions(mentions: List[dict]):
        """ Rearrange mentions based on concept IDs """
        clusters = {}
        for mention in mentions:
            clusters[mention['concept']] = clusters.get(mention['concept'], []) + [mention]

        return clusters

    def _parse_(self, content: dict, split_name: str) -> Document:

        content = self.manually_fix(content)

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
        empty_clusters = self.get_empty_clusters(clusters, content['concepts'])
        spans, tags = self.parse_relations(clusters, content['relations'], empty_clusters)
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
    def parse_relations(clusters: Dict[str, List[dict]], relations: List[dict], empty_clusters: Dict[str, dict]):
        """  """

        spans, tags = [], []

        for relation in relations:
            try:
                sub_mentions = clusters[relation['s']]
                obj_mentions = clusters[relation['o']]
            except KeyError as e:

                # This key doesnt' appear because its an empty cluster (zero mentions in text, for some reason)
                if e.args[0] in empty_clusters:
                    continue
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
