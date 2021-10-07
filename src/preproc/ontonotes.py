"""
As always, we start with trying to parse ontonotes well.
Specifically, we want
    - coref clusters
    - noun phrases
    - named entities (if annotated)

    - and predicted version of these properties? maybe not.

The dataclass is going to be document based. That is, one instance is one document.
"""
import pickle
import jsonlines
from pathlib import Path
from tqdm.auto import tqdm
from dataclasses import asdict
from typing import Iterable, Union, List, Optional

from config import LOCATIONS as LOC
from utils.data import CorefDocument
from utils.misc import to_toks, AnnotationBlockStack


class CoNLLOntoNotesParser:

    def __init__(self, ontonotes_dir: Path, splits: Iterable[str] = ('train',), ignore_empty_documents: bool = False):
        """
            :param ontonotes_dir: Path to the folder containing `development`, `train`, `test` subfolders.
            :param splits: a tuple of which subfolders should we process
            :param ignore_empty_documents: flag which if true
                will prevent documents with no coref clusters from being included
        """
        self.dir: Path = ontonotes_dir
        self.parsed: dict = {split_nm: [] for split_nm in splits}
        self.splits = ['train', 'development', 'test', 'conll-2012-test'] if not splits else splits

        self.flag_ignore_empty_documents: bool = ignore_empty_documents
        self.write_dir = LOC.parsed / 'ontonotes' / 'conll-2012'

    def write_to_disk(self, split: Union[str, Path], instances: List[CorefDocument]):
        """ Write a (large) list of documents to disk """

        # Assert that folder exists
        write_dir = self.write_dir / split
        write_dir.mkdir(parents=True, exist_ok=True)

        with (write_dir / 'dump.pkl').open('wb+') as f:
            pickle.dump(instances, f)

        with (write_dir / 'dump.jsonl').open('w+', encoding='utf8') as f:
            with jsonlines.Writer(f) as writer:
                writer.write_all([asdict(instance) for instance in instances])



    def run(self):

        for split in self.splits:
            outputs = self.parse(split)

            # Dump them to disk

    def parse(self, split_nm: Union[Path, str]):
        """ Where the actual parsing happens. One split at a time. """

        outputs: List[CorefDocument] = []

        folder_dir: Path = self.dir / split_nm
        assert folder_dir.exists(), f"The split {split_nm} does not exist in {self.dir}."

        folder_dir: Path = folder_dir / 'data' / 'english' / 'annotations'

        for docid, fname in enumerate(tqdm(folder_dir.glob('*/*/*/*gold_conll'))):
            # Iterate through all the files in this dir
            genre: str = str(fname).split('/')[-4]
            documents, clusters, speakers, docnames, docparts, docpos = self._parse_document_(path=fname)

            # Check if we want to ignore empty documents
            if self.flag_ignore_empty_documents:
                ne_documents, ne_clusters, ne_speakers, ne_docnames, ne_docparts = [], [], [], [], []

                for i, cluster in enumerate(clusters):
                    try:
                        if max(to_toks(cluster)) == -1:
                            continue
                    except TypeError:
                        # The cluster is not empty
                        ...

                    ne_documents += documents[i]
                    ne_clusters += clusters[i]
                    ne_speakers += speakers[i]
                    ne_docnames += docnames[i]
                    ne_docparts += docparts[i]

                documents = ne_documents
                clusters = ne_clusters
                speakers = ne_speakers
                docnames = ne_docnames
                docparts = ne_docparts

            # The remaining documents are to be converted into nice objects.
            for i in range(len(documents)):

                # Convert cluster info to spans and text sequences
                cluster_spans, cluster_text = self.convert_clusters(documents[i], clusters[i])

                doc = CorefDocument(
                    document=documents[i],
                    pos=docpos[i],
                    docname=docnames[i],
                    split=split_nm,
                    docpart=docparts[i],
                    clusters=cluster_spans.values(),
                    clusters_=cluster_text.values()
                )
                outputs.append(doc)

        return outputs

    @staticmethod
    def _normalize_word_(word, language):
        if language == "arabic":
            word = word[:word.find("#")]
        if word == "/." or word == "/?" or word == "/-":
            return word[1:]
        else:
            return word

    def _parse_document_(self, path, language="english"):
        """
        # Code taken from Mangoes

        Parses a single data file
        Returns the data from whatever documents are in the file.

        returns:
            words: Lists of Lists of Lists of strings. list of sentences. One sentence is a list of words.
            cluster_ids: Lists of Lists of Lists of ints or tuple(ints). Words that aren't mentions have either -1 as id
            speaker_ids: Lists of Lists of Lists of ints.
            doc_keys: List of document keys
        """
        doc_keys = []
        doc_sents = []
        doc_cluster_ids = []
        doc_speaker_ids = []
        doc_pos = []
        sentences = []
        sentence_cluster_ids = []
        sentence_speaker_ids = []
        sentence_pos_tags = []
        cur_sentence_words = []
        cur_sentence_cluster_ids = []
        cur_sentence_speaker_ids = []
        cur_sentence_pos_tags = []
        current_clusters = []
        docs = 0
        parts: List[int] = []
        with open(path, "r") as input_file:
            for line in input_file:
                if line.startswith("#begin document"):
                    doc_key = line.split()[2][:-1]
                    doc_keys.append(doc_key[1:-1])
                    assert line.split()[-1].isdigit()
                    parts.append(int(line.split()[-1]))
                    docs += 1
                elif line.startswith("#end document"):
                    assert len(sentences) == len(sentence_cluster_ids) == len(sentence_speaker_ids)
                    assert cur_sentence_words == []
                    doc_sents.append(sentences)
                    doc_cluster_ids.append(sentence_cluster_ids)
                    doc_speaker_ids.append(sentence_speaker_ids)
                    doc_pos.append(sentence_pos_tags)
                    sentences = []
                    sentence_cluster_ids = []
                    sentence_speaker_ids = []
                    sentence_pos_tags = []
                else:
                    data = line.split()
                    sentence_end = len(data) == 0
                    if sentence_end:
                        sentences.append(cur_sentence_words)
                        sentence_cluster_ids.append(cur_sentence_cluster_ids)
                        sentence_speaker_ids.append(cur_sentence_speaker_ids)
                        sentence_pos_tags.append(cur_sentence_pos_tags)
                        cur_sentence_words = []
                        cur_sentence_cluster_ids = []
                        cur_sentence_speaker_ids = []
                        cur_sentence_pos_tags = []
                    else:
                        cur_sentence_words.append(self._normalize_word_(data[3], language))
                        cur_sentence_pos_tags.append(data[4])
                        cur_sentence_speaker_ids.append(data[9])
                        raw_cluster_id = data[-1]
                        if raw_cluster_id == "-":
                            if len(current_clusters) == 0:
                                cluster_id = -1
                            elif len(current_clusters) == 1:
                                cluster_id = int(list(current_clusters)[0])
                            else:
                                cluster_id = tuple(int(item) for item in current_clusters)
                        else:
                            for part in raw_cluster_id.split("|"):
                                if "(" in part:
                                    current_clusters.append(part[1:-1] if ")" in part else part[1:])
                            if len(current_clusters) == 1:
                                cluster_id = int(list(current_clusters)[0])
                            else:
                                cluster_id = tuple(int(item) for item in current_clusters)
                            for part in raw_cluster_id.split("|"):
                                if ")" in part:
                                    current_clusters.remove(part[1:-1] if "(" in part else part[:-1])
                        cur_sentence_cluster_ids.append(cluster_id)
            assert len(doc_sents) == docs
            for i in range(docs):
                doc_keys[i] += f"_{i}"
            return doc_sents, doc_cluster_ids, doc_speaker_ids, doc_keys, parts, doc_pos

    @staticmethod
    def convert_clusters(document: List[List[str]], cluster: List[List[int]]) -> (dict, dict):
        """ based on the cluster info, get the span IDs and tokens """

        # If the clusters are empty, return empty lists
        try:
            if max(to_toks(cluster)) == -1:
                return {}, {}
        except TypeError:
            # The cluster is not empty
            ...

        cluster_spans, cluster_tokens = {}, {}

        # If the clusters are not empty,
        #   1. Flatten the document and cluster info from a list of list of tokens to a list of tokens
        #   2. Use a stack based mechanism to sort through the nested annotations - class AnnotationBlockStack
        #       ( btw which may be a int or a tuple for a token)

        flat_cluster = to_toks(cluster)
        flat_document = to_toks(document)

        stack = AnnotationBlockStack()
        for i, (token_cluster, token_text) in enumerate(zip(flat_cluster, flat_document)):

            completed_annotations: Optional[List] = stack.process(token=token_text, cluster=token_cluster, span_id=i)

            # All these completed annotations are to be noted in cluster_spans, cluster_tokens
            for block in completed_annotations:
                cluster_spans.setdefault(block.tag, []).append((block.start, block.end))
                cluster_tokens.setdefault(block.tag, []).append(block.words)

        return cluster_spans, cluster_tokens


if __name__ == '__main__':

    parser = CoNLLOntoNotesParser(LOC.ontonotes_conll, ignore_empty_documents=False)
    parser.run()
