"""
As always, we start with trying to parse ontonotes well.
Specifically, we want
    - coref clusters
    - noun phrases
    - named entities (if annotated)

    - and predicted version of these properties? maybe not.

The dataclass is going to be document based. That is, one instance is one document.
"""
import re
import json
import click
import spacy
import warnings
import unidecode
from pathlib import Path
from copy import deepcopy
from tqdm.auto import tqdm
from spacy.tokens import Token
from typing import Iterable, Union, List

# Local imports
try:
    import _pathfix
except ImportError:
    from . import _pathfix
from utils.data import Document, Clusters, NamedEntities, TypedRelations
from utils.misc import NERAnnotationBlockStack
from utils.nlp import to_toks, NullTokenizer
from preproc.commons import GenericParser
from config import LOCATIONS as LOC


class CoNLLOntoNotesParser(GenericParser):

    def __init__(
            self,
            raw_dir: Path,
            suffixes: Iterable[str] = (),
            ignore_empty_documents: bool = False,
    ):
        """
        :param raw_dir: Path to the folder containing `development`, `train`, `test` subfolders.
        :param suffixes: a tuple of which sub folders should we process
        :param ignore_empty_documents: flag which if true
            will prevent documents with no coref clusters from being included
        """
        super().__init__(raw_dir=raw_dir, suffixes=suffixes, ignore_empty_documents=ignore_empty_documents)
        self.dir: Path = raw_dir
        self.splits = (
            ["train", "development", "test", "conll-2012-test"]
            if not self.suffixes
            else self.suffixes
        )
        self.parsed: dict = {split_nm: [] for split_nm in self.splits}

        self.flag_ignore_empty_documents: bool = ignore_empty_documents
        self.write_dir = self.write_dir / "ontonotes"

        self.re_ner_tags = r"\([a-zA-Z]*|\)"

        self.nlp = spacy.load("en_core_web_sm")
        # This tokenizer DOES not tokenize documents.
        # Use this is the document is already tokenized.
        self.nlp.tokenizer = NullTokenizer(self.nlp.vocab)

    def run(self):

        for split in self.splits:
            # First, clear out all the previously processed things from the disk
            self.delete_preprocessed_files(split)
            outputs = self.parse(split)

            # Dump them to disk
            self.write_to_disk(split, outputs)

    def parse(self, split_nm: Union[Path, str]):
        """Where the actual parsing happens. One split at a time."""

        outputs: List[Document] = []

        folder_dir: Path = self.dir / split_nm
        assert (
            folder_dir.exists()
        ), f"The split {split_nm} does not exist in {self.dir}."

        folder_dir: Path = folder_dir / "data" / "english" / "annotations"

        n_files = len([0 for _ in folder_dir.rglob("*.gold_conll")])
        for doc_id, f_name in enumerate(
                tqdm(folder_dir.rglob("*.gold_conll"), total=n_files)
        ):

            # Iterate through all the files in this dir
            genre: str = str(f_name).split("/")[-4]
            (
                documents,
                clusters,
                speakers,
                doc_names,
                doc_parts,
                doc_pos,
                doc_ner_raw,
            ) = self._parse_conll_document_(path=f_name)

            # Check if we want to ignore empty documents
            if self.flag_ignore_empty_documents:
                ne_documents, ne_clusters, ne_speakers, ne_docnames, ne_docparts = (
                    [],
                    [],
                    [],
                    [],
                    [],
                )

                for i, cluster in enumerate(clusters):
                    if not cluster:
                        continue

                    ne_documents += documents[i]
                    ne_clusters += clusters[i]
                    ne_speakers += speakers[i]
                    ne_docnames += doc_names[i]
                    ne_docparts += doc_parts[i]

                documents = ne_documents
                clusters = ne_clusters
                # speakers = ne_speakers
                doc_names = ne_docnames
                doc_parts = ne_docparts

            # The remaining documents are to be converted into nice objects.
            for i in range(len(documents)):

                # Convert cluster info to spans and text sequences
                # cluster_spans, cluster_text = self.convert_clusters(documents[i], clusters[i])

                # Add +1 to span ends in all cluster annotations
                for j, cluster in enumerate(clusters[i]):
                    clusters[i][j] = [[sp[0], sp[1] + 1] for sp in cluster]

                # Convert cluster spans to text
                flat_doc = to_toks(documents[i])
                # noinspection PyTypeChecker
                spacy_doc = self.nlp(flat_doc)
                clusters_ = []
                for cluster_id, cluster in enumerate(clusters[i]):
                    clusters_.append([])
                    for span in cluster:
                        clusters_[cluster_id].append(flat_doc[span[0]: span[1]])

                # Make a coref clusters object
                coref = Clusters(spans=list(clusters[i]))

                # Convert NER tags into cluster-like things
                (
                    ner_gold_spans,
                    ner_gold_words,
                    ner_gold_tags,
                ) = self._onto_process_ner_tags_(doc_ner_raw[i], documents[i])
                # (
                #     ner_spacy_spans,
                #     ner_spacy_words,
                #     ner_spacy_tags,
                # ) = self._spacy_process_ner_tags_(spacy_doc)

                # Make NER objects
                ner_gold = NamedEntities(
                    spans=ner_gold_spans, tags=ner_gold_tags, words=ner_gold_words
                )
                # ner_spacy = NamedEntities(
                #     spans=ner_spacy_spans, tags=ner_spacy_tags, words=ner_spacy_words
                # )

                # Span heads calculation
                span_heads = self.get_span_heads(
                    spacy_doc, ner_gold_spans + coref.get_all_spans()
                    # spacy_doc, ner_gold_spans + ner_spacy_spans + coref.get_all_spans()
                )

                # Add span heads, words, and pos in NER and Coref objects
                coref.allocate_span_heads(span_heads=span_heads)
                coref.add_words(documents[i])
                coref.add_pos(doc_pos[i])

                ner_gold.allocate_span_heads(span_heads=span_heads)
                ner_gold.add_words(documents[i])
                ner_gold.add_pos(doc_pos[i])

                # ner_spacy.allocate_span_heads(span_heads=span_heads)
                # ner_spacy.add_words(documents[i])
                # ner_spacy.add_pos(doc_pos[i])

                doc = Document(
                    document=documents[i],
                    pos=doc_pos[i],
                    docname=doc_names[i],
                    split=split_nm,
                    genre=genre,
                    docpart=doc_parts[i],
                    ner=ner_gold,
                    # ner_spacy=ner_spacy,
                    coref=coref,
                    rel=TypedRelations([], [])
                )
                outputs.append(doc)

        return outputs

    @staticmethod
    def _spacy_process_ner_tags_(
            doc: spacy.tokens.Doc,
    ) -> (List[str], List[Union[int]], List[str]):
        ents = [[ent.start, ent.end] for ent in doc.ents]
        ents_ = [[tok.text for tok in ent] for ent in doc.ents]
        ents_tags = [ent.label_ for ent in doc.ents]
        return ents, ents_, ents_tags

    def _normalize_word_(self, word, language):
        # We normalise unicode etc here. It will make working with HF tokenizers down the road much easier.
        backup_word = deepcopy(word)
        word = self.replacements.get(word, word)
        word = unidecode.unidecode_expect_ascii(word, errors="strict")

        # DEBUG
        if word == "" and backup_word != "":
            warnings.warn(
                f"The word {backup_word} got changed to an empty string. This could create problems down the line. "
                f"You should get in debug mode and investigate."
            )

        if language == "arabic":
            word = word[: word.find("#")]
        if word == "/." or word == "/?" or word == "/-":
            return word[1:]
        else:
            return word

    def _parse_conll_document_(self, path, language="english"):
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
        doc_clusters = []
        doc_speaker_ids = []
        doc_pos = []
        doc_ner_raw = []
        sentences = []
        clusters = {}
        sentence_cluster_ids = []
        sentence_speaker_ids = []
        sentence_pos_tags = []
        sentence_ner_tags = []
        cur_sentence_words = []
        cur_sentence_cluster_ids = []
        cur_sentence_speaker_ids = []
        cur_sentence_pos_tags = []
        cur_sentence_ner_tags = []
        # current_clusters = []
        docs = 0
        parts: List[int] = []
        with open(path, "r") as input_file:
            for line in input_file:
                if line.startswith("#begin document"):
                    doc_key = line.split()[2][:-1]
                    doc_keys.append(doc_key[1:-1])
                    num_words = 0
                    assert line.split()[-1].isdigit()
                    parts.append(int(line.split()[-1]))
                    docs += 1
                elif line.startswith("#end document"):
                    assert (
                            len(sentences)
                            == len(sentence_cluster_ids)
                            == len(sentence_speaker_ids)
                    )
                    assert cur_sentence_words == []
                    doc_sents.append(sentences)
                    merged_clusters = []
                    for c1 in clusters.values():
                        existing = None
                        for m in c1:
                            for c2 in merged_clusters:
                                if tuple(m) in c2:
                                    existing = c2
                                    break
                            if existing is not None:
                                break
                        if existing is not None:
                            print("Merging clusters (shouldn't happen very often.)")
                            existing.update([tuple(x) for x in c1])
                        else:
                            merged_clusters.append(set([tuple(x) for x in c1]))
                    merged_clusters = [list(c) for c in merged_clusters]
                    doc_clusters.append(merged_clusters)
                    doc_speaker_ids.append(sentence_speaker_ids)
                    doc_pos.append(sentence_pos_tags)
                    doc_ner_raw.append(sentence_ner_tags)
                    sentences = []
                    sentence_cluster_ids = []
                    sentence_speaker_ids = []
                    sentence_pos_tags = []
                    sentence_ner_tags = []
                    clusters = {}
                else:
                    data = line.split()
                    sentence_end = len(data) == 0
                    if sentence_end:
                        sentences.append(cur_sentence_words)
                        sentence_cluster_ids.append(cur_sentence_cluster_ids)
                        sentence_speaker_ids.append(cur_sentence_speaker_ids)
                        sentence_pos_tags.append(cur_sentence_pos_tags)
                        sentence_ner_tags.append(cur_sentence_ner_tags)
                        cur_sentence_words = []
                        cur_sentence_cluster_ids = []
                        cur_sentence_speaker_ids = []
                        cur_sentence_pos_tags = []
                        cur_sentence_ner_tags = []
                    else:
                        cur_sentence_words.append(
                            self._normalize_word_(data[3], language)
                        )
                        cur_sentence_pos_tags.append(data[4])
                        cur_sentence_ner_tags.append(data[10])
                        cur_sentence_speaker_ids.append(data[9])
                        raw_cluster_id = data[-1]
                        for part in raw_cluster_id.split("|"):
                            if "(" in part:
                                clus_id = part[1:-1] if ")" in part else part[1:]
                                if clus_id not in clusters:
                                    clusters[clus_id] = []
                                clusters[clus_id].append([num_words])
                        for part in raw_cluster_id.split("|"):
                            if ")" in part:
                                clus_id = part[1:-1] if "(" in part else part[:-1]
                                for i in range(len(clusters[clus_id]) - 1, -1, -1):
                                    if len(clusters[clus_id][i]) == 1:
                                        clusters[clus_id][i].append(num_words)
                                        break
                        num_words += 1
            assert len(doc_sents) == docs
            for i in range(docs):
                doc_keys[i] += f"_{i}"
            return (
                doc_sents,
                doc_clusters,
                doc_speaker_ids,
                doc_keys,
                parts,
                doc_pos,
                doc_ner_raw,
            )

    def _onto_process_ner_tags_(self, tags: list, document: list) -> (list, list, list):
        """Convert raw columns of annotations into proper tags. Expect nested tags, use a stack."""
        finalised_annotations_span = []
        finalised_annotations_text = []
        finalised_annotation_tag = []
        stack = NERAnnotationBlockStack()

        # Flatten the sentence structure
        tags_f = to_toks(tags)
        document_f = to_toks(document)
        assert len(tags_f) == len(
            document_f
        ), f"There are {len(document_f)} words but only {tags_f} tags."

        # iterate over the tags, and handle the annotations using a AnnotationBlockStack
        # For every token, if it begins with bracket open (can be nested)
        # Note the tag inside the bracket open and make a new block on the stack
        # For every bracket close, pop the last entry in the stack
        # For every other annotation, just register the word inside the tag's words list
        for word_id, (tag, token) in enumerate(zip(tags_f, document_f)):

            # Begin condition
            for match in re.findall(self.re_ner_tags, tag):
                if "(" not in match:
                    continue
                match_ = match[:].replace("(", "")
                stack.add(word_id, match_.strip())

            # Just register the word, regardless of whatever happens
            stack.register_word(token)

            # End condition
            for match in re.findall(self.re_ner_tags, tag):
                if ")" not in match:
                    continue
                ner_span, ner_text, ner_tag = stack.pop(word_id + 1)
                finalised_annotations_span.append(ner_span)
                finalised_annotations_text.append(ner_text)
                finalised_annotation_tag.append(ner_tag)

        return (
            finalised_annotations_span,
            finalised_annotations_text,
            finalised_annotation_tag,
        )


@click.command()
@click.option(
    "--suffix",
    "-s",
    type=str,
    help="The name of the dataset SPLIT e.g. train, test, development, conll-2012-test etc",
)
@click.option(
    "--ignore-empty",
    "-i",
    is_flag=True,
    help="If True, we ignore the documents without any coref annotation",
)
def run(suffix: str, ignore_empty: bool):
    parser = CoNLLOntoNotesParser(
        LOC.ontonotes_conll, suffixes=[suffix], ignore_empty_documents=ignore_empty
    )
    parser.run()


if __name__ == "__main__":
    run()
