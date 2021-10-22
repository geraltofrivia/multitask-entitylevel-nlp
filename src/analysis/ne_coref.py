"""

# Goals

To uncover the interplay between NER annotations and coref annotations.
Specifically can finding NER directly lead to an estimation of num of clusters, do they all act as an antecedent?

# Specifics

1. Try to find num. mentions per coref cluster
2. Try to coalesce NER mentions (within one coref cluster) into one 'entity' and re-do this exp.
3. Try to do this coalesce thing globally within one document.
"""

# Imports
import spacy
from thefuzz import fuzz
from pprint import pprint
from copy import deepcopy
from spacy.tokens import Doc
from collections import Counter
from typing import List, Tuple, Dict, Union

from utils.misc import pop
from utils.data import Document
from dataloader import DataLoader
from utils.nlp import to_toks, remove_pos, NullTokenizer, is_nchunk


def _get_textual_exact_match_ners_(key: List[str], ners: List[List[str]]) -> List[int]:
    """ If an element in spans is completely subsumed by the span in key, we return it."""
    popids: List[int] = []
    for i, ner in enumerate(ners):

        if ner[0] == 'FAKE':
            continue

        if key == ner or key == ner[1:]:
            popids.append(i)

    return popids


def _get_overlaps_ners_(key: List[int], ners: List[Tuple[str, int, int]]) -> List[int]:
    """ If an element in spans is completely subsumed by the span in key, we return it. Same if key subsumes ner. """
    popids: List[int] = []
    for i, ner in enumerate(ners):

        if ner[0] == 'FAKE':
            continue

        if key[0] <= ner[1] < ner[2] <= key[1] or ner[1] <= key[0] < key[1] <= ner[2]:
            popids.append(i)

    return popids


def _get_exact_match_ners_(key: List[int], ners: List[List[Union[str, int]]]) -> List[int]:
    """ If an element in spans is completely subsumed by the span in key, we return it."""
    popids: List[int] = []
    for i, ner in enumerate(ners):
        if key[0] == ner[1] and ner[2] == key[1]:
            popids.append(i)

    return popids


# noinspection PyTypeChecker
def review(clustered_spans: Dict[int, list], clustered_spans_: Dict[int, list], ners, ners_, doc: Document,
           skip_unclustered: bool = False):
    """
        At a glance, take a gander at
            - entities which have been assigned to clusters
            - entities which have not (and to which they may still belong)
    """
    global nlp

    n_clustered = sum(len(x) for x in clustered_spans.values())
    n_total = len(doc.named_entities_gold)

    print(doc.docname, doc.genre)
    print(f"Amongst {n_total} entities, {n_clustered} have been assigned to a coref cluster.")
    print("-------Clustered-------")
    # First lets demonstrate the entities which have been grouped by clusters
    for cluster_id, cluster_entities in clustered_spans_.items():
        if not cluster_entities:
            continue

        print(f"{cluster_id:2d}: ")
        for ent in cluster_entities:
            print(' '.join(ent))
        print(doc.clusters_[cluster_id])

    if skip_unclustered:
        return

    print("-------Not Clustered-------")
    for ent, ent_ in zip(ners, ners_):
        matches = {}
        # Calculate fuzzy distance to all spans in all clusters
        for cluster_id, (cluster_spans, cluster_texts) in enumerate(zip(doc.clusters, doc.clusters_)):

            # Calculate fuzzy string sim
            matches[cluster_id] = sorted(
                [(fuzz.token_set_ratio(' '.join(ent_[1:]), ' '.join(text)), ' '.join(text), span)
                    for span, text in zip(cluster_spans, cluster_texts)],
                key=lambda x: -x[0])[:4]

        # Choose the three best clusters
        matches = dict(sorted(matches.items(), key=lambda kv: -max(tupl[0] for tupl in kv[1]))[:3])

        print(ent, ent_[1:])
        pprint(matches)
        print('--')

    print("-------Done-------")


def match_entities_to_coref_clusters(doc: Document, spacy_doc: Doc):
    """
        1. Find exact matches between coref and entity spans.
        2. If entities are left over, consider the ones which resemble a coref span
                if you remove determiners from the start. And punctuations from the end. And adjectives.
        3. For those NERs that are Noun chunks (spacy),
            find coref spans that are noun chunks and check if the string matches!
    :param spacy_doc: a spacy object corresponding to the text of the current instance
    :param doc: A utils/data/Document dataclass instance
    """

    # Declare the dict which stores clustered spans (entities)
    clustered_spans = {i: [] for i, _ in enumerate(doc.clusters)}
    clustered_spans_ = {i: [] for i, _ in enumerate(doc.clusters)}

    noun_chunks = [[chunk.start, chunk.end] for chunk in spacy_doc]
    lemmas = [token.lemma for token in spacy_doc]
    def lemmatize(span, lemmas): return lemmas[span[0]: span[1]]

    # Make a copy of named entities
    ners = deepcopy(doc.named_entities_gold)
    ners_ = deepcopy(doc.named_entities_gold_)
    print("Unresolved entities: ", len(ners))

    # In the first run, find exact matches

    # Iterate through every cluster, Iterate through every span in it
    #   and pull out the 'span' which is completely overlapped
    for i, cluster in enumerate(doc.clusters):
        for span in cluster:

            # Get overlapped stuff
            matched = _get_exact_match_ners_(span, ners)
            matched_spans = pop(ners, matched)
            matched_spans_ = pop(ners_, matched)
            clustered_spans[i] += matched_spans
            clustered_spans_[i] += matched_spans_

    print("Unresolved entities: ", len(ners))

    """ 
        In the second run, find overlaps
        Repeat the same block as above. Except this time, make sure that the span is not prefixed with 
            adj/det and suffixed with punctuations. Do this for both, the NER and the COREF span. 
            
        Implementation:
            create a copy of NER spans, where this filtering has been done and use this list to find NER ids
    """
    ners_filtered = [[ner[0]]+remove_pos(ner[1:], doc.pos) for ner in ners]
    ners_filtered_ = [[ner[0]]+to_toks(doc.document)[ner[1]: ner[2]] for ner in ners_filtered]
    for i, cluster in enumerate(doc.clusters):
        for span in cluster:

            # Do the pos based filtering on the coref span
            span = remove_pos(span, doc.pos)

            # Get overlapped stuff
            matched = _get_exact_match_ners_(span, ners_filtered)
            matched_spans = pop(ners, matched)
            _ = pop(ners_filtered, matched)
            _ = pop(ners_filtered_, matched)
            matched_spans_ = pop(ners_, matched)
            clustered_spans[i] += matched_spans
            clustered_spans_[i] += matched_spans_

    print("Unresolved entities: ", len(ners))

    """
        In the third run, find noun chunks with the same things.
        For an unresolved entity which is a noun chunk, 
            try to find coref spans which are noun chunks and also match exactly.
                
        Note: do not do any form of "POS based filtering in this step"
    """
    ners_chunks = [span if span[1:] in noun_chunks or is_nchunk(span[1:], doc.pos) else ['FAKE', -2, -1]
                   for span in ners]
    ners_chunks_ = [ners_[i] if ners_chunks[i][0] != 'FAKE' else ['FAKE', 'alpha'] for i in range(len(ners))]
    ners_chunks_lemmatized = [[span[0]]+lemmatize(span[1:], lemmas) if span[0] != 'FAKE' else span
                              for span in ners_chunks]
    for cluster_id, cluster in enumerate(doc.clusters):
        for span_id, span in enumerate(cluster):

            # Check if the span is a noun chunk
            if not (span in noun_chunks or is_nchunk(span, doc.pos)):
                continue

            span_ = doc.clusters_[cluster_id][span_id]

            # Get overlapped stuff
            matched = _get_textual_exact_match_ners_(span_, ners_chunks_)
            matched_spans = pop(ners, matched)
            _ = pop(ners_filtered, matched)
            _ = pop(ners_filtered_, matched)
            _ = pop(ners_chunks, matched)
            _ = pop(ners_chunks_, matched)
            _ = pop(ners_chunks_lemmatized, matched)
            matched_spans_ = pop(ners_, matched)
            clustered_spans[cluster_id] += matched_spans
            clustered_spans_[cluster_id] += matched_spans_

            span_ = lemmatize(doc.clusters_[cluster_id][span_id], lemmas)

            # Repeated for lemmatized version of the text
            matched = _get_textual_exact_match_ners_(span_, ners_chunks_lemmatized)
            matched_spans = pop(ners, matched)
            _ = pop(ners_filtered, matched)
            _ = pop(ners_filtered_, matched)
            _ = pop(ners_chunks, matched)
            _ = pop(ners_chunks_, matched)
            _ = pop(ners_chunks_lemmatized, matched)
            matched_spans_ = pop(ners_, matched)
            clustered_spans[cluster_id] += matched_spans
            clustered_spans_[cluster_id] += matched_spans_

    print("Unresolved entities: ", len(ners))

    """
        In the fourth run, do the same 
            but when both of the things i.e. entities and coreferent phrases are POS filtered 
    """
    ners_filtered_chunks = [span if span[1:] in noun_chunks or is_nchunk(span[1:], doc.pos) else ['FAKE', -2, -1]
                            for span in ners_filtered]
    ners_filtered_chunks_ = [ners_[i] if ners_filtered_chunks[i][0] != 'FAKE' else ['FAKE', 'alpha']
                             for i in range(len(ners))]
    ners_chunks_lemmatized = [[span[0]]+lemmatize(span[1:], lemmas) if span[0] != 'FAKE' else span
                              for span in ners_chunks]

    for cluster_id, cluster in enumerate(doc.clusters):
        for span_id, span in enumerate(cluster):

            # Check if the span is a noun chunk
            if not (span in noun_chunks or is_nchunk(span, doc.pos)):
                continue

            span = remove_pos(span, doc.pos)
            span_ = to_toks(doc.document)[span[0]: span[1]]

            # Get overlapped stuff
            matched = _get_textual_exact_match_ners_(span_, ners_filtered_chunks_)
            matched_spans = pop(ners, matched)
            _ = pop(ners_filtered, matched)
            _ = pop(ners_filtered_, matched)
            _ = pop(ners_chunks, matched)
            _ = pop(ners_chunks_, matched)
            _ = pop(ners_chunks_lemmatized, matched)
            _ = pop(ners_filtered_chunks, matched)
            _ = pop(ners_filtered_chunks_, matched)
            matched_spans_ = pop(ners_, matched)
            clustered_spans[cluster_id] += matched_spans
            clustered_spans_[cluster_id] += matched_spans_

    print("Unresolved entities: ", len(ners))

    """
        In the fifth run, target fuzzy noun chunks.
        Condition: 
            both candidates are noun chunks
            either one completely subsumes the other
            
        Repeat for both, fitered and unfiltered ones.        
    """
    for cluster_id, cluster in enumerate(doc.clusters):
        for span_id, span in enumerate(cluster):

            # Check if the span is a noun chunk
            # if doc.docname == 'wb/eng/00/eng_0005_0' and span == [104, 108]:
            #     print('potato')

            if not (span in noun_chunks or is_nchunk(span, doc.pos)):
                # Get overlapped stuff
                continue

            matched = _get_overlaps_ners_(span, ners_chunks)
            matched_spans = pop(ners, matched)
            _ = pop(ners_filtered, matched)
            _ = pop(ners_filtered_, matched)
            _ = pop(ners_chunks, matched)
            _ = pop(ners_chunks_, matched)
            _ = pop(ners_chunks_lemmatized, matched)
            _ = pop(ners_filtered_chunks, matched)
            _ = pop(ners_filtered_chunks_, matched)
            matched_spans_ = pop(ners_, matched)
            clustered_spans[cluster_id] += matched_spans
            clustered_spans_[cluster_id] += matched_spans_

            # Repeat but for filtered stuff
            span = remove_pos(span, doc.pos)

            # Get overlapped stuff
            matched = _get_overlaps_ners_(span, ners_filtered_chunks)
            matched_spans = pop(ners, matched)
            _ = pop(ners_filtered, matched)
            _ = pop(ners_chunks, matched)
            _ = pop(ners_chunks_, matched)
            _ = pop(ners_filtered_chunks, matched)
            _ = pop(ners_filtered_chunks_, matched)
            matched_spans_ = pop(ners_, matched)
            clustered_spans[cluster_id] += matched_spans
            clustered_spans_[cluster_id] += matched_spans_

    print("Unresolved entities: ", len(ners))
    review(clustered_spans, clustered_spans_, ners, ners_, doc)


def count_cluster_cardinality(doc: Document) -> Dict[int, int]:
    cardinalities = [len(cluster) for cluster in doc.clusters]
    return dict(Counter(cardinalities))


def run():
    """
        Iterate over all documents and try to
            - match entities to coref clusters
            - find other info about coref clusters
    """
    summary = {}
    nlp = spacy.load('en_core_web_sm')
    # This tokenizer DOES not tokenize documents.
    # Use this is the document is already tokenized.
    nlp.tokenizer = NullTokenizer(nlp.vocab)
    dl = DataLoader('ontonotes', 'train', ignore_empty_coref=True)

    summary['num_instances'] = len(dl)
    cardinalities = {}
    ignored = 0
    for i, doc in enumerate(dl):

        # Get the spacy doc object for this one. It will be needed. Trust me.
        spacy_doc = nlp(doc.document)

        for num_elem, num_clus in count_cluster_cardinality(doc).items():
            cardinalities[num_elem] = cardinalities.get(num_elem, 0) + num_clus

        if doc.clusters != [] and doc.named_entities_gold != []:
            match_entities_to_coref_clusters(doc, spacy_doc)
        else:
            ignored += 1
            continue


if __name__ == "__main__":
    run()