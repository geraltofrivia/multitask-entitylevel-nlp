"""

# Goals

To uncover the interplay between NER annotations and coref annotations. Specifically can finding NER directly lead to an estimation of num of clusters, do they all act as an antecedent?

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
from collections import Counter
from typing import List, Tuple, Dict, Union

from utils.misc import pop
from utils.data import Document
from dataloader import DataLoader
from config import LOCATIONS as LOC
from preproc.ontonotes import CoNLLOntoNotesParser
from utils.nlp import to_toks, remove_pos, NullTokenizer

nlp = spacy.load('en_core_web_sm')
# This tokenizer DOES not tokenize documents.
# Use this is the document is already tokenized.
nlp.tokenizer = NullTokenizer(nlp.vocab)


def _get_overlapped_ners_(key: Tuple[int, int], ners: List[Tuple[str, int, int]]) -> List[int]:
    """ If an element in spans is completely subsumed by the span in key, we return it."""
    popids: List[int] = []
    for i, ner in enumerate(ners):
        if key[0] <= ner[1] < ner[2] <= key[1]:
            popids.append(i)

    return popids


def _get_exact_match_ners_(key: List[int], ners: List[List[Union[str, int]]]) -> List[int]:
    """ If an element in spans is completely subsumed by the span in key, we return it."""
    popids: List[int] = []
    for i, ner in enumerate(ners):
        if key[0] == ner[1]  and ner[2] == key[1]:
            popids.append(i)

    return popids


def review(clustered_spans: Dict[int, list], clustered_spans_: Dict[int, list], ners, ners_, doc: Document):
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

    print("-------Not Clustered-------")
    for ent, ent_ in zip(ners, ners_):
        matches = {}
        # Calculate fuzzy distance to all spans in all clusters
        for cluster_id, (cluster_spans, cluster_texts) in enumerate(zip(doc.clusters, doc.clusters_)):

            # Calculate fuzzy string sim
            matches[cluster_id] = sorted(
                [(fuzz.partial_ratio(' '.join(ent_[1:]), ' '.join(text)), ' '.join(text), span)
                    for span, text in zip(cluster_spans, cluster_texts)],
                key= lambda x: -x[0])[:4]

        # Choose the three best clusters
        matches = dict(sorted(matches.items(), key=lambda kv: -max(tupl[0] for tupl in kv[1]))[:3])

        print(ent, ent_[1:])
        pprint(matches)
        print('--')

    print('potato')


def match_entities_to_coref_clusters(doc: Document):
    """
        1. Find exact matches between coref and entity spans.
        2. If entities are left over, consider the ones which resemble a coref span
                if you remove determiners from the start. And punctuations from the end. And adjectives.
        3. For those NERs that are Noun chunks (spacy),
            find coref spans that are noun chunks and check if the string matches!
    :param doc: A utils/data/Document dataclass instance
    """

    # Declare the dict which stores clustered spans (entities)
    clustered_spans = {i:[] for i,_ in enumerate(doc.clusters)}
    clustered_spans_ = {i:[] for i,_ in enumerate(doc.clusters)}

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
    for i, cluster in enumerate(doc.clusters):
        for span in cluster:

            # Do the pos based filtering on the coref span
            span = remove_pos(span, doc.pos)

            # Get overlapped stuff
            matched = _get_exact_match_ners_(span, ners_filtered)
            matched_spans = pop(ners, matched)
            _ = pop(ners_filtered, matched)
            matched_spans_ = pop(ners_, matched)
            clustered_spans[i] += matched_spans
            clustered_spans_[i] += matched_spans_

    print("Unresolved entities: ", len(ners))

    review(clustered_spans, clustered_spans_, ners, ners_, doc)

    # Iterate through every cluster, Iterate through every span in it
    #   and pull out the 'span' which is completely overlapped (subsumed)
    #
    # for i, cluster in enumerate(doc.clusters):
    #     for span in cluster:
    #
    #         # Get overlapped stuff
    #         overlapped = _get_overlapped_ners_(span, ners)
    #         overlapped_spans = pop(ners, overlapped)
    #         overlapped_spans_ = pop(ners_, overlapped)
    #         clustered_spans[i] += overlapped_spans
    #         clustered_spans_[i] += overlapped_spans_
    #
    # print("Unresolved entities: ", len(ners))

    # In the third run, try again but this time try to exact match span heads
    ner_heads: list = [['']+doc.head(ner[1:]) for ner in ners]

    # Iterate through every cluster,
    # Iterate through every span in it and pull out the ner span which is completely overlapped
    for i, cluster in enumerate(doc.clusters):
        for span in cluster:

            # Get overlapped stuff
            overlapped_ids = _get_overlapped_ners_(doc.head(span), ner_heads)
            if overlapped_ids:

                clustered_spans[i] += pop(ners, ids=overlapped_ids)
                print(pop(ners_, overlapped_ids))
                ner_heads: list = [['']+doc.head(ner[1:]) for ner in ners]

    print("Unresolved entities: ", len(ners))


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
    dl = DataLoader('ontonotes', 'train', ignore_empty_coref=True)

    summary['num_instances'] = len(dl)
    cardinalities = {}
    ignored = 0
    for i, doc in enumerate(dl):

        for num_elem, num_clus in count_cluster_cardinality(doc).items():
            cardinalities[num_elem] = cardinalities.get(num_elem, 0) + num_clus

        if doc.clusters != [] and doc.named_entities_gold != []:
            match_entities_to_coref_clusters(doc)
        else:
            ignored += 1
            continue


if __name__ == "__main__":
    run()