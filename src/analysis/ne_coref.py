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
import json
import click
import spacy
from thefuzz import fuzz
from pprint import pprint
from copy import deepcopy
from tqdm.auto import tqdm
from spacy.tokens import Doc
from typing import List, Tuple, Dict, Union

from utils.misc import pop
from utils.data import Document
from dataiter import RawDataset
from config import LOCATIONS as LOC
from utils.nlp import to_toks, remove_pos, NullTokenizer, is_nchunk

DEBUG = False
ENTITY_TAG_BLACKLIST = {
    'named_entities_gold': ['CARDINAL', 'MONEY', 'PERCENT', 'ORDINAL', 'TIME', 'QUANTITY'],
    'named_entities_spacy': ['CARDINAL', 'MONEY', 'PERCENT', 'ORDINAL', 'TIME', 'QUANTITY']
}


def _get_textual_exact_match_ners_(key: List[str], ners: List[List[str]]) -> List[int]:
    """ If an element in spans is completely subsumed by the span in key, we return it."""
    popids: List[int] = []
    for i, ner in enumerate(ners):

        if ner == [-2, -1]:
            continue

        if key == ner:
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
        if key[0] == ner[0] and key[1] == ner[1]:
            popids.append(i)

    return popids


# noinspection PyTypeChecker
def review(clustered_spans: Dict[int, list], clustered_spans_: Dict[int, list], ners, ners_, doc: Document,
           ent_src: str, skip_unclustered: bool = False):
    """
        At a glance, take a gander at
            - entities which have been assigned to clusters
            - entities which have not (and to which they may still belong)
    """

    n_clustered = sum(len(x) for x in clustered_spans.values())
    n_total = len(getattr(doc, ent_src))

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
        print(doc.coref.words[cluster_id])

    if skip_unclustered:
        return

    print("-------Not Clustered-------")
    for ent, ent_ in zip(ners, ners_):
        matches = {}
        # Calculate fuzzy distance to all spans in all clusters
        for cluster_id, (cluster_spans, cluster_texts) in enumerate(zip(doc.coref.spans, doc.coref.words)):
            # Calculate fuzzy string sim
            matches[cluster_id] = sorted(
                [(fuzz.token_set_ratio(' '.join(ent_[1:]), ' '.join(text)), ' '.join(text), span)
                 for span, text in zip(cluster_spans, cluster_texts)],
                key=lambda x: -x[0])[:4]

        # Choose the three best clusters
        matches = dict(sorted(matches.items(), key=lambda kv: -max(tuple_[0] for tuple_ in kv[1]))[:3])

        print(ent, ent_[1:])
        pprint(matches)
        print('--')

    print("-------Done-------")


def match_entities_to_coref_clusters(doc: Document, spacy_doc: Doc, ent_src: str, filter_entities: bool) \
        -> (Dict[int, List[int]], List[Union[str, int]]):
    """
        1. Find exact matches between coref and entity spans.
        2. If entities are left over, consider the ones which resemble a coref span
                if you remove determiners from the start. And punctuations from the end. And adjectives.
        3. For those NERs that are Noun chunks (spacy),
            find coref spans that are noun chunks and check if the string matches!
    """

    # Declare the dict which stores clustered spans (entities)
    clustered_spans = {i: [] for i, _ in enumerate(doc.coref.spans)}
    clustered_spans_ = {i: [] for i, _ in enumerate(doc.coref.spans)}

    noun_chunks = [[chunk.start, chunk.end] for chunk in spacy_doc.noun_chunks]
    lemmas = [token.lemma_ for token in spacy_doc]

    def lemmatize(span, lemmas):
        return lemmas[span[0]: span[1]]

    # Make a copy of named entities
    ners = deepcopy(getattr(doc, ent_src).spans)
    ners_ = deepcopy(getattr(doc, ent_src).words)
    nertags = deepcopy(getattr(doc, ent_src).tags)

    if filter_entities:
        popids = []
        for i, ner in enumerate(ners):
            if nertags[i] in ENTITY_TAG_BLACKLIST[ent_src]:
                popids.append(i)

        _ = pop(ners, popids)
        _ = pop(ners_, popids)
        _ = pop(nertags, popids)

    if DEBUG: print("Unresolved entities: ", len(ners))

    # In the first run, find exact matches

    # Iterate through every cluster, Iterate through every span in it
    #   and pull out the 'span' which is completely overlapped
    for i, cluster in enumerate(doc.coref.spans):
        for span in cluster:
            # Get overlapped stuff
            matched = _get_exact_match_ners_(span, ners)
            matched_spans = pop(ners, matched)
            matched_spans_ = pop(ners_, matched)
            matched_spans_tags = pop(nertags, matched)
            clustered_spans[i] += matched_spans
            clustered_spans_[i] += matched_spans_

    if DEBUG: print("Unresolved entities: ", len(ners))

    """ 
        In the second run, find overlaps
        Repeat the same block as above. Except this time, make sure that the span is not prefixed with 
            adj/det and suffixed with punctuations. Do this for both, the NER and the COREF span. 
            
        Implementation:
            create a copy of NER spans, where this filtering has been done and use this list to find NER ids
    """

    # Note: the number of actual spans in ners, ners_filtered etc etc is always the same.
    ners_filtered = [remove_pos(ner, doc.pos) for ner in ners]
    ners_filtered_ = [to_toks(doc.document)[ner[0]: ner[1]] for ner in ners_filtered]

    for i, cluster in enumerate(doc.coref.spans):
        for span in cluster:
            # Do the pos based filtering on the coref span
            span = remove_pos(span, doc.pos)

            # Get overlapped stuff
            matched = _get_exact_match_ners_(span, ners_filtered)
            matched_spans = pop(ners, matched)
            _ = pop(nertags, matched)
            _ = pop(ners_filtered, matched)
            _ = pop(ners_filtered_, matched)
            matched_spans_ = pop(ners_, matched)
            clustered_spans[i] += matched_spans
            clustered_spans_[i] += matched_spans_

    if DEBUG: print("Unresolved entities: ", len(ners))

    """
        In the third run, find noun chunks with the same things.
        For an unresolved entity which is a noun chunk, 
            try to find coref spans which are noun chunks and also match exactly.
                
        Note: do not do any form of "POS based filtering in this step"
    """
    ners_chunks = [span if span in noun_chunks or is_nchunk(span, doc.pos) else [-2, -1]
                   for span in ners]
    ners_chunks_ = [ners_[i] if ners_chunks[i] != [-2, -1] else ['alphabetagamma'] for i in range(len(ners))]
    ners_chunks_lemmatized = [lemmatize(span, lemmas) if span != [-2, -1] else 'alphabetagamma' for span in ners_chunks]
    for cluster_id, cluster in enumerate(doc.coref.spans):
        for span_id, span in enumerate(cluster):

            # Check if the span is a noun chunk
            if not (span in noun_chunks or is_nchunk(span, doc.pos)):
                continue

            span_ = doc.coref.words[cluster_id][span_id]

            # Get overlapped stuff
            matched = _get_textual_exact_match_ners_(span_, ners_chunks_)
            matched_spans = pop(ners, matched)
            _ = pop(ners_filtered, matched)
            _ = pop(ners_filtered_, matched)
            _ = pop(nertags, matched)
            _ = pop(ners_chunks, matched)
            _ = pop(ners_chunks_, matched)
            _ = pop(ners_chunks_lemmatized, matched)
            matched_spans_ = pop(ners_, matched)
            clustered_spans[cluster_id] += matched_spans
            clustered_spans_[cluster_id] += matched_spans_

            span_ = lemmatize(doc.coref.spans[cluster_id][span_id], lemmas)

            # Repeated for lemmatized version of the text
            matched = _get_textual_exact_match_ners_(span_, ners_chunks_lemmatized)
            matched_spans = pop(ners, matched)
            _ = pop(ners_filtered, matched)
            _ = pop(ners_filtered_, matched)
            _ = pop(ners_chunks, matched)
            _ = pop(nertags, matched)
            _ = pop(ners_chunks_, matched)
            _ = pop(ners_chunks_lemmatized, matched)
            matched_spans_ = pop(ners_, matched)
            clustered_spans[cluster_id] += matched_spans
            clustered_spans_[cluster_id] += matched_spans_

    if DEBUG: print("Unresolved entities: ", len(ners))

    """
        In the fourth run, do the same 
            but when both of the things i.e. entities and coreferent phrases are POS filtered 
    """
    ners_filtered_chunks = [span if span in noun_chunks or is_nchunk(span, doc.pos) else [-2, -1]
                            for span in ners_filtered]
    ners_filtered_chunks_ = [ners_[i] if ners_filtered_chunks[i] != [-2, -1] else ['alphabetagamma']
                             for i in range(len(ners))]
    ners_chunks_lemmatized = [ lemmatize(span, lemmas) if span != [-2, -1] else span
                              for span in ners_chunks]

    for cluster_id, cluster in enumerate(doc.coref.spans):
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
            _ = pop(nertags, matched)
            _ = pop(ners_chunks_, matched)
            _ = pop(ners_chunks_lemmatized, matched)
            _ = pop(ners_filtered_chunks, matched)
            _ = pop(ners_filtered_chunks_, matched)
            matched_spans_ = pop(ners_, matched)
            clustered_spans[cluster_id] += matched_spans
            clustered_spans_[cluster_id] += matched_spans_

    if DEBUG: print("Unresolved entities: ", len(ners))

    """
        In the fifth run, target fuzzy noun chunks.
        Condition: 
            both candidates are noun chunks
            either one completely subsumes the other
            
        Repeat for both, filtered and unfiltered ones.        
    """
    for cluster_id, cluster in enumerate(doc.coref.spans):
        for span_id, span in enumerate(cluster):

            if not (span in noun_chunks or is_nchunk(span, doc.pos)):
                # Get overlapped stuff
                continue

            matched = _get_overlaps_ners_(span, ners_chunks)
            matched_spans = pop(ners, matched)
            _ = pop(ners_filtered, matched)
            _ = pop(ners_filtered_, matched)
            _ = pop(ners_chunks, matched)
            _ = pop(nertags, matched)
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
            _ = pop(ners_filtered_, matched)
            _ = pop(ners_chunks, matched)
            _ = pop(ners_chunks_, matched)
            _ = pop(nertags, matched)
            _ = pop(ners_filtered_chunks, matched)
            _ = pop(ners_filtered_chunks_, matched)
            _ = pop(ners_chunks_lemmatized, matched)
            matched_spans_ = pop(ners_, matched)
            clustered_spans[cluster_id] += matched_spans
            clustered_spans_[cluster_id] += matched_spans_

    if DEBUG:
        print("Unresolved entities: ", len(ners))
        review(clustered_spans, clustered_spans_, ners, ners_, doc, ent_src=ent_src)

    """
        Returning information
            - which entities were unclustered
            - which entities went to which cluster
            
        In both information, we provide the index of the NER tag in getattr(doc, ent_src) ...
    """
    if not (len(ners) == len(ners_) == len(ners_filtered) == len(ners_filtered_) == len(ners_filtered_chunks)
            == len(ners_filtered_chunks_) == len(ners_chunks) == len(ners_chunks_)
            == len(ners_chunks_lemmatized) == len(nertags)):
        print(f"ners: {len(ners)}, {len(ners_)}")
        print(f"ners filtered: {len(ners_filtered)}, {len(ners_filtered_)}")
        print(f"ners chunks: {len(ners_chunks)}, {len(ners_chunks_)}")
        print(f"ners chunks filtered: {len(ners_filtered_chunks)}, {len(ners_filtered_chunks_)}")
        print(f"ners chunks lemmatized: {len(ners_chunks_lemmatized)}")
        print(f"ners tags: {len(nertags)}")
        raise AssertionError("We are not counting the leftover entities correctly somewhere. ")

    clustered_span_indices = {_id: [getattr(doc, ent_src).spans.index(span) for span in _ents]
                              for _id, _ents in clustered_spans.items()}
    return clustered_span_indices, ners, nertags


def count_cluster_cardinality(doc: Document) -> List[int]:
    """ Returns a dict where k: num of elements in one cluster; v: num of such clusters"""
    return [len(cluster) for cluster in doc.coref.spans]


def count_doc_n_clusters(doc: Document) -> int:
    """ Returns an int: num of clusters in this document """
    return len(doc.coref.spans)


def count_doc_n_entities(doc: Document, ent_src: str) -> int:
    """ Returns an int: num of named entities (gold) in this document """
    return len(getattr(doc, ent_src))


def count_doc_n_coref_entities(doc: Document) -> int:
    """ Returns an int: num of all spans in all clusters in this document """
    return sum(len(cluster_spans) for cluster_spans in doc.coref.spans)


def count_tag_n_entities(doc: Document, ent_src: str) -> List[str]:
    """ Returns a dict where k: ner tag, v: num of elements in the doc with this tag"""
    return [tag for tag in getattr(doc, ent_src).tags]


def count_coref_span_length(doc: Document) -> List[int]:
    """ Returns a list containing the length of all coref spans """
    return [span[1] - span[0] for cluster in doc.coref.spans for span in cluster]


def count_ner_span_length(doc: Document, ent_src: str) -> List[int]:
    """ Returns a list containing the length of all named entity spans (based on the entity source) """
    return [span[2] - span[1] for span in getattr(doc, ent_src).spans]


def get_ungrounded_clusters(doc: Document) -> List[int]:
    """ Returns the index of ungrounded clusters in the document. Done based on POS tags"""
    # clusters_pos = [[to_toks(doc.pos)[span[0]: span[1]] for span in cluster] for cluster in doc.coref.clusters]
    return [i for i, cluster_pos in enumerate(doc.coref.pos) if not ('NN' in cluster_pos or 'NNP' in cluster_pos)]


@click.command()
@click.option('--split', '-s', type=str, default='train',
              help="The name of the ontonotes (CoNLL) SPLIT e.g. train, test, development, conll-2012-test etc")
@click.option('--entity-source', '-es', type=str, default='gold',
              help="`gold` would use Gold NER annotations in Ontonotes. `spacy` would use Spacy's annotations.")
# @click.option('--name', '-n', type=str,
#               help="The name of the summary object to be written to disk")
@click.option('--filter-named-entities', '-fner', is_flag=True,
              help="If enabled, we consider all named entities. If not we consider a subset ignoring cardinals etc.")
@click.option('--debug', '-d', is_flag=True,
              help="If enabled, we print a whole bunch of things "
                   "and also review all the entities which are not clustered")
def run(split: str, entity_source: str, filter_named_entities: bool, debug: bool):
    """
        Iterate over all documents and try to
            - match entities to coref clusters
            - find other info about coref clusters
    """
    global DEBUG

    DEBUG = debug
    summary = {}
    nlp = spacy.load('en_core_web_sm')
    # This tokenizer DOES not tokenize documents.
    # Use this is the document is already tokenized.
    nlp.tokenizer = NullTokenizer(nlp.vocab)
    ds = RawDataset('ontonotes', split, tasks=('coref',))

    assert entity_source in ['spacy', 'gold'], f"Unknown entity source: {entity_source}"
    ent_src = f'ner_{entity_source}'

    # Create experiment name
    name = f"{entity_source}ner_{'all' if not filter_named_entities else 'some'}.json"

    summary['ignored_instances'] = 0
    summary['num_instances'] = len(ds)
    summary['tokens_per_doc'] = []
    summary['clusters_per_doc'] = []
    summary['elements_per_cluster'] = []
    summary['coref_entities_per_doc'] = []
    summary['named_entities_per_doc'] = []
    summary['named_entities_per_tag'] = []
    summary['ungrounded_clusters_per_doc'] = {}
    summary['length_coref_per_span'] = []
    summary['length_ner_per_span'] = []

    summary['named_entities_unmatched_per_doc'] = []
    summary['named_entities_unmatched_per_tag'] = []
    summary['named_entities_matched_per_tag'] = []
    summary['named_entities_matched_per_cluster'] = []
    summary['clusters_unmatched_per_doc'] = []
    summary['clusters_matched_per_doc'] = []
    summary['clusters_matched_different_tags_per_doc'] = []

    for i, doc in enumerate(tqdm(ds)):

        if getattr(doc, ent_src).isempty:
            summary['ignored_instances'] += 1
            continue

        # Get the spacy doc object for this one. It will be needed. Trust me.
        # noinspection PyTypeChecker
        spacy_doc = nlp(to_toks(doc.document))

        # Find the number of tokens in a document
        summary['tokens_per_doc'].append(len(to_toks(doc.document)))

        # Find statistics on the number of elements per cluster
        cardinalities = count_cluster_cardinality(doc)
        summary['elements_per_cluster'] += cardinalities

        # Find statistics on the number of coref clusters in one document
        summary['clusters_per_doc'].append(count_doc_n_clusters(doc))

        # Find statistics on the number of named entities in a document
        summary['named_entities_per_doc'].append(count_doc_n_entities(doc, ent_src=ent_src))

        # Find statistics on the number of coreferent entities in a doc
        summary['coref_entities_per_doc'].append(count_doc_n_coref_entities(doc))

        # Lets try and find ungrounded clusters in the document using simple metrics
        summary['ungrounded_clusters_per_doc'][doc.docname] = get_ungrounded_clusters(doc)

        # Find statistics on the number of named entities per named entity tags
        summary['named_entities_per_tag'] += count_tag_n_entities(doc, ent_src=ent_src)

        raise NotImplementedError

        matched_entity_ids, unmatched_entities = match_entities_to_coref_clusters(doc, spacy_doc, ent_src=ent_src,
                                                                                  filter_entities=filter_named_entities)
        unmatched_clusters = [k for k, v in matched_entity_ids.items() if not v]
        matched_clusters = [k for k, v in matched_entity_ids.items() if v]
        matched_entities = {k: [getattr(doc, ent_src).words[i] for i in v] for k, v in matched_entity_ids.items()}
        clus_matched_diff_tags_per_doc = len([1 for matched in matched_entity_ids.values()
                                              if len(set(getattr(doc, ent_src).spans[epos][0] for epos in matched)) > 1])

        summary['named_entities_unmatched_per_doc'].append(len(unmatched_entities))
        summary['named_entities_unmatched_per_tag'] += [tupl[0] for tupl in unmatched_entities]
        summary['named_entities_matched_per_tag'] += [ent[0] for entities in matched_entities.values()
                                                      for ent in entities]
        summary['named_entities_matched_per_cluster'] += [len(v) for k, v in matched_entity_ids.items()]
        summary['clusters_unmatched_per_doc'].append(len(unmatched_clusters))
        summary['clusters_matched_per_doc'].append(len(matched_clusters))
        summary['clusters_matched_different_tags_per_doc'].append(clus_matched_diff_tags_per_doc)

        # Lets also calculate the length of coref and entity spans
        summary['length_coref_per_span'] += count_coref_span_length(doc)
        summary['length_ner_per_span'] += count_ner_span_length(doc, ent_src=ent_src)

        # # Let's look at some clusters to which no entities have been matched
        # if i % 10 == 0:
        #     for cluster_id in unmatched_clusters:
        #         print(doc.coref.clusters_[cluster_id])

    # Write this summary to disk using whatever name we chose to provide here.
    with (LOC.runs / 'ne_coref' / name).open('w+', encoding='utf8') as f:
        json.dump(summary, f)


if __name__ == "__main__":
    run()
