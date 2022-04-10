"""
    Collection of evaluation functions for different tasks.
"""

import torch
import numpy as np
from collections import Counter
from typing import Dict, Callable
from scipy.optimize import linear_sum_assignment as linear_assignment

"""
    Make a overall (macro) eval system. 
    Configure it like you configure the training loop and throw it in the loop. Do not give it the model.
    The model is fed every time the loop is run. 
    
    All the metrics below, all helper functions are situated strategically.
    
    At every point it can return a dict,
        or append to a dict
        or summarise a dict ( across epochs ) 
    
    and has a nice _repr_ function. This is your goal for 09/04/2022!
    
    In a way that can return everything that needs to be returned. At any point of time.
    
"""


def compute_metrics(
        metrics: Dict[str, Callable],
        logits=None,
        labels=None,
        clusters=None,
        gold_clusters=None,
        mention_to_predicted=None,
        mention_to_gold=None
) -> Dict[str, float]:
    results = {}
    for metric_nm, metric_fn in metrics.items():
        outputs = metric_fn(
            logits=logits,
            labels=labels,
            clusters=clusters,
            gold_clusters=gold_clusters,
            mention_to_predicted=mention_to_predicted,
            mention_to_gold=mention_to_gold
        )
        for metric_suffix, metric_val in outputs.items():
            results[metric_nm + '_' + metric_suffix] = metric_val

        return results

    #
    # return {metric_nm: metric_fn(
    #     logits=logits,
    #     labels=labels,
    #     clusters=clusters,
    #     gold_clusters=gold_clusters,
    #     mention_to_predicted=mention_to_predicted,
    #     mention_to_gold=mention_to_gold
    # ).cpu().detach().item()
    #         for metric_nm, metric_fn in metrics.items()}


# noinspection PyUnusedLocal
def ner_acc(logits, labels, *args, **kwargs):
    """
        Does not distinguish b/w invalid spans, and actually annotated spans.
    :param logits: n_spans, n_classes
    :param labels: n_spans
    :return: scalar
    """
    return {
        'acc_all': torch.mean((torch.argmax(logits, dim=1) == labels).float()),
        'acc_only_annotated': torch.mean((torch.argmax(logits[labels != 0], dim=1) == labels[labels != 0]).float())
    }


# noinspection PyUnusedLocal
def pruner_pr(logits, labels, *args, **kwargs):
    """
    :param logits: n_spans
    :param labels: n_spans
    :return: scalar
    """
    p = torch.sum((logits > 0).to(float) * (labels > 0).to(float)) / torch.sum((labels > 0).to(float))
    r = torch.sum((logits > 0).to(float) * (labels > 0).to(float)) / torch.sum((logits > 0).to(float))
    # TODO: add f1
    return {'p': p, 'r': r}


# noinspection PyUnusedLocal
def ner_span_recog_pr(logits: torch.Tensor, labels: torch.Tensor, *args, **kwargs):
    """
        Treat as binary clf. And find proportion of spans which were correctly recognized as being spans
        (regardless of the label).
    """
    _logits = torch.argmax(logits, dim=1)  # n_spans, 1
    p = torch.sum((_logits > 0).to(float) * (labels > 0).to(float)) / torch.sum((labels > 0).to(float))
    r = torch.sum((_logits > 0).to(float) * (labels > 0).to(float)) / torch.sum((_logits > 0).to(float))
    return {'p': p, 'r': r}


# # noinspection PyUnusedLocal
# def coref_b_cubed_prf(clusters, mention_to_gold, *args, **kwargs):
#
#     evaluators = _coref_b_cubed_(clusters=clusters, mention_to_gold=mention_to_gold)


def _coref_b_cubed_(clusters, mention_to_gold):
    num, dem = 0, 0

    for c in clusters:
        if len(c) == 1:
            continue

        gold_counts = Counter()
        correct = 0
        for m in c:
            if m in mention_to_gold:
                gold_counts[tuple(mention_to_gold[m])] += 1
        for c2, count in gold_counts.items():
            if len(c2) != 1:
                correct += count * count

        num += correct / float(len(c))
        dem += len(c)

    return num, dem


# noinspection PyUnusedLocal
def _coref_muc_(clusters, mention_to_gold, *args, **kwargs):
    tp, p = 0, 0
    for c in clusters:
        p += len(c) - 1
        tp += len(c)
        linked = set()
        for m in c:
            if m in mention_to_gold:
                linked.add(mention_to_gold[m])
            else:
                tp -= 1
        tp -= len(linked)
    return tp, p


# noinspection PyUnusedLocal
def _coref_phi4_(clusters, gold_clusters, *args, **kwargs):
    return 2 * len([m for m in clusters if m in gold_clusters]) / float(len(clusters) + len(gold_clusters))


# noinspection PyUnusedLocal
def _coref_ceafe_(clusters, gold_clusters, *args, **kwargs):
    clusters = [c for c in clusters if len(c) != 1]
    scores = np.zeros((len(gold_clusters), len(clusters)))
    for i in range(len(gold_clusters)):
        for j in range(len(clusters)):
            scores[i, j] = _coref_phi4_(gold_clusters=gold_clusters[i], clusters=clusters[j])
    matching = linear_assignment(-scores)
    similarity = sum(scores[matching[0], matching[1]])

    # similarity = sum(scores[matching[:, 0], matching[:, 1]])
    return similarity, len(clusters), similarity, len(gold_clusters)
