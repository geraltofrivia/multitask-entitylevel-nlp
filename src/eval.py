"""
    Collection of evaluation functions for different tasks.
"""

import torch
import numpy as np
from collections import Counter
from typing import Dict, Callable
from scipy.optimize import linear_sum_assignment as linear_assignment


def compute_metrics(
        metrics: Dict[str, Callable],
        logits=None,
        labels=None,
        clusters=None,
        gold_clusters=None,
        mention_to_predicted=None,
        mention_to_gold=None
) -> Dict[str, float]:
    return {metric_nm: metric_fn(
        logits=logits,
        labels=labels,
        clusters=clusters,
        gold_clusters=gold_clusters,
        mention_to_predicted=mention_to_predicted,
        mention_to_gold=mention_to_gold
    ).cpu().detach().item()
            for metric_nm, metric_fn in metrics.items()}


# noinspection PyUnusedLocal
def ner_all(logits, labels, *args, **kwargs):
    """
        Does not distinguish b/w invalid spans, and actually annotated spans.
    :param logits: n_spans, n_classes
    :param labels: n_spans
    :return: scalar
    """
    return torch.mean((torch.argmax(logits, dim=1) == labels).float())


# noinspection PyUnusedLocal
def pruner_p(logits, labels, *args, **kwargs):
    """
    :param logits: n_spans
    :param labels: n_spans
    :return: scalar
    """
    return torch.sum((logits > 0).to(float) * (labels > 0).to(float)) / torch.sum((labels > 0).to(float))


# noinspection PyUnusedLocal
def pruner_r(logits, labels, *args, **kwargs):
    """
    :param logits: n_spans
    :param labels: n_spans
    :return: scalar
    """
    return torch.sum((logits > 0).to(float) * (labels > 0).to(float)) / torch.sum((logits > 0).to(float))


# noinspection PyUnusedLocal
def ner_only_annotated(logits, labels, *args, **kwargs):
    """
        Only care about the accuracy of spans which are actually annotated in text.
    :param logits: n_spans, n_classes
    :param labels: n_spans
    :return: scalar
    """
    op = torch.mean(
        (torch.argmax(logits[labels != 0], dim=1) == labels[labels != 0]).float()
    )
    return op


# noinspection PyUnusedLocal
def ner_span_recog_precision(logits: torch.Tensor, labels: torch.Tensor, *args, **kwargs):
    """
        Treat as binary clf. And find proportion of spans which were correctly recognized as being spans
        (regardless of the label).
    """
    _logits = torch.argmax(logits, dim=1)  # n_spans, 1
    return torch.sum((_logits > 0).to(float) * (labels > 0).to(float)) / torch.sum((labels > 0).to(float))


# noinspection PyUnusedLocal
def ner_span_recog_recall(logits: torch.Tensor, labels: torch.Tensor, *args, **kwargs):
    """
        Treat as binary clf. And find proportion of spans which were correctly recognized as being spans
         (regardless of the label).
    """
    _logits = torch.argmax(logits, dim=1)  # n_spans, 1
    return torch.sum((_logits > 0).to(float) * (labels > 0).to(float)) / torch.sum((_logits > 0).to(float))


# noinspection PyUnusedLocal
def coref_b_cubed(clusters, mention_to_gold, *args, **kwargs):
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
def coref_muc(clusters, mention_to_gold, *args, **kwargs):
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
def coref_phi4(clusters, gold_clusters, *args, **kwargs):
    return 2 * len([m for m in clusters if m in gold_clusters]) / float(len(clusters) + len(gold_clusters))


# noinspection PyUnusedLocal
def coref_ceafe(clusters, gold_clusters, *args, **kwargs):
    clusters = [c for c in clusters if len(c) != 1]
    scores = np.zeros((len(gold_clusters), len(clusters)))
    for i in range(len(gold_clusters)):
        for j in range(len(clusters)):
            scores[i, j] = coref_phi4(gold_clusters=gold_clusters[i], clusters=clusters[j])
    matching = linear_assignment(-scores)
    similarity = sum(scores[matching[0], matching[1]])

    # similarity = sum(scores[matching[:, 0], matching[:, 1]])
    return similarity, len(clusters), similarity, len(gold_clusters)
