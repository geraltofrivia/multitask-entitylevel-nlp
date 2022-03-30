"""
    Collection of evaluation functions for different tasks.
"""

import torch


def ner_all(logits, labels):
    """
        Does not distinguish b/w invalid spans, and actually annotated spans.
    :param logits: n_spans, n_classes
    :param labels: n_spans
    :return: scalar
    """
    return torch.mean((torch.argmax(logits, dim=1) == labels).float())


def pruner_p(logits, labels):
    """
    :param logits: n_spans
    :param labels: n_spans
    :return: scalar
    """
    return torch.sum((logits > 0).to(float) * (labels > 0).to(float)) / torch.sum((labels > 0).to(float))


def pruner_r(logits, labels):
    """
    :param logits: n_spans
    :param labels: n_spans
    :return: scalar
    """
    return torch.sum((logits > 0).to(float) * (labels > 0).to(float)) / torch.sum((logits > 0).to(float))


def ner_only_annotated(logits, labels):
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


def ner_span_recog_precision(logits: torch.Tensor, labels: torch.Tensor):
    """
        Treat as binary clf. And find proportion of spans which were correctly recognized as being spans (regardless of the label).
    """
    _logits = torch.argmax(logits, dim=1)  # n_spans, 1
    return torch.sum((_logits > 0).to(float) * (labels > 0).to(float)) / torch.sum((labels > 0).to(float))


def ner_span_recog_recall(logits: torch.Tensor, labels: torch.Tensor):
    """
        Treat as binary clf. And find proportion of spans which were correctly recognized as being spans (regardless of the label).
    """
    _logits = torch.argmax(logits, dim=1)  # n_spans, 1
    return torch.sum((_logits > 0).to(float) * (labels > 0).to(float)) / torch.sum((_logits > 0).to(float))
