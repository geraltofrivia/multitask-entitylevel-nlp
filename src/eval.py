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
    return torch.mean((torch.argmax(logits, dim=1) == labels).float()).item()


def ner_only_annotated(logits, labels):
    """
        Only care about the accuracy of spans which are actually annotated in text.
    :param logits: n_spans, n_classes
    :param labels: n_spans
    :return: scalar
    """
    return torch.mean(
        (torch.argmax(logits[labels != 0], dim=1) == labels[labels != 0]).float()
    )
