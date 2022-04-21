"""
    This file will contain all custom warnings and their explanations and nothing else
"""


class SpanHeadNotFoundError(Exception):
    """This is raised by the Document dataclass when an unforeseen spans' head is demanded"""
    ...


class NoValidAnnotations(Exception):
    """This exception is raised by Data Iters.
    Specifically if the given instance has no valid gold std annotations for the task."""
    ...


class LabelDictNotFound(FileNotFoundError):
    """Raised by the data iter.
    Specifically, if we don't find the label dict for a given task for a given dataset in data/manual"""
    ...
