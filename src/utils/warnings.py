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
