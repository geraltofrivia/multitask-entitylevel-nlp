"""
    Its a good idea to keep custom exceptions together.
"""


class BadParameters(Exception):
    """
    This exception is supposed to indicate that the cli params passed did not meet the standards for some reason.
    Not particularly helpful in figuring out why on its own. Look closely at the error message generated.
    """

    pass


class ImproperDumpDir(Exception):
    """
        A broad family invoked when some items are missing/improperly dumped in a dir.
    """
    pass


class UnknownTaskException(ValueError):
    """ Raised when a given task in a set of tasks is unknown. """
    pass


class UnknownDataSplitException(ValueError):
    """ Raised when a given data split is not recognized """
    ...


class SpanHeadNotFoundError(Exception):
    """This is raised by the Document dataclass when an unforeseen spans' head is demanded"""
    ...


class NoValidAnnotations(Exception):
    """This exception is raised by Data Iters.
    Specifically if the given instance has no valid gold std annotations for the task."""
    ...


class LabelDictNotFound(FileNotFoundError):
    """Raised by Data Iters.
    Specifically, if we don't find the label dict for a given task for a given dataset in data/manual"""
    ...
