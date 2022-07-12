"""
    Its a good idea to keep custom exceptions together.
"""
from typing import Union, Optional

import torch
from mytorch.utils.goodies import estimate_memory


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


class UnknownDomainException(ValueError):
    """ Raised when a given domain is unknown. """
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


class NANsFound(ValueError):
    """ There are nans where we didn't expect them (which is everywhere, pretty much). HALP! """
    ...


class AnticipateOutOfMemException(Exception):
    """ A model may throw this if it seems to be going out of memory. If not, well, bien. """

    def __init__(self, reason: str, device: Optional[Union[str, torch.device]], *args):
        super().__init__(*args)
        self.reason = reason

    def __str__(self):
        return f"{self.reason}\n\t Currently Available Memory: {estimate_memory()} GBs."
