"""
    Its a good idea to keep custom exceptions together.
"""
from pathlib import Path
from typing import Union

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

    def __init__(self, reason: str, *args):
        super().__init__(*args)
        self.reason = reason

    def __str__(self):
        return f"{self.reason}\n\t Currently Available Memory: {estimate_memory()} GBs."


class DatasetNotEncoded(FileNotFoundError):
    """ It seems this dataset was never encoded """

    def __init__(self, reason: Path, *args):
        super().__init__(*args)
        self.reason = reason

    def __str__(self):
        return f"The location: {self.reason} does not exist. We need to execute run.py."


class InstanceNotEncoded(FileNotFoundError):
    """ It seems that this particular instance was never encoded"""

    def __init__(self, loc: Path, hash: Union[int, float], *args):
        super().__init__(*args)
        self.loc = loc
        self.hash = hash

    def __str__(self):
        return f"The file: {self.hash} does not exist in {self.loc}. " \
               f"You probably ran the cacher with a sampling ratio or with trim."


class MismatchedConfig(Exception):
    """ """
    ...
