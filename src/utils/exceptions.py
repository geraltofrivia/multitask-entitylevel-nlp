"""
    Its a good idea to keep custom exceptions together.
"""


class BadParameters(Exception):
    """
    This exception is supposed to indicate that the cli params passed did not meet the standards for some reason.
    Not particularly helpful in figuring out why on its own. Look closely at the error message generated.
    """

    pass
