"""
Contains an NN Module which is responsible for, given a set of vector repr of text,
    iterate through it and classify spans.

Decisions:

1. We ignore sentence boundaries for now, and just assume that periods (sent. boundaries) are also just tokens.
"""
