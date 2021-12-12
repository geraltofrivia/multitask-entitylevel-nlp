"""
    This file contains some autoregressive (recurrent) models for text encoding.
    Creative applications would have to be underway to use them in a compositional manner.
"""
import torch.nn as nn


class LSTMEncoder(nn.Module):
    ...