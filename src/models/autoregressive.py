"""
    This file contains some autoregressive (recurrent) models for text encoding.
    Creative applications would have to be underway to use them in a compositional manner.
"""
import torch
import torch.nn as nn
import transformers


class LSTMEncoder(nn.Module):
    ...


class BertBase(transformers.BertPreTrainedModel):
    """
    Encapsulates a regular BERT encoder.
    It takes word/subword IDs (done in a huggingface fashion), and encodes them.

    IMPORTANT: for documents longer than max len (usually 512), in terms of subword tokens,
        we expect the doc to be broken down from (1, sl) to (num_subseq, max_seq_len).
    """

    def __init__(self, transformers_config):
        super().__init__(transformers_config)
        self.model = transformers.BertModel(
            transformers_config, add_pooling_layer=False
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
            Simple encapsulation of Huggingface Transformers' BERTModel.
        :param input_ids: tensor (num_subseq, max_seq_len)
        :param attention_mask: (num_subseq, max_seq_len)
        :return:
        """
        return self.model(input_ids, attention_mask)
