"""
    Contains different NN modules that can be combined for the MTL task.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List
import transformers

# Local imports
try:
    import _pathfix
except ImportError:
    from . import _pathfix


class TextEncoder(nn.Module):
    """
        Thin wrapper around huggingface BERT which manages the fact that we get one document divided by length
            in a way that looks like a batch of inputs.
    """

    def __init__(self,
                 config: transformers.BertConfig):
        """
            :param config:  The config, apart from being a usual BertConfig, also contains some domain specific parts.
            For sanity's sake, I recommend using this snippet to init it:

                ```py
                config = transformers.BertConfig('bert-base-uncased')
                config.max_span_width = 5
                config.device = 'cpu'
                config.name = 'bert-base-uncased'
                ```
        """
        super().__init__()

        self.config = config
        self.n_max_len = self.config.max_position_embeddings
        self.h_dim = self.config.hidden_size

        # Encoder responsible for giving contextual vectors to subword tokens
        self.encoder = transformers.BertModel.from_pretrained(self.config.name).to(self.config.device)

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                token_type_ids: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        """
            Just run the tokenized, sparsely encoded sequence through a BERT model

            It takes (n, 512) tensors and returns a (n, 512, 768) summary (each word has a 768 dim vec).
            We reshape it back to (n*512, 768) dim vec.

            Using masked select, we remove the padding tokens from encoded (and corresponding input ids).

            Glossary for dimensions:
            **n_seq**: the number of segments in the document (on the batch position)
            **m_len**: the length of each segment (usually the same as encoder's max length)
            **h_dim**: the hidden dim of the model output. E.g. 768 for pretrained bert base.
            **n_swords**: the actual number of subwords that exist (less than n_seq * m_len)

            :param input_ids: [n_seq, m_len]; output of BertTokenizer
            :param attention_mask: [n_seq, m_len]; output of BertTokenizer
            :param token_type_ids: [n_seq, m_len]; output of BertTokenizer; not used but doesn't hurt to send it.
        """
        encoded = self.encoder(input_ids, attention_mask)[0]  # [n_seq, m_len, h_dim]
        encoded = encoded.reshape((-1, self.h_dim))  # [n_seq * m_len, h_dim]

        # Remove all the padded tokens, using info from attention masks
        encoded = torch.masked_select(encoded, attention_mask.bool().view(-1, 1)) \
            .view(-1, self.h_dim)  # [n_words, h_dim]
        input_ids = torch.masked_select(input_ids, attention_mask.bool()) \
            .view(-1, 1)  # [n_swords, h_dim]

        return encoded, input_ids


class SpanEncoder(nn.Module):
    """
        The module which takes BertEncoder outputs as input and candidate spans (should be generated during data iter)
            and creates representations for all of them.
    """

    def __init__(self,
                 config: transformers.BertConfig):
        """
        :param config: BertConfig which as more things added to it, namely:
            ```py
                config = transformers.BertConfig('bert-base-uncased')
                config.max_span_width = 5
                config.metadata_feature_size = 20
                config.use_span_width = True
                config.use_span_intra_attention = True
                config.ner_dropout = 0.3
            ```

        use_span_width: include an embedding repr. the width of a given span or not
        use_span_intra_attention: also include a weighted sum of different subwords inside the span or not
        max_span_width: used to init embeddings (if span width is being used)
        metadata_feature_size: the output dim of span width embeddings
        ner_dropout: the dropout prob. used in multiple places
            TODO: do we need different dropout ratio for different positions?
        """
        super().__init__()

        self.config = config
        self.use_span_width: bool = config.use_span_width
        self.use_span_intra_attention: bool = config.use_span_intra_attention

        if self.use_span_width:
            # Span width embedding give a fix dim score to width of spans
            self.span_width_embeddings = nn.Embedding(num_embeddings=self.config.max_span_width,
                                                      embedding_dim=config.metadata_feature_size)

        if self.use_span_intra_attention:
            # Used to push 768dim contextual vecs to 1D vectors for attention computation during span embedding creation
            self.span_attend_projection = torch.nn.Linear(config.hidden_size, 1)

    def forward(self,
                encoded: torch.Tensor,
                span_starts: torch.Tensor,
                span_ends: torch.Tensor
                ):
        """

        Span embeddings: based on candidate_starts, candidate_ends go through encoded
            and find start and end subword embeddings. Concatenate them.
            add embedding corresponding to the width of the span
            add an attention weighted sum of vectors within the span.

        Glossary of dimension symbols:
        n_swords: number of actual subwords in the document
        h_dim: encoded subword vectors' dim (e.g. 768)
        n_cand: number of candidates
        m_dim: meta features dim (not used anywhere else but this block)
        s_dim: span embeddings dimension. Depending on whether we use attention weighted sum of words in span,
            and whether we use span_width, this could be:
            h_dim * 2 (if no extra things are added)
                + m_dim (if using span width embeddings)
                + h_dim (if using attention weighted sum of sw vectors in span)

        :param encoded: [n_swords, h_dim], the reshaped output of BERT
        :param span_starts: [n_cand], the precomputed span start indices
        :param span_ends: [n_cand], the precomputed span end indices

        :return: a tensor [n_cand, s_dim] corresponding to each span as denoted in span_start and span_end indices.
        """
        emb = [encoded[span_starts], encoded[span_ends]]  # ([n_cand, h_dim], [n_cand, h_dim])

        if self.use_span_width:
            span_width = 1 + span_ends - span_starts  # [n_cand] (width is 1 -> config.max_span_width)
            span_width_index = span_width - 1  # [n_cand] (index is 0 -> config.max_span_width - 1)

            # Embed and dropout
            span_width_emb = self.span_width_embeddings(span_width_index)
            span_width_emb = F.dropout(span_width_emb, p=self.config.ner_dropout, training=self.training)

            # Add these to the emb list as well
            emb.append(span_width_emb)

        if self.use_span_intra_attention:
            document_range = torch.arange(start=0, end=encoded.shape[0], device=self.config.device).unsqueeze(0) \
                .repeat(span_starts.shape[0], 1)  # [n_cand, n_swords]
            token_mask = torch.logical_and(document_range >= span_starts.unsqueeze(1),
                                           document_range <= span_ends.unsqueeze(1))  # [n_cand, n_swords]
            token_attn = self.span_attend_projection(encoded).squeeze(1).unsqueeze(0)  # [1, n_swords]
            token_attn = F.softmax(torch.log(token_mask.float()) + token_attn, 1)  # [n_cand, n_swords]

            attended_word_representations = torch.mm(token_attn, encoded)  # [n_cand, h_dim]

            # Add these to the emb list as well
            emb.append(attended_word_representations)

        # Concat and return
        return torch.cat(emb, dim=1)  # [n_cand, s_dim]


class NERDecoder(nn.Module):
    """
        Takes span representations (as outputted by the span encoder),
            and simply runs them through a 2 layer clf to get a distribution over number of classes
    """

    def __init__(self, config: transformers.BertConfig):
        """
            :param config: BertConfig which as more things added to it, namely:
            ```py
                config = transformers.BertConfig('bert-base-uncased')
                config.max_span_width = 5
                config.metadata_feature_size = 20
                config.use_span_width = True
                config.use_span_intra_attention = True
                config.use_heuristic_decoding = True
            ```

        use_span_width: include an embedding repr. the width of a given span or not.
        use_span_intra_attention: also include a weighted sum of different subwords inside the span or not.
        max_span_width: used to init embeddings (if span width is being used).
        metadata_feature_size: the output dim of span width .
        heuristic_decoding: if spans are overlapping, only keep the ones with the maximal prediction score.
        """
        super().__init__()

        self.config = config

        # The span embedding dim can be inferred from the config.
        s_dim = 2 * self.config.hidden_size
        if self.config.use_span_width:
            s_dim += self.config.metadata_feature_size
        if self.config.use_span_intra_attention:
            s_dim += self.config.hidden_size

        self.use_heuristic_decoding: bool = self.config.use_heuristic_decoding

        self.clf = nn.Sequential(
            nn.Linear(s_dim, config.clf_h_dim),
            nn.ReLU(),
            nn.Dropout(config.ner_dropout),
            nn.Linear(config.clf_h_dim, config.ner_classes)
        )

    def heuristic_decoder(self,
                          span_emb: torch.Tensor,
                          span_starts: torch.Tensor,
                          span_ends: torch.Tensor):
        # TODO: dis
        raise NotImplementedError
        # return span_emb

    def forward(self,
                span_emb: torch.Tensor,
                span_starts: torch.Tensor,
                span_ends: torch.Tensor):
        """
            As simple as forwards gets.


            :param span_emb: a [n_cand, s_dim] matrix
            :param span_starts: [n_cand], the precomputed span start indices
            :param span_ends: [n_cand], the precomputed span end indices
            :return: a [n_cand, ner_classes] matrix
        """
        predictions = self.clf(span_emb)

        if self.use_heuristic_decoding:
            return self.heuristic_decoder(predictions, span_starts, span_ends)
