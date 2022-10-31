"""
    File containing span pruner(s) for regular coref stuff
    and span predictor for word level stuff
"""

from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Local imports
try:
    import _pathfix
except ImportError:
    from . import _pathfix
from models.modules import Utils


class SpanPrunerHOI(torch.nn.Module):
    def __init__(
            self,
            hidden_size: int,
            unary_hdim: int,
            max_span_width: int,
            coref_metadata_feature_size: int,
            pruner_dropout: float,
            pruner_use_metadata: bool,
            pruner_max_num_spans: int,
            bias_in_last_layers: bool,
            pruner_top_span_ratio: float,
    ):
        super().__init__()

        # Some constants
        self._dropout: float = pruner_dropout
        self._use_metadata: bool = pruner_use_metadata
        self._max_num_spans: int = pruner_max_num_spans
        self._top_span_ratio: float = pruner_top_span_ratio
        self.dropout = nn.Dropout(p=self._dropout)

        # Parameter Time!
        _feat_dim = coref_metadata_feature_size
        _span_dim = (hidden_size * 3) + (_feat_dim if self._use_metadata else 0)

        self.span_emb_score_ffnn = Utils.make_ffnn(_span_dim, [unary_hdim], 1, self.dropout,
                                                   bias_in_last_layers=bias_in_last_layers)

        if self._use_metadata:
            self.span_width_score_ffnn = Utils.make_ffnn(_feat_dim, [unary_hdim], 1, self.dropout)
            self.emb_span_width_prior = Utils.make_embedding(max_span_width, coref_metadata_feature_size)

    def forward(self,
                candidate_span_emb: torch.tensor,  # [num_cand, new emb size]
                candidate_width_idx: torch.tensor,  # [num_cand, 1]
                candidate_starts: torch.tensor,  # [num_cand, ]
                candidate_ends: torch.tensor,  # [num_cand, ]
                speaker_ids: torch.tensor,  # [num_swords, 1]
                device: Union[str, torch.device],
                num_words: int,
                ):

        # Get span score
        candidate_mention_scores = torch.squeeze(self.span_emb_score_ffnn(candidate_span_emb), 1)
        if self._use_metadata:
            width_score = torch.squeeze(self.span_width_score_ffnn(self.emb_span_width_prior.weight), 1)
            # noinspection PyUnboundLocalVariable
            candidate_width_score = width_score[candidate_width_idx]
            candidate_mention_scores += candidate_width_score

        # Extract top spans
        candidate_idx_sorted_by_score = torch.argsort(candidate_mention_scores, descending=True).tolist()
        candidate_starts_cpu, candidate_ends_cpu = candidate_starts.tolist(), candidate_ends.tolist()
        num_top_spans = int(min(self._max_num_spans, num_words * self._top_span_ratio))
        selected_idx_cpu = Utils.extract_top_spans_hoi(candidate_idx_sorted_by_score, candidate_starts_cpu,
                                                       candidate_ends_cpu, num_top_spans)
        assert len(selected_idx_cpu) == num_top_spans
        selected_idx = torch.tensor(selected_idx_cpu, device=device)
        pruned_span_starts, pruned_span_ends = candidate_starts[selected_idx], candidate_ends[selected_idx]
        pruned_span_emb = candidate_span_emb[selected_idx]
        # top_span_cluster_ids = candidate_labels[selected_idx] if do_loss else None  #TODO: move this line
        pruned_span_scores = candidate_mention_scores[selected_idx]
        pruned_span_speaker_ids = speaker_ids[pruned_span_starts]

        return {
            'pruned_span_indices': selected_idx,  # HOI calls it 'selected_idx'
            'pruned_span_starts': pruned_span_starts,
            'pruned_span_ends': pruned_span_ends,
            'pruned_span_emb': pruned_span_emb,
            'pruned_span_scores': pruned_span_scores,
            'pruned_span_speaker_ids': pruned_span_speaker_ids,
            'num_top_spans': num_top_spans
        }


class SpanPrunerMangoes(torch.nn.Module):
    """
        Give me your candidate starts, your candidate ends
        Give me your transformer encoded tokens
        Give me your constants like max span width
        And I shall give you pruned span embeddings and corresponding maps.
    """

    def __init__(
            self,
            hidden_size: int,
            unary_hdim: int,
            max_span_width: int,
            coref_metadata_feature_size: int,
            pruner_dropout: float,
            pruner_use_metadata: bool,
            pruner_max_num_spans: int,
            bias_in_last_layers: bool,
            pruner_top_span_ratio: float,
    ):
        super().__init__()

        # Some constants
        self._dropout: float = pruner_dropout
        self._use_metadata: bool = pruner_use_metadata
        self._max_num_spans: int = pruner_max_num_spans
        self._top_span_ratio: float = pruner_top_span_ratio

        # Parameter Time!
        _feat_dim = coref_metadata_feature_size
        _span_dim = (hidden_size * 3) + (_feat_dim if self._use_metadata else 0)

        self.span_attend_projection = nn.Linear(hidden_size, 1)
        self.span_scorer = nn.Sequential(
            nn.Linear(_span_dim, unary_hdim),
            nn.ReLU(),
            nn.Dropout(self._dropout),
            nn.Linear(unary_hdim, 1, bias=bias_in_last_layers),
        )

        if self._use_metadata:
            self.span_width_scorer = Utils.make_ffnn(_feat_dim, [unary_hdim], 1, self._dropout)
            self.emb_span_width = Utils.make_embedding(max_span_width, coref_metadata_feature_size, )
            self.emb_span_width_prior = Utils.make_embedding(max_span_width, coref_metadata_feature_size)

    def get_span_word_attention_scores(self, hidden_states, span_starts, span_ends):
        """

        Parameters
        ----------
        hidden_states: tensor of size (num_tokens, emb_size)
            outputs of BERT model, reshaped
        span_starts, span_ends: tensor of size (num_candidates)
            indices of starts and ends of spans

        Returns
        -------
        tensor of size (num_candidates, span_embedding_size)
        """
        document_range = torch.arange(start=0, end=hidden_states.shape[0], device=hidden_states.device).unsqueeze(
            0).repeat(span_starts.shape[0], 1)  # [num_cand, num_words]
        # noinspection PyTypeChecker
        token_mask = torch.logical_and(document_range >= span_starts.unsqueeze(1),
                                       document_range <= span_ends.unsqueeze(1))  # [num_cand, num_words]
        token_atten = self.span_attend_projection(hidden_states).squeeze(1).unsqueeze(0)  # [1, num_words]
        token_attn = F.softmax(torch.log(token_mask.float()) + token_atten, 1)  # [num_cand, num_words]span
        return token_attn

    def get_span_embeddings(
            self,
            hidden_states: torch.Tensor,  # [num_swords, bert_emb_size] (2000, 732 e.g.)
            span_starts: torch.Tensor,  # [num_cand, ]
            span_ends: torch.Tensor  # [num_cand, ]
    ):
        """
        Obtains representations of the spans

        Parameters
        ----------
        hidden_states: tensor of size (num_tokens, bert_emb_size)
            outputs of BERT model, reshaped
        span_starts, span_ends: tensor of size (num_cand, )
            indices of starts and ends of spans

        Returns
        -------
        tensor of size (num_cand, span_embedding_size)
        """
        emb = [hidden_states[span_starts], hidden_states[span_ends]]

        if self._use_metadata:
            # Calculate span width embeddings
            span_width = 1 + span_ends - span_starts  # [num_cand]
            span_width_index = span_width - 1  # [num_cand]
            span_width_emb = self.emb_span_width(span_width_index)  # [num_cand, emb_size]
            span_width_emb = F.dropout(span_width_emb, p=self._dropout, training=self.training)

            # Append to Emb
            emb.append(span_width_emb)

        # Calculate attention weighted summary of different tokens
        token_attention_scores = self.get_span_word_attention_scores(hidden_states, span_starts,
                                                                     span_ends)  # [num_cand, num_words]
        attended_word_representations = torch.mm(token_attention_scores, hidden_states)  # [num_cand, emb_size]
        emb.append(attended_word_representations)
        return torch.cat(emb, dim=1)

    def forward(
            self,
            hidden_states: torch.tensor,  # [num_swords, bert_emb_size]
            candidate_starts: torch.tensor,  # [num_cand, ]
            candidate_ends: torch.tensor,  # [num_cand, ]
            speaker_ids: torch.tensor,  # [num_swords, 1]
            device: Union[str, torch.device]

    ):
        _num_words: int = hidden_states.shape[0]
        span_emb = self.get_span_embeddings(hidden_states, candidate_starts, candidate_ends)  # [num_cand, emb_size]
        span_scores = self.span_scorer(span_emb).squeeze(1)  # [num_cand,]

        if self._use_metadata:
            # Get span with scores (using embeddings with priors), and add them to candidate scores
            span_width_indices = candidate_ends - candidate_starts
            span_width_emb = self.emb_span_width_prior(span_width_indices)  # [num_cand, meta]
            span_width_scores = self.span_width_scorer(span_width_emb).squeeze(1)  # [num_cand, ]
            span_scores += span_width_scores  # [num_cand, ]

        # Get beam size (its a function of top span ratio, and length of document, capped by a threshold
        # noinspection PyTypeChecker
        num_top_spans = int(min(self._max_num_spans, _num_words * self._top_span_ratio))

        # Get top mention scores and sort by span order

        pruned_span_indices = Utils.extract_spans(candidate_starts, candidate_ends, span_scores, num_top_spans)
        pruned_span_starts = candidate_starts[pruned_span_indices]
        pruned_span_ends = candidate_ends[pruned_span_indices]
        pruned_span_emb = span_emb[pruned_span_indices]
        pruned_span_scores = span_scores[pruned_span_indices]

        if speaker_ids is not None:
            pruned_span_speaker_ids = speaker_ids[pruned_span_starts]
        else:
            pruned_span_speaker_ids = None

        return {
            'span_emb': span_emb,
            'pruned_span_indices': pruned_span_indices,  # HOI calls it 'selected_idx'
            'pruned_span_starts': pruned_span_starts,
            'pruned_span_ends': pruned_span_ends,
            'pruned_span_emb': pruned_span_emb,
            'pruned_span_scores': pruned_span_scores,
            'pruned_span_speaker_ids': pruned_span_speaker_ids,
            'num_top_spans': num_top_spans
        }


class SpanPredictor(torch.nn.Module):
    def __init__(self, input_size: int, distance_emb_size: int):
        super().__init__()
        self.ffnn = torch.nn.Sequential(
            torch.nn.Linear(input_size * 2 + 64, input_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(input_size, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 64),
        )
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(64, 4, 3, 1, 1),
            torch.nn.Conv1d(4, 2, 3, 1, 1)
        )
        self.emb = torch.nn.Embedding(128, distance_emb_size)  # [-63, 63] + too_far
        self._loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")

    @property
    def device(self) -> torch.device:
        """ A workaround to get current device (which is assumed to be the
        device of the first parameter of one of the submodules) """
        return next(self.ffnn.parameters()).device

    def forward(self,  # type: ignore  # pylint: disable=arguments-differ  #35566 in pytorch
                sentence_map: torch.Tensor,  #: Doc,
                words: torch.Tensor,
                heads_ids: torch.Tensor) -> torch.Tensor:
        """
        Calculates span start/end scores of words for each span head in
        heads_ids

        Args:
            doc (Doc): the document data
            words (torch.Tensor): contextual embeddings for each word in the
                document, [n_words, emb_size]
            heads_ids (torch.Tensor): word indices of span heads

        Returns:
            torch.Tensor: span start/end scores, [n_heads, n_words, 2]
        """
        # Obtain distance embedding indices, [n_heads, n_words]
        relative_positions = (heads_ids.unsqueeze(1) - torch.arange(words.shape[0], device=words.device).unsqueeze(0))
        emb_ids = relative_positions + 63  # make all valid distances positive
        emb_ids[(emb_ids < 0) + (emb_ids > 126)] = 127  # "too_far"

        # Obtain "same sentence" boolean mask, [n_heads, n_words]
        sent_id = torch.tensor(sentence_map, device=words.device)
        same_sent = (sent_id[heads_ids].unsqueeze(1) == sent_id.unsqueeze(0))

        # To save memory, only pass candidates from one sentence for each head
        # pair_matrix contains concatenated span_head_emb + candidate_emb + distance_emb
        # for each candidate among the words in the same sentence as span_head
        # [n_heads, input_size * 2 + distance_emb_size]
        rows, cols = same_sent.nonzero(as_tuple=True)
        pair_matrix = torch.cat((
            words[heads_ids[rows]],
            words[cols],
            self.emb(emb_ids[rows, cols]),
        ), dim=1)

        lengths = same_sent.sum(dim=1)
        padding_mask = torch.arange(0, lengths.max(), device=words.device).unsqueeze(0)
        padding_mask = (padding_mask < lengths.unsqueeze(1))  # [n_heads, max_sent_len]

        # [n_heads, max_sent_len, input_size * 2 + distance_emb_size]
        # This is necessary to allow the convolution layer to look at several
        # word scores
        padded_pairs = torch.zeros(*padding_mask.shape, pair_matrix.shape[-1], device=words.device)
        padded_pairs[padding_mask] = pair_matrix

        res = self.ffnn(padded_pairs)  # [n_heads, n_candidates, last_layer_output]
        res = self.conv(res.permute(0, 2, 1)).permute(0, 2, 1)  # [n_heads, n_candidates, 2]

        scores = torch.full((heads_ids.shape[0], words.shape[0], 2), float('-inf'), device=words.device)
        scores[rows, cols] = res[padding_mask]

        # Make sure that start <= head <= end during inference
        if not self.training:
            valid_starts = torch.log((relative_positions >= 0).to(torch.float))
            valid_ends = torch.log((relative_positions <= 0).to(torch.float))
            valid_positions = torch.stack((valid_starts, valid_ends), dim=2)
            return scores + valid_positions
        return scores

    def predict(self,
                doc,  #: Doc,
                words: torch.Tensor,
                clusters: List[List[int]]) -> List[list]:
        raise NotImplementedError
        """
        Predicts span clusters based on the word clusters.

        Args:
            doc (Doc): the document data
            words (torch.Tensor): [n_words, emb_size] matrix containing
                embeddings for each of the words in the text
            clusters (List[List[int]]): a list of clusters where each cluster
                is a list of word indices

        Returns:
            List[List[Span]]: span clusters
        """
        if not clusters:
            return []

        heads_ids = torch.tensor(
            sorted(i for cluster in clusters for i in cluster),
            device=self.device
        )

        scores = self(doc, words, heads_ids)
        starts = scores[:, :, 0].argmax(dim=1).tolist()
        ends = (scores[:, :, 1].argmax(dim=1) + 1).tolist()

        head2span = {
            head: (start, end)
            for head, start, end in zip(heads_ids.tolist(), starts, ends)
        }

        return [[head2span[head] for head in cluster]
                for cluster in clusters]
