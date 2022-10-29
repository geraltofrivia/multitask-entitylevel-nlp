"""
    Contains different NN modules that can be combined for the MTL task.
    Namely:

        - NOT A module for TransformerEncoder (which takes chunked text and combines them internally
            to create vectors for tokens)
        - A coref decoder module
        - A NER decoder module
        - A pruner? decoder module (or not)

    Everyone of these should have a forward function.
    But inference and pred with labels need to be outside.
"""
import math
from collections import Iterable
from typing import Union, List, Optional

import torch
import torch.nn as nn
import torch.nn.init as init

# Local imports
try:
    import _pathfix
except ImportError:
    from . import _pathfix
from utils.exceptions import BadParameters


class Utils(object):
    """
        Contain some elements that are used across different modules
    """

    @staticmethod
    def make_embedding(num_embeddings: int, embedding_dim: int, std: float = 0.02) -> torch.nn.Module:
        emb = nn.Embedding(num_embeddings, embedding_dim)
        init.normal_(emb.weight, std=std)
        return emb

    @staticmethod
    def make_ffnn(input_dim: int, hidden_dim: Optional[Union[int, List[int]]],
                  output_dim: int, dropout: Union[float, nn.Module],
                  bias_in_last_layers: bool = True, nonlin: str = 'relu', ):
        if nonlin.lower().strip() == 'relu':
            nonlin_fn = nn.ReLU
        elif nonlin.lower().strip() in ['leakyrelu', 'leaky_relu']:
            nonlin_fn = nn.LeakyReLU
        else:
            raise BadParameters(f'Unknown Non Linearity fn: `{nonlin}`.')

        if hidden_dim is None or hidden_dim == 0 or hidden_dim == [] or hidden_dim == [0]:
            return Utils.make_linear(input_dim, output_dim)

        if not isinstance(hidden_dim, Iterable):
            hidden_dim = [hidden_dim]

        if not isinstance(dropout, nn.Module):
            dropout = nn.Dropout(p=dropout)

        ffnn = [Utils.make_linear(input_dim, hidden_dim[0]), nonlin_fn(), dropout]
        for i in range(1, len(hidden_dim)):
            ffnn += [Utils.make_linear(hidden_dim[i - 1], hidden_dim[i]), nonlin_fn(), dropout]
        ffnn.append(Utils.make_linear(hidden_dim[-1], output_dim, bias=bias_in_last_layers))
        return nn.Sequential(*ffnn)

    @staticmethod
    def make_linear(in_features, out_features, bias=True, std=0.02):
        linear = nn.Linear(in_features, out_features, bias)
        init.normal_(linear.weight, std=std)
        if bias:
            init.zeros_(linear.bias)
        return linear

    @staticmethod
    def extract_top_spans_hoi(candidate_idx_sorted, candidate_starts, candidate_ends, num_top_spans):
        """ Keep top non-cross-overlapping candidates ordered by scores; compute on CPU because of loop """
        selected_candidate_idx = []
        start_to_max_end, end_to_min_start = {}, {}
        for candidate_idx in candidate_idx_sorted:
            if len(selected_candidate_idx) >= num_top_spans:
                break
            # Perform overlapping check
            span_start_idx = candidate_starts[candidate_idx]
            span_end_idx = candidate_ends[candidate_idx]
            cross_overlap = False
            for token_idx in range(span_start_idx, span_end_idx + 1):
                max_end = start_to_max_end.get(token_idx, -1)
                if token_idx > span_start_idx and max_end > span_end_idx:
                    cross_overlap = True
                    break
                min_start = end_to_min_start.get(token_idx, -1)
                if token_idx < span_end_idx and 0 <= min_start < span_start_idx:
                    cross_overlap = True
                    break
            if not cross_overlap:
                # Pass check; select idx and update dict stats
                selected_candidate_idx.append(candidate_idx)
                max_end = start_to_max_end.get(span_start_idx, -1)
                if span_end_idx > max_end:
                    start_to_max_end[span_start_idx] = span_end_idx
                min_start = end_to_min_start.get(span_end_idx, -1)
                if min_start == -1 or span_start_idx < min_start:
                    end_to_min_start[span_end_idx] = span_start_idx
        # Sort selected candidates by span idx
        selected_candidate_idx = sorted(selected_candidate_idx,
                                        key=lambda idx: (candidate_starts[idx], candidate_ends[idx]))
        if len(selected_candidate_idx) < num_top_spans:  # Padding
            selected_candidate_idx += ([selected_candidate_idx[0]] * (num_top_spans - len(selected_candidate_idx)))
        return selected_candidate_idx

    @staticmethod
    def extract_spans_old(candidate_starts, candidate_ends, candidate_mention_scores, num_top_spans):
        """
        Extracts the candidate spans with the highest mention scores, who's spans don't cross over other spans.

        Parameters:
        ----------
            candidate_starts: tensor of size (candidates)
                Indices of the starts of spans for each candidate.
            candidate_ends: tensor of size (candidates)
                Indices of the ends of spans for each candidate.
            candidate_mention_scores: tensor of size (candidates)
                Mention score for each candidate.
            num_top_spans: int
                Number of candidates to extract
        Returns:
        --------
            top_span_indices: tensor of size (num_top_spans)
                Span indices of the non-crossing spans with the highest mention scores
        """
        # sort based on mention scores
        top_span_indices = torch.argsort(candidate_mention_scores, descending=True)
        # add highest scores that don't cross
        end_to_earliest_start = {}
        start_to_latest_end = {}
        selected_spans = []
        current_span_index = 0
        while len(selected_spans) < num_top_spans and current_span_index < candidate_starts.size(0):
            ind = top_span_indices[current_span_index]
            any_crossing = False
            cand_start = candidate_starts[ind].item()
            cand_end = candidate_ends[ind].item()
            for j in range(cand_start, cand_end + 1):
                if j > cand_start and j in start_to_latest_end and start_to_latest_end[j] > cand_end:
                    any_crossing = True
                    break
                if j < cand_end and j in end_to_earliest_start and end_to_earliest_start[j] < cand_start:
                    any_crossing = True
                    break
            if not any_crossing:
                selected_spans.append(ind)
                if cand_start not in start_to_latest_end or start_to_latest_end[cand_start] < cand_end:
                    start_to_latest_end[cand_start] = cand_end
                if cand_end not in end_to_earliest_start or end_to_earliest_start[cand_end] > cand_start:
                    end_to_earliest_start[cand_end] = cand_start
            current_span_index += 1
        return torch.tensor(sorted(selected_spans)).long().to(candidate_starts.device)

    @staticmethod
    def batch_select(tensor, idx, device=torch.device('cpu')):
        """ Do selection per row (first axis). """
        assert tensor.shape[0] == idx.shape[0]  # Same size of first dim
        dim0_size, dim1_size = tensor.shape[0], tensor.shape[1]

        tensor = torch.reshape(tensor, [dim0_size * dim1_size, -1])
        idx_offset = torch.unsqueeze(torch.arange(0, dim0_size, device=device) * dim1_size, 1)
        new_idx = idx + idx_offset
        selected = tensor[new_idx]

        if tensor.shape[-1] == 1:  # If selected element is scalar, restore original dim
            selected = torch.squeeze(selected, -1)

        return selected

    @staticmethod
    def bucket_distance(distances):
        """
        Places the given values (designed for distances) into 10 semi-logscale buckets:
        [0, 1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+].

        Parameters
        ----------
        distances: tensor of size (candidates, candidates)
            token distances between pairs

        Returns
        -------
        distance buckets
            tensor of size (candidates, candidates)
        """
        logspace_idx = torch.floor(torch.log(distances.float()) / math.log(2)).int() + 3
        use_identity = (distances <= 4).int()
        combined_idx = use_identity * distances + (1 - use_identity) * logspace_idx
        return torch.clamp(combined_idx, 0, 9)

    @staticmethod
    def get_candidate_labels(candidate_starts, candidate_ends, labeled_starts, labeled_ends, labels):
        """
        get labels of candidates from gold ground truth

        Parameters
        ----------
        candidate_starts, candidate_ends: tensor of size (candidates)
            start and end token indices (in flattened document) of candidate spans
        labeled_starts, labeled_ends: tensor of size (labeled)
            start and end token indices (in flattened document) of labeled spans
        labels: tensor of size (labeled)
            cluster ids

        Returns
        -------
        candidate_labels: tensor of size (candidates)
        """
        same_start = torch.eq(labeled_starts.unsqueeze(1),
                              candidate_starts.unsqueeze(0))  # [num_labeled, num_candidates]
        same_end = torch.eq(labeled_ends.unsqueeze(1), candidate_ends.unsqueeze(0))  # [num_labeled, num_candidates]
        same_span = torch.logical_and(same_start, same_end)  # [num_labeled, num_candidates]
        # type casting in next line is due to torch not supporting matrix multiplication for Long tensors
        if labels.shape.__len__() == 1:
            candidate_labels = torch.mm(labels.unsqueeze(0).float(), same_span.float()).long()  # [1, num_candidates]
        else:
            candidate_labels = torch.mm(same_span.transpose(1, 0).float(), labels.float())  # [nclasses, num_candidates]
        return candidate_labels.squeeze(0)  # [num_candidates] or [nclasses, num_candidate]


class SharedDense(torch.nn.Module):
    """
        A linear-esque layer combination, configurable to include batchnorm, dropout, activation.
    """

    def __init__(
            self,
            input_size: int,
            output_size: int = -1,
            depth: int = 1,
            dropout_factor: float = 0.0,
            batchnorm: bool = False,
            dropout: bool = True,
            activation: bool = True,
    ):

        super().__init__()

        if output_size > input_size:
            raise BadParameters(f"Shared Module: Unexpected: inputdim: {input_size}. outputdim: {output_size}. "
                                f"Input dim should be better than output dim.")

        if depth == 0:
            self.params = nn.Sequential()

        elif depth > 0:

            """
                If inputdim is 300, output dim is 100, and n = 1 you want linear layers like:
                    [300, 100]
                    
                If n == 2:
                    [300, 200], [200, 100]
                    
                if n == 4:
                    [300, 250], [250, 200], [200, 150], [150, 100]
                    
                Basically, linearly stepping down
            """
            _h = input_size  # 768
            _l = output_size  # 256
            _n = depth  # 2
            _d = (_h - _l) // _n  # 256
            _arr = [_h] + [int(_h - (_d * i)) for i in range(1, _n)] + [_l]  # [768, 512, 256

            layers: List[torch.nn.Module] = []

            for indim, outdim in zip(_arr[:-1], _arr[1:]):
                layer = [Utils.make_linear(indim, outdim)]
                if batchnorm:
                    layer.append(nn.BatchNorm1d(outdim))
                if activation:
                    layer.append(nn.ReLU())
                if dropout:
                    layer.append(nn.Dropout(dropout_factor))
                layers += layer

            self.params = nn.Sequential(*layers)

        else:
            raise BadParameters(f"Depth of {depth} not understood!")

    def forward(self, input_tensor: torch.tensor) -> torch.tensor:
        return self.params(input_tensor)


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

# class SpanPrunerMangoes(torch.nn.Module):
#     """
#         Give me your candidate starts, your candidate ends
#         Give me your transformer encoded tokens
#         Give me your constants like max span width
#         And I shall give you pruned span embeddings and corresponding maps.
#     """
#
#     def __init__(
#             self,
#             hidden_size: int,
#             unary_hdim: int,
#             max_span_width: int,
#             coref_metadata_feature_size: int,
#             pruner_dropout: float,
#             pruner_use_metadata: bool,
#             pruner_max_num_spans: int,
#             bias_in_last_layers: bool,
#             pruner_top_span_ratio: float,
#     ):
#         super().__init__()
#
#         # Some constants
#         self._dropout: float = pruner_dropout
#         self._use_metadata: bool = pruner_use_metadata
#         self._max_num_spans: int = pruner_max_num_spans
#         self._top_span_ratio: float = pruner_top_span_ratio
#
#         # Parameter Time!
#         _feat_dim = coref_metadata_feature_size
#         _span_dim = (hidden_size * 3) + (_feat_dim if self._use_metadata else 0)
#
#         self.span_attend_projection = nn.Linear(hidden_size, 1)
#         self.span_scorer = nn.Sequential(
#             nn.Linear(_span_dim, unary_hdim),
#             nn.ReLU(),
#             nn.Dropout(self._dropout),
#             nn.Linear(unary_hdim, 1, bias=bias_in_last_layers),
#         )
#
#         if self._use_metadata:
#             self.span_width_scorer = Utils.make_ffnn(_feat_dim, [unary_hdim], 1, self._dropout)
#             self.emb_span_width = Utils.make_embedding(max_span_width, coref_metadata_feature_size, )
#             self.emb_span_width_prior = Utils.make_embedding(max_span_width, coref_metadata_feature_size)
#
#     def get_span_word_attention_scores(self, hidden_states, span_starts, span_ends):
#         """
#
#         Parameters
#         ----------
#         hidden_states: tensor of size (num_tokens, emb_size)
#             outputs of BERT model, reshaped
#         span_starts, span_ends: tensor of size (num_candidates)
#             indices of starts and ends of spans
#
#         Returns
#         -------
#         tensor of size (num_candidates, span_embedding_size)
#         """
#         document_range = torch.arange(start=0, end=hidden_states.shape[0], device=hidden_states.device).unsqueeze(
#             0).repeat(span_starts.shape[0], 1)  # [num_cand, num_words]
#         # noinspection PyTypeChecker
#         token_mask = torch.logical_and(document_range >= span_starts.unsqueeze(1),
#                                        document_range <= span_ends.unsqueeze(1))  # [num_cand, num_words]
#         token_atten = self.span_attend_projection(hidden_states).squeeze(1).unsqueeze(0)  # [1, num_words]
#         token_attn = F.softmax(torch.log(token_mask.float()) + token_atten, 1)  # [num_cand, num_words]span
#         return token_attn
#
#     def get_span_embeddings(
#             self,
#             hidden_states: torch.Tensor,  # [num_swords, bert_emb_size] (2000, 732 e.g.)
#             span_starts: torch.Tensor,  # [num_cand, ]
#             span_ends: torch.Tensor  # [num_cand, ]
#     ):
#         """
#         Obtains representations of the spans
#
#         Parameters
#         ----------
#         hidden_states: tensor of size (num_tokens, bert_emb_size)
#             outputs of BERT model, reshaped
#         span_starts, span_ends: tensor of size (num_cand, )
#             indices of starts and ends of spans
#
#         Returns
#         -------
#         tensor of size (num_cand, span_embedding_size)
#         """
#         emb = [hidden_states[span_starts], hidden_states[span_ends]]
#
#         if self._use_metadata:
#             # Calculate span width embeddings
#             span_width = 1 + span_ends - span_starts  # [num_cand]
#             span_width_index = span_width - 1  # [num_cand]
#             span_width_emb = self.emb_span_width(span_width_index)  # [num_cand, emb_size]
#             span_width_emb = F.dropout(span_width_emb, p=self._dropout, training=self.training)
#
#             # Append to Emb
#             emb.append(span_width_emb)
#
#         # Calculate attention weighted summary of different tokens
#         token_attention_scores = self.get_span_word_attention_scores(hidden_states, span_starts,
#                                                                      span_ends)  # [num_cand, num_words]
#         attended_word_representations = torch.mm(token_attention_scores, hidden_states)  # [num_cand, emb_size]
#         emb.append(attended_word_representations)
#         return torch.cat(emb, dim=1)
#
#     def forward(
#             self,
#             hidden_states: torch.tensor,  # [num_swords, bert_emb_size]
#             candidate_starts: torch.tensor,  # [num_cand, ]
#             candidate_ends: torch.tensor,  # [num_cand, ]
#             speaker_ids: torch.tensor,  # [num_swords, 1]
#             device: Union[str, torch.device]
#
#     ):
#         _num_words: int = hidden_states.shape[0]
#         span_emb = self.get_span_embeddings(hidden_states, candidate_starts, candidate_ends)  # [num_cand, emb_size]
#         span_scores = self.span_scorer(span_emb).squeeze(1)  # [num_cand,]
#
#         if self._use_metadata:
#             # Get span with scores (using embeddings with priors), and add them to candidate scores
#             span_width_indices = candidate_ends - candidate_starts
#             span_width_emb = self.emb_span_width_prior(span_width_indices)  # [num_cand, meta]
#             span_width_scores = self.span_width_scorer(span_width_emb).squeeze(1)  # [num_cand, ]
#             span_scores += span_width_scores  # [num_cand, ]
#
#         # Get beam size (its a function of top span ratio, and length of document, capped by a threshold
#         # noinspection PyTypeChecker
#         num_top_spans = int(min(self._max_num_spans, _num_words * self._top_span_ratio))
#
#         # Get top mention scores and sort by span order
#
#         pruned_span_indices = Utils.extract_spans(candidate_starts, candidate_ends, span_scores, num_top_spans)
#         pruned_span_starts = candidate_starts[pruned_span_indices]
#         pruned_span_ends = candidate_ends[pruned_span_indices]
#         pruned_span_emb = span_emb[pruned_span_indices]
#         pruned_span_scores = span_scores[pruned_span_indices]
#
#         if speaker_ids is not None:
#             pruned_span_speaker_ids = speaker_ids[pruned_span_starts]
#         else:
#             pruned_span_speaker_ids = None
#
#         return {
#             'span_emb': span_emb,
#             'pruned_span_indices': pruned_span_indices,  # HOI calls it 'selected_idx'
#             'pruned_span_starts': pruned_span_starts,
#             'pruned_span_ends': pruned_span_ends,
#             'pruned_span_emb': pruned_span_emb,
#             'pruned_span_scores': pruned_span_scores,
#             'pruned_span_speaker_ids': pruned_span_speaker_ids,
#             'num_top_spans': num_top_spans
#         }
