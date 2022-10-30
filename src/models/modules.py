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
from utils.exceptions import BadParameters, NANsFound


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
    def get_candidate_labels_wl(coreferent_word_indices, coreferent_word_cluster_ids, wordspace=None, n_words=None):
        """
            Similar to Utils.get_candidate_labels.
            If you have two lists like:
                [2,  7,   0,  4,  6, ...], size = n_coreferent_words
                [ 1,  2,  1,  1,  3, ...], size = n_coreferent_words
                where the first is word ID of the spanhead, and second is the cluster ID of the corresponding span,

            This fn gives you a list like
                [1, 0, 1, 0, 1, 0, 3, 2 ...], size = n_words, num_nonzero = n_coreferent_words
                where the index is words, the value is cluster ID.

            Optionally, you can pass another tensor representing the word space.
            For instance, you might have pruned out such that some words can never exist.
            In that case, give us a wordspace. This is an list of arbitary integers.
            We only consider those coreferent_words which appear in this list.


        :param coreferent_word_indices:
        :param coreferent_word_cluster_ids:
        :param wordspace:
        :return:
        """
        if not wordspace and not n_words:
            raise BadParameters("You need to specify either the number of words, or give an explicit word list.")
        if wordspace:
            raise NotImplementedError(f"Effectively, this you should never need this. Can be implemented though."
                                      f"Take reference from the fn below.")

        wordspace = torch.zeros(n_words, dtype=torch.long, device=coreferent_word_indices.device)
        wordspace[coreferent_word_indices] = coreferent_word_cluster_ids
        return wordspace

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

    @staticmethod
    def check_for_nans(data: Union[dict, list, torch.Tensor]):

        if isinstance(data, torch.Tensor) and data.isnan().any():
            raise NANsFound

        elif isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, torch.Tensor) and v.isnan().any():
                    raise NANsFound(f"Found a NaN in the key `{k}`")

                elif isinstance(v, list) or isinstance(v, dict):
                    Utils.check_for_nans(v)

        elif isinstance(data, list):
            for item in data:
                if isinstance(item, torch.Tensor) and item.isnan().any():
                    raise NANsFound

                elif isinstance(data, list) or isinstance(data, dict):
                    Utils.check_for_nans(data)


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

