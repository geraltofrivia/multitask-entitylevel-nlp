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
from typing import Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

# Local imports
try:
    import _pathfix
except ImportError:
    from . import _pathfix
from utils.exceptions import BadParameters


class Utils(object):

    @staticmethod
    def make_embeddings(num_embeddings: int, embedding_dim: int, std: float = 0.02) -> torch.nn.Module:
        emb = nn.Embedding(num_embeddings, embedding_dim)
        init.normal_(emb.weight, std=std)
        return emb

    @staticmethod
    def extract_spans(candidate_starts, candidate_ends, candidate_mention_scores, num_top_mentions):
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
            num_top_mentions: int
                Number of candidates to extract
        Returns:
        --------
            top_span_indices: tensor of size (num_top_mentions)
                Span indices of the non-crossing spans with the highest mention scores
        """
        # sort based on mention scores
        top_span_indices = torch.argsort(candidate_mention_scores, descending=True)
        # add highest scores that don't cross
        end_to_earliest_start = {}
        start_to_latest_end = {}
        selected_spans = []
        current_span_index = 0
        while len(selected_spans) < num_top_mentions and current_span_index < candidate_starts.size(0):
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
            _arr = [_h] + [int(_h - (_d * i)) for i in range(1, _n)] + [_d]  # [768, 512, 256

            layers: List[torch.nn.Module] = []

            for indim, outdim in zip(_arr[:-1], _arr[1:]):
                layer = [nn.Linear(indim, outdim)]
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

    def forward(self, input: torch.tensor) -> torch.tensor:
        return self.params(input)


class SpanPruner(torch.nn.Module):
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
            pruner_use_width: bool,
            pruner_max_num_spans: int,
            pruner_top_span_ratio: float,
    ):
        super().__init__()

        # Some constants
        self._dropout: float = pruner_dropout
        self._use_width: bool = pruner_use_width
        self._max_num_spans: int = pruner_max_num_spans
        self._top_span_ratio: float = pruner_top_span_ratio

        # Parameter Time!
        self.span_attend_projection = nn.Linear(hidden_size, 1)
        span_embedding_dim = (hidden_size * 3) + coref_metadata_feature_size
        self.span_scorer = nn.Sequential(
            nn.Linear(span_embedding_dim, unary_hdim),
            nn.ReLU(),
            nn.Dropout(self._dropout),
            nn.Linear(unary_hdim, 1),
        )

        if self._use_width:
            self.span_width_scorer = nn.Sequential(
                nn.Linear(coref_metadata_feature_size, unary_hdim),
                nn.ReLU(),
                nn.Dropout(self._dropout),
                nn.Linear(unary_hdim, 1),
            )
            self.emb_span_width = Utils.make_embeddings(max_span_width, coref_metadata_feature_size)
            self.emb_span_width_prior = Utils.make_embeddings(max_span_width, coref_metadata_feature_size)

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

        if self._use_width:
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

    ):
        _num_words: int = hidden_states.shape[0]
        span_emb = self.get_span_embeddings(hidden_states, candidate_starts, candidate_ends)  # [num_cand, emb_size]
        span_scores = self.span_scorer(span_emb).squeeze(1)  # [num_cand,]

        if self._use_width:
            # Get span with scores (using embeddings with priors), and add them to candidate scores
            span_width_indices = candidate_ends - candidate_starts
            span_width_emb = self.emb_span_width_prior(span_width_indices)  # [num_cand, meta]
            span_width_scores = self.span_width_scorer(span_width_emb).squeeze(1)  # [num_cand, ]
            span_scores += span_width_scores  # [num_cand, ]

        # Get beam size (its a function of top span ratio, and length of document, capped by a threshold
        # noinspection PyTypeChecker
        num_top_mentions = int(min(self._max_num_spans, _num_words * self._top_span_ratio))

        # Get top mention scores and sort by span order
        pruned_span_indices = Utils.extract_spans(candidate_starts, candidate_ends, span_scores, num_top_mentions)
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
            'pruned_span_indices': pruned_span_indices,
            'pruned_span_starts': pruned_span_starts,
            'pruned_span_ends': pruned_span_ends,
            'pruned_span_emb': pruned_span_emb,
            'pruned_span_scores': pruned_span_scores,
            'pruned_span_speaker_ids': pruned_span_speaker_ids,
            'num_top_mentions': num_top_mentions
        }


class CorefDecoder(torch.nn.Module):

    def __init__(
            self,
            max_top_antecedents: int,
            unary_hdim: int,
            hidden_size: int,
            ignore_speakers: bool,
            max_training_segments: int,
            coref_metadata_feature_size: int,
            coref_dropout: float,
            coref_higher_order: int,
            coref_num_speakers: int = 2
    ):
        super().__init__()

        self._dropout = coref_dropout

        # Config Time
        _span_embedding_dim = (hidden_size * 3) + coref_metadata_feature_size
        _final_metadata_size = coref_metadata_feature_size * (2 if ignore_speakers else 3)

        self.max_top_antecedents: int = max_top_antecedents
        self.max_training_segments: int = max_training_segments
        self.coref_depth: int = coref_higher_order
        self._ignore_speakers = ignore_speakers

        # Parameters Time
        self.fast_antecedent_projection = torch.nn.Linear(_span_embedding_dim, _span_embedding_dim)
        self.distance_projection = nn.Linear(coref_metadata_feature_size, 1)
        self.slow_antecedent_scorer = nn.Sequential(
            nn.Linear((_span_embedding_dim * 3) + _final_metadata_size, unary_hdim),
            nn.ReLU(),
            nn.Dropout(self._dropout),
            nn.Linear(unary_hdim, 1),
        )
        self.slow_antecedent_projection = torch.nn.Linear(_span_embedding_dim * 2, _span_embedding_dim)
        # TODO: wire this up as well
        # metadata embeddings
        # self.use_metadata = use_metadata
        # self.genres = {g: i for i, g in enumerate(genres)}
        # self.genre_embeddings = nn.Embedding(num_embeddings=len(self.genres),
        #                                       embedding_dim=coref_metadata_feature_size)

        self.emb_segment_dist = Utils.make_embeddings(max_training_segments, coref_metadata_feature_size)
        self.emb_fast_distance = Utils.make_embeddings(num_embeddings=10, embedding_dim=coref_metadata_feature_size)
        self.emb_slow_distance = Utils.make_embeddings(num_embeddings=10, embedding_dim=coref_metadata_feature_size)
        if self._ignore_speakers:
            self.emb_same_speaker = None
        else:
            self.emb_same_speaker = Utils.make_embeddings(2, coref_metadata_feature_size)

    @staticmethod
    def batch_gather(emb, indices):
        batch_size, seq_len = emb.shape
        flattened_emb = emb.view(-1, 1)
        offset = (torch.arange(start=0, end=batch_size, device=indices.device) * seq_len).unsqueeze(1)
        return flattened_emb[indices + offset].squeeze(2)

    def get_fast_antecedent_scores(self, span_emb):
        """
        Obtains representations of the spans

        Parameters
        ----------
        span_emb: tensor of size (candidates, emb_size)
            span representations

        Returns
        -------
        fast antecedent scores
            tensor of size (candidates, span_embedding_size)
        """
        source_emb = F.dropout(self.fast_antecedent_projection(span_emb),
                               p=self._dropout, training=self.training)  # [cand, emb]
        target_emb = F.dropout(span_emb, p=self._dropout, training=self.training)  # [cand, emb]
        return torch.mm(source_emb, target_emb.t())  # [cand, cand]

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

    def coarse_to_fine_pruning(self, span_emb, mention_scores, num_top_antecedents):
        """
        Compute fast estimate antecedent scores and prune based on these scores.

        Parameters
        ----------
        span_emb: tensor of size (candidates, emb_size)
            span representations
        mention_scores: tensor of size (candidates)
            mention scores of spans
        num_top_antecedents: int
            number of antecedents

        Returns
        -------
        top_antecedents: tensor of shape (mentions, antecedent_candidates)
            indices of top antecedents for each mention
        top_antecedents_mask: tensor of shape (mentions, antecedent_candidates)
            boolean mask for antecedent candidates
        top_antecedents_fast_scores: tensor of shape (mentions, antecedent_candidates)
            fast scores for each antecedent candidate
        top_antecedent_offsets: tensor of shape (mentions, antecedent_candidates)
            offsets for each mention/antecedent pair
        """
        num_candidates = span_emb.shape[0]
        top_span_range = torch.arange(start=0, end=num_candidates, device=span_emb.device)
        antecedent_offsets = top_span_range.unsqueeze(1) - top_span_range.unsqueeze(0)  # [cand, cand]
        antecedents_mask = antecedent_offsets >= 1  # [cand, cand]
        fast_antecedent_scores = mention_scores.unsqueeze(1) + mention_scores.unsqueeze(0)  # [cand, cand]
        fast_antecedent_scores += torch.log(antecedents_mask.float())  # [cand, cand]
        fast_antecedent_scores += self.get_fast_antecedent_scores(span_emb)  # [cand, cand]
        # add distance scores
        antecedent_distance_buckets = self.bucket_distance(antecedent_offsets).to(span_emb.device)  # [cand, cand]
        bucket_embeddings = F.dropout(self.emb_fast_distance(torch.arange(start=0, end=10, device=span_emb.device)),
                                      p=self._dropout, training=self.training)  # [10, feature_size]
        bucket_scores = self.distance_projection(bucket_embeddings)  # [10, 1]
        fast_antecedent_scores += bucket_scores[antecedent_distance_buckets].squeeze(-1)  # [cand, cand]
        # get top antecedent scores/features
        top_antecedents: torch.tensor = torch.topk(fast_antecedent_scores, num_top_antecedents, sorted=False,
                                                   dim=1)[1]  # [cand, num_ant]
        top_antecedents_mask = self.batch_gather(antecedents_mask, top_antecedents)  # [cand, num_ant]
        top_antecedents_fast_scores = self.batch_gather(fast_antecedent_scores, top_antecedents)  # [cand, num_ant]
        top_antecedents_offsets = self.batch_gather(antecedent_offsets, top_antecedents)  # [cand, num_ant]
        return top_antecedents, top_antecedents_mask, top_antecedents_fast_scores, top_antecedents_offsets

    def get_slow_antecedent_scores(self, top_span_emb, top_antecedents, top_antecedent_emb, top_antecedent_offsets,
                                   top_span_speaker_ids, segment_distance, domain: str):
        """
        Compute slow antecedent scores

        Parameters
        ----------
        top_span_emb: tensor of size (candidates, emb_size)
            span representations
        top_antecedents: tensor of size (candidates, antecedents)
            indices of antecedents for each candidate
        top_antecedent_emb: tensor of size (candidates, antecedents, emb)
            embeddings of top antecedents for each candidate
        top_antecedent_offsets: tensor of size (candidates, antecedents)
            offsets for each mention/antecedent pair
        top_span_speaker_ids: tensor of size TODO: complete this one
        segment_distance: tensor of size (candidates, antecedents)
            segment distances for each candidate antecedent pair
        domain: a string representing the domain from which this instance hails.

        Returns
        -------
        tensor of shape (candidates, antecedents)
            antecedent scores
        """
        num_cand, num_ant = top_antecedents.shape
        feature_emb_list = []

        if (not self._ignore_speakers) and (top_span_speaker_ids is not None):
            top_antecedent_speaker_ids = top_span_speaker_ids[top_antecedents]  # [top_cand, top_ant]
            # [top_cand, top_ant]
            same_speaker = torch.eq(top_span_speaker_ids.view(-1, 1), top_antecedent_speaker_ids)
            same_speaker_embedded = self.emb_same_speaker(same_speaker.long())
            feature_emb_list.append(same_speaker_embedded)
            # genre_embs = genre_emb.view(1, 1, -1).repeat(num_cand, num_ant, 1)  # [top_cand, top_ant, feature_size]
            # feature_emb_list.append(genre_embs)

        # span distance
        antecedent_distance_buckets = self.bucket_distance(top_antecedent_offsets).to(
            top_span_emb.device)  # [cand, cand]
        bucket_embeddings = self.emb_slow_distance(
            torch.arange(start=0, end=10, device=top_span_emb.device))  # [10, feature_size]
        feature_emb_list.append(bucket_embeddings[antecedent_distance_buckets])  # [cand, ant, feature_size]

        # segment distance
        segment_distance_emb = self.emb_segment_dist(
            torch.arange(start=0, end=self.max_training_segments, device=top_span_emb.device))
        feature_emb_list.append(segment_distance_emb[segment_distance])  # [cand, ant, feature_size]

        feature_emb = torch.cat(feature_emb_list, 2)  # [cand, ant, emb]
        feature_emb = F.dropout(feature_emb, p=self._dropout,
                                training=self.training)  # [cand, ant, emb]
        target_emb = top_span_emb.unsqueeze(1)  # [cand, 1, emb]
        similarity_emb = top_antecedent_emb * target_emb  # [cand, ant, emb]
        target_emb = target_emb.repeat(1, num_ant, 1)  # [cand, ant, emb]

        pair_emb = torch.cat([target_emb, top_antecedent_emb, similarity_emb, feature_emb], 2)  # [cand, ant, emb]

        return self.slow_antecedent_scorer(pair_emb).squeeze(-1)

    def forward(
            self,

            attention_mask: torch.tensor,  # [num_seg, len_seg]

            # Pruner outputs
            pruned_span_starts: torch.tensor,  # [num_cand, ]
            pruned_span_ends: torch.tensor,  # [num_cand, ]
            pruned_span_indices: torch.tensor,  # [num_cand, ]
            pruned_span_scores: torch.tensor,  # [num_cand, ]
            pruned_span_speaker_ids: torch.tensor,  # [num_cand, ]
            pruned_span_emb: torch.tensor,  # [num_cand, emb_size]

            # Some input ID things
            num_top_mentions: int,
            num_segments: int,
            len_segment: int,
            domain: str,
            device: Union[str, torch.device],
    ):

        num_top_antecedents = min(self.max_top_antecedents, num_top_mentions)

        # Start with coarse model
        dummy_scores = torch.zeros([num_top_mentions, 1], device=device)
        top_ante, top_ante_mask, top_ante_fast_scores, top_ante_offsets = self.coarse_to_fine_pruning(
            pruned_span_emb,
            pruned_span_scores,
            num_top_antecedents
        )

        # num_seg, len_seg
        word_segments = torch.arange(start=0, end=num_segments, device=device).view(-1, 1).repeat([1, len_segment])
        flat_word_segments = torch.masked_select(word_segments.view(-1), attention_mask.bool().view(-1))  # [num_swords]
        mention_segments = flat_word_segments[pruned_span_starts].view(-1, 1)  # [num_cand, 1]
        antecedent_segments = flat_word_segments[pruned_span_starts][top_ante]  # [ num_cand, num_ante]
        segment_distance = torch.clamp(mention_segments - antecedent_segments, 0,
                                       self.max_training_segments - 1)  # [num_cand, num_ante]

        # calculate final slow scores
        for i in range(self.coref_depth):
            top_ante_emb = pruned_span_emb[top_ante]  # [top_cand, top_ant, emb]
            try:
                top_ante_scores = top_ante_fast_scores + self.get_slow_antecedent_scores(
                    pruned_span_emb,
                    top_ante,
                    top_ante_emb,
                    top_ante_offsets,
                    pruned_span_speaker_ids,
                    segment_distance,
                    domain=domain)
            except RuntimeError as e:
                # This usually happens due to out of mem errors
                # print(f"Input IDS: {input_ids.shape}")
                # print(f"Spans Emb: ({candidate_starts.shape[0]}, something)")
                print("You need input IDs and other things here to figure out what went wrong. Pass 'em here!")
                raise e

            # [top_cand, top_ant]
            top_ante_weights = F.softmax(
                torch.cat([dummy_scores, top_ante_scores], 1), dim=-1
            )  # [top_cand, top_ant + 1]
            top_ante_emb = torch.cat([pruned_span_emb.unsqueeze(1), top_ante_emb],
                                     1)  # [top_cand, top_ant + 1, emb]
            attended_span_emb = torch.sum(top_ante_weights.unsqueeze(2) * top_ante_emb, 1)  # [top_cand, emb]
            gate_vectors = torch.sigmoid(
                self.slow_antecedent_projection(
                    torch.cat([pruned_span_emb, attended_span_emb], 1)))  # [top_cand, emb]
            pruned_span_emb = gate_vectors * attended_span_emb + (
                    1 - gate_vectors) * pruned_span_emb  # [top_cand, emb]

        # noinspection PyUnboundLocalVariable
        top_ante_scores = torch.cat([dummy_scores, top_ante_scores], 1)

        return {
            "coref_top_antecedents": top_ante,
            "coref_top_antecedent_scores": top_ante_scores,
            "coref_top_antecedents_mask": top_ante_mask,
            "pruned_candidate_mention_scores": pruned_span_scores,
            "pruned_span_starts": pruned_span_starts,
            "pruned_span_ends": pruned_span_ends,
            "pruned_span_indices": pruned_span_indices,
            # "coref_top_span_cluster_ids": top_span_cluster_ids,
        }
