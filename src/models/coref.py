"""
    Here we keep all our coref models.

"""
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Local imports
from models.modules import Utils


class CorefDecoderWL(torch.nn.Module):
    """
        This corresponds to the Word-Level coref paper - https://aclanthology.org/2021.emnlp-main.605.pdf
        Functionally, this can not be swapped in with CorefDecoderHOI as it does not even work at a span level
        Does not need pruning in the typical sense.

        This class has a lot of copy-pasted code from the repository - https://github.com/vdobrovolskii/wl-coref
        I claim no credit for the stuff here.
    """

    def __init__(
            self,
            use_speakers: bool,
            unary_hdim: int,
            hidden_size: int,

            max_training_segments: int,
            coref_metadata_feature_size: int,
            coref_num_genres: int,
            coref_use_metadata: bool,
            coref_dropout: float,
            coref_spanemb_size: int,

    ):
        super().__init__()

        self._n_genres = coref_num_genres
        self._use_metadata = coref_use_metadata
        self._use_speakers = use_speakers
        self._max_training_segments = max_training_segments
        self._dropout = coref_dropout

        self.dropout = nn.Dropout(self._dropout)

        _feat_dim = coref_metadata_feature_size
        _pair_dim = (hidden_size * 3) + _feat_dim + (_feat_dim if self._use_metadata else 0) + \
                    (_feat_dim if self._use_speakers else 0)

        """
        Pairwise Encoder stuff
            obtain feature embeddings for pairwise features
        """
        self.emb_genre = Utils.make_embedding(self._n_genres, _feat_dim) if self._use_metadata else None
        self.emb_same_speaker = Utils.make_embedding(2, _feat_dim) if self._use_speakers else None
        self.emb_segment_distance = Utils.make_embedding(max_training_segments, _feat_dim)

        """
        Anaphoricity Scorer stuff
            Calculates anaphoricity scores by passing the inputs into a FFNN
    
            TODO: This is supposed to do a fair bit of things in the forward fn. We'll see then    
        """
        self.anaphoricity_ffnn_hidden = Utils.make_ffnn(_pair_dim, [unary_hdim], 1, dropout=self._dropout,
                                                        nonlin='leakyrelu')

        """
         Word Encoder stuff
            Receives bert contextual embeddings of a text, extracts all the possible mentions in that text.
        """
        self.attn = Utils.make_linear(hidden_size, 1)

        """
            Rough Scorer stuff
            Is needed to give a roughly estimate of the anaphoricity of two candidates,
            only top scoring candidates are considered on later steps to reduce
            computational complexity.
        """
        self.bilinear = Utils.make_linear(hidden_size, hidden_size)

        """
            Span Predictor stuff
        """
        self.span_pred_ffnn = Utils.make_ffnn(hidden_size * 2 + 64, [hidden_size, 256], 64,
                                              nonlin='relu', dropout=self._dropout)
        self.span_pred_conv = torch.nn.Sequential(
            torch.nn.Conv1d(64, 4, 3, 1, 1),
            torch.nn.Conv1d(4, 2, 3, 1, 1)
        )
        self.emb_span_pred = Utils.make_embedding(128, coref_spanemb_size)
        '''self.ffnn = torch.nn.Sequential(
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
        self.emb = torch.nn.Embedding(128, distance_emb_size) # [-63, 63] + too_far'''

    def forward(
            self
    ):
        ...


class CorefDecoderHOI(torch.nn.Module):

    def __init__(
            self,
            max_top_antecedents: int,
            unary_hdim: int,
            hidden_size: int,
            use_speakers: bool,
            max_training_segments: int,
            coref_metadata_feature_size: int,
            coref_dropout: float,
            coref_higher_order: str,
            coref_depth: int,
            coref_num_genres: int,
            coref_easy_cluster_first: bool,
            coref_cluster_reduce: str,
            coref_cluster_dloss: bool,
            bias_in_last_layers: bool,
            coref_use_metadata: bool,
            coref_loss_type: str,
            coref_false_new_delta: float,
            coref_num_speakers: int = 2,
    ):
        """
        :param coref_depth: int indicating the number of higher order iterations in loop. 1 is not really higher order.
            Stuff I need to add here:
                genre stuff: num_genres

            Coref loss stuff comes here as well

        """
        super().__init__()
        self._dropout = coref_dropout
        self._n_genres = coref_num_genres
        self._use_speakers = use_speakers
        self._use_metadata = coref_use_metadata
        self._depth = coref_depth
        self._ho = coref_higher_order
        self._max_top_antecedents: int = max_top_antecedents
        self._easy_cluster_first = coref_easy_cluster_first
        self._cluster_reduce = coref_cluster_reduce
        self._cluster_dloss = coref_cluster_dloss
        self._loss_type = coref_loss_type

        _feat_dim = coref_metadata_feature_size
        _span_dim = (hidden_size * 3) + (_feat_dim if self._use_metadata else 0)
        _pair_dim = _span_dim * 3 + (_feat_dim * 2 if self._use_metadata else 0) + \
                    (_feat_dim * 2 if self._use_speakers else 0)

        # more variables
        self.debug = True
        self._span_dim = _span_dim
        self._feat_dim = _feat_dim
        self._pair_dim = _pair_dim
        self._max_training_segments = max_training_segments
        self._false_new_delta = coref_false_new_delta
        self.dropout = nn.Dropout(p=self._dropout)

        self.emb_same_speaker = Utils.make_embedding(2, _feat_dim) if self._use_speakers else None
        self.emb_segment_distance = Utils.make_embedding(max_training_segments, _feat_dim)
        self.emb_top_antecedent_distance = Utils.make_embedding(10, _feat_dim)
        self.emb_genre = Utils.make_embedding(self._n_genres, _feat_dim) if self._use_metadata else None
        self.emb_antecedent_distance_prior = Utils.make_embedding(10, _feat_dim) if self._use_metadata else None
        self.emb_cluster_size = Utils.make_embedding(10, _feat_dim) if self._ho == 'cluster_merging' else None

        self.antecedent_distance_score_ffnn = Utils.make_ffnn(_feat_dim, 0, 1,
                                                              self._dropout) if self._use_metadata else None
        self.span_attn_ffnn = Utils.make_ffnn(_span_dim, 0, 1, self._dropout) if self._ho == 'span_clustering' else None
        self.cluster_score_ffnn = Utils.make_ffnn(3 * _span_dim + _feat_dim, unary_hdim, 1, self._dropout,
                                                  bias_in_last_layers) if self._ho == 'cluster_merging' else None
        self.coarse_bilinear = Utils.make_ffnn(_span_dim, 0, _span_dim, dropout=self._dropout)
        self.coref_score_ffnn = Utils.make_ffnn(_pair_dim, [unary_hdim], 1, self._dropout, bias_in_last_layers)
        self.gate_ffnn = Utils.make_ffnn(2 * _span_dim, 0, _pair_dim, self._dropout) if self._depth > 1 else None

    @staticmethod
    def _merge_span_to_cluster(cluster_emb, cluster_sizes, cluster_to_merge_id, span_emb, reduce):
        cluster_size = cluster_sizes[cluster_to_merge_id].item()
        if reduce == 'mean':
            cluster_emb[cluster_to_merge_id] = (cluster_emb[cluster_to_merge_id] * cluster_size + span_emb) / (
                    cluster_size + 1)
        elif reduce == 'max':
            cluster_emb[cluster_to_merge_id], _ = torch.max(torch.stack([cluster_emb[cluster_to_merge_id], span_emb]),
                                                            dim=0)
        else:
            raise ValueError('reduce value is invalid: %s' % reduce)
        cluster_sizes[cluster_to_merge_id] += 1

    def cluster_merging(self, top_span_emb, top_antecedent_idx, top_antecedent_scores, emb_cluster_size,
                        cluster_score_ffnn,
                        cluster_transform, dropout, device, reduce='mean', easy_cluster_first=False):
        num_top_spans, max_top_antecedents = top_antecedent_idx.shape[0], top_antecedent_idx.shape[1]
        span_emb_size = top_span_emb.shape[-1]
        max_num_clusters = num_top_spans

        span_to_cluster_id = torch.zeros(num_top_spans, dtype=torch.long, device=device)  # id 0 as dummy cluster
        cluster_emb = torch.zeros(max_num_clusters, span_emb_size, dtype=torch.float,
                                  device=device)  # [max num clusters, emb size]
        num_clusters = 1  # dummy cluster
        cluster_sizes = torch.ones(max_num_clusters, dtype=torch.long, device=device)

        merge_order = torch.arange(0, num_top_spans)
        if easy_cluster_first:
            max_antecedent_scores, _ = torch.max(top_antecedent_scores, dim=1)
            merge_order = torch.argsort(max_antecedent_scores, descending=True)
        cluster_merging_scores = [None] * num_top_spans

        for i in merge_order.tolist():
            # Get cluster scores
            antecedent_cluster_idx = span_to_cluster_id[top_antecedent_idx[i]]
            antecedent_cluster_emb = cluster_emb[antecedent_cluster_idx]
            # antecedent_cluster_emb = dropout(cluster_transform(antecedent_cluster_emb))

            antecedent_cluster_size = cluster_sizes[antecedent_cluster_idx]
            antecedent_cluster_size = Utils.bucket_distance(antecedent_cluster_size)
            cluster_size_emb = dropout(emb_cluster_size(antecedent_cluster_size))

            span_emb = top_span_emb[i].unsqueeze(0).repeat(max_top_antecedents, 1)
            similarity_emb = span_emb * antecedent_cluster_emb
            pair_emb = torch.cat([span_emb, antecedent_cluster_emb, similarity_emb, cluster_size_emb],
                                 dim=1)  # [max top antecedents, pair emb size]
            cluster_scores = torch.squeeze(cluster_score_ffnn(pair_emb), 1)
            cluster_scores_mask = (antecedent_cluster_idx > 0).to(torch.float)
            cluster_scores *= cluster_scores_mask
            cluster_merging_scores[i] = cluster_scores

            # Get predicted antecedent
            antecedent_scores = top_antecedent_scores[i] + cluster_scores
            max_score, max_score_idx = torch.max(antecedent_scores, dim=0)
            if max_score < 0:
                continue  # Dummy antecedent
            max_antecedent_idx = top_antecedent_idx[i, max_score_idx]

            if not easy_cluster_first:  # Always add span to antecedent's cluster
                # Create antecedent cluster if needed
                antecedent_cluster_id = span_to_cluster_id[max_antecedent_idx]
                if antecedent_cluster_id == 0:
                    antecedent_cluster_id = num_clusters
                    span_to_cluster_id[max_antecedent_idx] = antecedent_cluster_id
                    cluster_emb[antecedent_cluster_id] = top_span_emb[max_antecedent_idx]
                    num_clusters += 1
                # Add span to cluster
                span_to_cluster_id[i] = antecedent_cluster_id
                self._merge_span_to_cluster(cluster_emb, cluster_sizes, antecedent_cluster_id, top_span_emb[i],
                                            reduce=reduce)
            else:  # current span can be in cluster already
                antecedent_cluster_id = span_to_cluster_id[max_antecedent_idx]
                curr_span_cluster_id = span_to_cluster_id[i]
                if antecedent_cluster_id > 0 and curr_span_cluster_id > 0:
                    # Merge two clusters
                    span_to_cluster_id[max_antecedent_idx] = curr_span_cluster_id
                    self._merge_clusters(cluster_emb, cluster_sizes, antecedent_cluster_id, curr_span_cluster_id,
                                         reduce=reduce)
                elif curr_span_cluster_id > 0:
                    # Merge antecedent to span's cluster
                    span_to_cluster_id[max_antecedent_idx] = curr_span_cluster_id
                    self._merge_span_to_cluster(cluster_emb, cluster_sizes, curr_span_cluster_id,
                                                top_span_emb[max_antecedent_idx], reduce=reduce)
                else:
                    # Create antecedent cluster if needed
                    if antecedent_cluster_id == 0:
                        antecedent_cluster_id = num_clusters
                        span_to_cluster_id[max_antecedent_idx] = antecedent_cluster_id
                        cluster_emb[antecedent_cluster_id] = top_span_emb[max_antecedent_idx]
                        num_clusters += 1
                    # Add span to cluster
                    span_to_cluster_id[i] = antecedent_cluster_id
                    self._merge_span_to_cluster(cluster_emb, cluster_sizes, antecedent_cluster_id, top_span_emb[i],
                                                reduce=reduce)

        cluster_merging_scores = torch.stack(cluster_merging_scores, dim=0)
        return cluster_merging_scores

    @staticmethod
    def attended_antecedent(top_span_emb, top_antecedent_emb, top_antecedent_scores, device):
        num_top_spans = top_span_emb.shape[0]
        top_antecedent_weights = torch.cat([torch.zeros(num_top_spans, 1, device=device), top_antecedent_scores], dim=1)
        top_antecedent_weights = nn.functional.softmax(top_antecedent_weights, dim=1)
        top_antecedent_emb = torch.cat([torch.unsqueeze(top_span_emb, 1), top_antecedent_emb], dim=1)
        refined_span_emb = torch.sum(torch.unsqueeze(top_antecedent_weights, 2) * top_antecedent_emb,
                                     dim=1)  # [num top spans, span emb size]
        return refined_span_emb

    @staticmethod
    def max_antecedent(top_span_emb, top_antecedent_emb, top_antecedent_scores, device):
        num_top_spans = top_span_emb.shape[0]
        top_antecedent_weights = torch.cat([torch.zeros(num_top_spans, 1, device=device), top_antecedent_scores], dim=1)
        top_antecedent_emb = torch.cat([torch.unsqueeze(top_span_emb, 1), top_antecedent_emb], dim=1)
        max_antecedent_idx = torch.argmax(top_antecedent_weights, dim=1, keepdim=True)
        refined_span_emb = Utils.batch_select(top_antecedent_emb, max_antecedent_idx, device=device).squeeze(
            1)  # [num top spans, span emb size]
        return refined_span_emb

    @staticmethod
    def span_clustering(top_span_emb, top_antecedent_idx, top_antecedent_scores, span_attn_ffnn, device):
        # Get predicted antecedents
        num_top_spans, max_top_antecedents = top_antecedent_idx.shape[0], top_antecedent_idx.shape[1]
        predicted_antecedents = []
        top_antecedent_scores = torch.cat([torch.zeros(num_top_spans, 1, device=device), top_antecedent_scores], dim=1)
        for i, idx in enumerate((torch.argmax(top_antecedent_scores, dim=1) - 1).tolist()):
            if idx < 0:
                predicted_antecedents.append(-1)
            else:
                predicted_antecedents.append(top_antecedent_idx[i, idx].item())
        # Get predicted clusters
        predicted_clusters = []
        span_to_cluster_id = [-1] * num_top_spans
        for i, predicted_idx in enumerate(predicted_antecedents):
            if predicted_idx < 0:
                continue
            assert i > predicted_idx, f'span idx: {i}; antecedent idx: {predicted_idx}'
            # Check antecedent's cluster
            antecedent_cluster_id = span_to_cluster_id[predicted_idx]
            if antecedent_cluster_id == -1:
                antecedent_cluster_id = len(predicted_clusters)
                predicted_clusters.append([predicted_idx])
                span_to_cluster_id[predicted_idx] = antecedent_cluster_id
            # Add mention to cluster
            predicted_clusters[antecedent_cluster_id].append(i)
            span_to_cluster_id[i] = antecedent_cluster_id
        if len(predicted_clusters) == 0:
            return top_span_emb

        # Pad clusters
        max_cluster_size = max([len(c) for c in predicted_clusters])
        cluster_sizes = []
        for cluster in predicted_clusters:
            cluster_sizes.append(len(cluster))
            cluster += [0] * (max_cluster_size - len(cluster))
        predicted_clusters_mask = torch.arange(0, max_cluster_size, device=device).repeat(len(predicted_clusters), 1)
        predicted_clusters_mask = predicted_clusters_mask < torch.tensor(cluster_sizes, device=device).unsqueeze(
            1)  # [num clusters, max cluster size]
        # Get cluster repr
        predicted_clusters = torch.tensor(predicted_clusters, device=device)
        cluster_emb = top_span_emb[predicted_clusters]  # [num clusters, max cluster size, emb size]
        span_attn = torch.squeeze(span_attn_ffnn(cluster_emb), 2)
        span_attn += torch.log(predicted_clusters_mask.to(torch.float))
        span_attn = nn.functional.softmax(span_attn, dim=1)
        cluster_emb = torch.sum(cluster_emb * torch.unsqueeze(span_attn, 2), dim=1)  # [num clusters, emb size]
        # Get refined span
        refined_span_emb = []
        for i, cluster_idx in enumerate(span_to_cluster_id):
            if cluster_idx < 0:
                refined_span_emb.append(top_span_emb[i])
            else:
                refined_span_emb.append(cluster_emb[cluster_idx])
        refined_span_emb = torch.stack(refined_span_emb, dim=0)
        return refined_span_emb

    def forward(
            self,

            # Actual input stuff
            attention_mask: torch.tensor,  # [num_seg, len_seg]

            # Pruner outputs
            pruned_span_starts: torch.tensor,  # [num_cand, ]
            pruned_span_ends: torch.tensor,  # [num_cand, ]
            pruned_span_indices: torch.tensor,  # [num_cand, ]
            pruned_span_scores: torch.tensor,  # [num_cand, ]
            pruned_span_speaker_ids: torch.tensor,  # [num_cand, ]
            pruned_span_emb: torch.tensor,  # [num_cand, emb_size]

            # Some input ID things
            num_top_spans: int,
            num_segments: int,
            len_segment: int,
            domain: str,
            genre: int,
            device: Union[str, torch.device],
    ):
        """ We pick up after span pruning done by SpanPruner forward """

        attention_mask = attention_mask.to(bool)

        # TODO: figure out if we need attention mask to still be broken into segments or linearized?

        # Used to limit how many antecedents we consider, per pruned span
        num_top_antecedents = min(self._max_top_antecedents, num_top_spans)

        # Coarse pruning on each mention's antecedents
        top_span_range = torch.arange(0, num_top_spans, device=device)
        antecedent_offsets = torch.unsqueeze(top_span_range, 1) - torch.unsqueeze(top_span_range, 0)
        antecedent_mask = (antecedent_offsets >= 1)
        pairwise_mention_score_sum = torch.unsqueeze(pruned_span_scores, 1) + torch.unsqueeze(pruned_span_scores, 0)
        source_span_emb = self.dropout(self.coarse_bilinear(pruned_span_emb))
        target_span_emb = self.dropout(torch.transpose(pruned_span_emb, 0, 1))
        pairwise_coref_scores = torch.matmul(source_span_emb, target_span_emb)
        pairwise_fast_scores = pairwise_mention_score_sum + pairwise_coref_scores
        pairwise_fast_scores += torch.log(antecedent_mask.to(torch.float))
        if self._use_metadata:
            distance_score = torch.squeeze(
                self.antecedent_distance_score_ffnn(self.dropout(self.emb_antecedent_distance_prior.weight)), 1)
            bucketed_distance = Utils.bucket_distance(antecedent_offsets)
            antecedent_distance_score = distance_score[bucketed_distance]
            pairwise_fast_scores += antecedent_distance_score
        top_pairwise_fast_scores, top_antecedent_idx = torch.topk(pairwise_fast_scores, k=num_top_antecedents)
        top_antecedent_mask = Utils.batch_select(antecedent_mask, top_antecedent_idx,
                                                 device)  # [num top spans, max top antecedents]
        top_antecedent_offsets = Utils.batch_select(antecedent_offsets, top_antecedent_idx, device)

        # Slow Mention Ranking
        if True:
            # if conf['fine_grained']:
            same_speaker_emb, genre_emb, seg_distance_emb, top_antecedent_distance_emb = None, None, None, None
            if self._use_speakers:
                top_antecedent_speaker_id = pruned_span_speaker_ids[top_antecedent_idx]
                same_speaker = torch.unsqueeze(pruned_span_speaker_ids, 1) == top_antecedent_speaker_id
                same_speaker_emb = self.emb_same_speaker(same_speaker.to(torch.long))
                genre_emb = self.emb_genre(genre)
                genre_emb = torch.unsqueeze(torch.unsqueeze(genre_emb, 0), 0).repeat(num_top_spans,
                                                                                     num_top_antecedents, 1)
            if self._use_metadata:
                num_segs, seg_len = attention_mask.shape[0], attention_mask.shape[1]
                token_seg_ids = torch.arange(0, num_segs, device=device).unsqueeze(1).repeat(1, seg_len)
                token_seg_ids = token_seg_ids[attention_mask]
                top_span_seg_ids = token_seg_ids[pruned_span_starts]
                top_antecedent_seg_ids = token_seg_ids[pruned_span_starts[top_antecedent_idx]]
                top_antecedent_seg_distance = torch.unsqueeze(top_span_seg_ids, 1) - top_antecedent_seg_ids
                top_antecedent_seg_distance = torch.clamp(top_antecedent_seg_distance, 0,
                                                          self._max_training_segments - 1)
                seg_distance_emb = self.emb_segment_distance(top_antecedent_seg_distance)
            # if conf['use_features']:  # Antecedent distance
            top_antecedent_distance = Utils.bucket_distance(top_antecedent_offsets)
            top_antecedent_distance_emb = self.emb_top_antecedent_distance(top_antecedent_distance)

            for depth in range(self._depth):
                top_antecedent_emb = pruned_span_emb[top_antecedent_idx]  # [n_spans, max top antecedents, emb size]
                feature_list = []
                if self._use_metadata:  # speaker, genre
                    feature_list.append(same_speaker_emb)
                    feature_list.append(genre_emb)
                    feature_list.append(seg_distance_emb)
                # if conf['use_features']:  # Antecedent distance
                feature_list.append(top_antecedent_distance_emb)

                feature_emb = torch.cat(feature_list, dim=2)
                feature_emb = self.dropout(feature_emb)
                target_emb = torch.unsqueeze(pruned_span_emb, 1).repeat(1, num_top_antecedents, 1)
                similarity_emb = target_emb * top_antecedent_emb
                pair_emb = torch.cat([target_emb, top_antecedent_emb, similarity_emb, feature_emb], 2)
                top_pairwise_slow_scores = torch.squeeze(self.coref_score_ffnn(pair_emb), 2)
                top_pairwise_scores = top_pairwise_slow_scores + top_pairwise_fast_scores
                if self._ho == 'cluster_merging':
                    cluster_merging_scores = self.cluster_merging(pruned_span_emb, top_antecedent_idx,
                                                                  top_pairwise_scores,
                                                                  self.emb_cluster_size, self.cluster_score_ffnn, None,
                                                                  self.dropout,
                                                                  device=device, reduce=self._cluster_reduce,
                                                                  easy_cluster_first=self._easy_cluster_first)
                    break
                elif depth != self._depth - 1:
                    cluster_merging_scores = None
                    if self._ho == 'attended_antecedent':
                        refined_span_emb = self.attended_antecedent(pruned_span_emb, top_antecedent_emb,
                                                                    top_pairwise_scores, device)
                    elif self._ho == 'max_antecedent':
                        refined_span_emb = self.max_antecedent(pruned_span_emb, top_antecedent_emb, top_pairwise_scores,
                                                               device)
                    elif self._ho == 'entity_equalization':
                        raise NotImplementedError(f"Did not implement entity equalization yet")
                        # refined_span_emb = ho.entity_equalization(top_span_emb, top_antecedent_emb, top_antecedent_idx,
                        #                                           top_pairwise_scores, device)
                    elif self._ho == 'span_clustering':
                        refined_span_emb = self.span_clustering(pruned_span_emb, top_antecedent_idx,
                                                                top_pairwise_scores, self.span_attn_ffnn, device)
                    else:
                        raise ValueError(f"Unknown value for self._higher_order: {self._ho}")

                    gate = self.gate_ffnn(torch.cat([pruned_span_emb, refined_span_emb], dim=1))
                    gate = torch.sigmoid(gate)
                    pruned_span_emb = gate * refined_span_emb + (
                            1 - gate) * pruned_span_emb  # [num top spans, span emb size]
        else:
            # noinspection PyUnreachableCode
            top_pairwise_scores = top_pairwise_fast_scores  # [num top spans, max top antecedents]

        top_antecedent_scores = torch.cat([torch.zeros(num_top_spans, 1, device=device), top_pairwise_scores], dim=1)

        """
            TODO: there's a bunch of stuff here that seems to be here for the purpose of making the loss compuation
                only.
            Once we figure its purpose out, we will proceed to add a pred_with_labels or do_loss fn or something ihre.                
        """

        return {
            "coref_top_antecedents": top_antecedent_idx,
            "coref_top_antecedents_score": top_antecedent_scores,
            "coref_top_antecedents_mask": top_antecedent_mask,
            "coref_top_pairwise_scores": top_pairwise_scores,
            "coref_cluster_merging_scores": cluster_merging_scores,  # only valid when cluster merging
            # "coref_top_span_cluster_ids": top_span_cluster_ids,
        }

    def get_coref_loss(
            self,
            top_span_cluster_ids: torch.tensor,
            top_antecedents: torch.tensor,
            top_antecedents_mask: torch.tensor,
            top_antecedents_score: torch.tensor,
            cluster_merging_scores: torch.tensor,  # For cluster merging
            top_pairwise_scores: torch.tensor,  # For cluster merging
            num_top_spans: int,
            device: Union[str, torch.device],
    ):
        """
            This is to be called with elements from CorefDecoderHoi forward, but also with other stuff (gold)
            AGAIN: THIS DOES NOT CALL FORWARD INSIDE IT. Call it from your 'main' module, whatever that is.
        """
        # same as their `candidate_labels`
        top_antecedent_cluster_ids = top_span_cluster_ids[top_antecedents]
        top_antecedent_cluster_ids += (top_antecedents_mask.to(
            torch.long) - 1) * 100000  # Mask id on invalid antecedents
        same_gold_cluster_indicator = (top_antecedent_cluster_ids == torch.unsqueeze(top_span_cluster_ids, 1))
        non_dummy_indicator = torch.unsqueeze(top_span_cluster_ids > 0, 1)
        pairwise_labels = same_gold_cluster_indicator & non_dummy_indicator
        dummy_antecedent_labels = torch.logical_not(pairwise_labels.any(dim=1, keepdims=True))
        top_antecedent_gold_labels = torch.cat([dummy_antecedent_labels, pairwise_labels], dim=1)

        # Get loss
        if self._loss_type == 'marginalized':
            log_marginalized_antecedent_scores = torch.logsumexp(
                top_antecedents_score + torch.log(top_antecedent_gold_labels.to(torch.float)), dim=1)
            log_norm = torch.logsumexp(top_antecedents_score, dim=1)
            loss = torch.sum(log_norm - log_marginalized_antecedent_scores)
        elif self._loss_type == 'hinge':
            top_antecedent_mask = torch.cat([torch.ones(num_top_spans, 1, dtype=torch.bool, device=device),
                                             top_antecedents_mask], dim=1)
            top_antecedents_score += torch.log(top_antecedent_mask.to(torch.float))
            highest_antecedent_scores, highest_antecedent_idx = torch.max(top_antecedents_score, dim=1)
            gold_antecedent_scores = top_antecedents_score + torch.log(top_antecedent_gold_labels.to(torch.float))
            highest_gold_antecedent_scores, highest_gold_antecedent_idx = torch.max(gold_antecedent_scores, dim=1)
            slack_hinge = 1 + highest_antecedent_scores - highest_gold_antecedent_scores
            # Calculate delta
            highest_antecedent_is_gold = (highest_antecedent_idx == highest_gold_antecedent_idx)
            mistake_false_new = (highest_antecedent_idx == 0) & torch.logical_not(dummy_antecedent_labels.squeeze())
            delta = ((3 - self._false_new_delta) / 2) * torch.ones(num_top_spans, dtype=torch.float, device=device)
            delta -= (1 - self._false_new_delta) * mistake_false_new.to(torch.float)
            delta *= torch.logical_not(highest_antecedent_is_gold).to(torch.float)
            loss = torch.sum(slack_hinge * delta)
        else:
            raise ValueError(f"Unknown Loss type: `{self._loss_type}`. Must be `marginalized` or `hinge`.")

        if self._ho == 'cluster_merging':
            top_pairwise_scores += cluster_merging_scores
            top_antecedent_scores = torch.cat([torch.zeros(num_top_spans, 1, device=device), top_pairwise_scores],
                                              dim=1)
            log_marginalized_antecedent_scores2 = torch.logsumexp(
                top_antecedent_scores + torch.log(top_antecedent_gold_labels.to(torch.float)), dim=1)
            log_norm2 = torch.logsumexp(top_antecedent_scores, dim=1)  # [num top spans]
            loss_cm = torch.sum(log_norm2 - log_marginalized_antecedent_scores2)
            if self._cluster_dloss:
                loss += loss_cm
            else:
                loss = loss_cm

        return loss


class CorefDecoderMangoes(torch.nn.Module):

    def __init__(
            self,
            max_top_antecedents: int,
            unary_hdim: int,
            hidden_size: int,
            use_speakers: bool,
            max_training_segments: int,
            coref_metadata_feature_size: int,
            coref_dropout: float,
            coref_higher_order: int,
            bias_in_last_layers: bool,
            coref_num_speakers: int = 2
    ):
        super().__init__()

        self._dropout = coref_dropout
        self.max_top_antecedents: int = max_top_antecedents
        self.max_training_segments: int = max_training_segments
        self.coref_depth: int = coref_higher_order
        self._ignore_speakers = not use_speakers

        # Config Time
        _span_embedding_dim = (hidden_size * 3) + coref_metadata_feature_size
        _final_metadata_size = coref_metadata_feature_size * (2 if self._ignore_speakers else 3)

        # Parameters Time
        self.fast_antecedent_projection = torch.nn.Linear(_span_embedding_dim, _span_embedding_dim)
        self.distance_projection = nn.Linear(coref_metadata_feature_size, 1)
        self.slow_antecedent_scorer = nn.Sequential(
            nn.Linear((_span_embedding_dim * 3) + _final_metadata_size, unary_hdim),
            nn.ReLU(),
            nn.Dropout(self._dropout),
            nn.Linear(unary_hdim, 1, bias=bias_in_last_layers),
        )
        self.slow_antecedent_projection = torch.nn.Linear(_span_embedding_dim * 2, _span_embedding_dim)
        # TODO: wire this up as well
        # metadata embeddings
        # self.use_metadata = use_metadata
        # self.genres = {g: i for i, g in enumerate(genres)}
        # self.genre_embeddings = nn.Embedding(num_embeddings=len(self.genres),
        #                                       embedding_dim=coref_metadata_feature_size)

        self.emb_segment_dist = Utils.make_embedding(max_training_segments, coref_metadata_feature_size)
        self.emb_fast_distance = Utils.make_embedding(num_embeddings=10, embedding_dim=coref_metadata_feature_size)
        self.emb_slow_distance = Utils.make_embedding(num_embeddings=10, embedding_dim=coref_metadata_feature_size)
        if self._ignore_speakers:
            self.emb_same_speaker = None
        else:
            self.emb_same_speaker = Utils.make_embedding(2, coref_metadata_feature_size)

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
        antecedent_distance_buckets = Utils.bucket_distance(antecedent_offsets).to(span_emb.device)  # [cand, cand]
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
        antecedent_distance_buckets = Utils.bucket_distance(top_antecedent_offsets).to(
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
            num_top_spans: int,
            num_segments: int,
            len_segment: int,
            domain: str,
            genre: int,
            device: Union[str, torch.device],
    ):

        num_top_antecedents = min(self.max_top_antecedents, num_top_spans)

        # Start with coarse model
        dummy_scores = torch.zeros([num_top_spans, 1], device=device)
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
            "coref_top_antecedents_score": top_ante_scores,
            "coref_top_antecedents_mask": top_ante_mask,
            # "coref_top_span_cluster_ids": top_span_cluster_ids,
        }
