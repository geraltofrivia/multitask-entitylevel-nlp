"""
    The core model is here. It does Coref. It does NER. It does everything. Come one come all.
    Modularity is so 2021. I'll shoot myself in the foot instead thank you very much.
"""

import math
import random
from typing import List, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertPreTrainedModel

# Local imports
try:
    import _pathfix
except ImportError:
    from . import _pathfix
from config import _SEED_ as SEED
from utils.data import Tasks
from utils.exceptions import AnticipateOutOfMemException, UnknownDomainException
from utils.misc import SerializedBertConfig

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# noinspection PyUnusedLocal
class MangoesMTL(BertPreTrainedModel):

    def __init__(
            self,
            enc_nm: str,
            vocab_size: int,
            hidden_size: int,
            max_span_width: int,
            top_span_ratio: int,
            max_top_antecedents: int,
            coref_dropout: float,
            unary_hdim: int,
            coref_higher_order: int,
            coref_metadata_feature_size: int,
            coref_max_training_segments: int,
            coref_loss_mean: bool,
            task_1: dict,
            task_2: dict,
            bias_in_last_layers: bool = False,
            skip_instance_after_nspan: int = -1,
            *args, **kwargs
    ):

        base_config = SerializedBertConfig(vocab_size=vocab_size)
        super().__init__(base_config)

        # Convert task, task2 to Tasks object again (for now)
        task_1 = Tasks(**task_1)
        task_2 = Tasks(**task_2)

        self.bert = BertModel(base_config, add_pooling_layer=False)
        # self.bert = BertModel.from_pretrained(enc_nm, add_pooling_layer=False)

        ffnn_hidden_size = unary_hdim
        bert_hidden_size = hidden_size

        self.span_attend_projection = torch.nn.Linear(bert_hidden_size, 1)
        span_embedding_dim = (bert_hidden_size * 3) + coref_metadata_feature_size
        self.coref_dropout = coref_dropout
        self.mention_scorer = nn.Sequential(
            nn.Linear(span_embedding_dim, ffnn_hidden_size),
            nn.ReLU(),
            nn.Dropout(coref_dropout),
            nn.Linear(ffnn_hidden_size, 1),
        )
        self.width_scores = nn.Sequential(
            nn.Linear(coref_metadata_feature_size, ffnn_hidden_size),
            nn.ReLU(),
            nn.Dropout(coref_dropout),
            nn.Linear(ffnn_hidden_size, 1),
        )
        self.fast_antecedent_projection = torch.nn.Linear(span_embedding_dim, span_embedding_dim)
        self.slow_antecedent_scorer = nn.Sequential(
            nn.Linear((span_embedding_dim * 3) + (coref_metadata_feature_size * 2), ffnn_hidden_size),
            nn.ReLU(),
            nn.Dropout(coref_dropout),
            nn.Linear(ffnn_hidden_size, 1),
        )
        self.slow_antecedent_projection = torch.nn.Linear(span_embedding_dim * 2, span_embedding_dim)
        # metadata embeddings
        # self.use_metadata = use_metadata
        # self.genres = {g: i for i, g in enumerate(genres)}
        # self.genre_embeddings = nn.Embedding(num_embeddings=len(self.genres), embedding_dim=metadata_feature_size)

        self.distance_embeddings = nn.Embedding(num_embeddings=10, embedding_dim=coref_metadata_feature_size)
        self.slow_distance_embeddings = nn.Embedding(num_embeddings=10, embedding_dim=coref_metadata_feature_size)
        self.distance_projection = nn.Linear(coref_metadata_feature_size, 1)
        # self.same_speaker_embeddings = nn.Embedding(num_embeddings=2, embedding_dim=coref_metadata_feature_size)
        self.span_width_embeddings = nn.Embedding(num_embeddings=max_span_width,
                                                  embedding_dim=coref_metadata_feature_size)
        self.span_width_prior_embeddings = nn.Embedding(num_embeddings=max_span_width,
                                                        embedding_dim=coref_metadata_feature_size)
        self.segment_dist_embeddings = nn.Embedding(num_embeddings=coref_max_training_segments,
                                                    embedding_dim=coref_metadata_feature_size)

        """
            NER Stuff is domain specific.
            Corresponding to domain, we will have individual "final" classifiers for NER.
            
            This of course because the final classes for NERs differ from each other.
            However, this will be done in a two step process.
            
            The two layers of NER will be broken down into a common, and domain specific variant.
        """
        self.unary_ner_common = nn.Sequential(
            nn.Linear(span_embedding_dim, ffnn_hidden_size),
            nn.ReLU(),
            nn.Dropout(coref_dropout),
            # nn.Linear(ffnn_hidden_size, n_classes_ner, bias=bias_in_last_layers)
        )
        self.unary_ner_specific = nn.ModuleDict({
            task.dataset: nn.Linear(ffnn_hidden_size, task.n_classes_ner, bias=bias_in_last_layers)
            for task in [task_1, task_2] if (not task.isempty() and 'ner' in task)
        })

        # Loss management for pruner
        self.pruner_loss = self._rescaling_weights_bce_loss_
        self.ner_loss = nn.functional.cross_entropy

        self.max_span_width = max_span_width
        self.top_span_ratio = top_span_ratio
        self.max_top_antecedents = max_top_antecedents
        self.coref_max_training_segments = coref_max_training_segments
        self.coref_depth = coref_higher_order
        self.coref_loss_mean = coref_loss_mean
        self._skip_instance_after_nspan = skip_instance_after_nspan
        self._tasks_: List[Tasks] = [task_1, task_2]

        self.init_weights()

    def task_separate_gradient_clipping(self):
        # noinspection PyAttributeOutsideInit
        self.clip_grad_norm_ = self.separate_max_norm_base_task

    # noinspection PyProtectedMember
    def is_unweighted(self, task, domain):
        task_obj = None
        for stored_task_obj in self._tasks_:
            if stored_task_obj.dataset == domain:
                task_obj = stored_task_obj

        if task_obj is None:
            raise UnknownDomainException(f"Domain {domain} was probably not passed to this model.")

        return task_obj._task_unweighted_(task)

    def separate_max_norm_base_task(self, max_norm):
        base_params = [p for n, p in self.named_parameters() if "bert" in n]
        task_params = [p for n, p in self.named_parameters() if "bert" not in n]
        torch.nn.utils.clip_grad_norm_(base_params, max_norm)
        torch.nn.utils.clip_grad_norm_(task_params, max_norm)

    @staticmethod
    def _rescaling_weights_bce_loss_(logits, labels, weight: Optional[torch.Tensor] = None):
        # if weights are provided, scale them based on labels
        if weight is not None:
            _weight = torch.zeros_like(labels, dtype=torch.float) + weight[0]
            _weight[labels == 1] = weight[1]
            return nn.functional.binary_cross_entropy_with_logits(logits, labels, _weight)
        else:
            return nn.functional.binary_cross_entropy_with_logits(logits, labels)

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

    def get_slow_antecedent_scores(self, top_span_emb, top_antecedents, top_antecedent_emb, top_antecedent_offsets,
                                   segment_distance):
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
        segment_distance: tensor of size (candidates, antecedents)
            segment distances for each candidate antecedent pair

        Returns
        -------
        tensor of shape (candidates, antecedents)
            antecedent scores
        """
        num_cand, num_ant = top_antecedents.shape
        feature_emb_list = []

        # if self.use_metadata:
        #     top_antecedent_speaker_ids = top_span_speaker_ids[top_antecedents]  # [top_cand, top_ant]
        #     # [top_cand, top_ant]
        #     same_speaker = torch.eq(top_span_speaker_ids.view(-1, 1), top_antecedent_speaker_ids)
        #     speaker_pair_emb = self.same_speaker_embeddings(
        #         torch.arange(start=0, end=2, device=top_span_emb.device))  # [2, feature_size]
        #     feature_emb_list.append(speaker_pair_emb[same_speaker.long()])
        #     genre_embs = genre_emb.view(1, 1, -1).repeat(num_cand, num_ant, 1)  # [top_cand, top_ant, feature_size]
        #     feature_emb_list.append(genre_embs)

        # span distance
        antecedent_distance_buckets = self.bucket_distance(top_antecedent_offsets).to(
            top_span_emb.device)  # [cand, cand]
        bucket_embeddings = self.slow_distance_embeddings(
            torch.arange(start=0, end=10, device=top_span_emb.device))  # [10, feature_size]
        feature_emb_list.append(bucket_embeddings[antecedent_distance_buckets])  # [cand, ant, feature_size]

        # segment distance
        segment_distance_emb = self.segment_dist_embeddings(
            torch.arange(start=0, end=self.coref_max_training_segments, device=top_span_emb.device))
        feature_emb_list.append(segment_distance_emb[segment_distance])  # [cand, ant, feature_size]

        feature_emb = torch.cat(feature_emb_list, 2)  # [cand, ant, emb]
        feature_emb = F.dropout(feature_emb, p=self.coref_dropout,
                                training=self.training)  # [cand, ant, emb]
        target_emb = top_span_emb.unsqueeze(1)  # [cand, 1, emb]
        similarity_emb = top_antecedent_emb * target_emb  # [cand, ant, emb]
        target_emb = target_emb.repeat(1, num_ant, 1)  # [cand, ant, emb]

        pair_emb = torch.cat([target_emb, top_antecedent_emb, similarity_emb, feature_emb], 2)  # [cand, ant, emb]

        return self.slow_antecedent_scorer(pair_emb).squeeze(-1)

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
        bucket_embeddings = F.dropout(self.distance_embeddings(torch.arange(start=0, end=10, device=span_emb.device)),
                                      p=self.coref_dropout, training=self.training)  # [10, feature_size]
        bucket_scores = self.distance_projection(bucket_embeddings)  # [10, 1]
        fast_antecedent_scores += bucket_scores[antecedent_distance_buckets].squeeze(-1)  # [cand, cand]
        # get top antecedent scores/features
        _, top_antecedents = torch.topk(fast_antecedent_scores, num_top_antecedents, sorted=False,
                                        dim=1)  # [cand, num_ant]
        top_antecedents_mask = self.batch_gather(antecedents_mask, top_antecedents)  # [cand, num_ant]
        top_antecedents_fast_scores = self.batch_gather(fast_antecedent_scores, top_antecedents)  # [cand, num_ant]
        top_antecedents_offsets = self.batch_gather(antecedent_offsets, top_antecedents)  # [cand, num_ant]
        return top_antecedents, top_antecedents_mask, top_antecedents_fast_scores, top_antecedents_offsets

    @staticmethod
    def batch_gather(emb, indices):
        batch_size, seq_len = emb.shape
        flattened_emb = emb.view(-1, 1)
        offset = (torch.arange(start=0, end=batch_size, device=indices.device) * seq_len).unsqueeze(1)
        return flattened_emb[indices + offset].squeeze(2)

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
        candidate_labels = torch.mm(labels.unsqueeze(0).float(), same_span.float()).long()  # [1, num_candidates]
        return candidate_labels.squeeze(0)  # [num_candidates]

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
                               p=self.coref_dropout, training=self.training)  # [cand, emb]
        target_emb = F.dropout(span_emb, p=self.coref_dropout, training=self.training)  # [cand, emb]
        return torch.mm(source_emb, target_emb.t())  # [cand, cand]

    def get_span_embeddings(self, hidden_states, span_starts, span_ends):
        """
        Obtains representations of the spans

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
        emb = [hidden_states[span_starts], hidden_states[span_ends]]

        span_width = 1 + span_ends - span_starts  # [num_cand]
        span_width_index = span_width - 1  # [num_cand]
        span_width_emb = self.span_width_embeddings(span_width_index)  # [num_cand, emb]
        span_width_emb = F.dropout(span_width_emb, p=self.coref_dropout, training=self.training)
        emb.append(span_width_emb)

        token_attention_scores = self.get_span_word_attention_scores(hidden_states, span_starts,
                                                                     span_ends)  # [num_cand, num_words]
        attended_word_representations = torch.mm(token_attention_scores, hidden_states)  # [num_cand, emb_size]
        emb.append(attended_word_representations)
        return torch.cat(emb, dim=1)

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

    @staticmethod
    def coref_loss(top_antecedent_scores, top_antecedent_labels):
        """
        Calculate softmax loss

        Parameters
        ----------
        top_antecedent_scores: tensor of size [top_cand, top_ant + 1]
            scores of each antecedent for each mention candidate
        top_antecedent_labels: tensor of size [top_cand, top_ant + 1]
            labels for each antecedent

        Returns
        -------
        tensor of size (num_candidates)
            loss for each mention
        """
        gold_scores = top_antecedent_scores + torch.log(top_antecedent_labels.float())  # [top_cand, top_ant+1]
        marginalized_gold_scores = torch.logsumexp(gold_scores, 1)  # [top_cand]
        log_norm = torch.logsumexp(top_antecedent_scores, 1)  # [top_cand]
        return log_norm - marginalized_gold_scores  # [top_cand]

    def todel_get_predicted_antecedents(self, antecedent_idx, antecedent_scores):
        """ CPU list input """
        predicted_antecedents = []
        for i, idx in enumerate(np.argmax(antecedent_scores, axis=1) - 1):
            if idx < 0:
                predicted_antecedents.append(-1)
            else:
                predicted_antecedents.append(antecedent_idx[i][idx])
        return predicted_antecedents

    def todel_get_predicted_clusters(self, pruned_span_starts, pruned_span_ends, coref_top_antecedents,
                                     coref_top_antecedent_scores):
        """ CPU list input """
        # Get predicted antecedents
        predicted_antecedents = self.todel_get_predicted_antecedents(coref_top_antecedents, coref_top_antecedent_scores)

        # Get predicted clusters
        mention_to_cluster_id = {}
        predicted_clusters = []
        for i, predicted_idx in enumerate(predicted_antecedents):
            if predicted_idx < 0:
                continue
            assert i > predicted_idx, f'span idx: {i}; antecedent idx: {predicted_idx}'
            # Check antecedent's cluster
            antecedent = (int(pruned_span_starts[predicted_idx]), int(pruned_span_ends[predicted_idx]))
            antecedent_cluster_id = mention_to_cluster_id.get(antecedent, -1)
            if antecedent_cluster_id == -1:
                antecedent_cluster_id = len(predicted_clusters)
                predicted_clusters.append([antecedent])
                mention_to_cluster_id[antecedent] = antecedent_cluster_id
            # Add mention to cluster
            mention = (int(pruned_span_starts[i]), int(pruned_span_ends[i]))
            predicted_clusters[antecedent_cluster_id].append(mention)
            mention_to_cluster_id[mention] = antecedent_cluster_id

        predicted_clusters = [tuple(c) for c in predicted_clusters]
        return predicted_clusters, mention_to_cluster_id, predicted_antecedents

    def forward(
            self,
            input_ids: torch.tensor,
            attention_mask: torch.tensor,
            sentence_map: List[int],
            tasks: Iterable[str],
            domain: str,
            *args,
            **kwargs
    ):
        device = input_ids.device
        bert_outputs = self.bert(input_ids, attention_mask)

        mention_doc = bert_outputs[0]  # [num_seg, max_seg_len, emb_len]
        num_seg, max_seg_len, emb_len = mention_doc.shape
        mention_doc = torch.masked_select(mention_doc.view(num_seg * max_seg_len, emb_len),
                                          attention_mask.bool().view(-1, 1)).view(-1, emb_len)  # [num_words, emb_len]
        flattened_ids = torch.masked_select(input_ids, attention_mask.bool()).view(-1)  # [num_words]
        num_words = mention_doc.shape[0]

        # calculate all possible spans
        candidate_starts = torch.arange(start=0,
                                        end=num_words,
                                        device=mention_doc.device).view(-1, 1) \
            .repeat(1, self.max_span_width)  # [num_words, max_span_width]
        candidate_ends = candidate_starts + torch.arange(start=0, end=self.max_span_width,
                                                         device=mention_doc.device).unsqueeze(
            0)  # [num_words, max_span_width]
        candidate_start_sentence_indices = sentence_map[candidate_starts]  # [num_words, max_span_width]
        candidate_end_sentence_indices = sentence_map[
            torch.clamp(candidate_ends, max=num_words - 1)]  # [num_words, max_span_width]
        # find spans that are in the same segment and don't run past the end of the text
        # noinspection PyTypeChecker
        candidate_mask = torch.logical_and(candidate_ends < num_words,
                                           torch.eq(candidate_start_sentence_indices,
                                                    candidate_end_sentence_indices)).view(
            -1).bool()  # [num_words *max_span_width]
        candidate_starts = torch.masked_select(candidate_starts.view(-1), candidate_mask)  # [candidates]
        candidate_ends = torch.masked_select(candidate_ends.view(-1), candidate_mask)  # [candidates]

        """ At this point, if there are more candidates than expected, SKIP this op."""
        if 0 < self._skip_instance_after_nspan < candidate_starts.shape[0]:
            raise AnticipateOutOfMemException(f"There are {candidate_starts.shape[0]} candidates", device)

        # if coref_gold_ends is not None and coref_gold_starts is not None and coref_gold_cluster_ids is not None:
        #     candidate_cluster_ids = self.get_candidate_labels(candidate_starts, candidate_ends,
        #                                                       coref_gold_starts, coref_gold_ends,
        #                                                       coref_gold_cluster_ids)  # [candidates]

        """
            That's the Span Pruner.
            Next we need to break out into Coref and NER parts
        """

        # get span embeddings and mention scores
        span_emb = self.get_span_embeddings(mention_doc, candidate_starts, candidate_ends)  # [candidates, span_emb]

        if 'coref' in tasks or 'pruner' in tasks:

            pruned_candidate_mention_scores = self.mention_scorer(span_emb).squeeze(1)  # [candidates]
            # get span width scores and add to candidate mention scores
            span_width_index = candidate_ends - candidate_starts  # [candidates]
            span_width_emb = self.span_width_prior_embeddings(span_width_index)  # [candidates, emb]
            pruned_candidate_mention_scores += self.width_scores(span_width_emb).squeeze(1)  # [candidates]

            # get beam size
            num_top_mentions = int(float(num_words * self.top_span_ratio))
            num_top_antecedents = min(self.max_top_antecedents, num_top_mentions)

            # get top mention scores and sort by sort by span order
            pruned_span_indices = self.extract_spans(candidate_starts, candidate_ends, pruned_candidate_mention_scores,
                                                     num_top_mentions)
            pruned_span_starts = candidate_starts[pruned_span_indices]  # [top_cand]
            pruned_span_ends = candidate_ends[pruned_span_indices]  # [top_cand]
            top_span_emb = span_emb[pruned_span_indices]  # [top_cand, span_emb]
            top_span_mention_scores = pruned_candidate_mention_scores[pruned_span_indices]  # [top_cand]

            # # COREF Specific things
            # if coref_gold_ends is not None and coref_gold_starts is not None and coref_gold_cluster_ids is not None:
            #     # noinspection PyUnboundLocalVariable
            #     top_span_cluster_ids = candidate_cluster_ids[pruned_span_indices]  # [top_cand]

            # course to fine pruning
            dummy_scores = torch.zeros([num_top_mentions, 1], device=pruned_span_indices.device)  # [top_cand, 1]
            top_antecedents, top_antecedents_mask, top_ante_fast_scores, top_antecedent_offsets = \
                self.coarse_to_fine_pruning(top_span_emb, top_span_mention_scores, num_top_antecedents)

            num_segments = input_ids.shape[0]
            segment_length = input_ids.shape[1]
            word_segments = torch.arange(start=0, end=num_segments, device=input_ids.device).view(-1, 1).repeat(
                [1, segment_length])  # [segments, segment_len]
            flat_word_segments = torch.masked_select(word_segments.view(-1), attention_mask.bool().view(-1))
            mention_segments = flat_word_segments[pruned_span_starts].view(-1, 1)  # [top_cand, 1]
            antecedent_segments = flat_word_segments[pruned_span_starts[top_antecedents]]  # [top_cand, top_ant]
            segment_distance = torch.clamp(mention_segments - antecedent_segments, 0,
                                           self.coref_max_training_segments - 1)  # [top_cand, top_ant]

            # calculate final slow scores
            for i in range(self.coref_depth):
                top_ante_emb = top_span_emb[top_antecedents]  # [top_cand, top_ant, emb]
                top_ante_scores = top_ante_fast_scores + self.get_slow_antecedent_scores(top_span_emb,
                                                                                         top_antecedents,
                                                                                         top_ante_emb,
                                                                                         top_antecedent_offsets,
                                                                                         segment_distance)
                # [top_cand, top_ant]
                top_ante_weights = F.softmax(
                    torch.cat([dummy_scores, top_ante_scores], 1), dim=-1
                )  # [top_cand, top_ant + 1]
                top_ante_emb = torch.cat([top_span_emb.unsqueeze(1), top_ante_emb], 1)  # [top_cand, top_ant + 1, emb]
                attended_span_emb = torch.sum(top_ante_weights.unsqueeze(2) * top_ante_emb, 1)  # [top_cand, emb]
                gate_vectors = torch.sigmoid(
                    self.slow_antecedent_projection(torch.cat([top_span_emb, attended_span_emb], 1)))  # [top_cand, emb]
                top_span_emb = gate_vectors * attended_span_emb + (1 - gate_vectors) * top_span_emb  # [top_cand, emb]

            # noinspection PyUnboundLocalVariable
            top_ante_scores = torch.cat([dummy_scores, top_ante_scores], 1)

            coref_specific = {
                "coref_top_antecedents": top_antecedents,
                "coref_top_antecedent_scores": top_ante_scores,
                "coref_top_antecedents_mask": top_antecedents_mask,
                "pruned_candidate_mention_scores": pruned_candidate_mention_scores,
                "pruned_span_starts": pruned_span_starts,
                "pruned_span_ends": pruned_span_ends,
                "pruned_span_indices": pruned_span_indices,
                # "coref_top_span_cluster_ids": top_span_cluster_ids,
            }
        else:
            coref_specific = {}

        if 'ner' in tasks:
            # We just need span embeddings here

            fc1 = self.unary_ner_common(span_emb)

            # Depending on the domain, select the right decoder
            logits = self.unary_ner_specific[domain](fc1)
            logits = torch.nn.functional.softmax(logits, dim=1)
            ner_specific = {"ner_logits": logits}

        else:
            ner_specific = {}

        # noinspection PyUnboundLocalVariable
        return {
            "candidate_starts": candidate_starts,
            "candidate_ends": candidate_ends,
            "flattened_ids": flattened_ids,
            **coref_specific,
            **ner_specific
        }

    def pred_with_labels(
            self,
            input_ids: torch.tensor,
            attention_mask: torch.tensor,
            token_type_ids: torch.tensor,
            sentence_map: List[int],
            word_map: List[int],
            n_words: int,
            n_subwords: int,
            tasks: Iterable[str],
            domain: str,
            coref: dict = None,
            ner: dict = None,
            pruner: dict = None,
            *args, **kwargs
    ):

        # Run the model.
        # this will get all outputs regardless of whatever tasks are thrown at ya
        predictions = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            sentence_map=sentence_map,
            word_map=word_map,
            n_words=n_words,
            domain=domain,
            tasks=tasks,
            n_subwords=n_subwords,
            coref_gold_starts=coref.get('gold_starts', None),
            coref_gold_ends=coref.get('gold_ends', None),
            coref_gold_cluster_ids=coref.get('gold_label_values', None),
            pruner_gold_labels=pruner.get('gold_labels', None),
            ner_gold_labels=ner.get('gold_labels', None)
        )

        candidate_starts = predictions['candidate_starts']
        candidate_ends = predictions['candidate_ends']

        outputs = {
            "loss": {},
            "num_candidates": candidate_starts.shape[0]
        }

        if "pruner" in tasks:
            pred_starts = predictions["pruned_span_starts"]
            pred_ends = predictions["pruned_span_ends"]
            pred_indices = predictions["pruned_span_indices"]
            gold_starts = pruner["gold_starts"]
            gold_ends = pruner["gold_ends"]

            logits_after_pruning = torch.zeros_like(candidate_starts, device=candidate_starts.device, dtype=torch.float)
            logits_after_pruning[pred_indices] = 1

            # Find which candidates (in the unpruned candidate space) correspond to actual gold candidates
            cand_gold_starts = torch.eq(gold_starts.repeat(candidate_starts.shape[0], 1),
                                        candidate_starts.unsqueeze(1))
            cand_gold_ends = torch.eq(gold_ends.repeat(candidate_ends.shape[0], 1),
                                      candidate_ends.unsqueeze(1))
            # noinspection PyArgumentList
            labels_after_pruning = torch.logical_and(cand_gold_starts, cand_gold_ends).any(dim=1).float()

            # Calculate the loss !
            if self.is_unweighted(task='pruner', domain=domain):
                pruner_loss = self.pruner_loss(logits_after_pruning, labels_after_pruning)
            else:
                pruner_loss = self.pruner_loss(logits_after_pruning, labels_after_pruning, weight=pruner["weights"])

            # DEBUG
            try:
                assert not torch.isnan(pruner_loss), \
                    f"Found nan in loss. Here are some details - \n\tLogits shape: {logits_after_pruning.shape}, " \
                    f"\n\tLabels shape: {labels_after_pruning.shape}, " \
                    f"\n\tNonZero lbls: {labels_after_pruning[labels_after_pruning != 0].shape}"
            except AssertionError:
                print('potato')

            outputs["loss"]["pruner"] = pruner_loss
            outputs["pruner"] = {"logits": logits_after_pruning, "labels": labels_after_pruning}

            # labels_after_pruning = self.get_candidate_labels(candidate_starts, candidate_ends,
            #                                                  gold_starts, gold_ends,)

            # # Repeat pred to gold dims and then collapse the eq.
            # start_eq = torch.eq(pred_starts.repeat(gold_starts.shape[0],1), gold_starts.unsqueeze(1)).any(dim=0)
            # end_eq = torch.eq(pred_ends.repeat(gold_ends.shape[0],1), gold_ends.unsqueeze(1)).any(dim=0)
            # same_spans = torch.logical_and(start_eq, end_eq)

        if "coref" in tasks:

            # top_span_cluster_ids = predictions["coref_top_span_cluster_ids"]
            top_antecedents = predictions["coref_top_antecedents"]
            top_antecedents_mask = predictions["coref_top_antecedents_mask"]
            top_antecedent_scores = predictions["coref_top_antecedent_scores"]
            top_span_indices = predictions["pruned_span_indices"]

            gold_starts = coref["gold_starts"]
            gold_ends = coref["gold_ends"]
            gold_cluster_ids = coref["gold_label_values"]

            gold_candidate_cluster_ids = self.get_candidate_labels(candidate_starts, candidate_ends,
                                                                   gold_starts, gold_ends,
                                                                   gold_cluster_ids)
            top_span_cluster_ids = gold_candidate_cluster_ids[top_span_indices]

            # Unpack everything we need
            top_antecedent_cluster_ids = top_span_cluster_ids[top_antecedents]  # [top_cand, top_ant]
            top_antecedent_cluster_ids += torch.log(top_antecedents_mask.float()).int()  # [top_cand, top_ant]
            same_cluster_indicator = torch.eq(top_antecedent_cluster_ids,
                                              top_span_cluster_ids.unsqueeze(1))  # [top_cand, top_ant]
            non_dummy_indicator = (top_span_cluster_ids > 0).unsqueeze(1)  # [top_cand, 1]
            pairwise_labels = torch.logical_and(same_cluster_indicator, non_dummy_indicator)  # [top_cand, top_ant]
            # noinspection PyArgumentList
            dummy_labels = torch.logical_not(pairwise_labels.any(1, keepdims=True))  # [top_cand, 1]
            top_antecedent_labels = torch.cat([dummy_labels, pairwise_labels], 1)  # [top_cand, top_ant + 1]
            coref_loss = self.coref_loss(top_antecedent_scores, top_antecedent_labels)  # [top_cand]

            if self.coref_loss_mean:
                coref_loss = torch.mean(coref_loss)
            else:
                coref_loss = torch.sum(coref_loss)

            # And now, code that helps with eval
            gold_clusters = {}
            _cluster_ids = coref["gold_label_values"]
            _gold_starts = coref["gold_starts"]
            _gold_ends = coref["gold_ends"]
            ids = predictions["flattened_ids"]
            pruned_span_starts = predictions["pruned_span_starts"]
            pruned_span_ends = predictions["pruned_span_ends"]
            top_antecedents = predictions["coref_top_antecedents"]

            for i in range(len(_cluster_ids)):
                assert len(_cluster_ids) == len(_gold_starts) == len(_gold_ends)
                cid = _cluster_ids[i].item()
                if cid in gold_clusters:
                    gold_clusters[cid].append((_gold_starts[i].item(),
                                               _gold_ends[i].item()))
                else:
                    gold_clusters[cid] = [(_gold_starts[i].item(),
                                           _gold_ends[i].item())]

            gold_clusters = [tuple(v) for v in gold_clusters.values()]
            mention_to_gold = {}
            for c in gold_clusters:
                for mention in c:
                    mention_to_gold[mention] = c

            top_indices = torch.argmax(top_antecedent_scores, dim=-1, keepdim=False)
            mention_indices = []
            antecedent_indices = []
            predicted_antecedents = []
            for i in range(len(pruned_span_ends)):
                if top_indices[i] > 0:
                    mention_indices.append(i)
                    antecedent_indices.append(top_antecedents[i][top_indices[i] - 1].item())
                    predicted_antecedents.append(top_indices[i] - 1)

            cluster_sets = []
            for i in range(len(mention_indices)):
                new_cluster = True
                for j in range(len(cluster_sets)):
                    if mention_indices[i] in cluster_sets[j] or antecedent_indices[i] in cluster_sets[j]:
                        cluster_sets[j].add(mention_indices[i])
                        cluster_sets[j].add(antecedent_indices[i])
                        new_cluster = False
                        break
                if new_cluster:
                    cluster_sets.append({mention_indices[i], antecedent_indices[i]})

            cluster_dicts = []
            clusters = []
            for i in range(len(cluster_sets)):
                cluster_mentions = sorted(list(cluster_sets[i]))
                current_ids = []
                current_start_end = []
                for mention_index in cluster_mentions:
                    current_ids.append(ids[pruned_span_starts[mention_index]:pruned_span_ends[mention_index] + 1])
                    current_start_end.append(
                        (pruned_span_starts[mention_index].item(), pruned_span_ends[mention_index].item()))
                cluster_dicts.append({"cluster_ids": current_ids})
                clusters.append(tuple(current_start_end))

            mention_to_predicted = {}
            for c in clusters:
                for mention in c:
                    mention_to_predicted[mention] = c

            coref_eval = {
                "clusters": clusters,
                "gold_clusters": gold_clusters,
                "mention_to_predicted": mention_to_predicted,
                "mention_to_gold": mention_to_gold
            }

            outputs["loss"]["coref"] = coref_loss
            outputs["coref"] = coref_eval

        if "ner" in tasks:
            ner_gold_starts = ner["gold_starts"]
            ner_gold_ends = ner["gold_ends"]
            ner_gold_label_values = ner["gold_label_values"]
            ner_logits = predictions["ner_logits"]  # n_spans, n_classes
            ner_labels = self.get_candidate_labels(candidate_starts, candidate_ends,
                                                   ner_gold_starts, ner_gold_ends,
                                                   ner_gold_label_values)

            # Calculating the loss
            # if self.ner_unweighted:
            if self.is_unweighted(task='ner', domain=domain):
                ner_loss = self.ner_loss(ner_logits, ner_labels)
            else:
                ner_loss = self.ner_loss(ner_logits, ner_labels, weight=ner["weights"])

            assert not torch.isnan(ner_loss), \
                f"Found nan in loss. Here are some details - \n\tLogits shape: {ner_logits.shape}, " \
                f"\n\tLabels shape: {ner_labels.shape}, " \
                f"\n\tNonZero lbls: {ner_labels[ner_labels != 0].shape}"

            outputs["loss"]["ner"] = ner_loss
            outputs["ner"] = {"logits": ner_logits, "labels": ner_labels}

        return outputs


if __name__ == "__main__":
    ...
