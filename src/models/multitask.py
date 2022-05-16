"""
    The core model is here. It does Coref. It does NER. It does everything. Come one come all.
    Modularity is so 2021. I'll shoot myself in the foot instead thank you very much.
"""

import math
import torch
import transformers
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Iterable, Optional

# Local imports
try:
    import _pathfix
except ImportError:
    from . import _pathfix


class BasicMTL(nn.Module):
    def __init__(self, enc_modelnm: str, config: transformers.BertConfig, n_classes_ner: int = 0):
        super().__init__()

        self.config = config
        self.n_max_len: int = self.config.max_position_embeddings
        self.n_hid_dim: int = self.config.hidden_size
        self.n_classes_ner = n_classes_ner
        n_coref_metadata_dim = self.config.coref_metadata_feature_size

        # Encoder responsible for giving contextual vectors to subword tokens
        self.encoder = transformers.BertModel.from_pretrained(enc_modelnm).to(config.device)

        # Span width embeddings give a fix dim score to width of spans (usually 1 to config.max_span_width (~5))
        self.span_width_embeddings = nn.Embedding(
            num_embeddings=self.config.max_span_width, embedding_dim=n_coref_metadata_dim,
        ).to(config.device)
        self.segment_dist_embeddings = nn.Embedding(num_embeddings=config.coref_max_training_segments,
                                                    embedding_dim=n_coref_metadata_dim).to(config.device)

        # Used to push 768dim contextual vecs to 1D vectors for attention computation during span embedding creation
        self.span_attend_projection = torch.nn.Linear(config.hidden_size, 1).to(config.device)

        self.distance_embeddings = nn.Embedding(num_embeddings=10, embedding_dim=n_coref_metadata_dim).to(config.device)
        self.slow_distance_embeddings = nn.Embedding(num_embeddings=10, embedding_dim=n_coref_metadata_dim).to(
            config.device)
        self.distance_projection = nn.Linear(n_coref_metadata_dim, 1).to(config.device)

        # Mention scorer (Unary, hdim) takes span representations and passes them through a 2 layer FF NN to score
        #   whether they are valid spans or not.
        #   NOTE: its use is conflated because it tries to do two things
        #       (i) find syntactically incoherent spans
        #       (ii) find spans that are not anaphoric
        span_embedding_dim = 3 * config.hidden_size + n_coref_metadata_dim
        self.unary_coref = nn.Sequential(
            nn.Linear(span_embedding_dim, config.unary_hdim),
            nn.ReLU(),
            nn.Dropout(config.coref_dropout),
            nn.Linear(config.unary_hdim, 1, bias=self.config.bias_in_last_layers),
        ).to(config.device)

        self.binary_coref_slow = nn.Sequential(
            nn.Linear((span_embedding_dim * 3) + 2 * n_coref_metadata_dim, config.unary_hdim),
            nn.ReLU(),
            nn.Dropout(config.coref_dropout),
            nn.Linear(config.unary_hdim, 1, bias=self.config.bias_in_last_layers),
        ).to(config.device)

        # self.binary_coref = nn.Sequential(
        #     nn.Linear((span_embedding_dim * 3), config.binary_hdim),
        #     nn.ReLU(),
        #     nn.Dropout(config.coref_dropout),
        #     nn.Linear(config.binary_hdim, 1),
        # ).to(config.device)

        self.unary_ner = nn.Sequential(
            nn.Linear(span_embedding_dim, config.unary_hdim),
            nn.ReLU(),
            nn.Dropout(config.coref_dropout),
            nn.Linear(config.unary_hdim, n_classes_ner, bias=self.config.bias_in_last_layers),
        ).to(config.device)

        self.fast_antecedent_projection = torch.nn.Linear(span_embedding_dim, span_embedding_dim).to(config.device)
        # TODO: why are no grads coming here?
        self.slow_antecedent_projection = torch.nn.Linear(span_embedding_dim * 2, span_embedding_dim).to(config.device)

        # Initialising Losses
        self.ner_loss = nn.functional.cross_entropy
        self.pruner_loss = self._rescaling_weights_bce_loss_
        self.coref_loss_agg = torch.mean if config.coref_loss_mean else torch.sum
        try:
            self.ner_unweighted = self.config.ner_unweighted
        except AttributeError:
            self.ner_unweighted = False
        try:
            self.pruner_unweighted = self.config.pruner_unweighted
        except AttributeError:
            self.pruner_unweighted = False

    @staticmethod
    def _rescaling_weights_bce_loss_(logits, labels, weight: Optional[torch.Tensor] = None):
        # if weights are provided, scale them based on labels
        if weight is not None:
            _weight = torch.zeros_like(labels, dtype=torch.float) + weight[0]
            _weight[labels == 1] = weight[1]
            return nn.functional.binary_cross_entropy_with_logits(logits, labels, _weight)
        else:
            return nn.functional.binary_cross_entropy_with_logits(logits, labels)

    def _get_span_word_attention_scores_(self, hidden_states, span_starts, span_ends):
        """
        CODE copied from https://gitlab.inria.fr/magnet/mangoes/-/blob/coref_exp/mangoes/modeling/coref.py#L564
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
        document_range = (
            torch.arange(
                start=0, end=hidden_states.shape[0], device=self.config.device
            ).unsqueeze(0).repeat(span_starts.shape[0], 1)
        )  # [num_cand, num_words]
        # noinspection PyTypeChecker
        token_mask = torch.logical_and(
            document_range >= span_starts.unsqueeze(1),
            document_range <= span_ends.unsqueeze(1),
        )  # [num_cand, num_words]
        token_attn = (
            self.span_attend_projection(hidden_states).squeeze(1).unsqueeze(0)
        )  # [1, num_words]
        token_attn = F.softmax(
            torch.log(token_mask.float()) + token_attn, 1
        )  # [num_cand, num_words]span
        return token_attn

    @staticmethod
    def _bucket_distance_(distances):
        """
        Places the given values (designed for distances) into 10 semi-log-scale buckets:
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
    def _batch_gather_(emb, indices):
        batch_size, seq_len = emb.shape
        flattened_emb = emb.view(-1, 1)
        offset = (torch.arange(start=0, end=batch_size, device=indices.device) * seq_len).unsqueeze(1)
        return flattened_emb[indices + offset].squeeze(2)

    def _get_span_embeddings_(self, hidden_states, span_starts, span_ends):
        """
        CODE copied from https://gitlab.inria.fr/magnet/mangoes/-/blob/coref_exp/mangoes/modeling/coref.py#L535

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
        span_width_emb = F.dropout(
            span_width_emb, p=self.config.coref_dropout, training=self.training
        )
        emb.append(span_width_emb)

        token_attention_scores = self._get_span_word_attention_scores_(
            hidden_states, span_starts, span_ends
        )  # [num_cand, num_words]
        attended_word_representations = torch.mm(
            token_attention_scores, hidden_states
        )  # [num_cand, emb_size]
        emb.append(attended_word_representations)
        return torch.cat(emb, dim=1)

    def _get_fast_antecedent_scores_(self, span_emb):
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
                               p=self.config.coref_dropout, training=self.training)  # [cand, emb]
        target_emb = F.dropout(span_emb, p=self.config.coref_dropout, training=self.training)  # [cand, emb]
        return torch.mm(source_emb, target_emb.t())  # [cand, cand]

    def _get_slow_antecedent_scores_(
            self,
            top_span_emb: torch.tensor,
            top_antecedents: torch.tensor,
            top_antecedent_emb: torch.tensor,
            top_antecedent_offsets: torch.tensor,
            segment_distance: torch.tensor):
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
        #     same_speaker = torch.eq(top_span_speaker_ids.view(-1, 1), top_antecedent_speaker_ids)  # [top_cand, top_ant]
        #     speaker_pair_emb = self.same_speaker_embeddings(
        #         torch.arange(start=0, end=2, device=top_span_emb.device))  # [2, feature_size]
        #     feature_emb_list.append(speaker_pair_emb[same_speaker.long()])
        #     genre_embs = genre_emb.view(1, 1, -1).repeat(num_cand, num_ant, 1)  # [top_cand, top_ant, feature_size]
        #     feature_emb_list.append(genre_embs)

        # span distance
        antecedent_distance_buckets = self._bucket_distance_(top_antecedent_offsets).to(
            top_span_emb.device)  # [cand, cand]
        bucket_embeddings = self.slow_distance_embeddings(
            torch.arange(start=0, end=10, device=top_span_emb.device))  # [10, feature_size]
        feature_emb_list.append(bucket_embeddings[antecedent_distance_buckets])  # [cand, ant, feature_size]

        # segment distance
        segment_distance_emb = self.segment_dist_embeddings(
            torch.arange(start=0, end=self.config.coref_max_training_segments, device=top_span_emb.device))
        feature_emb_list.append(segment_distance_emb[segment_distance])  # [cand, ant, feature_size]

        feature_emb = torch.cat(feature_emb_list, 2)  # [cand, ant, emb]
        feature_emb = F.dropout(feature_emb, p=self.config.coref_dropout,
                                training=self.training)  # [cand, ant, emb]
        target_emb = top_span_emb.unsqueeze(1)  # [cand, 1, emb]
        similarity_emb = top_antecedent_emb * target_emb  # [cand, ant, emb]
        target_emb = target_emb.repeat(1, num_ant, 1)  # [cand, ant, emb]

        pair_emb = torch.cat([target_emb, top_antecedent_emb, similarity_emb, feature_emb], 2)  # [cand, ant, emb]

        return self.binary_coref_slow(pair_emb).squeeze(-1)

    def _coarse_to_fine_pruning_(self, span_emb, mention_scores, num_top_antecedents):
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
        fast_antecedent_scores += self._get_fast_antecedent_scores_(span_emb)  # [cand, cand]
        # add distance scores
        antecedent_distance_buckets = self._bucket_distance_(antecedent_offsets).to(span_emb.device)  # [cand, cand]
        bucket_embeddings = F.dropout(self.distance_embeddings(torch.arange(start=0, end=10, device=span_emb.device)),
                                      p=self.config.coref_dropout, training=self.training)  # [10, feature_size]
        bucket_scores = self.distance_projection(bucket_embeddings)  # [10, 1]
        fast_antecedent_scores += bucket_scores[antecedent_distance_buckets].squeeze(-1)  # [cand, cand]
        # get top antecedent scores/features
        _, top_antecedents = torch.topk(fast_antecedent_scores, num_top_antecedents, sorted=False,
                                        dim=1)  # [cand, num_ant]
        top_antecedents_mask = self._batch_gather_(antecedents_mask, top_antecedents)  # [cand, num_ant]
        top_antecedents_fast_scores = self._batch_gather_(fast_antecedent_scores, top_antecedents)  # [cand, num_ant]
        top_antecedents_offsets = self._batch_gather_(antecedent_offsets, top_antecedents)  # [cand, num_ant]
        return top_antecedents, top_antecedents_mask, top_antecedents_fast_scores, top_antecedents_offsets

    @staticmethod
    def _extract_spans_(
            candidate_starts: torch.Tensor,
            candidate_ends: torch.Tensor,
            candidate_span_scores: torch.Tensor,
            num_top_mentions: int,
    ):
        """
            Extracts the candidate spans with the highest mention scores,
                whose spans do not overlap other spans ...
                TODO: (mais pourquoi avoir cette restriction)

        :param candidate_starts: n_cands,
        :param candidate_ends: n_cands,
        :param candidate_span_scores: n_cands,
        :param num_top_mentions: int
        :return: span indices of the non crossing spans with highest mention scores
        """
        # sort based on mention scores
        top_span_indices = torch.argsort(candidate_span_scores, descending=True)
        # add highest scores that don't cross
        end_to_earliest_start = {}
        start_to_latest_end = {}
        selected_spans = []
        current_span_index = 0
        # noinspection PyArgumentList
        while len(selected_spans) < num_top_mentions and \
                current_span_index < candidate_starts.size(0):
            ind = top_span_indices[current_span_index]
            any_crossing = False
            cand_start = candidate_starts[ind].item()
            cand_end = candidate_ends[ind].item()
            for j in range(cand_start, cand_end + 1):
                """
                    Making a change here:
                        if 338-340 has been seen, we would not block out 338-339 since previously,
                            we were concerned with only hard subsumption. Now, this one will get filtered out as well.
                """
                if j >= cand_start and j in start_to_latest_end and start_to_latest_end[j] > cand_end:
                    any_crossing = True
                    break
                if j <= cand_end and j in end_to_earliest_start and end_to_earliest_start[j] < cand_start:
                    any_crossing = True
                    break
            if not any_crossing:
                selected_spans.append(ind)
                if (
                        cand_start not in start_to_latest_end
                        or start_to_latest_end[cand_start] < cand_end
                ):
                    start_to_latest_end[cand_start] = cand_end
                if (
                        cand_end not in end_to_earliest_start
                        or end_to_earliest_start[cand_end] > cand_start
                ):
                    end_to_earliest_start[cand_end] = cand_start
            current_span_index += 1
        return torch.tensor(sorted(selected_spans)).long().to(candidate_starts.device)

    @staticmethod
    def get_candidate_labels_mangoes(
            candidate_starts, candidate_ends, labeled_starts, labeled_ends, labels
    ):
        """

        Function used to compare gold starts/ends and candidate starts/ends to create a correspondence index
            (whether a candidate is correct or not, BASED on the labels given in the last arg)

        CODE copied from https://gitlab.inria.fr/magnet/mangoes/-/blob/coref_exp/mangoes/modeling/coref.py#L470
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
        same_start = torch.eq(
            labeled_starts.unsqueeze(1), candidate_starts.unsqueeze(0)
        )  # [num_labeled, num_candidates]
        same_end = torch.eq(
            labeled_ends.unsqueeze(1), candidate_ends.unsqueeze(0)
        )  # [num_labeled, num_candidates]
        same_span = torch.logical_and(
            same_start, same_end
        )  # [num_labeled, num_candidates]

        # type casting in next line is due to torch not supporting matrix multiplication for Long tensors
        candidate_labels = torch.mm(
            labels.unsqueeze(0).float(), same_span.float()
        ).long()  # [1, num_candidates]
        return candidate_labels.squeeze(0)  # [num_candidates]

    @staticmethod
    def coref_softmax_loss(top_antecedent_scores, top_antecedent_labels):
        """
        Code copied from https://gitlab.inria.fr/magnet/mangoes/-/blob/coref_exp/mangoes/modeling/coref.py#L587
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
        gold_scores = top_antecedent_scores + torch.log(
            top_antecedent_labels.float()
        )  # [top_cand, top_ant+1]
        marginalized_gold_scores = torch.logsumexp(gold_scores, 1)  # [top_cand]
        log_norm = torch.logsumexp(top_antecedent_scores, 1)  # [top_cand]
        return log_norm - marginalized_gold_scores  # [top_cand]

    # @staticmethod
    # def ner_loss(logits: torch.tensor, labels: torch.tensor):
    #     return torch.nn.functional.nll_loss(input=logits, target=labels)

    # noinspection PyUnusedLocal
    def pruner(
            self,
            n_subwords: int,
            candidate_starts: torch.tensor,
            candidate_ends: torch.tensor,
            candidate_span_embeddings: torch.tensor,
            *args, **kwargs
    ):
        """
            Debugging notes: this seems reasonably correct. Manually went through it thrice or more.

            This takes span repr vectors and passes them through a unary classifier, scores them and returns
                1. logits - score for every span ( #n_cand, )
                1. top_span_indices: based on num of selected spans  ( #top_cand,)
                    (beam search based on candidates and overlaps and shit)
                2. top_span_starts: indices imposed on original candidate span starts ( #top_cand,)
                3. top_span_ends: same as above ( #top_cand,)
                4. top_span_scores (if needed) same as above but has logits' values for each selected span ( #top_cand,)
                5. top_span_emb: candidate span vecs selected based on the beam search thing (#top_cand, span_emb_dim)
        """
        """
            Step 3: Span Pruning (for Coref)
            
            candidate_span_scores: n_cands
            Pass the span embeddings through a 2L FF NN (w dropout) and get a scalar score indicating the models' 
                predictions about them. We now need through sort through the spans and keep top-k.
                
            num_top_antecedents: int
            Getting top-k: Either we stick with the hard limit (max_top_antecedents = ~50) 
                or we keep antecedents proportional to doc length (say 40% of doc length) 
                if we have under 50 antecedents to begin with 
        """
        # TODO: some gold stuff
        # TODO: some meta stuff

        # For Coref task, we pass the span embeddings through the span pruner
        # n_cands
        candidate_span_scores = self.unary_coref(candidate_span_embeddings).squeeze(1)
        # TODO: joe adds span width embeddings here but we skip it for simplicity's sake.

        # get beam size
        n_top_spans = int(float(n_subwords * self.config.top_span_ratio))
        n_top_ante = min(self.config.max_top_antecedents, n_top_spans)

        # Get top mention scores and sort by span order (avoiding overlaps in selected spans)
        top_span_indices = self._extract_spans_(
            candidate_starts, candidate_ends, candidate_span_scores, n_top_spans
        )
        top_span_starts = candidate_starts[top_span_indices]  # [top_cand]
        top_span_ends = candidate_ends[top_span_indices]  # [top_cand]
        top_span_emb = candidate_span_embeddings[
            top_span_indices
        ]  # [top_cand, span_emb]
        top_span_mention_scores = candidate_span_scores[top_span_indices]  # [top_cand]

        # Make another field which is one hot in cand. space, indicating the top_span_indices
        logits_after_pruning = torch.zeros_like(candidate_starts, device=self.config.device, dtype=torch.float)
        logits_after_pruning[top_span_indices] = 1

        return {
            "logits": candidate_span_scores,
            "top_span_starts": top_span_starts,
            "top_span_ends": top_span_ends,
            "top_span_mention_scores": top_span_mention_scores,
            "top_span_emb": top_span_emb,
            "n_top_spans": n_top_spans,
            "n_top_antecedents": n_top_ante,
            "top_span_indices": top_span_indices,
            "logits_after_pruning": logits_after_pruning
        }

    # noinspection PyUnusedLocal
    def coref(
            self,
            input_ids: torch.tensor,
            encoded: torch.tensor,
            attention_mask: torch.tensor,
            top_span_starts: torch.tensor,
            top_span_emb: torch.tensor,
            top_span_mention_scores: torch.tensor,
            n_top_spans: int,
            n_top_antecedents: int,
            *args, **kwargs
    ):
        """
            This is an adapted but largely unchanged code of BertForCoreferenceResolutionBase in mangoes,
                with c2f elements but no higher order stuff and
                TODO: no metadata finagling as of now. we'll include this soon-ish.
        :return:
            {
                "top_antecedents": tensor: indices in top span candidates of top antecedents for each mention
                ...
                # TODO: fill this
            }
        """

        # c2f block
        dummy_scores = torch.zeros([n_top_spans, 1], device=self.config.device)  # [top_cand, 1]
        top_antecedents, top_antecedents_mask, top_antecedents_fast_scores, top_antecedent_offsets = \
            self._coarse_to_fine_pruning_(top_span_emb, top_span_mention_scores, n_top_antecedents)

        num_segments = input_ids.shape[0]
        segment_length = input_ids.shape[1]
        flat_word_segments = torch.arange(start=0, end=num_segments, device=input_ids.device).view(-1, 1).repeat(
            [1, segment_length]).view(-1)  # [segments, segment_len]
        # flat_word_segments = torch.masked_select(word_segments.view(-1), attention_mask.bool().view(-1))
        mention_segments = flat_word_segments[top_span_starts].view(-1, 1)  # [top_cand, 1]
        antecedent_segments = flat_word_segments[top_span_starts[top_antecedents]]  # [top_cand, top_ant]
        segment_distance = torch.clamp(mention_segments - antecedent_segments, 0,
                                       self.config.coref_max_training_segments - 1)  # [top_cand, top_ant]

        # calculate final slow scores (this was a higher order for loop that we paved over)
        top_antecedent_emb = top_span_emb[top_antecedents]  # [top_cand, top_ant, emb]
        top_antecedent_scores = top_antecedents_fast_scores + \
                                self._get_slow_antecedent_scores_(
                                    top_span_emb,
                                    top_antecedents,
                                    top_antecedent_emb,
                                    top_antecedent_offsets,
                                    segment_distance
                                )  # [top_cand, top_ant]

        top_antecedent_weights = F.softmax(
            torch.cat([dummy_scores, top_antecedent_scores], 1), dim=-1)  # [top_cand, top_ant + 1]
        top_antecedent_emb = torch.cat([top_span_emb.unsqueeze(1), top_antecedent_emb],
                                       1)  # [top_cand, top_ant + 1, emb]
        attended_span_emb = torch.sum(top_antecedent_weights.unsqueeze(2) * top_antecedent_emb,
                                      1)  # [top_cand, emb]
        gate_vectors = torch.sigmoid(
            self.slow_antecedent_projection(torch.cat([top_span_emb, attended_span_emb], 1)))  # [top_cand, emb]
        top_span_emb = gate_vectors * attended_span_emb + (1 - gate_vectors) * top_span_emb  # [top_cand, emb]

        top_antecedent_scores = torch.cat([dummy_scores, top_antecedent_scores], 1)

        # This is the code block that i spent months debugging but couldn't fix so I've carpeted it over.
        # Not deleting it because... sentiments.
        #
        # """
        #     Step 4: Filtering antecedents for each anaphor
        #     1. The antecedent must occur before the anaphor: modeled by a lower triangular identity matrix based mask
        #     2. Only top K prev. antecedents are considered. They are sorted based on their mention scores.
        #         For this: we put the mask over mention scores and argsort to get top indices.
        #         These indices are then set as one, rest as zero in a new mask. E.g.
        #
        #         Let there are 4 spans. k (max antecedents to be considered) is 2
        #         mention_scores = [2,4,-1,3]
        #         all_prev = [
        #             [1, 0, 0, 0],
        #             [1, 1, 0, 0],
        #             [1, 1, 1, 0],
        #             [1, 1, 1, 1]
        #         ] # i.e. for the first span, the candidates must be only 1;
        #         # for third, the first three spans are candidates .. etc
        #         desired_mask = [
        #             [0, _, _, _], # though we can select k=2 for it, there is only 1 candidate before the span
        #             [1, 0, _, _], # we can select k=2, we have only two options,
        #             [1, 0, _, _], # though we can select the third one, the first two have higher mention score
        #             [1, 3, _, _] # the top k=2 mentions based on mention score
        #         ]
        # """
        # # Create an antecedent mask: lower triangular matrix =e, rest 0 indicating the candidates for each span
        # # [n_ana, n_ana]
        # top_antecedents_mask__filtered = torch.ones(
        #     n_top_spans, n_top_spans, device=encoded.device
        # ).tril()  # Lower triangular mat with filled diagonal
        # top_antecedents_mask__filtered = top_antecedents_mask__filtered - torch.eye(
        #     top_antecedents_mask__filtered.shape[0],
        #     top_antecedents_mask__filtered.shape[1],
        #     dtype=top_antecedents_mask__filtered.dtype,
        #     device=top_antecedents_mask__filtered.device,
        # )  # torch.eye is a diagonal. This op simply makes it so the mat has zeros in diagonals
        #
        # """
        #     At this point, this mask looks like:
        #     [[0., 0., 0.,  ..., 0., 0., 0.],
        #     [1., 0., 0.,  ..., 0., 0., 0.],
        #     [1., 1., 0.,  ..., 0., 0., 0.],
        #     ...,
        #     [1., 1., 1.,  ..., 0., 0., 0.],
        #     [1., 1., 1.,  ..., 1., 0., 0.],
        #     [1., 1., 1.,  ..., 1., 1., 0.]]
        #
        #
        #     Because when we take a log of it (used in argsort below) it will look like:
        #     [[-inf, -inf, -inf,  ..., -inf, -inf, -inf],
        #     [0., -inf, -inf,  ..., -inf, -inf, -inf],
        #     [0., 0., -inf,  ..., -inf, -inf, -inf],
        #     ...,
        #     [0., 0., 0.,  ..., -inf, -inf, -inf],
        #     [0., 0., 0.,  ..., 0., -inf, -inf],
        #     [0., 0., 0.,  ..., 0., 0., -inf]]
        #
        # """
        #
        # # Argsort mention scores for each span but ignored masked instances.
        # top_antecedents_ind__filtered = torch.argsort(
        #     top_span_mention_scores + torch.log(top_antecedents_mask__filtered),
        #     descending=True,
        #     dim=1,
        # )  # [n_ana, n_ana]
        #
        # # Add 1 to indices (temporarily, to distinguish between index '0',
        # #   and ignoring an index '_' (in the matrix in code comments above),
        # #   and the multiply it with the mask.
        # top_antecedents_ind__filtered = (top_antecedents_ind__filtered + 1) * \
        #                                 top_antecedents_mask__filtered  # [n_ana, n_ana]
        #
        # # Finally we subtract 1
        # # and hereon negative values indicates things which we don't want
        # top_antecedents_ind__filtered = (top_antecedents_ind__filtered - 1).to(
        #     torch.long
        # )  # [n_ana, n_ante]
        #
        # # The [n_ana, n_ana] lower triangular mat now has sorted span indices.
        # # We further need to clamp them to k.
        # # For that, we just crop top_antecedents_ind__filtered
        # top_antecedents_ind__filtered = top_antecedents_ind__filtered[:, : self.config.max_top_antecedents]
        # # [n_ana, n_ante]
        #
        # # # Below is garbage, ignore
        # # # For that, a simple col mul. will suffice
        # # column_mask = torch.zeros((num_top_mentions,), device=encoded.device, dtype=torch.float)
        # # column_mask[:config.max_top_antecedents] = 1
        # # top_antecedents_ind__filtered = top_antecedents_ind__filtered*column_mask
        #
        # # At this point, top_antecedents_ind__filtered has -1 representing masked out things,
        # #   and 0+ int to repr. actual indices
        # top_antecedents_emb__filtered = top_span_emb[
        #     top_antecedents_ind__filtered
        # ]  # [n_ana, n_ante, span_emb]
        # # This mask is needed to ignore the antecedents which occur after the anaphor (in span embeddings mat)
        # top_antecedents_emb__filtered[top_antecedents_ind__filtered < 0] = 0
        #
        # # Create a mask repr the -1 in top_antecedent_per_ana_ind
        # # top_antecedents_ind__filtered = torch.hstack(
        # #     [
        # #         torch.zeros((top_antecedents_ind__filtered.shape[0], 1),
        # #                     dtype=torch.int64, device=self.config.device) - 1,
        # #         top_antecedents_ind__filtered
        # #     ]
        # # )
        # top_antecedents_ind__filtered = torch.hstack(
        #     [
        #         top_antecedents_ind__filtered,
        #         torch.zeros((top_antecedents_ind__filtered.shape[0], 1),
        #                     dtype=torch.int64, device=self.config.device) - 1,
        #     ]
        # )  # TODO: try putting dummies to left and taking care of the rest of the code
        # top_antecedents_mask = torch.ones_like(top_antecedents_ind__filtered)
        # top_antecedents_mask[top_antecedents_ind__filtered < 0] = 0
        #
        # # We argsort this to yield a list of indices.
        #
        # """
        #     Step 5: Finally, let's do pairwise scoring of spans and their candidate antecedent scores.
        #     We concat
        #         - a (n_anaphor, 1, span_emb) mat repr anaphors
        #         - a (n_anaphor, max_antecedents, span_emb) mat repr antecedents for each anaphor
        #         - a (n_anaphor, max_antecedents, span_emb) mat repr element wise mul b/w the two.
        # """
        # similarity_emb = top_antecedents_emb__filtered * top_span_emb.unsqueeze(
        #     1
        # )  # [n_ana, n_ante, span_emb]
        # anaphor_emb = top_span_emb.unsqueeze(1).repeat(
        #     1, min(n_top_spans, self.config.max_top_antecedents), 1
        # )  # [n_ana, n_ante, span_emb]
        # pair_emb = torch.cat(
        #     [anaphor_emb, top_antecedents_emb__filtered, similarity_emb], 2
        # )  # [n_ana], n_ante, 3*span_emb]
        #
        # # Finally, pass it through the params and get the scores
        # top_antecedent_scores = self.binary_coref(pair_emb).squeeze(
        #     -1
        # )  # [n_ana, n_ante]
        #
        # # Dummy scores are set to zero (for reasons explained in Lee et al 2017 e2e coref)
        # dummy_scores = torch.zeros(
        #     [n_top_spans, 1], device=encoded.device
        # )  # [n_ana, 1]
        #
        # top_antecedent_scores = torch.cat(
        #     [dummy_scores, top_antecedent_scores], dim=1
        #     # [top_antecedent_scores, dummy_scores], dim=1
        # )  # [n_ana, n_ante + 1]
        #
        # # Now we transpose some things back from the space of individual span's candidates to the space of pruned spans
        #
        # # Now you have a notion of which are the likeliest antecedents for every given anaphor (sorted by scores).
        # top_antecedent_indices_in_each_span_cand_space = torch.argsort(
        #     top_antecedent_scores, descending=True
        # )  # [n_ana, n_ante + 1]
        #
        # # TODO: I'm not sure about this. Empirically decide.
        # # Since we're going to be using the top_antecedent_mask, need to get the mask arranged in the same fashion.
        # top_antecedent_mask = top_antecedents_mask.gather(
        #     index=top_antecedent_indices_in_each_span_cand_space,
        #     dim=1
        # )  # [n_ana, n_ante + 1]
        #
        # '''
        #      The previous code (commented below) was flawed. Like, objectively wrong.
        #      Previously, top_antecedent_indices were in not in the same space, but just argsort-ed scores
        #         so a value of 45 in index (3, 0) would mean that
        #         for the third span, antecedent candidate #45 is the most likely antecedent.
        #         However, this is not 45 in top_spans ..
        #         just 45th antecedent in the list of top 50 spans for THIS anaphor.
        #
        #      So, that's fixed now.
        # '''
        # top_antecedent_indices = top_antecedents_ind__filtered.gather(
        #     index=top_antecedent_indices_in_each_span_cand_space,
        #     dim=1
        # )  # [n_ana, n_ante + 1]

        # Now we just return them.top_antecedents_emb__filtered
        return {
            "top_antecedent_scores": top_antecedent_scores,
            "top_antecedent_mask": top_antecedents_mask,
            "top_antecedents": top_antecedents,  # this was top_antecedent_indices
            # "antecedent_map": top_antecedents_ind__filtered  # this mask brings things from 461x461 to 461x51 space
        }

    # noinspection PyUnusedLocal
    def ner(
            self,
            candidate_span_embeddings: torch.tensor,
            *args, **kwargs
    ):
        """
        Just a unary classifier over all spans.
        Just that. That's it.
        """
        logits = self.unary_ner(candidate_span_embeddings).squeeze(1)
        logits = torch.nn.functional.softmax(logits, dim=1)
        return {"logits": logits}

    # noinspection PyUnusedLocal
    def forward(
            self,
            input_ids: torch.tensor,
            attention_mask: torch.tensor,
            token_type_ids: torch.tensor,
            sentence_map: List[int],
            word_map: List[int],
            n_words: int,
            n_subwords: int,
            tasks: Iterable[str],
            *args,
            **kwargs
    ):
        """
        :param input_ids: tensor, shape (number of subsequences, max length), output of tokenizer, reshaped
        :param attention_mask: tensor, shape (number of subsequences, max length), output of tokenizer, reshaped
        :param token_type_ids: tensor, shape (number of subsequences, max length), output of tokenizer, reshaped
        :param sentence_map: list of sentence ID for each subword (excluding padded stuff)
        :param word_map: list of word ID for each subword (excluding padded stuff)
        :param n_words: number of words (not subwords) in the original doc
        :param n_subwords: number of subwords
        :param tasks: list containing either coref or ner or both (indicating which routes to follow for this batch)
        """

        """
            Step 1: Encode
            
            Just run the tokenized, sparsely encoded sequence through a BERT model
            
            It takes (n, 512) tensors and returns a (n, 512, 768) summary (each word has a 768 dim vec).
            We reshape it back to (n*512, 768) dim vec.
            
            Using masked select, we remove the padding tokens from encoded (and corresponding input ids).
        """
        assert self.encoder.device == input_ids.device == attention_mask.device, \
            f"encoder: {self.encoder.device}, input_ids: {input_ids.device}, attn_mask: {attention_mask.device}"

        encoded = self.encoder(input_ids, attention_mask)[0]  # n_seq, m_len, h_dim
        encoded = encoded.reshape((-1, self.n_hid_dim))  # n_seq * m_len, h_dim

        # Remove all the padded tokens, using info from attention masks
        encoded = torch.masked_select(encoded, attention_mask.bool().view(-1, 1)).view(
            -1, self.n_hid_dim
        )  # n_words, h_dim
        input_ids = torch.masked_select(input_ids, attention_mask.bool()).view(
            -1, 1
        )  # n_words, h_dim

        """
            Step 1.n
            
            Calculate spans locally. Ignore the ones passed by the dataiter
        """
        candidate_starts = torch.arange(start=0,
                                        end=input_ids.shape[0],
                                        device=self.config.device).view(-1, 1).repeat(1, self.config.max_span_width)
        candidate_ends = candidate_starts + torch.arange(start=0, end=self.config.max_span_width,
                                                         device=self.config.device).unsqueeze(0)
        candidate_start_sentence_indices = sentence_map[candidate_starts]  # [num_words, max_span_width]
        candidate_end_sentence_indices = sentence_map[
            torch.clamp(candidate_ends, max=input_ids.shape[0] - 1)]  # [num_words, max_span_width]
        # find spans that are in the same segment and don't run past the end of the text
        candidate_mask = torch.logical_and(candidate_ends < input_ids.shape[0],
                                           torch.eq(candidate_start_sentence_indices,
                                                    candidate_end_sentence_indices)).view(
            -1).bool()  # [num_words *max_span_width]
        candidate_starts = torch.masked_select(candidate_starts.view(-1), candidate_mask)  # [candidates]
        candidate_ends = torch.masked_select(candidate_ends.view(-1), candidate_mask)

        """
            Step 2: Span embeddings
            
            candidate_span_embeddings: 3*h_dim + meta_dim 
            Span embeddings: based on candidate_starts, candidate_ends go through encoded 
                and find start and end subword embeddings. Concatenate them. 
                add embedding corresponding to the width of the span
                add an attention weighted sum of vectors within the span.  
        """
        # n_cands, 3*h_dim + meta_dim
        candidate_span_embeddings = self._get_span_embeddings_(
            encoded, candidate_starts, candidate_ends
        )

        # Just init the return dict
        return_dict = {
            "encoded": encoded,
            "input_ids": input_ids,
            "candidate_span_embeddings": candidate_span_embeddings,
            "coref": None,
            "ner": None,
            "candidate_starts": candidate_starts,
            "candidate_ends": candidate_ends
        }

        # DEBUG
        if candidate_span_embeddings.isnan().any():
            raise AssertionError(f"Found {candidate_span_embeddings.isnan().float().sum()} nans in Can Span Emb!"
                                 f"\n\tAt this point there are {encoded.isnan().float().sum()} nans in encoded.")

        if "coref" in tasks or "pruner" in tasks:
            pruner_op = self.pruner(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                sentence_map=sentence_map,
                word_map=word_map,
                n_words=n_words,
                n_subwords=n_subwords,
                candidate_starts=candidate_starts,
                candidate_ends=candidate_ends,
                candidate_span_embeddings=candidate_span_embeddings,
                encoded=encoded
            )
            return_dict['pruner'] = pruner_op

            if "coref" in tasks:
                coref_op = self.coref(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    encoded=encoded,
                    top_span_starts=pruner_op['top_span_starts'],
                    top_span_emb=pruner_op['top_span_emb'],
                    top_span_mention_scores=pruner_op['top_span_mention_scores'],
                    n_top_spans=pruner_op['n_top_spans'],
                    n_top_antecedents=pruner_op['n_top_antecedents']
                )
                return_dict["coref"] = coref_op

        if "ner" in tasks:
            ner_op = self.ner(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                sentence_map=sentence_map,
                word_map=word_map,
                n_words=n_words,
                n_subwords=n_subwords,
                candidate_starts=candidate_starts,
                candidate_ends=candidate_ends,
                candidate_span_embeddings=candidate_span_embeddings,
                encoded=encoded,
            )
            return_dict["ner"] = ner_op

        return return_dict

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
            coref: dict = None,
            ner: dict = None,
            pruner: dict = None,
            prep_coref_eval: bool = True,
            *args, **kwargs
    ):
        """
        :param input_ids: tensor, shape (number of subsequences, max length), output of tokenizer, reshaped
        :param attention_mask: tensor, shape (number of subsequences, max length), output of tokenizer, reshaped
        :param token_type_ids: tensor, shape (number of subsequences, max length), output of tokenizer, reshaped
        :param sentence_map: list of sentence ID for each subword (excluding padded stuff)
        :param word_map: list of word ID for each subword (excluding padded stuff)
        :param n_words: number of words (not subwords) in the original doc
        :param n_subwords: number of subwords
        :param candidate_starts: subword ID for candidate span starts
        :param candidate_ends: subword ID for candidate span ends
        :param tasks: list of which forward functions to compute (and which labels to expect) e.g. ['coref', 'ner']
        :param coref: A dict containing gold annotations for coref for this document.
        It is expected to be there if 'coref' is a element in arg "tasks"
        Specifically,
            gold_starts: subword ID for actual span starts [n_gold_spans,]
            gold_ends: subword ID for actual span ends [n_gold_spans,]
            gold_cluster_ids_on_candidates: cluster ID for actual spans [n_gold_spans,]
                indexed according to the candidate spans i.e. 0 when candidate is not annotated,
                >0 for actual cluster ID of the candidate span
                (starts with 1)
        :param ner: A dict containing gold annotations for NER for this document.
        It is expected to be there if 'ner' is a element in arg "tasks"
        Specifically:
            gold_labels: one label corresponding to every candidate span generated.
        :param pruner: a dict containing gold annotations for Pruner submodule for this document.
        Specifically:
            gold_labels: 0/1 arr indicating which of all the candidate spans are named entities
                which are coreferent (in the case of ontonotes),
                or just valid mentions (singletons + coreferent) for other datasets.
        :param prep_coref_eval: if true, we run the coref outputs (and inputs) through a bunch of post processing,
            making it possible that we can use b-cubed, muc etc functions to actually eval coref.
        """
        predictions = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            sentence_map=sentence_map,
            word_map=word_map,
            n_words=n_words,
            n_subwords=n_subwords,
            tasks=tasks
        )

        outputs = {"loss": {}}
        candidate_starts = predictions['candidate_starts']
        candidate_ends = predictions['candidate_ends']

        if "pruner" in tasks:
            # Unpack pruner specific args (y labels)
            pruner_logits = predictions["pruner"]["logits"]
            pruner_logits_after_pruning = predictions["pruner"]["logits_after_pruning"]
            pruner_gold_label_values = pruner["gold_label_values"]
            pruner_gold_starts = pruner["gold_starts"]
            pruner_gold_ends = pruner["gold_ends"]
            pruner_labels = self.get_candidate_labels_mangoes(candidate_starts, candidate_ends,
                                                              pruner_gold_starts, pruner_gold_ends,
                                                              pruner_gold_label_values)

            if self.pruner_unweighted:
                pruner_loss = self.pruner_loss(pruner_logits, pruner_labels)
            else:
                pruner_loss = self.pruner_loss(pruner_logits, pruner_labels, weight=pruner["weights"])

            # DEBUG
            try:
                assert not torch.isnan(pruner_loss), \
                    f"Found nan in loss. Here are some details - \n\tLogits shape: {pruner_logits.shape}, " \
                    f"\n\tLabels shape: {pruner_labels.shape}, " \
                    f"\n\tNonZero lbls: {pruner_labels[pruner_labels != 0].shape}"
            except AssertionError:
                print('potato')

            outputs["loss"]["pruner"] = pruner_loss
            outputs["pruner"] = {"logits": pruner_logits, "labels": pruner_labels,
                                 "logits_after_pruning": pruner_logits_after_pruning}

        if "coref" in tasks:
            # Unpack Coref specific args (y labels)
            coref_gold_starts = coref["gold_starts"]
            coref_gold_ends = coref["gold_ends"]
            coref_gold_label_values = coref["gold_label_values"]
            gold_cluster_ids_on_candidates = self.get_candidate_labels_mangoes(candidate_starts, candidate_ends,
                                                                               coref_gold_starts, coref_gold_ends,
                                                                               coref_gold_label_values)

            # Unpack Coref specific model predictions
            top_span_starts = predictions["pruner"]["top_span_starts"]
            top_span_ends = predictions["pruner"]["top_span_ends"]
            pruned_space_map = predictions["pruner"]["top_span_indices"]  # mapping from 5383 -> 461
            top_antecedent_scores = predictions["coref"]["top_antecedent_scores"]
            top_antecedent_mask = predictions["coref"]["top_antecedent_mask"]
            top_antecedents = predictions["coref"]["top_antecedents"]
            # antecedents_space_map = predictions["coref"]["antecedent_map"]  # mapping from 461 -> 51

            # Again, this entire snippet I spent months debugging and am paving over with Mangoes code (below)
            # """
            #     Gold cluster ID on candidate is a vector in original candidate space (~5k)
            #         where each position is a candidate, and its value represents cluster ID.
            #     It's mostly zero.
            #
            #     In the first step, we bring this to the space of pruned candidates
            # """
            # gold_cluster_ids_on_pruned = gold_cluster_ids_on_candidates[pruned_space_map]
            #
            # # We next bring it to the antecedents space (which is different for every span)
            # gold_cluster_ids_on_antecedents = gold_cluster_ids_on_pruned[
            #     antecedents_space_map[:, 1:]
            #     # antecedents_space_map[:, : antecedents_space_map.shape[1] - 1]
            # ]
            #
            # # Make sure the masked candidates are suppressed.
            # # ### This again is problematic because the mask is over the top_antecedent_scores space
            # # ### But antecedent cluster IDs are a different sequence.
            # # ### That is, [2, 3] in the mask being zero does not mean that [2, 3] in cluster IDs should also be zero.
            # gold_cluster_ids_on_antecedents[top_antecedent_mask[:, 1:] == 0] = 0
            # # gold_cluster_ids_on_antecedents[top_antecedent_mask[:, : top_antecedent_mask.shape[1] - 1] == 0] = 0
            #
            # # top_antecedent_cluster_ids = top_span_cluster_ids[top_antecedent_scores]  # [top_cand, top_ant]
            # gold_cluster_ids_on_antecedents += torch.log(top_antecedent_mask.float()).int()[
            #                                    :, 1:
            #                                    # :, : top_antecedent_mask.shape[1] - 1
            #                                    ]  # [top_cand, top_ant]
            # same_cluster_indicator = torch.eq(
            #     gold_cluster_ids_on_antecedents, gold_cluster_ids_on_pruned.unsqueeze(1)
            # )  # [top_cand, top_ant]
            # non_dummy_indicator = (gold_cluster_ids_on_pruned > 0).unsqueeze(
            #     1
            # )  # [top_cand, 1]
            # pairwise_labels = torch.logical_and(
            #     same_cluster_indicator, non_dummy_indicator
            # )  # [top_cand, top_ant]
            # # noinspection PyArgumentList
            # dummy_labels = torch.logical_not(
            #     pairwise_labels.any(1, keepdims=True)
            # )  # [top_cand, 1]
            # # TODO: might have to concatenate it the other way round (pairwise first, dummy second)
            # top_antecedent_labels = torch.cat(
            #     # [pairwise_labels, dummy_labels], 1
            #     [dummy_labels, pairwise_labels], 1
            # )  # [top_cand, top_ant + 1]
            # coref_loss = self.coref_softmax_loss(
            #     top_antecedent_scores, top_antecedent_labels
            # )  # [top_cand]
            # coref_loss = torch.mean(coref_loss)
            # coref_loss = torch.sum(coref_loss)

            gold_cluster_ids_on_pruned = gold_cluster_ids_on_candidates[pruned_space_map]

            top_antecedent_cluster_ids = gold_cluster_ids_on_pruned[top_antecedents]  # [top_cand, top_ant]
            top_antecedent_cluster_ids += torch.log(top_antecedent_mask.float()).int()  # [top_cand, top_ant]
            same_cluster_indicator = torch.eq(top_antecedent_cluster_ids,
                                              gold_cluster_ids_on_pruned.unsqueeze(1))  # [top_cand, top_ant]
            non_dummy_indicator = (gold_cluster_ids_on_pruned > 0).unsqueeze(1)  # [top_cand, 1]
            pairwise_labels = torch.logical_and(same_cluster_indicator, non_dummy_indicator)  # [top_cand, top_ant]
            dummy_labels = torch.logical_not(pairwise_labels.any(1, keepdims=True))  # [top_cand, 1]
            top_antecedent_labels = torch.cat([dummy_labels, pairwise_labels], 1)  # [top_cand, top_ant + 1]
            coref_loss = self.coref_softmax_loss(top_antecedent_scores, top_antecedent_labels)  # [top_cand]
            coref_loss = self.coref_loss_agg(coref_loss)  # can be a mean or a sum depending on config.

            predictions["loss"] = coref_loss
            coref_logits = top_antecedent_scores
            coref_labels = top_antecedent_labels

            if prep_coref_eval:

                """
                     Prepping things for evaluation (muc, b-cubed etc).
                     Appropriating the second last cell of 
                        https://gitlab.inria.fr/magnet/mangoes/-/blob/master/notebooks/BERT%20for%20Co-reference%20Resolution%20-%20Ontonotes.ipynb
                """
                # We go for each cluster id. That's not too difficult
                gold_clusters = {}
                for i, val in enumerate(coref['gold_cluster_ids']):
                    cluster_id = val.item()

                    # Populate the dict
                    gold_clusters[cluster_id] = gold_clusters.get(cluster_id, []) + \
                                                [(coref['gold_starts'][i].item(), coref['gold_ends'][i].item())]

                # Gold clusters is a dict of tuple of (start, end), (start, end) corresponding to every cluster ID
                gold_clusters = [tuple(v) for v in gold_clusters.values()]

                mention_to_gold = {}
                for cluster in gold_clusters:
                    for mention in cluster:
                        mention_to_gold[mention] = cluster

                ids = input_ids
                top_span_starts = top_span_starts
                top_span_ends = top_span_ends
                mention_indices = []
                antecedent_indices = []
                for i in range(len(top_span_ends)):
                    # If ith span is predicted to be related to an actual antecedent
                    if top_antecedents[:, 0][i].item() > 0:
                        mention_indices.append(i)
                        antecedent_indices.append(top_antecedents[:, 0][i].item())

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
                        current_ids.append(ids[top_span_starts[mention_index]:top_span_ends[mention_index] + 1])
                        current_start_end.append(
                            (top_span_starts[mention_index].item(), top_span_ends[mention_index].item()))
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

            else:
                coref_eval = None

            outputs["loss"]["coref"] = coref_loss
            outputs["coref"] = {"logits": coref_logits, "labels": coref_labels, "eval": coref_eval}

        if "ner" in tasks:
            ner_gold_starts = ner["gold_starts"]
            ner_gold_ends = ner["gold_ends"]
            ner_gold_label_values = ner["gold_label_values"]
            ner_logits = predictions["ner"]["logits"]  # n_spans, n_classes
            ner_labels = self.get_candidate_labels_mangoes(candidate_starts, candidate_ends,
                                                           ner_gold_starts, ner_gold_ends,
                                                           ner_gold_label_values)

            # Calculating the loss
            if self.ner_unweighted:
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

    # ce ne marche pas, desole
    # config = transformers.BertConfig("bert-base-uncased")
    # config.max_span_width = 5
    # config.coref_dropout = 0.3
    # config.coref_metadata_feature_size = 20
    # config.unary_hdim = 1000
    # config.binary_hdim = 2000
    # config.top_span_ratio = 0.4
    # config.max_top_antecedents = 50
    # config.device = "cpu"
    # config.name = "bert-base-uncased"
    #
    # # Init the tokenizer
    # tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
    #
    # # Get the dataset up and running
    # ds = MultiTaskDataIter(
    #     "ontonotes", "train", tokenizer=tokenizer, config=config, tasks=("ner",)
    # )
    #
    # # Make the model
    # model = BasicMTL("bert-base-uncased", config=config)
    #
    # # Try to wrap it in a dataloader
    # for x in ds:
    #     pred = model.pred_with_labels(**x)
    #     print(pred)
    #     ...
