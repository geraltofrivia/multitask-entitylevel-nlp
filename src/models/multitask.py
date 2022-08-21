"""
    The core model is here. It does Coref. It does NER. It does everything. Come one come all.
    Modularity is so 2021. I'll shoot myself in the foot instead thank you very much.
"""

import random
from typing import List, Iterable, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

# Local imports
try:
    import _pathfix
except ImportError:
    from . import _pathfix
from utils.data import Tasks
from config import _SEED_ as SEED, NER_IS_MULTILABEL
from preproc.encode import Retriever
from models.modules import SpanPruner, CorefDecoder, SharedDense
from utils.exceptions import AnticipateOutOfMemException, UnknownDomainException, NANsFound

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# noinspection PyUnusedLocal
class MangoesMTL(nn.Module):

    def __init__(
            self,
            enc_nm: str,
            vocab_size: int,
            hidden_size: int,
            max_span_width: int,
            max_top_antecedents: int,
            device: Union[str, torch.device],
            unary_hdim: int,
            max_training_segments: int,
            encoder_dropout: float,
            dense_layers: int,
            task_1: dict,
            task_2: dict,

            # Pruner Specific Params
            pruner_dropout: float,
            pruner_use_width: bool,
            pruner_max_num_spans: int,
            pruner_top_span_ratio: float,

            # Coref Specific Params
            coref_dropout: float,
            coref_loss_mean: bool,
            coref_higher_order: int,
            coref_metadata_feature_size: int,

            bias_in_last_layers: bool,
            skip_instance_after_nspan: int,
            coref_num_speakers: int,
            ignore_speakers: bool,
            shared_compressor: bool,  # If True, it will reduce BERT embeddings from 768 to 256

            # This is a crucial flag which changes a lot of things
            freeze_encoder: bool,

            *args, **kwargs
    ):

        # base_config = SerializedBertConfig(vocab_size=vocab_size)
        super().__init__()

        # Convert task, task2 to Tasks object again (for now)
        task_1 = Tasks(**task_1)
        task_2 = Tasks(**task_2)

        if not freeze_encoder:
            self.bert = BertModel.from_pretrained(enc_nm)
            self.retriever = None
        else:
            self.bert = None
            self.retriever = Retriever(vocab_size=enc_nm, device=device)

        # # This dense thing is the one that takes the brunt of being cross task
        # linear_layers = []
        # for _ in range(dense_layers):
        #     linear_layers += [
        #         nn.Linear(hidden_size, hidden_size),
        #         # nn.BatchNorm1d(hidden_size),
        #         nn.ReLU(),
        #         nn.Dropout(encoder_dropout)
        #     ]
        self.shared = SharedDense(input_size=hidden_size,
                                  output_size=hidden_size // 3 if shared_compressor else hidden_size,
                                  depth=dense_layers,
                                  dropout_factor=encoder_dropout)

        # Hidden size is now compressed
        hidden_size = hidden_size // 3 if shared_compressor else hidden_size

        self.pruner = SpanPruner(
            hidden_size=hidden_size,
            unary_hdim=unary_hdim,
            max_span_width=max_span_width,
            coref_metadata_feature_size=coref_metadata_feature_size,
            pruner_dropout=pruner_dropout,
            pruner_use_width=pruner_use_width,
            pruner_max_num_spans=pruner_max_num_spans,
            pruner_top_span_ratio=pruner_top_span_ratio,
            bias_in_last_layers=bias_in_last_layers
        )
        self.coref = CorefDecoder(
            max_top_antecedents=max_top_antecedents,
            unary_hdim=unary_hdim,
            hidden_size=hidden_size,
            ignore_speakers=ignore_speakers,
            max_training_segments=max_training_segments,
            coref_metadata_feature_size=coref_metadata_feature_size,
            coref_dropout=coref_dropout,
            coref_higher_order=coref_higher_order,
            coref_num_speakers=coref_num_speakers,
            bias_in_last_layers=bias_in_last_layers
        )
        # self.bert = BertModel.from_pretrained(enc_nm, add_pooling_layer=False)

        ffnn_hidden_size = unary_hdim
        bert_hidden_size = hidden_size
        span_embedding_dim = (hidden_size * 3) + coref_metadata_feature_size

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
            task.dataset: nn.Linear(ffnn_hidden_size, task.n_classes_ner + 1, bias=bias_in_last_layers)
            for task in [task_1, task_2] if (not task.isempty() and 'ner' in task)
        })

        # Loss management for pruner
        self.pruner_loss = self._rescaling_weights_bce_loss_
        self.ner_loss = {}
        for task in [task_1, task_2]:
            if task.dataset in NER_IS_MULTILABEL:
                self.ner_loss[task.dataset] = nn.functional.binary_cross_entropy_with_logits
            else:
                self.ner_loss[task.dataset] = nn.functional.cross_entropy

        self.max_span_width = max_span_width
        self.max_top_antecedents = max_top_antecedents
        self.max_training_segments = max_training_segments
        self.coref_depth = coref_higher_order
        self.coref_loss_mean = coref_loss_mean
        self.ner_n_classes = {task.dataset: task.n_classes_ner for task in [task_1, task_2]}
        self._skip_instance_after_nspan = skip_instance_after_nspan
        self._ignore_speaker = ignore_speakers
        self._freeze_encoder = freeze_encoder
        self._tasks_: List[Tasks] = [task_1, task_2]

        # TODO: replace this
        # self.init_weights()

    # def get_gradnorm_param_group(self):
    #
    #     coref_params = nn.ModuleList([
    #         self.span_attend_projection,
    #         self.mention_scorer,
    #         self.width_scores,
    #         self.fast_antecedent_projection,
    #         self.slow_antecedent_scorer,
    #         self.slow_antecedent_projection,
    #         self.distance_embeddings,
    #         self.slow_distance_embeddings,
    #         self.distance_projection,
    #         self.span_width_embeddings,
    #         self.span_width_prior_embeddings,
    #         self.segment_dist_embeddings,
    #     ])
    #
    #     ner_params = nn.ModuleList([
    #         self.unary_ner
    #     ])
    #
    #     # TODO divide pruner and coref here !
    #     return nn.ModuleDict({'coref': coref_params, 'ner': ner_params})

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
    def _rescaling_weights_bce_loss_(logits, labels, weight: Optional[torch.tensor] = None):
        # if weights are provided, scale them based on labels
        if weight is not None:
            _weight = torch.zeros_like(labels, dtype=torch.float) + weight[0]
            _weight[labels == 1] = weight[1]
            return nn.functional.binary_cross_entropy_with_logits(logits, labels, _weight)
        else:
            return nn.functional.binary_cross_entropy_with_logits(logits, labels)

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

    def forward(
            self,
            input_ids: torch.tensor,
            attention_mask: torch.tensor,
            candidate_starts: torch.tensor,
            candidate_ends: torch.tensor,
            sentence_map: List[int],
            tasks: Iterable[str],
            domain: str,
            hash: int,
            speaker_ids: Optional[torch.tensor] = None,
            *args,
            **kwargs
    ):
        # TODO: make docstrings for this function

        device = input_ids.device

        """ At this point, if there are more candidates than expected, SKIP this op."""
        if 0 < self._skip_instance_after_nspan < candidate_starts.shape[0]:
            raise AnticipateOutOfMemException(f"There are {candidate_starts.shape[0]} candidates", device)

        # Pass through Text Encoder
        if not self._freeze_encoder:
            hidden_states = self.bert(input_ids, attention_mask)[0]  # [num_seg, max_seg_len, emb_len]
        else:
            hidden_states = self.retriever.load(domain=domain, hash=hash)  # [num_seg, max_seg_len, emb_len]
        num_segments, len_segment, len_embedding = hidden_states.shape

        # Re-arrange BERT outputs and input_ids to be a flat list: [num_words, *] from [num_segments, max_seg_len, *]
        hidden_states = torch.masked_select(hidden_states.view(num_segments * len_segment, len_embedding),
                                            attention_mask.bool().view(-1, 1)).view(-1,
                                                                                    len_embedding)  # [num_words, emb_len]
        flattened_ids = torch.masked_select(input_ids, attention_mask.bool()).view(-1)  # [num_words]
        if speaker_ids is not None:
            speaker_ids = torch.masked_select(speaker_ids.view(num_segments * len_segment),
                                              attention_mask.bool().view(-1))

        # Note the number of words
        num_words = hidden_states.shape[0]

        """
            Shared Parameter Stuff
        """
        hidden_states = self.shared(hidden_states)

        """
            That's the Span Pruner.
            Next we need to break out into Coref and NER parts
        """

        pruner_outputs = self.pruner(
            hidden_states=hidden_states,
            candidate_starts=candidate_starts,
            candidate_ends=candidate_ends,
            speaker_ids=speaker_ids
        )

        if 'coref' in tasks or 'pruner' in tasks:

            coref_specific = self.coref.forward(
                attention_mask=attention_mask,
                pruned_span_starts=pruner_outputs['pruned_span_starts'],
                pruned_span_ends=pruner_outputs['pruned_span_ends'],
                pruned_span_indices=pruner_outputs['pruned_span_indices'],
                pruned_span_scores=pruner_outputs['pruned_span_scores'],
                pruned_span_speaker_ids=pruner_outputs['pruned_span_speaker_ids'],
                pruned_span_emb=pruner_outputs['pruned_span_emb'],
                num_top_mentions=pruner_outputs['num_top_mentions'],
                num_segments=num_segments,
                len_segment=len_segment,
                domain=domain,
                device=device
            )
        else:
            coref_specific = {}

        if 'ner' in tasks:
            # We just need span embeddings here

            fc1 = self.unary_ner_common(pruner_outputs['span_emb'])

            # Depending on the domain, select the right decoder
            logits = self.unary_ner_specific[domain](fc1)
            # logits = torch.nn.functional.softmax(logits, dim=1)
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
            candidate_starts: torch.tensor,
            candidate_ends: torch.tensor,
            sentence_map: List[int],
            word_map: List[int],
            n_words: int,
            n_subwords: int,
            tasks: Iterable[str],
            domain: str,
            hash: int,
            speaker_ids: Optional[torch.tensor] = None,
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
            candidate_starts=candidate_starts,
            candidate_ends=candidate_ends,
            sentence_map=sentence_map,
            word_map=word_map,
            n_words=n_words,
            domain=domain,
            tasks=tasks,
            hash=hash,
            speaker_ids=speaker_ids,
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
            except AssertionError as e:
                raise e

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

            """
                At this point NER Labels is a n_spans, n_classes+1 matrix where most rows are zero.
                We want to turn all those zero rows into ones where the last element is active (not an entity class)
                If NER is not a multilabel class, then those positions are already at zero. Don't have to do anything.
            """
            if domain in NER_IS_MULTILABEL:
                zero_indices = torch.sum(ner_labels, dim=1) == 0  # n_spans
                ner_labels[zero_indices, 0] = 1  # n_spans, n_classes+1
            else:
                # Don't have to do this. The zero indices SHOULD have zero label.
                ...
                # zero_indices = ner_labels == 0
                # ner_labels[zero_indices] = self.ner_n_classes[domain]

            # Calculating the loss
            # if self.ner_unweighted:
            if self.is_unweighted(task='ner', domain=domain):
                ner_loss = self.ner_loss[domain](ner_logits, ner_labels)
            else:
                ner_loss = self.ner_loss[domain](ner_logits, ner_labels, weight=ner["weights"])

            if torch.isnan(ner_loss):
                raise NANsFound(
                    f"Found nan in loss. Here are some details - \n\tLogits shape: {ner_logits.shape}, "
                    f"\n\tLabels shape: {ner_labels.shape}, "
                    f"\n\tNonZero lbls: {ner_labels[ner_labels != 0].shape}"
                )

            outputs["loss"]["ner"] = ner_loss
            outputs["ner"] = {"logits": ner_logits, "labels": ner_labels}

        return outputs


if __name__ == "__main__":
    ...
