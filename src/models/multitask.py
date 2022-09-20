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
class MTLModel(nn.Module):

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

            # NER specific Params
            ner_dropout: float,

            # POS specific Params
            pos_dropout: float,

            bias_in_last_layers: bool,
            skip_instance_after_nspan: int,
            coref_num_speakers: int,
            use_speakers: bool,
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
        if 'coref' in task_1 or 'coref' in task_2:
            self.coref = CorefDecoder(
                max_top_antecedents=max_top_antecedents,
                unary_hdim=unary_hdim,
                hidden_size=hidden_size,
                use_speakers=use_speakers,
                max_training_segments=max_training_segments,
                coref_metadata_feature_size=coref_metadata_feature_size,
                coref_dropout=coref_dropout,
                coref_higher_order=coref_higher_order,
                coref_num_speakers=coref_num_speakers,
                bias_in_last_layers=bias_in_last_layers
            )

        span_embedding_dim = (hidden_size * 3) + coref_metadata_feature_size

        if 'ner' in task_1 or 'ner' in task_2:
            """
                NER Stuff is domain specific.
                Corresponding to domain, we will have individual "final" classifiers for NER.
                
                This of course because the final classes for NERs differ from each other.
                However, this will be done in a two step process.
                
                The two layers of NER will be broken down into a common, and domain specific variant.
            """
            self.unary_ner_common = nn.Sequential(
                nn.Linear(span_embedding_dim, unary_hdim),
                nn.ReLU(),
                nn.Dropout(ner_dropout),
                # nn.Linear(ffnn_hidden_size, n_classes_ner, bias=bias_in_last_layers)
            )
            self.unary_ner_specific = nn.ModuleDict({
                task.dataset: nn.Linear(unary_hdim, task.n_classes_ner + 1, bias=bias_in_last_layers)
                for task in [task_1, task_2] if (not task.isempty() and 'ner' in task)
            })  # classes + 1 -> not an entity (class zero)

        if 'pos' in task_1 or 'pos' in task_2:
            """
                Like NER, POS too is task specific. 
                Corresponding to domain, we will have individual "final" classifiers since POS tags may differ 
                    from dataset to dataset.
            """
            self.token_pos_common = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(pos_dropout)
            )
            self.token_pos_specific = nn.ModuleDict({
                task.dataset: nn.Linear(hidden_size // 2, task.n_classes_pos, bias=bias_in_last_layers)
                for task in [task_1, task_2] if (not task.isempty() and 'pos' in task)
            })  # every token must have a pos tag and so we don't add a faux class here

        # Loss management for pruner
        self.pruner_loss = self._rescaling_weights_bce_loss_
        self.ner_loss = {}
        for task in [task_1, task_2]:
            if task.dataset in NER_IS_MULTILABEL:
                self.ner_loss[task.dataset] = nn.functional.binary_cross_entropy_with_logits
            else:
                self.ner_loss[task.dataset] = nn.functional.cross_entropy
        self.pos_loss = nn.functional.cross_entropy

        self.max_span_width = max_span_width
        self.max_top_antecedents = max_top_antecedents
        self.max_training_segments = max_training_segments
        self.coref_depth = coref_higher_order
        self.coref_loss_mean = coref_loss_mean
        self.ner_n_classes = {task.dataset: task.n_classes_ner for task in [task_1, task_2]}
        self._skip_instance_after_nspan = skip_instance_after_nspan
        self._use_speaker = use_speakers
        self._freeze_encoder = freeze_encoder
        self._tasks_: List[Tasks] = [task_1, task_2]

        # TODO: replace this
        # self.init_weights()

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
                                     coref_top_antecedents_score):
        """ CPU list input """
        # Get predicted antecedents
        predicted_antecedents = self.todel_get_predicted_antecedents(coref_top_antecedents, coref_top_antecedents_score)

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
        TODO: this goes as well
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

    def get_coref_loss(
            self,
            candidate_starts: torch.tensor,
            candidate_ends: torch.tensor,
            gold_starts: torch.tensor,
            gold_ends: torch.tensor,
            gold_cluster_ids: torch.tensor,
            top_span_indices: torch.tensor,
            top_antecedents: torch.tensor,
            top_antecedents_mask: torch.tensor,
            top_antecedents_score: torch.tensor,
    ) -> torch.tensor:
        """ this is going to the module as well """
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
        coref_loss = self.coref_loss(top_antecedents_score, top_antecedent_labels)  # [top_cand]

        if self.coref_loss_mean:
            coref_loss = torch.mean(coref_loss)
        else:
            coref_loss = torch.sum(coref_loss)

        return coref_loss

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
        num_seg, len_seg, len_emb = hidden_states.shape

        # Re-arrange BERT outputs and input_ids to be a flat list: [num_words, *] from [num_segments, max_seg_len, *]
        hidden_states = torch.masked_select(hidden_states.view(num_seg * len_seg, len_emb),
                                            attention_mask.bool().view(-1, 1)).view(-1,
                                                                                    len_emb)  # [num_words, emb_len]
        flattened_ids = torch.masked_select(input_ids, attention_mask.bool()).view(-1)  # [num_words]
        if speaker_ids is not None:
            speaker_ids = torch.masked_select(speaker_ids.view(num_seg * len_seg),
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
                num_segments=num_seg,
                len_segment=len_seg,
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

        if 'pos' in tasks:
            # We just need token embeddings here
            fc1 = self.token_pos_common(hidden_states)
            logits = self.token_pos_specific[domain](fc1)
            pos_specific = {"pos_logits": logits}
        else:
            pos_specific = {}

        # noinspection PyUnboundLocalVariable
        return {
            "candidate_starts": candidate_starts,
            "candidate_ends": candidate_ends,
            "flattened_ids": flattened_ids,
            **coref_specific,
            **ner_specific,
            **pos_specific
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
            pos: dict = None,
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

            # Compute Loss
            coref_loss = self.get_coref_loss(
                candidate_starts=predictions['candidate_starts'],
                candidate_ends=predictions['candidate_ends'],
                gold_starts=coref['gold_starts'],
                gold_ends=coref['gold_ends'],
                gold_cluster_ids=coref['gold_label_values'],
                top_span_indices=predictions['pruned_span_indices'],
                top_antecedents=predictions['coref_top_antecedents'],
                top_antecedents_mask=predictions['coref_top_antecedents_mask'],
                top_antecedents_score=predictions['coref_top_antecedents_score']
            )

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

            top_indices = torch.argmax(predictions['coref_top_antecedents_score'], dim=-1, keepdim=False)
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

        if "pos" in tasks:
            pos_logits = predictions["pos_logits"]
            pos_labels = pos["gold_label_values"]
            if self.is_unweighted(task="pos", domain=domain):
                pos_loss = self.pos_loss(pos_logits, pos_labels)
            else:
                pos_loss = self.pos_loss(pos_logits, pos_labels, weight=pos["weights"])

            if torch.isnan(pos_loss):
                raise NANsFound(
                    f"Found nan in POS loss. Here are some details - \n\tLogits shape: {pos_logits.shape}, "
                    f"\n\tLabels shape: {pos_labels.shape}, "
                )

            outputs["loss"]["pos"] = pos_loss
            outputs["pos"] = {"logits": pos_logits, "labels": pos_labels}

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
                try:
                    ner_loss = self.ner_loss[domain](ner_logits, ner_labels, weight=ner["weights"])
                except IndexError as e:
                    print(ner_logits)
                    print(ner_logits.shape)
                    print(ner_labels)
                    print(ner_labels.max())
                    raise e

            if torch.isnan(ner_loss):
                raise NANsFound(
                    f"Found nan in NER loss. Here are some details - \n\tLogits shape: {ner_logits.shape}, "
                    f"\n\tLabels shape: {ner_labels.shape}, "
                    f"\n\tNonZero lbls: {ner_labels[ner_labels != 0].shape}"
                )

            outputs["loss"]["ner"] = ner_loss
            outputs["ner"] = {"logits": ner_logits, "labels": ner_labels}

        return outputs


class MangoesModel(nn.Module):
    def __init__(self, config, device, num_genres=None):
        """
        knowledge_base: "both" "wiki" or "wordnet"
        entity_emb_choice: "avg" or "max"
        """
        super().__init__()
        self.config = config
        self.device = device

        self.num_genres = num_genres if num_genres else len(config['genres'])
        self.max_seg_len = config['max_segment_len']
        self.max_span_width = config['max_span_width']
        assert config['loss_type'] in ['marginalized', 'hinge']
        if config['coref_depth'] > 1 or config['higher_order'] == 'cluster_merging':
            assert config['fine_grained']  # Higher-order is in slow fine-grained scoring

        # Model
        ####### entity embeddings
        self.knowledge_base = config["knowledge_base"]
        if config["knowledge_base"] == "both" or config["knowledge_base"] == "wiki":
            vocab = Vocabulary.from_files("./simple_kb/vocabulary")
            wiki_params = Params({
                "embedding_dim": 300,
                "pretrained_file": "https://allennlp.s3-us-west-2.amazonaws.com/knowbert/wiki_entity_linking/entities_glove_format.gz",
                "sparse": False,
                "trainable": False,
                "vocab_namespace": "entity_wiki"
            })
            self.wiki_embeddings = Embedding.from_params(vocab, wiki_params)
            self.wiki_null_embedding = torch.nn.Parameter(torch.zeros(300), requires_grad=True)
            self.wiki_null_embedding.data.normal_(mean=0.0, std=0.02)
            # if config["knowledge_base"] == "wordnet" or config["knowledge_base"] == "both":
            #     self.wn_embeddings = WordNetAllEmbedding(
            #         embedding_file="https://allennlp.s3-us-west-2.amazonaws.com/knowbert/wordnet/wordnet_synsets_mask_null_vocab_embeddings_tucker_gensen.hdf5",
            #         entity_dim=200,
            #         entity_file="https://allennlp.s3-us-west-2.amazonaws.com/knowbert/wordnet/entities.jsonl",
            #         vocab_file="https://allennlp.s3-us-west-2.amazonaws.com/knowbert/wordnet/wordnet_synsets_mask_null_vocab.txt",
            #         entity_h5_key="tucker_gensen",
            #         dropout=0.1,
            #         pos_embedding_dim=None,
            #         include_null_embedding=False)
            self.wn_null_embedding = torch.nn.Parameter(torch.zeros(200), requires_grad=True)
            self.wn_null_embedding.data.normal_(mean=0.0, std=0.02)
        self.entity_emb_choice = config["entity_emb_choice"]
        #######
        self.dropout = nn.Dropout(p=config['dropout_rate'])
        self.bert = BertModel.from_pretrained(config['bert_pretrained_name_or_path'])

        self.bert_emb_size = self.bert.config.hidden_size + \
                             (300 if config["knowledge_base"] == "both" or config["knowledge_base"] == "wiki" else 0) + \
                             (200 if config["knowledge_base"] == "wordnet" or config["knowledge_base"] == "both" else 0)
        self.span_emb_size = self.bert_emb_size * 3
        if config['use_features']:
            self.span_emb_size += config['feature_emb_size']
        self.pair_emb_size = self.span_emb_size * 3
        if config['use_metadata']:
            self.pair_emb_size += 2 * config['feature_emb_size']
        if config['use_features']:
            self.pair_emb_size += config['feature_emb_size']
        if config['use_segment_distance']:
            self.pair_emb_size += config['feature_emb_size']

        self.emb_span_width = self.make_embedding(self.max_span_width) if config['use_features'] else None
        self.emb_span_width_prior = self.make_embedding(self.max_span_width) if config['use_width_prior'] else None
        self.emb_antecedent_distance_prior = self.make_embedding(10) if config['use_distance_prior'] else None
        self.emb_genre = self.make_embedding(self.num_genres)
        self.emb_same_speaker = self.make_embedding(2) if config['use_metadata'] else None
        self.emb_segment_distance = self.make_embedding(config['max_training_sentences']) if config[
            'use_segment_distance'] else None
        self.emb_top_antecedent_distance = self.make_embedding(10)
        self.emb_cluster_size = self.make_embedding(10) if config['higher_order'] == 'cluster_merging' else None

        self.mention_token_attn = self.make_ffnn(self.bert_emb_size, 0, output_size=1) if config[
            'model_heads'] else None
        self.span_emb_score_ffnn = self.make_ffnn(self.span_emb_size, [config['ffnn_size']] * config['ffnn_depth'],
                                                  output_size=1)
        self.span_width_score_ffnn = self.make_ffnn(config['feature_emb_size'],
                                                    [config['ffnn_size']] * config['ffnn_depth'], output_size=1) if \
        config['use_width_prior'] else None
        self.coarse_bilinear = self.make_ffnn(self.span_emb_size, 0, output_size=self.span_emb_size)
        self.antecedent_distance_score_ffnn = self.make_ffnn(config['feature_emb_size'], 0, output_size=1) if config[
            'use_distance_prior'] else None
        self.coref_score_ffnn = self.make_ffnn(self.pair_emb_size, [config['ffnn_size']] * config['ffnn_depth'],
                                               output_size=1) if config['fine_grained'] else None

        self.gate_ffnn = self.make_ffnn(2 * self.span_emb_size, 0, output_size=self.span_emb_size) if config[
                                                                                                          'coref_depth'] > 1 else None
        self.span_attn_ffnn = self.make_ffnn(self.span_emb_size, 0, output_size=1) if config[
                                                                                          'higher_order'] == 'span_clustering' else None
        self.cluster_score_ffnn = self.make_ffnn(3 * self.span_emb_size + config['feature_emb_size'],
                                                 [config['cluster_ffnn_size']] * config['ffnn_depth'], output_size=1) if \
        config['higher_order'] == 'cluster_merging' else None

        self.update_steps = 0  # Internal use for debug
        self.debug = True

    def make_embedding(self, dict_size, std=0.02):
        emb = nn.Embedding(dict_size, self.config['feature_emb_size'])
        init.normal_(emb.weight, std=std)
        return emb

    def aggregate_entity_embeddings(self, entity_embeddings, entity_priors):
        # emb are of shape: (segments, num_candidate_spans, max_num_candidates (padded if less), emb_dim)
        # candidate priors: (segments, num_candidate_spans, max_num_candidates)
        emb_size = entity_embeddings.shape[-1]
        if self.entity_emb_choice == "max":
            max_entity_indices = torch.max(entity_priors, dim=-1)[1].unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1,
                                                                                                        emb_size)
            return torch.gather(entity_embeddings, 2, max_entity_indices).squeeze(-2)
        elif self.entity_emb_choice == "avg":
            return torch.matmul(entity_priors.unsqueeze(2), entity_embeddings).squeeze(2)

    @staticmethod
    def compute_sequence_entity_embeddings(entity_embeddings, entity_spans, input_id_shape, null_embedding, device):
        # entity_embeddings: shape (num_segments, max_entities, emb_dim)
        # entity_spans: shape (num_segments, max_entities, 2 (start, end)) (padded spans are [-1, -1])
        seq_emb = torch.zeros((input_id_shape[0], input_id_shape[1], entity_embeddings.shape[-1])).to(device)
        for i in range(input_id_shape[0]):
            null_indices = set(range(input_id_shape[1]))
            # valid_span_mask = entity_spans[i] >= 0
            # sel = entity_spans[i][valid_span_mask].view(-1, 2)
            # seq_emb[i, :, entity_spans[i][valid_span_mask]] += entity_embeddings[i]
            for j in range(entity_spans.shape[1]):
                if torch.sum(entity_spans[i, j]) == -2:
                    break
                seq_emb[i, entity_spans[i, j, 0]:entity_spans[i, j, 1] + 1] += entity_embeddings[i, j]
                null_indices -= set(range(entity_spans[i, j, 0], entity_spans[i, j, 1] + 1))
            seq_emb[i, list(null_indices)] = null_embedding
        return seq_emb

    def make_linear(self, in_features, out_features, bias=True, std=0.02):
        linear = nn.Linear(in_features, out_features, bias)
        init.normal_(linear.weight, std=std)
        if bias:
            init.zeros_(linear.bias)
        return linear

    def make_ffnn(self, feat_size, hidden_size, output_size):
        if hidden_size is None or hidden_size == 0 or hidden_size == [] or hidden_size == [0]:
            return self.make_linear(feat_size, output_size)

        if not isinstance(hidden_size, Iterable):
            hidden_size = [hidden_size]
        ffnn = [self.make_linear(feat_size, hidden_size[0]), nn.ReLU(), self.dropout]
        for i in range(1, len(hidden_size)):
            ffnn += [self.make_linear(hidden_size[i - 1], hidden_size[i]), nn.ReLU(), self.dropout]
        ffnn.append(self.make_linear(hidden_size[-1], output_size))
        return nn.Sequential(*ffnn)

    def get_params(self, named=False):
        bert_based_param, task_param = [], []
        for name, param in self.named_parameters():
            if name.startswith('bert'):
                to_add = (name, param) if named else param
                bert_based_param.append(to_add)
            else:
                to_add = (name, param) if named else param
                task_param.append(to_add)
        return bert_based_param, task_param

    def forward(self, *input):
        return self.get_predictions_and_loss(*input)

    def get_predictions_and_loss(self, input_ids, input_mask, speaker_ids, sentence_len, genre, sentence_map,
                                 is_training, gold_starts=None, gold_ends=None, gold_mention_cluster_map=None,
                                 candidates=None):
        """ Model and input are already on the device """
        device = self.device
        conf = self.config

        do_loss = False
        if gold_mention_cluster_map is not None:
            assert gold_starts is not None
            assert gold_ends is not None
            do_loss = True

        if self.knowledge_base == "both" or self.knowledge_base == "wiki":
            wiki_seq_emb = self.aggregate_entity_embeddings(
                self.wiki_embeddings(candidates["wiki"]["candidate_entities"]["ids"]),
                candidates["wiki"]["candidate_entity_priors"])
            wiki_seq_emb = self.compute_sequence_entity_embeddings(wiki_seq_emb, candidates["wiki"]["candidate_spans"],
                                                                   input_ids.shape, self.wiki_null_embedding,
                                                                   input_ids.device)
        if self.knowledge_base == "both" or self.knowledge_base == "wordnet":
            wn_seq_emb = self.aggregate_entity_embeddings(
                self.wn_embeddings(candidates["wordnet"]["candidate_entities"]["ids"]),
                candidates["wordnet"]["candidate_entity_priors"])
            wn_seq_emb = self.compute_sequence_entity_embeddings(wn_seq_emb, candidates["wordnet"]["candidate_spans"],
                                                                 input_ids.shape, self.wn_null_embedding,
                                                                 input_ids.device)

        # Get token emb
        mention_doc = self.bert(input_ids, attention_mask=input_mask, return_dict=True)[
            "last_hidden_state"]  # [num seg, num max tokens, emb size]

        if self.knowledge_base == "both" or self.knowledge_base == "wiki":
            mention_doc = torch.cat((mention_doc, wiki_seq_emb), dim=-1)
        if self.knowledge_base == "both" or self.knowledge_base == "wordnet":
            mention_doc = torch.cat((mention_doc, wn_seq_emb), dim=-1)

        input_mask = input_mask.to(torch.bool)
        mention_doc = mention_doc[input_mask]
        speaker_ids = speaker_ids[input_mask]
        num_words = mention_doc.shape[0]

        # Get candidate span
        sentence_indices = sentence_map  # [num tokens]
        candidate_starts = torch.unsqueeze(torch.arange(0, num_words, device=device), 1).repeat(1, self.max_span_width)
        candidate_ends = candidate_starts + torch.arange(0, self.max_span_width, device=device)
        candidate_start_sent_idx = sentence_indices[candidate_starts]
        candidate_end_sent_idx = sentence_indices[torch.min(candidate_ends, torch.tensor(num_words - 1, device=device))]
        candidate_mask = (candidate_ends < num_words) & (candidate_start_sent_idx == candidate_end_sent_idx)
        candidate_starts, candidate_ends = candidate_starts[candidate_mask], candidate_ends[
            candidate_mask]  # [num valid candidates]
        num_candidates = candidate_starts.shape[0]

        # Get candidate labels
        if do_loss:
            same_start = (torch.unsqueeze(gold_starts, 1) == torch.unsqueeze(candidate_starts, 0))
            same_end = (torch.unsqueeze(gold_ends, 1) == torch.unsqueeze(candidate_ends, 0))
            same_span = (same_start & same_end).to(torch.long)
            candidate_labels = torch.matmul(torch.unsqueeze(gold_mention_cluster_map, 0).to(torch.float),
                                            same_span.to(torch.float))
            candidate_labels = torch.squeeze(candidate_labels.to(torch.long),
                                             0)  # [num candidates]; non-gold span has label 0

        # Get span embedding
        span_start_emb, span_end_emb = mention_doc[candidate_starts], mention_doc[candidate_ends]
        candidate_emb_list = [span_start_emb, span_end_emb]
        if conf['use_features']:
            candidate_width_idx = candidate_ends - candidate_starts
            candidate_width_emb = self.emb_span_width(candidate_width_idx)
            candidate_width_emb = self.dropout(candidate_width_emb)
            candidate_emb_list.append(candidate_width_emb)
        # Use attended head or avg token
        candidate_tokens = torch.unsqueeze(torch.arange(0, num_words, device=device), 0).repeat(num_candidates, 1)
        candidate_tokens_mask = (candidate_tokens >= torch.unsqueeze(candidate_starts, 1)) & (
                    candidate_tokens <= torch.unsqueeze(candidate_ends, 1))
        if conf['model_heads']:
            token_attn = torch.squeeze(self.mention_token_attn(mention_doc), 1)
        else:
            token_attn = torch.ones(num_words, dtype=torch.float, device=device)  # Use avg if no attention
        candidate_tokens_attn_raw = torch.log(candidate_tokens_mask.to(torch.float)) + torch.unsqueeze(token_attn, 0)
        candidate_tokens_attn = nn.functional.softmax(candidate_tokens_attn_raw, dim=1)
        head_attn_emb = torch.matmul(candidate_tokens_attn, mention_doc)
        candidate_emb_list.append(head_attn_emb)
        candidate_span_emb = torch.cat(candidate_emb_list, dim=1)  # [num candidates, new emb size]

        # Get span score
        candidate_mention_scores = torch.squeeze(self.span_emb_score_ffnn(candidate_span_emb), 1)
        if conf['use_width_prior']:
            width_score = torch.squeeze(self.span_width_score_ffnn(self.emb_span_width_prior.weight), 1)
            candidate_width_score = width_score[candidate_width_idx]
            candidate_mention_scores += candidate_width_score

        # Extract top spans
        candidate_idx_sorted_by_score = torch.argsort(candidate_mention_scores, descending=True).tolist()
        candidate_starts_cpu, candidate_ends_cpu = candidate_starts.tolist(), candidate_ends.tolist()
        num_top_spans = int(min(conf['max_num_extracted_spans'], conf['top_span_ratio'] * num_words))
        selected_idx_cpu = self._extract_top_spans(candidate_idx_sorted_by_score, candidate_starts_cpu,
                                                   candidate_ends_cpu, num_top_spans)
        assert len(selected_idx_cpu) == num_top_spans
        selected_idx = torch.tensor(selected_idx_cpu, device=device)
        top_span_starts, top_span_ends = candidate_starts[selected_idx], candidate_ends[selected_idx]
        top_span_emb = candidate_span_emb[selected_idx]
        top_span_cluster_ids = candidate_labels[selected_idx] if do_loss else None
        top_span_mention_scores = candidate_mention_scores[selected_idx]

        # Coarse pruning on each mention's antecedents
        max_top_antecedents = min(num_top_spans, conf['max_top_antecedents'])
        top_span_range = torch.arange(0, num_top_spans, device=device)
        antecedent_offsets = torch.unsqueeze(top_span_range, 1) - torch.unsqueeze(top_span_range, 0)
        antecedent_mask = (antecedent_offsets >= 1)
        pairwise_mention_score_sum = torch.unsqueeze(top_span_mention_scores, 1) + torch.unsqueeze(
            top_span_mention_scores, 0)
        source_span_emb = self.dropout(self.coarse_bilinear(top_span_emb))
        target_span_emb = self.dropout(torch.transpose(top_span_emb, 0, 1))
        pairwise_coref_scores = torch.matmul(source_span_emb, target_span_emb)
        pairwise_fast_scores = pairwise_mention_score_sum + pairwise_coref_scores
        pairwise_fast_scores += torch.log(antecedent_mask.to(torch.float))
        if conf['use_distance_prior']:
            distance_score = torch.squeeze(
                self.antecedent_distance_score_ffnn(self.dropout(self.emb_antecedent_distance_prior.weight)), 1)
            bucketed_distance = util.bucket_distance(antecedent_offsets)
            antecedent_distance_score = distance_score[bucketed_distance]
            pairwise_fast_scores += antecedent_distance_score
        top_pairwise_fast_scores, top_antecedent_idx = torch.topk(pairwise_fast_scores, k=max_top_antecedents)
        top_antecedent_mask = util.batch_select(antecedent_mask, top_antecedent_idx,
                                                device)  # [num top spans, max top antecedents]
        top_antecedent_offsets = util.batch_select(antecedent_offsets, top_antecedent_idx, device)

        # Slow mention ranking
        if conf['fine_grained']:
            same_speaker_emb, genre_emb, seg_distance_emb, top_antecedent_distance_emb = None, None, None, None
            if conf['use_metadata']:
                top_span_speaker_ids = speaker_ids[top_span_starts]
                top_antecedent_speaker_id = top_span_speaker_ids[top_antecedent_idx]
                same_speaker = torch.unsqueeze(top_span_speaker_ids, 1) == top_antecedent_speaker_id
                same_speaker_emb = self.emb_same_speaker(same_speaker.to(torch.long))
                genre_emb = self.emb_genre(genre)
                genre_emb = torch.unsqueeze(torch.unsqueeze(genre_emb, 0), 0).repeat(num_top_spans, max_top_antecedents,
                                                                                     1)
            if conf['use_segment_distance']:
                num_segs, seg_len = input_ids.shape[0], input_ids.shape[1]
                token_seg_ids = torch.arange(0, num_segs, device=device).unsqueeze(1).repeat(1, seg_len)
                token_seg_ids = token_seg_ids[input_mask]
                top_span_seg_ids = token_seg_ids[top_span_starts]
                top_antecedent_seg_ids = token_seg_ids[top_span_starts[top_antecedent_idx]]
                top_antecedent_seg_distance = torch.unsqueeze(top_span_seg_ids, 1) - top_antecedent_seg_ids
                top_antecedent_seg_distance = torch.clamp(top_antecedent_seg_distance, 0,
                                                          self.config['max_training_sentences'] - 1)
                seg_distance_emb = self.emb_segment_distance(top_antecedent_seg_distance)
            if conf['use_features']:  # Antecedent distance
                top_antecedent_distance = util.bucket_distance(top_antecedent_offsets)
                top_antecedent_distance_emb = self.emb_top_antecedent_distance(top_antecedent_distance)

            for depth in range(conf['coref_depth']):
                top_antecedent_emb = top_span_emb[top_antecedent_idx]  # [num top spans, max top antecedents, emb size]
                feature_list = []
                if conf['use_metadata']:  # speaker, genre
                    feature_list.append(same_speaker_emb)
                    feature_list.append(genre_emb)
                if conf['use_segment_distance']:
                    feature_list.append(seg_distance_emb)
                if conf['use_features']:  # Antecedent distance
                    feature_list.append(top_antecedent_distance_emb)
                feature_emb = torch.cat(feature_list, dim=2)
                feature_emb = self.dropout(feature_emb)
                target_emb = torch.unsqueeze(top_span_emb, 1).repeat(1, max_top_antecedents, 1)
                similarity_emb = target_emb * top_antecedent_emb
                pair_emb = torch.cat([target_emb, top_antecedent_emb, similarity_emb, feature_emb], 2)
                top_pairwise_slow_scores = torch.squeeze(self.coref_score_ffnn(pair_emb), 2)
                top_pairwise_scores = top_pairwise_slow_scores + top_pairwise_fast_scores
                if conf['higher_order'] == 'cluster_merging':
                    cluster_merging_scores = ho.cluster_merging(top_span_emb, top_antecedent_idx, top_pairwise_scores,
                                                                self.emb_cluster_size, self.cluster_score_ffnn, None,
                                                                self.dropout,
                                                                device=device, reduce=conf['cluster_reduce'],
                                                                easy_cluster_first=conf['easy_cluster_first'])
                    break
                elif depth != conf['coref_depth'] - 1:
                    if conf['higher_order'] == 'attended_antecedent':
                        refined_span_emb = ho.attended_antecedent(top_span_emb, top_antecedent_emb, top_pairwise_scores,
                                                                  device)
                    elif conf['higher_order'] == 'max_antecedent':
                        refined_span_emb = ho.max_antecedent(top_span_emb, top_antecedent_emb, top_pairwise_scores,
                                                             device)
                    elif conf['higher_order'] == 'entity_equalization':
                        refined_span_emb = ho.entity_equalization(top_span_emb, top_antecedent_emb, top_antecedent_idx,
                                                                  top_pairwise_scores, device)
                    elif conf['higher_order'] == 'span_clustering':
                        refined_span_emb = ho.span_clustering(top_span_emb, top_antecedent_idx, top_pairwise_scores,
                                                              self.span_attn_ffnn, device)

                    gate = self.gate_ffnn(torch.cat([top_span_emb, refined_span_emb], dim=1))
                    gate = torch.sigmoid(gate)
                    top_span_emb = gate * refined_span_emb + (1 - gate) * top_span_emb  # [num top spans, span emb size]
        else:
            top_pairwise_scores = top_pairwise_fast_scores  # [num top spans, max top antecedents]

        if not do_loss:
            if conf['fine_grained'] and conf['higher_order'] == 'cluster_merging':
                top_pairwise_scores += cluster_merging_scores
            top_antecedent_scores = torch.cat([torch.zeros(num_top_spans, 1, device=device), top_pairwise_scores],
                                              dim=1)  # [num top spans, max top antecedents + 1]
            return candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends, top_antecedent_idx, top_antecedent_scores

        # Get gold labels
        top_antecedent_cluster_ids = top_span_cluster_ids[top_antecedent_idx]
        top_antecedent_cluster_ids += (top_antecedent_mask.to(
            torch.long) - 1) * 100000  # Mask id on invalid antecedents
        same_gold_cluster_indicator = (top_antecedent_cluster_ids == torch.unsqueeze(top_span_cluster_ids, 1))
        non_dummy_indicator = torch.unsqueeze(top_span_cluster_ids > 0, 1)
        pairwise_labels = same_gold_cluster_indicator & non_dummy_indicator
        dummy_antecedent_labels = torch.logical_not(pairwise_labels.any(dim=1, keepdims=True))
        top_antecedent_gold_labels = torch.cat([dummy_antecedent_labels, pairwise_labels], dim=1)

        # Get loss
        top_antecedent_scores = torch.cat([torch.zeros(num_top_spans, 1, device=device), top_pairwise_scores], dim=1)
        if conf['loss_type'] == 'marginalized':
            log_marginalized_antecedent_scores = torch.logsumexp(
                top_antecedent_scores + torch.log(top_antecedent_gold_labels.to(torch.float)), dim=1)
            log_norm = torch.logsumexp(top_antecedent_scores, dim=1)
            loss = torch.sum(log_norm - log_marginalized_antecedent_scores)
        elif conf['loss_type'] == 'hinge':
            top_antecedent_mask = torch.cat(
                [torch.ones(num_top_spans, 1, dtype=torch.bool, device=device), top_antecedent_mask], dim=1)
            top_antecedent_scores += torch.log(top_antecedent_mask.to(torch.float))
            highest_antecedent_scores, highest_antecedent_idx = torch.max(top_antecedent_scores, dim=1)
            gold_antecedent_scores = top_antecedent_scores + torch.log(top_antecedent_gold_labels.to(torch.float))
            highest_gold_antecedent_scores, highest_gold_antecedent_idx = torch.max(gold_antecedent_scores, dim=1)
            slack_hinge = 1 + highest_antecedent_scores - highest_gold_antecedent_scores
            # Calculate delta
            highest_antecedent_is_gold = (highest_antecedent_idx == highest_gold_antecedent_idx)
            mistake_false_new = (highest_antecedent_idx == 0) & torch.logical_not(dummy_antecedent_labels.squeeze())
            delta = ((3 - conf['false_new_delta']) / 2) * torch.ones(num_top_spans, dtype=torch.float, device=device)
            delta -= (1 - conf['false_new_delta']) * mistake_false_new.to(torch.float)
            delta *= torch.logical_not(highest_antecedent_is_gold).to(torch.float)
            loss = torch.sum(slack_hinge * delta)

        # Add mention loss
        if conf['mention_loss_coef']:
            gold_mention_scores = top_span_mention_scores[top_span_cluster_ids > 0]
            non_gold_mention_scores = top_span_mention_scores[top_span_cluster_ids == 0]
            loss_mention = -torch.sum(torch.log(torch.sigmoid(gold_mention_scores))) * conf['mention_loss_coef']
            loss_mention += -torch.sum(torch.log(1 - torch.sigmoid(non_gold_mention_scores))) * conf[
                'mention_loss_coef']
            loss += loss_mention

        if conf['higher_order'] == 'cluster_merging':
            top_pairwise_scores += cluster_merging_scores
            top_antecedent_scores = torch.cat([torch.zeros(num_top_spans, 1, device=device), top_pairwise_scores],
                                              dim=1)
            log_marginalized_antecedent_scores2 = torch.logsumexp(
                top_antecedent_scores + torch.log(top_antecedent_gold_labels.to(torch.float)), dim=1)
            log_norm2 = torch.logsumexp(top_antecedent_scores, dim=1)  # [num top spans]
            loss_cm = torch.sum(log_norm2 - log_marginalized_antecedent_scores2)
            if conf['cluster_dloss']:
                loss += loss_cm
            else:
                loss = loss_cm

        # Debug
        if self.debug:
            if self.update_steps % 20 == 0:
                logger.info('---------debug step: %d---------' % self.update_steps)
                # logger.info('candidates: %d; antecedents: %d' % (num_candidates, max_top_antecedents))
                logger.info('spans/gold: %d/%d; ratio: %.2f' % (
                num_top_spans, (top_span_cluster_ids > 0).sum(), (top_span_cluster_ids > 0).sum() / num_top_spans))
                if conf['mention_loss_coef']:
                    logger.info('mention loss: %.4f' % loss_mention)
                if conf['loss_type'] == 'marginalized':
                    logger.info(
                        'norm/gold: %.4f/%.4f' % (torch.sum(log_norm), torch.sum(log_marginalized_antecedent_scores)))
                else:
                    logger.info('loss: %.4f' % loss)
        self.update_steps += 1

        return [candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends,
                top_antecedent_idx, top_antecedent_scores], loss

    def _extract_top_spans(self, candidate_idx_sorted, candidate_starts, candidate_ends, num_top_spans):
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

    def get_predicted_antecedents(self, antecedent_idx, antecedent_scores):
        """ CPU list input """
        predicted_antecedents = []
        for i, idx in enumerate(np.argmax(antecedent_scores, axis=1) - 1):
            if idx < 0:
                predicted_antecedents.append(-1)
            else:
                predicted_antecedents.append(antecedent_idx[i][idx])
        return predicted_antecedents

    def get_predicted_clusters(self, span_starts, span_ends, antecedent_idx, antecedent_scores):
        """ CPU list input """
        # Get predicted antecedents
        predicted_antecedents = self.get_predicted_antecedents(antecedent_idx, antecedent_scores)

        # Get predicted clusters
        mention_to_cluster_id = {}
        predicted_clusters = []
        for i, predicted_idx in enumerate(predicted_antecedents):
            if predicted_idx < 0:
                continue
            assert i > predicted_idx, f'span idx: {i}; antecedent idx: {predicted_idx}'
            # Check antecedent's cluster
            antecedent = (int(span_starts[predicted_idx]), int(span_ends[predicted_idx]))
            antecedent_cluster_id = mention_to_cluster_id.get(antecedent, -1)
            if antecedent_cluster_id == -1:
                antecedent_cluster_id = len(predicted_clusters)
                predicted_clusters.append([antecedent])
                mention_to_cluster_id[antecedent] = antecedent_cluster_id
            # Add mention to cluster
            mention = (int(span_starts[i]), int(span_ends[i]))
            predicted_clusters[antecedent_cluster_id].append(mention)
            mention_to_cluster_id[mention] = antecedent_cluster_id

        predicted_clusters = [tuple(c) for c in predicted_clusters]
        return predicted_clusters, mention_to_cluster_id, predicted_antecedents

    def update_evaluator(self, span_starts, span_ends, antecedent_idx, antecedent_scores, gold_clusters, evaluator):
        predicted_clusters, mention_to_cluster_id, _ = self.get_predicted_clusters(span_starts, span_ends,
                                                                                   antecedent_idx, antecedent_scores)
        mention_to_predicted = {m: predicted_clusters[cluster_idx] for m, cluster_idx in mention_to_cluster_id.items()}
        gold_clusters = [tuple(tuple(m) for m in cluster) for cluster in gold_clusters]
        mention_to_gold = {m: cluster for cluster in gold_clusters for m in cluster}
        evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)
        return predicted_clusters


if __name__ == "__main__":
    ...
