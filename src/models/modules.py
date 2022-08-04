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
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class Utils(object):

    @staticmethod
    def make_embeddings(vocab_size: int, output_dim: int, std: float = 0.02) -> torch.nn.Module:
        emb = nn.Embedding(vocab_size, output_dim)
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
