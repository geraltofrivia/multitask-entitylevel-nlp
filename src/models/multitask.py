"""
    TODO:
        - in data sampler (pipeline, make sure to convert 1, sl -> n, max sl sequences.
            One doc per batch. Pad_to_multiple_of etc etc.

"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List
import transformers

# Local imports
try:
    import _pathfix
except ImportError:
    from . import _pathfix
from dataiter import MultiTaskDataset


class BasicMTL(nn.Module):

    def __init__(self, enc_modelnm: str, config: transformers.BertConfig):
        super().__init__()

        self.config = config
        self.n_max_len: int = self.config.max_position_embeddings
        self.n_hid_dim: int = self.config.hidden_size

        # Encoder responsible for giving contextual vectors to subword tokens
        self.encoder = transformers.BertModel.from_pretrained(enc_modelnm)

        # Span width embeddings give a fix dim score to width of spans (usually 1 to config.max_span_width (~5))
        self.span_width_embeddings = nn.Embedding(num_embeddings=config.max_span_width,
                                                  embedding_dim=config.metadata_feature_size)

        # Used to push 768dim contextual vecs to 1D vectors for attention computation during span embedding creation
        self.span_attend_projection = torch.nn.Linear(config.hidden_size, 1)

        # Mention scorer (Unary, hdim) takes span representations and passes them through a 2 layer FFNN to score
        #   whether they are valid spans or not.
        #   NOTE: its use is conflated because it tries to do two things
        #       (i) find syntactically incoherent spans
        #       (ii) find spans that are not anaphoric
        span_embedding_dim = 3 * config.hidden_size + config.metadata_feature_size
        self.unary_coref = nn.Sequential(
            nn.Linear(span_embedding_dim, config.unary_hdim),
            nn.ReLU(),
            nn.Dropout(config.coref_dropout),
            nn.Linear(config.unary_hdim, 1),
        )

        self.binary_coref = nn.Sequential(
            nn.Linear((span_embedding_dim * 3), config.binary_hdim),
            nn.ReLU(),
            nn.Dropout(config.coref_dropout),
            nn.Linear(config.binary_hdim, 1),
        )

    def get_span_word_attention_scores(self, hidden_states, span_starts, span_ends):
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
        document_range = torch.arange(start=0, end=hidden_states.shape[0], device=hidden_states.device).unsqueeze(
            0).repeat(span_starts.shape[0], 1)  # [num_cand, num_words]
        token_mask = torch.logical_and(document_range >= span_starts.unsqueeze(1),
                                       document_range <= span_ends.unsqueeze(1))  # [num_cand, num_words]
        token_attn = self.span_attend_projection(hidden_states).squeeze(1).unsqueeze(0)  # [1, num_words]
        token_attn = F.softmax(torch.log(token_mask.float()) + token_attn, 1)  # [num_cand, num_words]span
        return token_attn

    def get_span_embeddings(self, hidden_states, span_starts, span_ends):
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
        span_width_emb = F.dropout(span_width_emb, p=self.config.coref_dropout, training=self.training)
        emb.append(span_width_emb)

        token_attention_scores = self.get_span_word_attention_scores(hidden_states, span_starts,
                                                                     span_ends)  # [num_cand, num_words]
        attended_word_representations = torch.mm(token_attention_scores, hidden_states)  # [num_cand, emb_size]
        emb.append(attended_word_representations)
        return torch.cat(emb, dim=1)

    @staticmethod
    def bucket_distance(distances):
        """
        CODE copied from https://gitlab.inria.fr/magnet/mangoes/-/blob/coref_exp/mangoes/modeling/coref.py#L496
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
    def extract_spans(candidate_starts: torch.Tensor,
                      candidate_ends: torch.Tensor,
                      candidate_span_scores: torch.Tensor,
                      num_top_mentions: int):
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

    def forward(self,
                input_ids: torch.tensor,
                attention_mask: torch.tensor,
                token_type_ids: torch.tensor,
                sentence_map: List[int],
                word_map: List[int],
                n_words: int,
                n_subwords: int,
                candidate_starts: torch.tensor,
                candidate_ends: torch.tensor
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

        """

        '''
            Step 1: Encode
            
            Just run the tokenized, sparsely encoded sequence through a BERT model
            
            It takes (n, 512) tensors and returns a (n, 512, 768) summary (each word has a 768 dim vec).
            We reshape it back to (n*512, 768) dim vec.
            
            Using masked select, we remove the padding tokens from encoded (and corresponding input ids).
        '''
        encoded = self.encoder(input_ids, attention_mask)[0]  # n_seq, m_len, h_dim
        encoded = encoded.reshape((-1, self.n_hid_dim))  # n_seq * m_len, h_dim

        # Remove all the padded tokens, using info from attention masks
        encoded = torch.masked_select(encoded, attention_mask.bool().view(-1, 1)) \
            .view(-1, self.n_hid_dim)  # n_words, h_dim
        input_ids = torch.masked_select(input_ids, attention_mask.bool()) \
            .view(-1, 1)  # n_words, h_dim

        # TODO: some gold stuff

        '''
            Step 2: Span embeddings
            
            candidate_span_embeddings: 3*h_dim + meta_dim 
            Span embeddings: based on candidate_starts, candidate_ends go through encoded 
                and find start and end subword embeddings. Concatenate them. 
                add embedding corresponding to the width of the span
                add an attention weighted sum of vectors within the span.  
        '''
        # n_cands, 3*h_dim + meta_dim
        candidate_span_embeddings = self.get_span_embeddings(encoded, candidate_starts, candidate_ends)

        '''
            Step 3: Span Pruning (for Coref)
            
            candidate_span_scores: n_cands
            Pass the span embeddings through a 2L FFNN (w dropout) and get a scalar score indicating the models' 
                predictions about them. We now need through sort through the spans and keep top-k.
                
            num_top_antecedents: int
            Getting top-k: Either we stick with the hard limit (max_top_antecedents = ~50) 
                or we keep antecedents proportional to doc length (say 40% of doc length) 
                if we have under 50 antecedents to begin with 
        '''
        # For Coref task, we pass the span embeddings through the span pruner
        candidate_span_scores = self.unary_coref(candidate_span_embeddings).squeeze(1)  # n_cands
        # TODO: joe adds span width embeddings here but we skip it for simplicity's sake.

        # get beam size
        n_top_spans = int(float(n_subwords * self.config.top_span_ratio))
        n_top_ante = min(self.config.max_top_antecedents, n_top_spans)

        # Get top mention scores and sort by span order (avoiding overlaps in selected spans)
        top_span_indices = self.extract_spans(candidate_starts, candidate_ends,
                                              candidate_span_scores, n_top_spans)
        top_span_starts = candidate_starts[top_span_indices]  # [top_cand]
        top_span_ends = candidate_ends[top_span_indices]  # [top_cand]
        top_span_emb = candidate_span_embeddings[top_span_indices]  # [top_cand, span_emb]
        top_span_mention_scores = candidate_span_scores[top_span_indices]  # [top_cand]

        # TODO: some gold stuff
        # TODO: some meta stuff

        '''
            Step 4: Filtering antecedents for each anaphor
            1. The antecedent must occur before the anaphor: modeled by a lower triangular identity matrix based mask
            2. Only top K prev. antecedents are considered. They are sorted based on their mention scores.
                For this: we put the mask over mention scores and argsort to get top indices. 
                These indices are then set as one, rest as zero in a new mask. E.g.
                
                Let there are 4 spans. k (max antecedents to be considered) is 2
                mention_scores = [2,4,-1,3]
                all_prev = [
                    [1, 0, 0, 0],
                    [1, 1, 0, 0],
                    [1, 1, 1, 0],
                    [1, 1, 1, 1]
                ] # i.e. for the first span, the candidates must be only 1; 
                # for third, the first three spans are candidates .. etc
                desired_mask = [
                    [0, _, _, _], # though we can select k=2 for it, there is only 1 candidate before the span
                    [1, 0, _, _], # we can select k=2, we have only two options,
                    [1, 0, _, _], # though we can select the third one, the first two have higher mention score
                    [1, 3, _, _] # the top k=2 mentions based on mention score
                ]
        '''
        # Create an antecedent mask: lower triangular matrix =e, rest 0 indicating the candidates for each span
        # [n_ana, n_ana]
        antecedents_mask = torch.ones(n_top_spans, n_top_spans, device=encoded.device).tril()

        # Argsort mention scores for each span, cognizant of the fact that the choice for each should be
        antecedents_per_ana_ind = torch.argsort(top_span_mention_scores + torch.log(antecedents_mask),
                                                descending=True, dim=1)  # [n_ana, n_ana]

        # Add 1 to indices (temporarily, to distinguish between index '0', and ignoring an index '_'
        #   (in the matrix in code comments above), and take its lower triangular as well
        antecedents_per_ana_ind = (antecedents_per_ana_ind + 1).tril()  # [n_ana, n_ana]

        # The [n_ana, n_ana] lower triangular mat now has sorted span indices.
        # We further need to clamp them to k.
        # For that, we just crop antecedents_per_ana_ind
        antecedents_per_ana_ind = antecedents_per_ana_ind[:, :config.max_top_antecedents]  # [n_ana, n_ante]
        # # Below is garbage, ignore
        # # For that, a simple col mul. will suffice
        # column_mask = torch.zeros((num_top_mentions,), device=encoded.device, dtype=torch.float)
        # column_mask[:config.max_top_antecedents] = 1
        # antecedents_per_ana_ind = antecedents_per_ana_ind*column_mask

        # Finally we subtract 1 (and negative indicates things which we don't want)
        antecedents_per_ana_ind = (antecedents_per_ana_ind - 1).to(torch.long)  # [n_ana, n_ante]

        # At this point, antecedents_per_ana_ind has -1 representing masked out things,
        #   and 0+ int to repr. actual indices
        antecedents_per_ana_emb = top_span_emb[antecedents_per_ana_ind]  # [n_ana, n_ante, span_emb]
        # This mask is needed to ignore the antecedents which occur after the anaphor
        antecedents_per_ana_emb[antecedents_per_ana_ind < 0] = 0

        '''
            Step 5: Finally, let's do pairwise scoring of spans and their candidate antecedent scores.
            We concat
                - a (n_anaphor, 1, span_emb) mat repr anaphors
                - a (n_anaphor, max_antecedents, span_emb) mat repr antecedents for each anaphor
                - a (n_anaphor, max_antecedents, span_emb) mat repr element wise mul b/w the two.
        '''
        similarity_emb = antecedents_per_ana_emb * top_span_emb.unsqueeze(1)  # [n_ana, n_ante, span_emb]
        anaphor_emb = top_span_emb.unsqueeze(1).repeat(1, config.max_top_antecedents, 1)  # [n_ana, n_ante, span_emb]
        pair_emb = torch.cat([anaphor_emb, antecedents_per_ana_emb, similarity_emb], 2)  # [n_ana], n_ante, 3*span_emb]

        # Finally, pass it through the params and get the scores
        antecedent_scores = self.binary_coref(pair_emb).squeeze(-1)  # [n_ana, n_ante]

        # Dummy scores are set to zero (for reasons explained in Lee et al 2017 e2ecoref)
        dummy_scores = torch.zeros([n_top_spans, 1], device=encoded.device)  # [n_ana, 1]

        antecedent_scores = torch.cat([antecedent_scores, dummy_scores], dim=1)  # [n_ana, n_ante + 1]

        # Now we just return them.


if __name__ == '__main__':

    config = transformers.BertConfig('bert-base-uncased')
    config.max_span_width = 5
    config.coref_dropout = 0.3
    config.metadata_feature_size = 20
    config.unary_hdim = 1000
    config.binary_hdim = 2000
    config.top_span_ratio = 0.4
    config.max_top_antecedents = 50

    model = BasicMTL('bert-base-uncased', config=config)
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

    # Try to wrap it in a dataloader
    for x in MultiTaskDataset("ontonotes", "train", tokenizer=tokenizer, config=config, ignore_empty_coref=True):
        model(**x)
        print("haha")
        ...
