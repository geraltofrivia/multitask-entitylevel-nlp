"""
    TODO:
        - in data sampler (pipeline, make sure to convert 1, sl -> n, max sl sequences.
            One doc per batch. Pad_to_multiple_of etc etc.

"""
import torch
import torch.nn as nn
from typing import List
import transformers as tf

# Local imports
from dataiter import process_document


class BasicMTL(nn.Module):

    def __init__(self, enc_modelnm: str, max_span_width: int = 30, max_candidates_coref: int = 50):
        super().__init__()

        self.encoder = tf.BertModel.from_pretrained(enc_modelnm)

        self.n_max_len: int = self.encoder.config.max_position_embeddings
        self.n_hid_dim: int = self.encoder.config.hidden_size

    def forward(self,
                input_ids: torch.tensor,
                attention_mask: torch.tensor,
                token_type_ids: torch.tensor,
                sentence_map: List[int],
                word_map: List[int],
                n_words: int,
                n_subwords: int
                ):
        """
        :param input_ids: tensor, shape (number of subsequences, max length), output of tokenizer, reshaped
        :param attention_mask: tensor, shape (number of subsequences, max length), output of tokenizer, reshaped
        :param token_type_ids: tensor, shape (number of subsequences, max length), output of tokenizer, reshaped
        :param sentence_map: list of sentence ID for each subword (excluding padded stuff)
        :param word_map: list of word ID for each subword (excluding padded stuff)
        :param n_words: number of words (not subwords) in the original doc
        :param n_subwords: number of subwords

        """

        '''
            Step 1: Encode
            
            Just run the tokenized, sparsely encoded sequence through a BERT model
            
            It takes (n, 512) tensors and returns a (n, 512, 768) summary (each word has a 768 dim vec).
            We reshape it back to (n*512, 768) dim vec.
        '''
        encoded = self.encoder(input_ids, attention_mask)[0]  # n_seq, m_len, h_dim
        encoded = encoded.reshape((-1, self.n_hid_dim))  # n_seq * m_len, h_dim

        # Remove all the padded tokens, using info from attention masks
        encoded = torch.masked_select(encoded, attention_mask.bool().view(-1, 1)) \
            .view(-1, self.n_hid_dim)  # n_words, h_dim
        input_ids = torch.masked_select(input_ids, attention_mask.bool().view(-1, 1)) \
            .view(-1, 1)  # n_words, h_dim

        print('oh well')


if __name__ == '__main__':

    from dataiter import RawCorefDataset
    from utils.nlp import to_toks

    model = BasicMTL('bert-base-uncased')
    config = tf.BertConfig('bert-base-uncased')
    tokenizer = tf.BertTokenizer.from_pretrained('bert-base-uncased')
    for x in RawCorefDataset("ontonotes", "train"):
        # op = tokenizer(to_toks(x.document), add_special_tokens=False, padding=True, pad_to_multiple_of=

        preprocced = process_document(x, tokenizer, config)
        model(**preprocced)
        print("haha")
        ...
