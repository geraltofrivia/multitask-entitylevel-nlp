"""
    File containing classes which take text and return fixed or contextual embeddings corresponding to:
        - tokens
        - chars ? (TODO)
        - summary vec (TODO)

    So far, we have GloVe, and BERT embeddings implemented
"""

import torch
from typing import List, Dict, Callable
from transformers import BertTokenizer, BertModel

# Local imports
from utils.misc import pop


class BertEmbeddings:
    """

        Input should be a list of tokens. We are going to internally concatenate them and treat them as one string.

        ## Usage:

            embeddings = BertEmbeddings()
            text =  "Replace me by any text other you'd like."
            tokens =  text.split(' ')
            vectors = embeddings.encode(tokens)
    """

    def __init__(self, model_name='bert-base-uncased', subword_pooling='mean', debug: bool = True):

        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.uncased = 'uncased' in model_name
        self.subword_pooling_ = subword_pooling
        self.__debug = debug
        self.subword_pooling: Callable = {
            'first': self._pooling_first_,
            'mean': self._pooling_mean_
        }[subword_pooling]

    @staticmethod
    def _pooling_first_(subword_vectors: torch.Tensor) -> torch.Tensor:
        """ input is [n, hdim], output must be [1, hdim] """
        return subword_vectors[0, :].unsqueeze(0)

    @staticmethod
    def _pooling_mean_(subword_vectors: torch.Tensor) -> torch.Tensor:
        """ input is [n, hdim], output must be [1, hdim] """
        return torch.mean(subword_vectors, dim=0).unsqueeze(0)

    def _match_subwords_to_words(self, tokens: List[str], encoded_input: dict) -> Dict[int, int]:
        """
            Create a dictionary that matches subword indices to word indices
            TODO: make it handle UNKs
        """
        sw2w = {}
        sw_tokens = self.tokenizer.convert_ids_to_tokens(encoded_input['input_ids'].squeeze(0).tolist(),
                                                         skip_special_tokens=True)[:]
        tokens = tokens[:]
        curr_sw_index = 0
        curr_w_index = 0

        while True:

            # break if sw tokens are empty
            if not sw_tokens:
                break

            # print(sw_tokens, tokens)
            #     input()

            if sw_tokens[0] == tokens[0]:
                # exact match
                sw2w[curr_sw_index] = curr_w_index
                sw_tokens.pop(0)
                tokens.pop(0)
                curr_sw_index += 1
                curr_w_index += 1
            else:
                sw_phrase = ''
                sw_selected = -1
                for i, next_word in enumerate(sw_tokens):
                    next_word = next_word[:]
                    next_word = next_word if not next_word.startswith('##') else next_word[2:]
                    sw_phrase += next_word

                    if sw_phrase == tokens[0]:
                        sw_selected = i
                        break

                if sw_selected < 0:
                    raise AssertionError(f"It seems that no subwords ({sw_tokens[:3]}) can form the next token: {tokens[0]}")

                for i in range(sw_selected+1):
                    sw2w[curr_sw_index + i ] = curr_w_index

                curr_w_index += 1
                curr_sw_index += sw_selected + 1
                tokens.pop(0)
                pop(sw_tokens, list(range(sw_selected+1)))

        return sw2w

    def pool_subword_vectors(self, tokens: List[str], encoded_input: dict, output: torch.Tensor) -> torch.Tensor:

        if self.uncased:
            tokens = [tok.lower() for tok in tokens]

        subword_to_word_index = self._match_subwords_to_words(tokens, encoded_input)
        word_to_subword_index = {}
        for sw_in, w_in in subword_to_word_index.items():
            word_to_subword_index[w_in] = word_to_subword_index.get(w_in, []) + [sw_in]

        if self.__debug:
            # Check if the word to subword index is in order
            for i, word_i in zip(range(len(word_to_subword_index)), word_to_subword_index.keys()):
                if not i == word_i:
                    raise AssertionError("Word to subword index is not in order. This needs to be handled explicitly")

        # Break down the output vectors based on each word
        word_vectors = [output[subwords_in_this_word,:] for subwords_in_this_word in word_to_subword_index.values()]

        # Pool each of them
        word_vectors = [self.subword_pooling(word_vector) for word_vector in word_vectors]

        # Stack them into a torch tensor and send it back
        word_vectors_t = torch.stack(word_vectors, dim=1)

        return word_vectors_t

    def encode(self, tokens: List[str]) -> torch.tensor:
        """
            Works with one sequence at a time, not a batch of sequence.
            Input the text sequence, we're going to encode it
                and pool subword vectors into word vectors
                and return a sequence of contextual vectors corresponding to each token in the input
        """

        # Turn the tokens into continuous string
        # text = ' '.join(tokens)

        # Use the BERT subword tokenizer on the string, and get encoded vectors corresponding to subword tokens.
        encoded_input = self.tokenizer(tokens, return_tensors='pt', add_special_tokens=False, is_split_into_words=True)
        output = self.model(**encoded_input)[0]                                         # (1, subword seq len, hidden dim)

        # Since we are working with one batch, let's squeeze away the 0th dim out of outputs
        output = output.squeeze(0)                                                      # (subword seq len, hidden dim)

        # Figure out which subwords belong to which tokens, and do some sort of pooling, as predetermined.
        pooled_output = self.pool_subword_vectors(tokens, encoded_input, output)        # (1, word seq len, hidden dim)

        return pooled_output


if __name__ == '__main__':
    be = BertEmbeddings()
    print(be.encode("I see a potato".split()).shape)