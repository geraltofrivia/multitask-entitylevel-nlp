"""
    File containing classes which take text and return fixed or contextual embeddings corresponding to:
        - tokens
        - chars ? (TODO)
        - summary vec (TODO)

    So far, we have GloVe, and BERT embeddings implemented
"""
import torch
import numpy as np
import transformers
from gensim.models import KeyedVectors
from typing import List, Callable

# Local imports
from utils.nlp import match_subwords_to_words
from config import LOCATIONS as LOC


class BertEmbeddings:
    """

    Input should be a list of tokens. We are going to internally concatenate them and treat them as one string.

    ## Usage:

        embeddings = BertEmbeddings()
        text =  "Replace me by any text other you'd like."
        tokens =  text.split(' ')
        vectors = embeddings.encode(tokens)

    TODO: padding is set to zero. Should it be?
    TODO: what happens when a doc is larger than the max length? We seem to be truncating it.
        Should we id-fy sentence by sentence instead?
    """

    def __init__(
            self, model_name="bert-base-uncased", subword_pooling="mean", debug: bool = True
    ):

        self.tokenizer = transformers.BertTokenizer.from_pretrained(model_name)
        self.model = transformers.BertModel.from_pretrained(model_name)
        self.uncased = "uncased" in model_name
        self.subword_pooling_ = subword_pooling
        self.__debug = debug
        self.subword_pooling: Callable = {
            "first": self._pooling_first_,
            "mean": self._pooling_mean_,
        }[subword_pooling]

    @staticmethod
    def _pooling_first_(subword_vectors: torch.Tensor) -> torch.Tensor:
        """input is [n, hdim], output must be [1, hdim]"""
        return subword_vectors[0, :].unsqueeze(0)

    @staticmethod
    def _pooling_mean_(subword_vectors: torch.Tensor) -> torch.Tensor:
        """input is [n, hdim], output must be [1, hdim]"""
        return torch.mean(subword_vectors, dim=0).unsqueeze(0)

    def pool_subword_vectors(
            self, tokens: List[str], input_ids: dict, output: torch.Tensor
    ) -> torch.Tensor:

        if self.uncased:
            tokens = [tok.lower() for tok in tokens]

        subword_to_word_index = match_subwords_to_words(
            tokens, input_ids, self.tokenizer
        )
        word_to_subword_index = {}
        for sw_in, w_in in subword_to_word_index.items():
            word_to_subword_index[w_in] = word_to_subword_index.get(w_in, []) + [sw_in]

        if self.__debug:
            # Check if the word to subword index is in order
            for i, word_i in zip(
                    range(len(word_to_subword_index)), word_to_subword_index.keys()
            ):
                if not i == word_i:
                    raise AssertionError(
                        "Word to subword index is not in order. This needs to be handled explicitly"
                    )

        # Break down the output vectors based on each word
        word_vectors = [
            output[subwords_in_this_word, :]
            for subwords_in_this_word in word_to_subword_index.values()
        ]

        # Pool each of them
        word_vectors = [
            self.subword_pooling(word_vector) for word_vector in word_vectors
        ]

        # Stack them into a torch tensor and send it back
        word_vectors_t = torch.stack(word_vectors, dim=1)

        return word_vectors_t

    def encode(self, tokens: List[str]) -> torch.Tensor:
        """
        Works with one sequence at a time, not a batch of sequence.
        Input the text sequence, we're going to encode it
            and pool subword vectors into word vectors
            and return a sequence of contextual vectors corresponding to each token in the input
        """

        # Turn the tokens into continuous string
        # text = ' '.join(tokens)

        # Use the BERT subword tokenizer on the string, and get encoded vectors corresponding to subword tokens.
        encoded_input = self.tokenizer(
            tokens,
            return_tensors="pt",
            add_special_tokens=False,
            is_split_into_words=True,
            truncation=False,
        )
        output = self.model(**encoded_input)[0]  # (1, subword seq len, hidden dim)

        # Since we are working with one batch, let's squeeze away the 0th dim out of outputs
        output = output.squeeze(0)  # (subword seq len, hidden dim)

        # Figure out which subwords belong to which tokens, and do some sort of pooling, as predetermined.
        pooled_output = self.pool_subword_vectors(
            tokens, encoded_input["input_ids"], output
        )  # (1, word seq len, hidden dim)

        return pooled_output

    def batch_encode(self, tokens: List[List[str]]) -> torch.Tensor:
        """Similar to encode but works with a batch of text sequences (tokenized)"""
        batch_encoded_input = self.tokenizer(
            tokens,
            return_tensors="pt",
            add_special_tokens=False,
            is_split_into_words=True,
            padding=True,
            truncation=False,
        )
        batch_output = self.model(**batch_encoded_input)[
            0
        ]  # ( bs, subword seq len, hidden dim)

        pooled_output = []
        for i, output in enumerate(batch_output):
            encoded_input = batch_encoded_input["input_ids"][i]
            pooled_output.append(
                self.pool_subword_vectors(tokens[i], encoded_input, output)
            )

        # Pad all the tensors to be able to stack them.
        n_batch = batch_output.shape[0]
        hiddendim = batch_output.shape[-1]
        maxlen = max(output.shape[1] for output in pooled_output)
        pooled_output_ = torch.zeros(
            (n_batch, maxlen, hiddendim), dtype=batch_output.dtype
        )
        for i, output in enumerate(pooled_output):
            pooled_output_[i, : output.shape[1], :] = output

        return pooled_output_


class Word2VecEmbeddings:
    def __init__(self):
        self.vectors = KeyedVectors.load_word2vec_format(LOC.word2vec, binary=True)

    def encode(self):
        ...


class GloVeEmbeddings:
    """
    Checks if the GloVe embeddings are converted to the w2v format already or not.
    If not, it does the conversion and then loads them.

    Conversion snippet:
        from gensim.test.utils import datapath, get_tmpfile
        from gensim.models import KeyedVectors
        from gensim.scripts.glove2word2vec import glove2word2vec
        glove_file = datapath('test_glove.txt')
        tmp_file = get_tmpfile("test_word2vec.txt")
        _ = glove2word2vec(glove_file, tmp_file)
        model = KeyedVectors.load_word2vec_format(tmp_file)

    TODO: padding is set to zero. Should it be?
    """

    def __init__(self):

        glove_loc = LOC.glove / "glove.6B.300d.w2v.txt"

        # Check if the GloVe file is already converted
        if not glove_loc.exists():
            # Do the conversion
            from gensim.scripts.glove2word2vec import glove2word2vec

            glove_file_loc = str(glove_loc).replace(".w2v.txt", ".txt")
            glove_loc.touch()
            _ = glove2word2vec(glove_file_loc, str(glove_loc))

        self.vectors = KeyedVectors.load_word2vec_format(glove_loc)

    # noinspection PyTypeChecker
    def encode(self, tokens: List[str]) -> torch.Tensor:

        output: List[np.ndarray] = []
        for token in tokens:
            try:
                output.append(self.vectors[token.lower()])
            except KeyError:
                output.append(np.zeros_like(self.vectors["the"]))

        return torch.Tensor(output)

    def batch_encode(self, tokens: List[List[str]]) -> torch.Tensor:

        n_batch = len(tokens)
        maxlen = max(len(seq) for seq in tokens)
        # noinspection PyUnresolvedReferences
        hdim = self.vectors["the"].shape[0]

        outputs = torch.zeros((n_batch, maxlen, hdim), dtype=torch.float32)

        # Start filling it in
        for i, seq in enumerate(tokens):
            encoded_seq = self.encode(seq)
            outputs[i, : len(encoded_seq), :] = encoded_seq

        return outputs


if __name__ == "__main__":
    # Testing encode
    be = BertEmbeddings()
    # noinspection SpellCheckingInspection
    print(
        be.encode(
            "I see a potatoatoatoato in my house. Nopessorryimeant houeses.".split()
        ).shape
    )
    #
    # Testing batch encode
    # noinspection SpellCheckingInspection
    tokens = [
        "I see a little silhouette of a man.".split(),
        "Replace me with whatever text you seem to have agrowing inclinationing for.".split(),
        "this one is smol".split(),
        "grabbless".split(),
    ]
    # print(be.batch_encode(tokens).shape)
    #
    # Testing truncations
    tokens = tokens[0] * 200
    print(be.encode(tokens).shape)
    #
    # ge = GloVeEmbeddings()
    # print('potato')
