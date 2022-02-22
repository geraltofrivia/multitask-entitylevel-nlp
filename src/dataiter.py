"""
    Different parts of the pipeline may want their data in different ways.
    We store all of them here.
"""

import pickle
import numpy as np
from pathlib import Path
import transformers as tf
from typing import List, Optional

from utils.nlp import to_toks, match_subwords_to_words
from config import LOCATIONS as LOC, NPRSEED
from utils.data import Document

np.random.seed(NPRSEED)


class Dataset:
    """
        Provide a dataset name, and split, and we load it for you.
        We do lazy loading so you never have to worry about overwhelming the mem
            unless of course the data is stored in one giant pickle file.
            in which case, we can't help you and please contact the preprocessing department

        # Usage

        ### Iterating through processed ontonotes split
        ds = Dataset('ontonotes', split='train', ignore_empty_coref=True)
        for doc in ds:
            ...
    """

    def __init__(self, data: str, split: str, shuffle: bool = False, ignore_empty_coref: bool = False):
        super().__init__()

        self.dir: Path = Path()
        self._n_: int = -1
        self.data = data
        self.split = split
        self._shuffle_: bool = shuffle
        self._ignore_empty_coref_: bool = ignore_empty_coref

        self.fnames = self.get_fnames(data, split)

        self.data: Optional[List[Document]] = None

    def _count_instances_(self, fdir: Path, fnames: List[str]) -> List[int]:
        """ Return a list of bridging instances in each file by loading each and storing its length """
        n: List[int] = []
        for fname in fnames:
            n_doc = 0
            with (fdir / fname).open('rb') as f:
                for instance in pickle.load(f):

                    if self._ignore_empty_coref_ and instance.coref.isempty:
                        continue

                    n_doc += 1

            n.append(n_doc)

        return n

    def all(self):

        for fname in self.fnames:
            with fname.open('rb') as f:
                data_batch: List[Document] = pickle.load(f)

            if self._shuffle_:
                np.random.shuffle(data_batch)

            for instance in data_batch:

                if self._ignore_empty_coref_ and not instance.coref.isempty:
                    continue

                # instance.finalise()
                yield instance

    def __len__(self) -> int:
        """
            Go through all the files and count the number of instances.
            If you did do this before. don't reopen all files
        """

        if self._n_ < 0:
            self._n_ = sum(self._count_instances_(self.dir, self.fnames))

        return self._n_

    @staticmethod
    def get_fnames(dataset: str, split: str):
        return [fnm for fnm in (LOC.parsed / dataset / split).glob('*.pkl')]

    def __iter__(self):
        """ This function enables simply calling ds() instead of ds.all(). """
        return self.all()


class DataLoaderToHFTokenizer:
    """
        Wrap a dataloader in such a way that a single string is returned corresponding to a dataloader's document.

        Usage snippet:

        # Training a tokenizer from scratch
        ds = Dataset('ontonotes', split, ignore_empty_coref=True)
        docstrings = DataLoaderToHFTokenizer(dataloader=ds)
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        if uncased:
            tokenizer.normalizer = normalizers.Lowercase()
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        # Make a BPETrainer
        trainer = trainers.BpeTrainer(
            special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
            vocab_size=n_vocab, show_progress=True)

        tokenizer.train_from_iterator(docstrings, trainer=trainer)
    """

    def __init__(self, dataset: Dataset):
        self.ds = dataset

    @staticmethod
    def format_doc(doc: Document) -> str:
        text = doc.document  # is a list of list of strings
        text = to_toks(text)
        return ' '.join(text)

    def __iter__(self):
        for instance in self.ds:
            yield self.format_doc(instance)

    def __len__(self) -> int:
        return self.ds.__len__()


"""
    TODO: move this function to a sampler class down the road
"""


def process_document(instance: Document, tokenizer: tf.BertTokenizer, config: tf.BertConfig) -> dict:
    """
        PS: TODO: move it somewhere saner, in a iter or something.

        PSA:
        Tokenizer will return tensors padded to max length of the model. For instance,
            doc length - 350, output shape - (1, 512)
            doc length - 514, output shape - (1, 1024) ...

        We then reshape (1, n*512) to (n, 512) and treat one document with multiple max lengths as
            "multiple documents" with one max length.

        NOTE: Yes this process will lead to some loss in contextual vectors since we're truncating the context
            (n*2)-2 times but it seems to be the canonical way to treat these issues.
    """
    # Some params pulled from config
    n_words = len(to_toks(instance.document))
    n_mlen = config.max_position_embeddings  # max length of the encoder

    '''
        Tokenize the document (as per BERT requirements: subwords)
    '''
    tokens = to_toks(instance.document)
    tokenized = tokenizer(tokens, add_special_tokens=False, padding=True, truncation=False,
                          pad_to_multiple_of=n_mlen, is_split_into_words=True, return_tensors='pt',
                          return_length=True)

    n_subwords = tokenized.lengths

    '''
        Find the word and sentence corresponding to each subword in the tokenized document
    '''
    # Create a subword id to word id dictionary
    # TODO: conv to tensor
    subword2word = match_subwords_to_words(tokens, tokenized.input_ids, tokenizer)
    wordmap_subword = [subword2word[subwordid] for subwordid in range(n_subwords)]
    sentmap_subword = [instance.sentence_map[wordid] for wordid in wordmap_subword]

    # Resize these tokens as outlined in docstrings
    input_ids = tokenized.input_ids.reshape((-1, n_mlen))  # n_seq, m_len
    token_type_ids = tokenized.token_type_ids.reshape((-1, n_mlen))  # n_seq, m_len
    attention_mask = tokenized.attention_mask.reshape((-1, n_mlen))

    '''
        Span Iteration: find all valid contiguous sequences of inputs. 
    
        We create subsequences from it, upto length K and pass them all through a span scorer.
        1. Create dumb start, end indices
        2. Exclude the ones which exist at the end of sentences (i.e. start in one, end in another)
    '''

    # TODO: #spaniter add a custom field in config -> max span width
    # TODO: #spaniter copy over snippets of candidate start/end from notebook

    return_dict = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids,
        'word_map': wordmap_subword,
        'sentence_map': sentmap_subword,
        'n_words': n_words,
        'n_subwords': n_subwords
    }

    return return_dict


if __name__ == '__main__':

    tokenizer = tf.BertTokenizer.from_pretrained('bert-base-uncased')
    config = tf.BertConfig('bert-base-uncased')
    ds = Dataset('ontonotes', 'train')

    for x in ds:
        process_document(x, tokenizer, config)
        break
