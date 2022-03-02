"""
    Different parts of the pipeline may want their data in different ways.
    We store all of them here.
"""
import json
import torch
import pickle
import numpy as np
from typing import List
import transformers as tf
from tqdm.auto import tqdm
from torch.utils.data import Dataset

from utils.nlp import to_toks, match_subwords_to_words
from config import LOCATIONS as LOC, NPRSEED
from utils.data import Document

np.random.seed(NPRSEED)


class MultiTaskDataset(Dataset):

    def __init__(self, src: str, split: str, config, tokenizer: tf.BertTokenizer,
                 shuffle: bool = False, ignore_empty_coref: bool = False):
        self._src_ = src
        self._split_ = split
        self._shuffle_ = shuffle
        self._ignore_empty_coref_ = ignore_empty_coref
        self.tokenizer = tokenizer
        self.config = config

        # Pull word replacements from the manually entered list
        with (LOC.manual / 'replacements.json').open('r') as f:
            self.replacements = json.load(f)

        self.data = RawCorefDataset(src=src, split=split, ignore_empty_coref=ignore_empty_coref, shuffle=shuffle)
        self.process()

    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, item):
        return self.data[item]

    def process(self):
        self.data = [self.process_document(datum) for datum in tqdm(self.data)]

    def handle_replacements(self, tokens: List[str]) -> List[str]:
        return [self.replacements.get(tok, tok) for tok in tokens]

    def process_document(self, instance: Document) -> dict:
        """
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
        n_mlen = self.config.max_position_embeddings  # max length of the encoder

        '''
            Tokenize the document (as per BERT requirements: subwords)
            Edge cases to handle: noted in data/manual/replacements.json                
        '''
        tokens = to_toks(instance.document)
        tokens = self.handle_replacements(tokens)
        tokenized = self.tokenizer(tokens, add_special_tokens=False, padding=True, truncation=False,
                                   pad_to_multiple_of=n_mlen, is_split_into_words=True, return_tensors='pt',
                                   return_length=True)

        n_subwords = tokenized.attention_mask.sum().item()

        '''
            Find the word and sentence corresponding to each subword in the tokenized document
        '''
        # Create a subword id to word id dictionary
        subword2word = match_subwords_to_words(tokens, tokenized.input_ids, self.tokenizer)
        wordid_for_subword = torch.tensor([subword2word[subword_id] for subword_id in range(n_subwords)])
        # subwordid_for_word = torch.tensor([])     # TODO: #labels - fill this
        sentid_for_subword = torch.tensor([instance.sentence_map[word_id] for word_id in wordid_for_subword])

        # 1 marks that the index is a start of a new token, 0 marks that it is not.
        word_startmap_subword = wordid_for_subword != torch.roll(wordid_for_subword, 1)

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

        candidate_starts = torch.arange(start=0, end=n_subwords) \
            .unsqueeze(1) \
            .repeat(1, self.config.max_span_width)  # n_subwords, max_span_width
        candidate_ends = candidate_starts + torch.arange(start=0, end=self.config.max_span_width) \
            .unsqueeze(0)  # n_subwords, max_span_width

        """
            # Ignoring invalid spans
            1. Spans which end beyond the document size
            2. Spans which start and end in different sentences
            3. Spans which start OR end mid-word. 
                This one is going to be difficult. But we will be utilising the word map in some form.
                
            We combine all these boolean filters (filter_*: torch.Tensor) with a logical AND.
            NOTE: all filter_* tensors are of shape # n_subwords, max_span_width
        """
        filter_beyond_document = candidate_ends < n_subwords

        candidate_starts_sent_id = sentid_for_subword[candidate_starts]
        candidate_ends_sent_id = sentid_for_subword[torch.clamp(candidate_ends, max=n_subwords - 1)]
        filter_different_sentences = candidate_starts_sent_id == candidate_ends_sent_id

        filter_candidate_starts_midword = word_startmap_subword[candidate_starts]
        filter_candidate_ends_midword = word_startmap_subword[torch.clamp(candidate_ends, max=n_subwords - 1)]

        candidate_mask = filter_beyond_document & filter_different_sentences & filter_candidate_starts_midword & \
                         filter_candidate_ends_midword  # n_subwords, max_span_width

        # Now we flatten the candidate starts, ends and the mask and do an index select to ignore the invalid ones
        candidate_mask = candidate_mask.view(-1)  # n_subwords * max_span_width
        candidate_starts = torch.masked_select(candidate_starts.view(-1), candidate_mask)  # n_subwords*max_span_width
        candidate_ends = torch.masked_select(candidate_ends.view(-1), candidate_mask)  # n_subwords*max_span_width

        """
            # Label management:
                - for now, our only goal is to change the label from being a word level index 
                    to a subword (bert friendly) level index
                - down the line we need an eval function haha
        """
        # TODO: labels: fill this in.
        #  you will have to figure out 'how to interpret' a span like [2: 3] or even [2: 2] etc
        #  the way this done when making span vectors during the forward pass would dictate how these labels are done

        return_dict = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'word_map': wordid_for_subword,
            'sentence_map': sentid_for_subword,
            'n_words': n_words,
            'n_subwords': n_subwords,
            'candidate_starts': candidate_starts,
            'candidate_ends': candidate_ends,
            'instance': instance
        }

        return return_dict


class RawCorefDataset(Dataset):

    def __init__(self, src: str, split: str, shuffle: bool = False,
                 ignore_empty_coref: bool = False):

        # super().__init__()
        self._src_ = src
        self._split_ = split
        self._shuffle_ = shuffle
        self._ignore_empty_coref_ = ignore_empty_coref
        self.data: List[Document] = []

        self.pull_from_disk()

    @staticmethod
    def get_fnames(dataset: str, split: str):
        return [fnm for fnm in (LOC.parsed / dataset / split).glob('*.pkl')]

    def pull_from_disk(self):
        """ RIP ur mem """

        filenames = self.get_fnames(self._src_, self._split_)
        if self._shuffle_: np.random.shuffle(filenames)

        for fname in filenames:
            with fname.open('rb') as f:
                data_batch: List[Document] = pickle.load(f)

            if self._shuffle_: np.random.shuffle(data_batch)

            for instance in data_batch:

                if self._ignore_empty_coref_ and instance.coref.isempty:
                    continue

                self.data.append(instance)

    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, item):
        return self.data[item]

    def __iter__(self):
        return iter(self.data)


class DataLoaderToHFTokenizer:
    """
        IGNORE; Legacy uses
        Wrap a dataloader in such a way that a single string is returned corresponding to a dataloader's document.

        Usage snippet:

        # Training a tokenizer from scratch
        ds = RawCorefDataset('ontonotes', split, ignore_empty_coref=True)
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

    def __init__(self, dataset: RawCorefDataset):
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


if __name__ == '__main__':

    tokenizer = tf.BertTokenizer.from_pretrained('bert-base-uncased')
    config = tf.BertConfig('bert-base-uncased')
    config.max_span_width = 8

    ds = MultiTaskDataset('ontonotes', 'train', config=config, tokenizer=tokenizer, ignore_empty_coref=True)

    # Custom fields in config
    # config.

    for x in ds:
        # process_document(x, tokenizer, config)
        break
