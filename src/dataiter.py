"""
    Different parts of the pipeline may want their data in different ways.
    We store all of them here.
"""

from dataloader import DataLoader
from utils.data import Document
from utils.nlp import to_toks


class DataLoaderToHFTokenizer:
    """
        Wrap a dataloader in such a way that a single string is returned corresponding to a dataloader's document.

        Usage snippet:

        # Training a tokenizer from scratch
        dl = DataLoader('ontonotes', split, ignore_empty_coref=True)
        docstrings = DataLoaderToHFTokenizer(dataloader=dl)
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

    def __init__(self, dataloader: DataLoader):
        self.dl = dataloader

    @staticmethod
    def format_doc(doc: Document) -> str:
        text = doc.document # is a list of list of strings
        text = to_toks(text)
        return ' '.join(text)

    def __iter__(self):
        for instance in self.dl:
            yield self.format_doc(instance)

    def __len__(self) -> int:
        return self.dl.__len__()
