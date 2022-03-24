"""
    Different parts of the pipeline may want their data in different ways.
    We store all of them here.
"""
import json
import torch
import pickle
import warnings
import numpy as np
from pathlib import Path
import transformers
from tqdm.auto import tqdm
from typing import List, Iterable
from torch.utils.data import Dataset

# Local imports
try:
    import _pathfix
except ImportError:
    from . import _pathfix
from utils.nlp import to_toks, match_subwords_to_words
from config import LOCATIONS as LOC, NPRSEED
from utils.data import Document

np.random.seed(NPRSEED)


class MultiTaskDataset(Dataset):
    def __init__(
            self,
            src: str,
            split: str,
            config,
            tokenizer: transformers.BertTokenizer,
            shuffle: bool = False,
            tasks: Iterable[str] = (),
            rebuild_cache: bool = False,
    ):
        # TODO: make it such that multiple values can be passed in 'split'
        self._src_ = src
        self._split_ = split
        self._shuffle_ = shuffle
        self._tasks_ = sorted(tasks)
        self.tokenizer = tokenizer
        self.config = config

        self.ner_tag_dict: dict = {"__na__": 0}

        # Pull word replacements from the manually entered list
        with (LOC.manual / "replacements.json").open("r") as f:
            self.replacements = json.load(f)

        self.data, flag_successfully_pulled_from_disk = self.load_from_disk(
            rebuild_cache
        )

        if not flag_successfully_pulled_from_disk:
            self.data = RawDataset(
                src=src, split=split, tasks=self._tasks_, shuffle=shuffle
            )
            self.process()
            self.write_to_disk()

    def write_to_disk(self):
        """
        Write to MultiTaskDatasetDump_<task1>_<task2>[ad infinitum].pkl in /data/parsed/self._src_/self._split_
        File names could be:
             MultiTaskDatasetDump_coref.pkl
             MultiTaskDatasetDump_ner_gold.pkl
             MultiTaskDatasetDump_ner_spacy.pkl
             MultiTaskDatasetDump_coref_ner_gold.pkl
        """
        # Prep the file name
        dump_fname = LOC.parsed / self._src_ / self._split_ / 'MultiTaskDatasetDump'
        for task in self._tasks_:
            dump_fname = str(dump_fname) + f"_{task}"
        dump_fname = Path(dump_fname + ".pkl")

        with dump_fname.open("wb+") as f:
            pickle.dump((self.data, self.config), f)

    def load_from_disk(self, ignore_cache: bool) -> (list, bool):
        """
            Look for MultiTaskDatasetDump_<task1>_<task2>[ad infinitum].pkl in /data/parsed/self._src_/self._split_
            File names could be:
                 MultiTaskDatasetDump_coref.pkl
                 MultiTaskDatasetDump_ner_gold.pkl
                 MultiTaskDatasetDump_ner_spacy.pkl
                 MultiTaskDatasetDump_coref_ner_gold.pkl
        :return: a list of processed dicts (optional) and
            a bool indicating whether we successfully pulled sthing from the disk or not
        """
        success = False

        if ignore_cache:
            return None, success

        # Prep the file name
        dump_fname = LOC.parsed / self._src_ / self._split_ / 'MultiTaskDatasetDump'
        for task in self._tasks_:
            dump_fname = str(dump_fname) + f"_{task}"
        dump_fname = Path(dump_fname + ".pkl")

        # Check if file exists
        if not dump_fname.exists():
            warnings.warn(
                f"Processed (training ready) data not found on {dump_fname}."
                "Reprocessing will commence now but will take some time. Approx. 5 min."
            )
            return None, success

        # Pull the data, and the config
        with dump_fname.open("rb") as f:
            data, old_config = pickle.load(f)

        # Check if config matches
        # TODO: we need a way to iterate over all fields of the config somehow
        if (
                config.vocab_size == old_config.vocab_size
                and config.hidden_size == old_config.hidden_size
                and config.max_span_width == old_config.max_span_width
        ):
            return data, True
        else:
            warnings.warn(
                f"Processed (training ready) found on {dump_fname}. But the config files mismatch."
                "Reprocessing will commence now but will take some time. Approx. 5 min."
            )

            return None, False

    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, item):
        return self.data[item]

    def process(self):
        self.data = [self.process_document_generic(datum) for datum in tqdm(self.data)]

    def handle_replacements(self, tokens: List[str]) -> List[str]:
        return [self.replacements.get(tok, tok) for tok in tokens]

    @staticmethod
    def get_candidate_labels_mangoes(
            candidate_starts, candidate_ends, labeled_starts, labeled_ends, labels
    ):
        """
        CODE copied from https://gitlab.inria.fr/magnet/mangoes/-/blob/coref_exp/mangoes/modeling/coref.py#L470
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
        same_start = torch.eq(
            labeled_starts.unsqueeze(1), candidate_starts.unsqueeze(0)
        )  # [num_labeled, num_candidates]
        same_end = torch.eq(
            labeled_ends.unsqueeze(1), candidate_ends.unsqueeze(0)
        )  # [num_labeled, num_candidates]
        same_span = torch.logical_and(
            same_start, same_end
        )  # [num_labeled, num_candidates]

        # type casting in next line is due to torch not supporting matrix multiplication for Long tensors
        candidate_labels = torch.mm(
            labels.unsqueeze(0).float(), same_span.float()
        ).long()  # [1, num_candidates]
        return candidate_labels.squeeze(0)  # [num_candidates]

    @staticmethod
    def get_candidate_labels(
            candidate_starts, candidate_ends, labeled_starts, labeled_ends, labels
    ):
        """
        Based on a cluster ID, make a #can, #can matrix where each row represents a distribution (multi-hot, discrete)
            over the antecedents. I.e.
            <i, j> = 1 => span i (anaphor) and span j (antecedent) are coreferent
            <i, j> = 0 => span i and span j are not coreferent

        Limitations
        -----------

        1. No way to distinguish spans which are invalid in the candidates
        1. No way to penalize for gold spans not even being a candidate
        1. Very sparse matrix?
        1. Down the line, we trim this matrix based on the span pruner.
            Pruned away spans have no way to backprop from these.

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

        same_start = torch.eq(
            labeled_starts.unsqueeze(1), candidate_starts.unsqueeze(0)
        )  # [num_labeled, num_candidates]
        same_end = torch.eq(
            labeled_ends.unsqueeze(1), candidate_ends.unsqueeze(0)
        )  # [num_labeled, num_candidates]
        same_span = torch.logical_and(same_start, same_end)

        # # Making a labeled span -> candidate span dictionary. Easier to work with lists.
        #  candidate_starts_ = candidate_starts.tolist()
        #  candidate_ends_ = candidate_ends.tolist()
        #  labeled_starts_ = labeled_starts.tolist()
        #  labeled_ends_ = labeled_ends.tolist()
        #
        #  labeled_spans_to_candidate_spans: dict = {}
        #
        #  for labeled_start_, labeled_end_ in zip(labeled_starts_, labeled_ends_):
        #
        #      try:
        #          i = candidate_starts_.index(labeled_start_)
        #          j = candidate_ends_.index(labeled_end_)
        #      except ValueError:
        #          continue
        #
        #      labeled_spans_to_candidate_spans[(labeled_start_, labeled_end_)] = (i, j)

        # Now, we use this nxn mat to repr where spans are coreferent
        # if a row i is [0,1,1,0,0,0] that implies that span i is coreferent to span 1 and 2.
        truth = torch.zeros(
            (candidate_starts.shape[0], candidate_starts.shape[0]),
            dtype=torch.long,
            device=candidate_starts.device,
        )

        for cluster_id in range(1, 1 + torch.max(labels).item()):
            # Get indices corresponding to these labels in the candidate space
            indices = same_span[labels == cluster_id].nonzero()[:, 1]  # [n_spans,]

            # x: Repeat it like (a, b, c, a, b, c, a, b, c)
            # y: Repeat it like (a, a, a, b, b, b, c, c, c)
            ind_x = indices.repeat(
                indices.shape[0],
            )  # [n_spans^^2,]
            ind_y = (
                indices.repeat(indices.shape[0], 1).transpose(1, 0).reshape(-1)
            )  # [n_spans^^2,]

            # Using the x and y as coordinates, fill in the matrix with ones.
            truth[ind_x, ind_y] = 1

        # Make it a lower triangular matrix to avoid cases where anaphor appears before the antecedent
        # Next also,
        truth = truth.tril()
        truth = (
                truth - torch.eye(candidate_starts.shape[0], candidate_starts.shape[0])
        ).clamp(0)

        return truth

    def process_coref(
            self,
            instance: Document,
            generic_processed_stuff: dict,
            word2subword_starts: dict,
            word2subword_ends: dict,
    ) -> dict:
        """
            Work with 'generic' processed stuff and the raw instance to yield coref specific things
                - cluster IDs specific to gold annotations (but imposed on cluster space)
                - gold start and end spans and their corresponding cluster IDs
        :return: a dict which gets should be added to generic_processed_stuff
        """

        """
           # Label management:
               - change the label from being a word level index to a subword (bert friendly) level index
        """
        try:
            gold_starts = torch.tensor(
                [
                    word2subword_starts[span[0]]
                    for cluster in instance.coref.spans
                    for span in cluster
                ],
                dtype=torch.long,
                device=self.config.device,
            )  # n_gold
            gold_ends = torch.tensor(
                [
                    word2subword_ends[span[1] - 1]
                    for cluster in instance.coref.spans
                    for span in cluster
                ],
                dtype=torch.long,
                device=self.config.device,
            )  # n_gold
        except KeyError as e:
            print(e)
            print("oh shit")

        # 1 is added at cluster ID ends because zero represents "no link/no cluster". I think...
        gold_cluster_ids = (
                torch.tensor(
                    [
                        cluster_id
                        for cluster_id, cluster in enumerate(instance.coref.spans)
                        for _ in range(len(cluster))
                    ],
                    dtype=torch.long,
                    device=self.config.device,
                )
                + 1
        )  # n_gold

        candidate_starts = generic_processed_stuff["candidate_starts"]
        candidate_ends = generic_processed_stuff["candidate_ends"]

        cluster_labels = self.get_candidate_labels_mangoes(
            candidate_starts, candidate_ends, gold_starts, gold_ends, gold_cluster_ids
        )  # [n_cand, n_cand]

        coref_specific = {  # Pred stuff
            "gold_cluster_ids_on_candidates": cluster_labels,
            "gold_starts": gold_starts,
            "gold_ends": gold_ends,
            "gold_cluster_ids": gold_cluster_ids,
        }

        return coref_specific

    def process_ner_gold(
            self,
            instance: Document,
            generic_processed_stuff: dict,
            word2subword_starts: dict,
            word2subword_ends: dict,
    ) -> dict:
        """
            Work with generic processed stuff to also work with NER things
            TODO: what do we need ?
        :return:
        """
        gold_starts = torch.tensor(
            [word2subword_starts[span[0]] for span in instance.ner_gold.spans],
            dtype=torch.long,
            device=self.config.device,
        )  # n_gold
        gold_ends = torch.tensor(
            [word2subword_ends[span[1] - 1] for span in instance.ner_gold.spans],
            dtype=torch.long,
            device=self.config.device,
        )
        gold_labels = []
        for tag in instance.ner_gold.tags:
            if tag not in self.ner_tag_dict:
                self.ner_tag_dict[tag] = len(self.ner_tag_dict)
            gold_labels.append(self.ner_tag_dict[tag])
        gold_labels = torch.tensor(
            gold_labels, dtype=torch.long, device=self.config.device
        )

        # Now to superimpose this tensor on the candidate space.
        candidate_starts = generic_processed_stuff["candidate_starts"]
        candidate_ends = generic_processed_stuff["candidate_ends"]
        gold_labels = self.get_candidate_labels_mangoes(
            candidate_starts, candidate_ends, gold_starts, gold_ends, gold_labels
        )  # [n_cand, n_cand]

        ner_specific = {
            "gold_starts": gold_starts,
            "gold_ends": gold_ends,
            "gold_labels": gold_labels,
        }

        return ner_specific

    def process_document_generic(self, instance: Document) -> dict:
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

        """
            Tokenize the document (as per BERT requirements: subwords)
            Edge cases to handle: noted in data/manual/replacements.json                
        """
        tokens = to_toks(instance.document)
        tokens = self.handle_replacements(tokens)
        tokenized = self.tokenizer(
            tokens,
            add_special_tokens=False,
            padding=True,
            truncation=False,
            pad_to_multiple_of=n_mlen,
            is_split_into_words=True,
            return_tensors="pt",
            return_length=True,
        )

        n_subwords = tokenized.attention_mask.sum().item()

        """
            Find the word and sentence corresponding to each subword in the tokenized document
        """
        # Create a subword id to word id dictionary
        subword2word = match_subwords_to_words(
            tokens, tokenized.input_ids, self.tokenizer
        )
        word2subword_all = {}
        for sw_id, w_id in subword2word.items():
            word2subword_all[w_id] = word2subword_all.get(w_id, []) + [sw_id]
        word2subword_starts = {k: v[0] for k, v in word2subword_all.items()}
        word2subword_ends = {k: v[-1] for k, v in word2subword_all.items()}

        wordid_for_subword = torch.tensor(
            [subword2word[subword_id] for subword_id in range(n_subwords)],
            dtype=torch.long,
            device=self.config.device,
        )
        sentid_for_subword = torch.tensor(
            [instance.sentence_map[word_id] for word_id in wordid_for_subword],
            dtype=torch.long,
            device=self.config.device,
        )
        # subwordid_for_word_start = torch.tensor([word2subword_starts[word_id]
        #                                          for word_id in range(len(word2subword_starts))],
        #                                         dtype=torch.long, device=self.config.device)
        # subwordid_for_word_end = torch.tensor([word2subword_ends[word_id]
        #                                        for word_id in range(len(word2subword_ends))],
        #                                       dtype=torch.long, device=self.config.device)

        # 1 marks that the index is a start of a new token, 0 marks that it is not.
        word_startmap_subword = wordid_for_subword != torch.roll(wordid_for_subword, 1)

        # Resize these tokens as outlined in docstrings
        input_ids = tokenized.input_ids.reshape((-1, n_mlen))  # n_seq, m_len
        token_type_ids = tokenized.token_type_ids.reshape((-1, n_mlen))  # n_seq, m_len
        attention_mask = tokenized.attention_mask.reshape((-1, n_mlen))

        """
            Span Iteration: find all valid contiguous sequences of inputs. 
        
            We create subsequences from it, upto length K and pass them all through a span scorer.
            1. Create dumb start, end indices
            2. Exclude the ones which exist at the end of sentences (i.e. start in one, end in another)
        """

        candidate_starts = (
            torch.arange(start=0, end=n_subwords, device=self.config.device)
                .unsqueeze(1)
                .repeat(1, self.config.max_span_width)
        )  # n_subwords, max_span_width
        candidate_ends = candidate_starts + torch.arange(
            start=0, end=self.config.max_span_width, device=self.config.device
        ).unsqueeze(
            0
        )  # n_subwords, max_span_width

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
        candidate_ends_sent_id = sentid_for_subword[
            torch.clamp(candidate_ends, max=n_subwords - 1)
        ]
        filter_different_sentences = candidate_starts_sent_id == candidate_ends_sent_id

        # filter_candidate_starts_midword = word_startmap_subword[candidate_starts]
        # filter_candidate_ends_midword = word_startmap_subword[torch.clamp(candidate_ends, max=n_subwords - 1)]

        candidate_mask = (
                filter_beyond_document & filter_different_sentences
        )  # & filter_candidate_starts_midword & \
        #  filter_candidate_ends_midword  # n_subwords, max_span_width

        # Now we flatten the candidate starts, ends and the mask and do an index select to ignore the invalid ones
        candidate_mask = candidate_mask.view(-1)  # n_subwords * max_span_width
        candidate_starts = torch.masked_select(
            candidate_starts.view(-1), candidate_mask
        )  # n_subwords*max_span_width
        candidate_ends = torch.masked_select(
            candidate_ends.view(-1), candidate_mask
        )  # n_subwords*max_span_width

        # DEBUG
        if n_subwords > 512 and input_ids.shape != attention_mask.shape:
            print("potato")

        return_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "word_map": wordid_for_subword,
            "sentence_map": sentid_for_subword,
            "n_words": n_words,
            "n_subwords": n_subwords,
            "candidate_starts": candidate_starts,
            "candidate_ends": candidate_ends,
        }

        if "coref" in self._tasks_:
            return_dict["coref"] = self.process_coref(
                instance, return_dict, word2subword_starts, word2subword_ends
            )

        if "ner_gold" in self._tasks_:
            return_dict["ner_gold"] = self.process_ner_gold(
                instance, return_dict, word2subword_starts, word2subword_ends
            )

        return return_dict


class RawDataset(Dataset):
    def __init__(
            self, src: str, split: str, shuffle: bool = False, tasks: Iterable[str] = ()
    ):
        """
            Returns an iterable that yields one document at a time.
            It looks for pickled instances of class utils.data.Document in /data/parsed directories.

            You can configure it to filter out instances which do not have annotations for a particular thing e.g.,
                - coref by passing tasks=('coref',) (all instances w/o coref annotations will be skipped)
                - ner by passing tasks=('ner_gold',)
                - ner (silver std) by passing tasks=('ner_spacy',)
                - or both by tasks=('coref', 'ner_gold') etc etc


        :param src: the first child under /data/parsed, usually, the name of a dataset
        :param split: often different splits are stored in different directories, so this is the second child under
            /data/parsed/<src>
        :param shuffle: whether the dataset should be shuffled or not
        :param tasks: Read code comments but passing 'coref' and/or 'ner' as an item ensures that
            only instances which contain coref and/or ner annotations are returned.
        """

        # Sanity check params
        for task in tasks:
            if task not in ["coref", "ner_gold", "ner_spacy"]:
                raise AssertionError(
                    f"An unrecognized task name sent: {task}. "
                    "So far, we work with 'coref', 'ner_gold', 'ner_spacy'."
                )

        # super().__init__()
        self._src_ = src
        self._split_ = split
        self._shuffle_ = shuffle
        self._tasks = tasks
        self.data: List[Document] = []

        self.pull_from_disk()

    @staticmethod
    def get_fnames(dataset: str, split: str):
        return [fnm for fnm in (LOC.parsed / dataset / split).glob("dump*.pkl")]

    def pull_from_disk(self):
        """RIP ur mem"""

        filenames = self.get_fnames(self._src_, self._split_)
        if self._shuffle_:
            np.random.shuffle(filenames)

        for fname in filenames:

            with fname.open("rb") as f:
                data_batch: List[Document] = pickle.load(f)

            if self._shuffle_:
                np.random.shuffle(data_batch)

            for instance in data_batch:

                # See if the instance fits the criteria set in self._tasks
                if "coref" in self._tasks and instance.coref.isempty:
                    continue

                if "ner_gold" in self._tasks and instance.ner_gold.isempty:
                    continue

                if "ner_spacy" in self._tasks and instance.ner_spacy.isempty:
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
    ds = RawDataset('ontonotes', split, tasks=('coref',))
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

    def __init__(self, dataset: RawDataset):
        self.ds = dataset

    @staticmethod
    def format_doc(doc: Document) -> str:
        text = doc.document  # is a list of list of strings
        text = to_toks(text)
        return " ".join(text)

    def __iter__(self):
        for instance in self.ds:
            yield self.format_doc(instance)

    def __len__(self) -> int:
        return self.ds.__len__()


if __name__ == "__main__":

    # # Attempt to pull from disk
    # encoder = transformers.BertModel.from_pretrained(LOC.root / 'models' / 'huggingface'
    # / 'bert-base-uncased' / 'encoder')
    tokenizer = transformers.BertTokenizer.from_pretrained(
        LOC.root / "models" / "huggingface" / "bert-base-uncased" / "tokenizer"
    )
    config = transformers.BertConfig(
        LOC.root / "models" / "huggingface" / "bert-base-uncased" / "tokenizer"
    )

    # tokenizer = tf.BertTokenizer.from_pretrained('bert-base-uncased')
    # config = tf.BertConfig('bert-base-uncased')
    config.max_span_width = 8
    config.device = "cpu"
    tasks = ["ner_gold", "coref"]

    ds = MultiTaskDataset(
        "ontonotes",
        "train",
        config=config,
        tokenizer=tokenizer,
        tasks=tasks,
        # rebuild_cache=True,
    )

    # Custom fields in config
    # config.

    for x in ds:
        # process_document(x, tokenizer, config)
        break