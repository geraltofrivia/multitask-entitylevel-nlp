"""
    Different parts of the pipeline may want their data in different ways.
    We store all of them here.
"""
import json
import pickle
import random
import warnings
from pathlib import Path
from typing import List, Iterable, Union, Optional, Callable, Dict, Tuple

import numpy as np
import torch
import transformers
from mytorch.utils.goodies import FancyDict
from torch.utils.data import Dataset
from tqdm.auto import tqdm

# Local imports
try:
    import _pathfix
except ImportError:
    from . import _pathfix
from config import LOCATIONS as LOC, _SEED_ as SEED, KNOWN_TASKS, unalias_split, is_split_train
from utils.exceptions import NoValidAnnotations, LabelDictNotFound, UnknownTaskException
from utils.nlp import to_toks, match_subwords_to_words
from utils.data import Document, Tasks
from utils.misc import check_dumped_config, compute_class_weight_sparse, SerializedBertConfig

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class MultiTaskDataIter(Dataset):
    def __init__(
            self,
            src: str,
            split: str,
            tasks: Tasks,
            config: Union[FancyDict, SerializedBertConfig],
            tokenizer: transformers.BertTokenizer,
            allow_speaker_ids: bool = True,
            shuffle: bool = False,
            rebuild_cache: bool = False
    ):
        """
            Important thing to note is:
                - it will never move tensors to GPU. Will always stick to CPU (so that we dont have a mem leak).
                - ideally you should delete the Dataset and make a new one every time you start a new epoch.
        :param src: the name of the main dataset (e.g. ontonotes)
        :param split: the name of the dataset split (e.g. 'train', or 'development')
        :param config: the config dict (SerializedBertConfig + required params)
        :param tokenizer: a pretrained BertTokenizer
        :param shuffle: a flag which if true would shuffle instances within one batch. Should be turned on.
        :param tasks: a tasks object with loss scales, with ner and pruner classes etc all noted down.
        :param rebuild_cache: a flag which if True ignores pre-processed pickled files and reprocesses everything
        :param allow_speaker_ids: a bool which if turned to False will hide the speaker ID from __getitem__.
            in an ideal world, this should prevent the model from doing anything with speakers.
        """
        # TODO: make it such that multiple values can be passed in 'split'
        self._src_ = src
        self._split_ = split
        self._shuffle_ = shuffle
        self.tasks: Tasks = tasks
        self.tokenizer = tokenizer
        self.config = config
        self.uncased = config.uncased
        self._flag_allow_speaker_ids = allow_speaker_ids

        self.loss_scales = torch.tensor(tasks.loss_scales, dtype=torch.float)

        # Pull word replacements from the manually entered list
        with (LOC.manual / "replacements.json").open("r") as f:
            self.replacements = json.load(f)

        if 'ner' in self.tasks:
            # We need a tag dictionary

            try:
                with (LOC.manual / f"ner_{self._src_}_tag_dict.json").open("r") as f:
                    self.ner_tag_dict = json.load(f)
            except FileNotFoundError:
                # The tag dictionary does not exist. Report and quit.
                raise LabelDictNotFound(f"No label dict found for ner task for {self._src_}")

        if 'rel' in self.tasks:

            try:
                with (LOC.manual / f"rel_{self._src_}_tag_dict.json").open("r") as f:
                    self.rel_tag_dict = json.load(f)
            except FileNotFoundError:
                # The tag dictionary does not exist. Report and quit.
                raise LabelDictNotFound(f"No label dict found for ner task for {self._src_}")

        (self.data, flag_successfully_pulled_from_disk) = self.load_from_disk(rebuild_cache)
        self.task_weights = {}

        if not flag_successfully_pulled_from_disk:

            # Init the list of documents (pull it from disk)
            self.dataset = DocumentReader(src=src, split=split, tasks=self.tasks, shuffle=shuffle)

            # Temporarily set class weights for all tasks as [0.5, 0.5]
            for task_nm in self.tasks:
                if task_nm in ['ner', 'pruner']:
                    self.task_weights[task_nm] = torch.ones(2, dtype=torch.float) / 2

            # Process the dataset
            self.process()

            # Calculate actual class weights
            if not unalias_split(self._split_) == 'test':
                for task_nm in self.tasks:
                    if (task_nm == 'ner' and not self.tasks.ner_unweighted()) or \
                            (task_nm == 'pruner' and not self.tasks.pruner_unweighted()):
                        self.task_weights[task_nm] = torch.tensor(self.estimate_class_weights(task_nm),
                                                                  dtype=torch.float)
            # Update self.data with these class weights
            for i, item in enumerate(self.data):
                if item['ner']:
                    self.data[i]['ner']['weights'] = self.task_weights['ner']
                if item['pruner']:
                    self.data[i]['pruner']['weights'] = self.task_weights['pruner']

            # Write this to disk
            self.write_to_disk()

        if self.config.trim:
            warnings.warn("The dataset has been trimmed to only 50 instances. This is NOT a legit experiment any more!")
            self.data = self.data[:50]

    def estimate_class_weights(self, task: str) -> List[float]:
        """
            A sense of prior prob of predicting a class, based on the data available.
            Expect them to be extremely heavily twisted in the case of __na__ of course.
        """

        def approx_n_spans(n: int, k: int) -> int:
            """n is total words, k is max span len"""
            return int(n * k - (k * k * 0.5) + (k * 0.5))

        # Create a flat (long, long) list of all labels
        # print(self.data[0][task]['gold_labels'])
        # print(self.data[0][task]['gold_labels'])
        y = torch.cat([datum[task]['gold_label_values'] for datum in self.data]).to('cpu')
        zero_spans = sum(approx_n_spans(datum['n_subwords'], self.config.max_span_width) for datum in self.data) - len(
            y)
        return compute_class_weight_sparse(np.unique(y), class_frequencies=np.bincount(y), class_zero_freq=zero_spans)
        # return compute_class_weight('balanced', np.unique(you), y.numpy()).tolist()

    def write_to_disk(self):
        """
        Write to MultiTaskDatasetDump_<task1>_<task2>[ad infinitum].pkl in /data/parsed/self._src_/self._split_
        File names could be:
             MultiTaskDatasetDump_coref.pkl
             MultiTaskDatasetDump_ner.pkl
             MultiTaskDatasetDump_coref_ner.pkl
        """
        # Prep the file name
        dump_fname = LOC.parsed / self._src_ / self._split_ / "MultiTaskDatasetDump"
        for task in self.tasks:
            dump_fname = str(dump_fname) + f"_{task}"
        dump_fname = Path(dump_fname + ".pkl")

        with dump_fname.open("wb+") as f:
            pickle.dump((self.data, self.config), f)

    def load_from_disk(self, ignore_cache: bool) -> (list, bool):
        """
        Look for MultiTaskDatasetDump_<task1>_<task2>[ad infinitum].pkl in /data/parsed/self._src_/self._split_
        File names could be:
             MultiTaskDatasetDump_coref.pkl
             MultiTaskDatasetDump_ner.pkl
             MultiTaskDatasetDump_coref_ner.pkl
        :return: a list of processed dicts (optional) and
            a bool indicating whether we successfully pulled something from the disk or not
        """
        success = False

        if ignore_cache:
            return None, success

        # Prep the file name
        dump_fname = LOC.parsed / self._src_ / self._split_ / "MultiTaskDatasetDump"
        for task in self.tasks:
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
        if check_dumped_config(self.config, old=old_config, find_alternatives=False, verbose=True):
            print(f"Pulled {len(data)} instances from {dump_fname}.")

            # This is the only time where we actually return data.
            # In this situation, we want to update the loss scales, and maybe something else in the future.
            for datum in data:
                datum['loss_scales'] = self.loss_scales

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
        obj = self.data[item]

        # Append relevant things to it
        if not self._flag_allow_speaker_ids:
            # And this automatically makes it so the model doesn't have to worry about it
            obj['speaker_ids'] = None

        return obj

    def __setitem__(self, i, item):
        self.data[i] = item

    def process(self):
        self.data = []
        for datum in tqdm(self.dataset):
            try:
                self.data.append(self.process_document_generic(datum))
            except NoValidAnnotations:
                # Maybe due to the way spans are generated, this instance has no valid annotations
                # ### for one of the tasks we want to extract from it. We ignore this instance.
                continue

        del self.dataset

    def handle_replacements(self, tokens: List[str]) -> List[str]:
        return [self.replacements.get(tok, tok) for tok in tokens]

    @staticmethod
    def _get_speaker_ids_(attention_mask, sentid_for_subword, speakers_list: List[int]):
        """ Extrapolate the speakers across tokens. use attn mask for padding. Use sentid for sentences per swtoken """

        # This is commented out because speakers will never be None
        # if len(speakers_list) == 0:
        #     return None

        # n_subwords,
        speakerid_for_subword = torch.tensor(speakers_list, dtype=torch.long, device='cpu')[sentid_for_subword]

        # Pad it out to attention mask
        # 1, len
        padded = torch.zeros_like(attention_mask) - 1
        padded[0, :speakerid_for_subword.shape[0]] = speakerid_for_subword

        return padded

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
            device='cpu',
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

    # noinspection PyUnusedLocal
    def process_pruner(
            self,
            instance: Document,
            generic_processed_stuff: dict,
            coref_processed_stuff: dict
    ):
        """
            Just create a 1/0 vec representing all valid candidates in gold candidate stuff.
        """
        # unpack things
        # candidate_starts = generic_processed_stuff["candidate_starts"]
        # candidate_ends = generic_processed_stuff["candidate_ends"]
        gold_starts = coref_processed_stuff["gold_starts"]
        gold_ends = coref_processed_stuff["gold_ends"]

        # replace gold labels with all ones.
        new_cluster_ids = torch.ones_like(gold_starts)

        # gold_labels = self.get_candidate_labels_mangoes(
        #     candidate_starts, candidate_ends, gold_starts, gold_ends, new_cluster_ids
        # ).to(float)

        pruner_specific = {  # Pred stuff
            # "gold_labels": gold_labels,
            "gold_starts": gold_starts,
            "gold_ends": gold_ends,
            "gold_label_values": new_cluster_ids,
            "weights": self.task_weights['pruner']
        }

        return pruner_specific

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

        if instance.coref.isempty:
            raise NoValidAnnotations("Coref")

        """
            # Label management:
               - change the label from being a word level index to a subword (bert friendly) level index
            NOTE: the document may be truncated. 
                So, if you don't find a particular span there, check if the document can be truncated (is train split)
                If not, raise KeyError. If yes, ignore span.
        """
        gold_starts, gold_ends, gold_cluster_ids = [], [], []
        for cluster_id, cluster in enumerate(instance.coref.spans):
            for span in cluster:

                if span[0] < len(word2subword_starts) and span[1] - 1 < len(word2subword_ends):
                    gold_starts.append(word2subword_starts[span[0]])
                    gold_ends.append(word2subword_ends[span[1] - 1])
                    gold_cluster_ids.append(cluster_id)
                else:
                    if not is_split_train(dataset=self._src_, split=self._split_):
                        raise KeyError(
                            f"No sw found for span {span}: {to_toks(instance.document)[span[0]: span[1]]}"
                            f"in {instance.docname}."
                        )

        gold_starts = torch.tensor(gold_starts, dtype=torch.long, device='cpu')
        gold_ends = torch.tensor(gold_ends, dtype=torch.long, device='cpu')
        gold_cluster_ids = torch.tensor(gold_cluster_ids, dtype=torch.long, device='cpu') + 1
        """ This +1 may be important, in distingushing actual clusters from singletones/non mentions"""

        coref_specific = {  # Pred stuff
            # "gold_cluster_ids_on_candidates": cluster_labels,
            "gold_starts": gold_starts,
            "gold_ends": gold_ends,
            "gold_label_values": gold_cluster_ids,
        }

        return coref_specific

    def process_ner(
            self,
            instance: Document,
            generic_processed_stuff: dict,
            word2subword_starts: dict,
            word2subword_ends: dict,
    ) -> dict:

        if instance.ner.isempty:
            raise NoValidAnnotations("NER")

        """
            Work with generic processed stuff to also work with NER things
z        """
        gold_starts, gold_ends, gold_labels = [], [], []
        for span, tag in zip(instance.ner.spans, instance.ner.tags):

            if span[0] < len(word2subword_starts) and span[1] - 1 < len(word2subword_ends):
                gold_starts.append(word2subword_starts[span[0]])
                gold_ends.append(word2subword_ends[span[1] - 1])
                if tag not in self.ner_tag_dict:
                    raise AssertionError(f"Tag {tag} not found in Tag dict!")
                gold_labels.append(self.ner_tag_dict[tag])
            else:
                if not is_split_train(dataset=self._src_, split=self._split_):
                    raise KeyError(
                        f"No sw found for span {span}: {to_toks(instance.document)[span[0]: span[1]]}"
                        f"in {instance.docname}."
                    )

        gold_starts = torch.tensor(gold_starts, dtype=torch.long, device='cpu')
        gold_ends = torch.tensor(gold_ends, dtype=torch.long, device='cpu')
        gold_labels = torch.tensor(gold_labels, dtype=torch.long, device='cpu')

        ner_specific = {
            "gold_starts": gold_starts,
            "gold_ends": gold_ends,
            "gold_label_values": gold_labels,
            "weights": self.task_weights['ner']
        }

        # Finally, check if gold_labels are empty (maybe all spans are larger than max span width or something)
        if gold_labels.nonzero().shape[0] == 0:
            raise NoValidAnnotations("NER")

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

        """ 
            Truncation logic: if we are working with the train set, truncate the documents. Not otherwise.
        """
        tokenized = self.tokenizer(
            tokens,
            add_special_tokens=False,
            padding=True,
            truncation=is_split_train(dataset=self._src_, split=self._split_),
            pad_to_multiple_of=n_mlen,
            is_split_into_words=True,
            return_tensors="pt",
            return_length=True,
            max_length=n_mlen * self.config.max_document_segments
        )

        n_subwords = tokenized.attention_mask.sum().item()

        """
            Find the word and sentence corresponding to each subword in the tokenized document
        """
        # Create a subword id to word id dictionary
        subword2word = match_subwords_to_words(
            tokens, tokenized.input_ids, self.tokenizer,
            ignore_cases=self.uncased
        )
        word2subword_all = {}
        for sw_id, w_id in subword2word.items():
            word2subword_all[w_id] = word2subword_all.get(w_id, []) + [sw_id]
        word2subword_starts = {k: v[0] for k, v in word2subword_all.items()}
        word2subword_ends = {k: v[-1] for k, v in word2subword_all.items()}

        wordid_for_subword = torch.tensor(
            [subword2word[subword_id] for subword_id in range(n_subwords)],
            dtype=torch.long,
            device="cpu",
        )
        sentid_for_subword = torch.tensor(
            [instance.sentence_map[word_id] for word_id in wordid_for_subword],
            dtype=torch.long,
            device="cpu",
        )
        speaker_ids = self._get_speaker_ids_(tokenized.attention_mask, sentid_for_subword, instance.speakers)

        # subwordid_for_word_start = torch.tensor([word2subword_starts[word_id]
        #                                          for word_id in range(len(word2subword_starts))],
        #                                         dtype=torch.long, device="cpu")
        # subwordid_for_word_end = torch.tensor([word2subword_ends[word_id]
        #                                        for word_id in range(len(word2subword_ends))],
        #                                       dtype=torch.long, device="cpu")

        # noinspection PyUnusedLocal
        # Commented out, DO NOT DELETE (you never know ;) )
        # 1 marks that the index is a start of a new token, 0 marks that it is not.
        # word_startmap_subword = wordid_for_subword != torch.roll(wordid_for_subword, 1)

        # Resize these tokens as outlined in docstrings
        input_ids = tokenized.input_ids.reshape((-1, n_mlen))  # n_seq, m_len
        token_type_ids = tokenized.token_type_ids.reshape((-1, n_mlen))  # n_seq, m_len
        attention_mask = tokenized.attention_mask.reshape((-1, n_mlen))
        speaker_ids = speaker_ids.reshape((-1, n_mlen)) if speaker_ids is not None else speaker_ids

        """
            Span Iteration: find all valid contiguous sequences of inputs. 
        
            We create subsequences from it, up to length K and pass them all through a span scorer.
            1. Create dumb start, end indices
            2. Exclude the ones which exist at the end of sentences (i.e. start in one, end in another)
        """

        # candidate_starts = (
        #     torch.arange(start=0, end=n_subwords, device="cpu").unsqueeze(1).repeat(1, self.config.max_span_width)
        # )  # n_subwords, max_span_width
        # candidate_ends = candidate_starts + torch.arange(
        #     start=0, end=self.config.max_span_width, device="cpu"
        # ).unsqueeze(
        #     0
        # )  # n_subwords, max_span_width

        """
            # Ignoring invalid spans
            1. Spans which end beyond the document size
            2. Spans which start and end in different sentences
            3. Spans which start OR end mid-word. 
                This one is going to be difficult. But we will be utilising the word map in some form.
                
            We combine all these boolean filters (filter_*: torch.Tensor) with a logical AND.
            NOTE: all filter_* tensors are of shape # n_subwords, max_span_width
        """
        # filter_beyond_document = candidate_ends < n_subwords
        #
        # candidate_starts_sent_id = sentid_for_subword[candidate_starts]
        # candidate_ends_sent_id = sentid_for_subword[
        #     torch.clamp(candidate_ends, max=n_subwords - 1)
        # ]
        # filter_different_sentences = candidate_starts_sent_id == candidate_ends_sent_id
        #
        # # filter_candidate_starts_midword = word_startmap_subword[candidate_starts]
        # # filter_candidate_ends_midword = word_startmap_subword[torch.clamp(candidate_ends, max=n_subwords - 1)]
        #
        # candidate_mask = (filter_beyond_document & filter_different_sentences)
        # # & filter_candidate_starts_midword & filter_candidate_ends_midword
        # # & filter_candidate_starts_midword & \
        # #  filter_candidate_ends_midword  # n_subwords, max_span_width
        #
        # # Now we flatten the candidate starts, ends and the mask and do an index select to ignore the invalid ones
        # candidate_mask = candidate_mask.view(-1)  # n_subwords * max_span_width
        #
        # """
        #     Final Candidate filtering:
        #         - if there are more than the specified num. of candidates (self.filter_candidates_threshold),
        #         - look at POS tags inside the documents and for each candidate,
        #         - if the POS tag contains VB, skip it.
        # """
        # if candidate_mask.int().sum().item() > self.filter_candidates_pos_threshold:
        #     # Get pos tags for each instance
        #     pos = to_toks(instance.pos)
        #     for i, (cs, ce) in enumerate(zip(candidate_starts.reshape(-1), candidate_ends.reshape(-1))):
        #         if not candidate_mask[i]:
        #             continue
        #         _cs = wordid_for_subword[cs]
        #         _ce = wordid_for_subword[ce]
        #         _pos = pos[_cs: _ce]
        #         if 'VB' in _pos \
        #                 or 'VBD' in _pos:
        #             candidate_mask[i] = False
        #
        # candidate_starts = torch.masked_select(
        #     candidate_starts.view(-1), candidate_mask
        # )  # n_subwords*max_span_width
        # candidate_ends = torch.masked_select(
        #     candidate_ends.view(-1), candidate_mask
        # )  # n_subwords*max_span_width
        tasks = [task if not task.startswith("ner") else "ner" for task in self.tasks.names]

        return_dict = {
            "tasks": tasks,
            "domain": self.tasks.dataset,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "word_map": wordid_for_subword,
            "speaker_ids": speaker_ids,
            "sentence_map": sentid_for_subword,
            "n_words": n_words,
            "n_subwords": n_subwords,
            # "candidate_starts": candidate_starts,
            # "candidate_ends": candidate_ends,
            "loss_scales": self.loss_scales,
            "coref": {},
            "ner": {},
            "pruner": {}
        }

        if "coref" in self.tasks or "pruner" in self.tasks:

            coref_op = self.process_coref(
                instance, return_dict, word2subword_starts, word2subword_ends
            )

            if "coref" in self.tasks:
                return_dict["coref"] = coref_op

            if "pruner" in self.tasks:
                return_dict["pruner"] = self.process_pruner(
                    instance, return_dict, coref_op
                )

        if "ner" in self.tasks:
            return_dict["ner"] = self.process_ner(
                instance, return_dict, word2subword_starts, word2subword_ends
            )

        return return_dict


class DocumentReader(Dataset):
    def __init__(
            self, src: str, split: Optional[str] = None, shuffle: bool = False, tasks: Tasks = Tasks.create()
    ):
        """
            Returns an iterable that yields one document at a time.
            It looks for pickled instances of class utils.data.Document in /data/parsed directories.

            You can configure it to filter out instances which do not have annotations for a particular thing e.g.,
                - coref by passing tasks=('coref',) (all instances w/o coref annotations will be skipped)
                - ner by passing tasks=('ner',)
                - or both by tasks=('coref', 'ner') etc etc


        :param src: the first child under /data/parsed, usually, the name of a dataset
        :param split: often different splits are stored in different directories, so this is the second child under
            /data/parsed/<src>
        :param shuffle: whether the dataset should be shuffled or not
        :param tasks: Read code comments but passing 'coref' and/or 'ner' as an item ensures that
            only instances which contain coref and/or ner annotations are returned.
        """

        # Sanity check params
        for task in tasks:
            if task not in KNOWN_TASKS:
                raise UnknownTaskException(
                    f"An unrecognized task name sent: {task}. "
                    f"So far, we can work with {KNOWN_TASKS}."
                )
            # if "ner" in task and "ner_spacy" in task:
            #     raise AssertionError("Multiple NER specific names passed. Pas bon!")

        # super().__init__()
        self._src_ = src
        self._split_ = split
        self._shuffle_ = shuffle
        self._tasks = tasks
        self.data: List[Document] = []

        self.pull_from_disk()

    @staticmethod
    def get_fnames(dataset: str, split: str):
        if split:
            return [fnm for fnm in (LOC.parsed / dataset / split).glob("dump*.pkl")]
        else:
            return [fnm for fnm in (LOC.parsed / dataset).glob("dump*.pkl")]

    def pull_from_disk(self):
        """RIP ur mem lol"""

        filenames = self.get_fnames(self._src_, self._split_)
        if self._shuffle_:
            np.random.shuffle(filenames)

        if len(filenames) == 0:
            raise FileNotFoundError(f"No preprocessed documents found in the desired location."
                                    f" - {self._src_}, {self._split_}")

        for fname in filenames:

            with fname.open("rb") as f:
                data_batch: List[Document] = pickle.load(f)

            if self._shuffle_:
                np.random.shuffle(data_batch)

            for instance in data_batch:

                # See if the instance fits the criteria set in self._tasks
                if "coref" in self._tasks and instance.coref.isempty:
                    continue

                if "pruner" in self._tasks and instance.coref.isempty:
                    continue

                if "ner" in self._tasks and instance.ner.isempty:
                    continue

                if "rel" in self._tasks and instance.rel.isempty:
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
    ds = DocumentReader('ontonotes', split, tasks=('coref',))
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

    def __init__(self, dataset: DocumentReader):
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


class MultiDomainDataCombiner(Dataset):
    def __init__(
            self,
            srcs: List[Callable],
            sampling_ratio: Optional[Iterable[float]] = None,
            speaker_offsets: Optional[List[int]] = None,
    ):
        """
            A data iter that should be given multiple data iter callables.
            Sampling ratio, if specified will oversample or undersample based on the ratio.

            Usage:

            ```py
            ontonotes_di_partial = partial(MultiTaskDataIter, src='ontonotes', split='train', ...)
            scierc_di_partial = partial(MultiTaskDataIter, src='scierc', split='train', ...)

            dic = MultiDomainDataCombiner(src=[ontonotes_di_partial, scierc_di_partial])
            for inst in dic:
                # sometime you get instances from ontonotes, sometimes from scierc
                ...
            ```

        :param srcs: partials set up to init a dataiter whenever needed
        :param sampling_ratio: list of weights for each dataiter. e.g. [0.5, 0.5] implies we sample equally from both.
        """

        # Init all data iterators.
        self.dataiters: List[MultiTaskDataIter] = [iter_partial() for iter_partial in srcs]
        self.tasks: List[Tasks] = [dataiter.tasks for dataiter in self.dataiters]
        self._speaker_offsets = speaker_offsets if speaker_offsets is not None else [0] * len(self.tasks)
        """
            Set dataset sampling indices.
            The list should be roughly the same size as combined length of all dataiters.
        """
        self.source_indices = []
        self.source_pointers = [-1] * len(self.dataiters)

        # TODO: interpret sampling ratios properly.
        if not sampling_ratio:
            sampling_ratio = [1] * len(self.dataiters)

        # Normalise sampling_ratios, weighted by aggregate length of all dataiters
        weights = [len(dataiter) for dataiter in self.dataiters]
        self.weights = [int(weight * ratio) for weight, ratio in zip(weights, sampling_ratio)]

        # Create a list of ints based on these weights which dictate which iter to choose the next sample from

        for i, dataset_specific_weight in enumerate(self.weights):
            self.source_indices += [i] * dataset_specific_weight

        # Now shuffle these
        np.random.shuffle(self.source_indices)

        # Maintain a history so __getitem__ is deterministic
        # ie. {0: [0, 0], 1: [1,0]} -> combined_ds[0] -> data source 0, item 0; combined_ds[1] -> data source 1, item 0
        self.history: Dict[int, Tuple[int, int]] = {}

    def __len__(self):
        return sum(self.weights)

    # def __iter__(self):
    #     for dataiter_index in self.source_indices:
    #         self.source_pointers[dataiter_index] += 1
    #         di = self.dataiters[dataiter_index]
    #         yield di[self.source_pointers[dataiter_index] % len(di)]

    def __getitem__(self, i) -> dict:

        if i in self.history:
            dataiter_index, pointer_index = self.history[i]
            return self.dataiters[dataiter_index][pointer_index]

        else:
            # This item is being asked for the first time.
            # Select the data source, and then based on the last seem item in that data source,
            # ### ask for the next item.

            # Which source to sample from
            dataiter_index = self.source_indices[i]
            di = self.dataiters[dataiter_index]

            # Which item to yield
            pointer_index = self.source_pointers[dataiter_index]
            pointer_index += 1

            # Pull the index
            instance = di[pointer_index % len(di)]

            """
                Now, for some custom logic
            """
            # Apply speaker offset if speaker IDs are not none
            if instance['speaker_ids'] is not None:
                instance['speaker_ids'] += self._speaker_offsets[dataiter_index]

            # Update pointer registry
            self.source_pointers[dataiter_index] = pointer_index

            # Update the history
            self.history[i] = (dataiter_index, pointer_index)

            return instance

    def __setitem__(self, i, item):
        try:
            dataiter_index, pointer_index = self.history[i]
            di = self.dataiters[dataiter_index]
            self.dataiters[dataiter_index][pointer_index % len(di)] = item
        except KeyError:
            raise KeyError(f"Tried to set item in position {i}, when we've only been through {len(self.history)} items")


if __name__ == '__main__':

    task = Tasks.parse(datasrc='codicrac-persuasion', tuples=[('coref', 1.0, True)])
    di = DocumentReader('codicrac-persuasion', 'train', tasks=task)

    for x in di:
        print('potato')
        break
