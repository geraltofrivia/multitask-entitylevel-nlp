import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass, field, is_dataclass, asdict
from pathlib import Path
from typing import List, Union, Optional, Dict, Any, Type

import numpy as np
import torch
import transformers
from mytorch.utils.goodies import FancyDict
from transformers import BertConfig


class SerializedBertConfig(transformers.BertConfig):

    def to_dict(self):
        output = super().to_dict()
        for k, v in output.items():
            if is_dataclass(v):
                output[k] = asdict(v)

        return output


def pop(data: list, ids: Union[np.ndarray, List[int]]) -> Optional[list]:
    """Pop multiple elements from a list"""
    if len(ids) == 0:
        return []

    popped: list = []

    ids = sorted(ids, reverse=True)
    # Fix datatype of ids
    ids = [int(x) for x in ids]

    assert ids[0] < len(data), f"Data has only {len(data)} elements, but we have to pop {ids[0]}."

    for _id in ids:
        popped.append(data.pop(_id))

    return popped


def change_device(instance: dict, device: Union[str, torch.device]) -> dict:
    """ Go through every k, v in a dict and change its device (make it recursive) """
    for k, v in instance.items():
        if type(v) is torch.Tensor:
            # if 'device' == 'cpu':
            instance[k] = v.detach().to(device)
            # else:
            #     instance[k] = v.to(device)
        elif type(v) is dict:
            instance[k] = change_device(v, device)

    return instance


def weighted_addition_losses(losses, tasks, scales):
    # Sort the tasks
    stacked = torch.hstack([losses[task_nm] for task_nm in tasks])
    weighted = stacked * scales
    return torch.sum(weighted)


def load_speaker_tag_dict(parentdir: Path, src: str) -> Optional[Dict[str, int]]:
    """
        If a dict containing self._src_ is in parentdir for speakers, pull it else return None
        PS: Parentdir should be config.LOCATIONS.manual
    """
    loc = parentdir / f'speaker_{src}_tag_dict.json'
    if not loc.exists():
        return None
    with loc.open('r') as f:
        return json.load(f)


def load_genre_tag_dict(parentdir: Path, src: str) -> Optional[Dict[str, int]]:
    """
        If a dict containing self._src_ is in parentdir for genres, pull it else return None
        PS: Parentdir should be config.LOCATIONS.manual
    """
    loc = parentdir / f'genre_{src}_tag_dict.json'
    if not loc.exists():
        return None
    with loc.open('r') as f:
        return json.load(f)


@dataclass
class AnnotationBlock:
    """
    Variation of AnnotationBlock, below. These aren't expected to be nested (or its fine if they are)
    Used for Ontonotes' NER parsing, specifically.

    an onf block may look like
    start = _, end = _+2, w

    """

    start: int
    words: List[str]
    tag: Union[str, int]

    end: int = field(default=-1)

    # Optional field for more text
    metadata: str = field(default_factory=str)

    def finalise(self):
        assert self.end >= self.start
        if self.start == self.end:
            self.end += 1

        assert len(self.words) == self.end - self.start

    @property
    def span(self) -> List[Union[str, int]]:
        return [self.start, self.end]


@dataclass
class NestedAnnotationBlock:
    """
        Dataclass used exclusively for searching for a pos tag in raw ONF ontonotes file.
        This dataclass represents one annotation / pair of brackets.

        E.g. when searching for NPs in
            (NP (CD six)
                (NNS years))

        an onf block may look like
        start = _, end = _+2, w

    """

    start: int
    words: List[str]
    tag: str

    end: int = field(default_factory=int)

    open_inside: int = 0
    closed_inside: int = 0

    # Optional field for more text
    metadata: str = field(default_factory=str)

    def finalise(self):
        assert self.open_inside == self.closed_inside, \
            f"Bracket Mismatch. Open: {self.open_inside}, Closed: {self.closed_inside}"

        assert self.end >= self.start
        if self.start == self.end:
            self.end += 1

    @property
    def span(self) -> List[int]:
        return [self.start, self.end]


class NERAnnotationBlockStack:
    def __init__(self):
        self.blocks: List[AnnotationBlock] = []

    def register_word(self, word: str) -> None:
        for block in self.blocks:
            block.words.append(word)

    def __len__(self):
        return self.blocks.__len__()

    def pop(self, span: int) -> (List[Union[str, int]], List[str]):
        assert len(self.blocks) > 0, "Cannot pop from an empty stack."
        self.blocks[-1].end = span

        block = self.blocks.pop(-1)
        return block.span, block.words, block.tag

    def add(self, span: int, tag: str) -> None:
        block = AnnotationBlock(tag=tag, start=span, words=[])
        self.blocks.append(block)


class NestedAnnotationBlockStack:
    """
        A collection of aforementioned NestedAnnotationBlockStack.
        This will be used for markable extraction from crac datasets/
    """

    def __init__(self):
        self.blocks: List[NestedAnnotationBlock] = []

    def append(self, block: NestedAnnotationBlock) -> None:
        self.blocks.append(block)

    def register_word(self, word: str) -> None:
        for block in self.blocks:
            block.words.append(word)

    def add(self, n: int = 1) -> None:
        for block in self.blocks:
            block.open_inside += n

    def sub(self, n: int, span: int) -> Optional[List[NestedAnnotationBlock]]:
        """
            Note that n bracket ended.
            Iteratively add one closed brackets to all blocks.
            If at any point open brackets == closed brackets,
                mark that block as finished and pop it
        """

        to_pop: List[int] = []

        for i in range(n):

            # Close one bracket in all blocks
            for block_id, block in enumerate(self.blocks):

                # If the block is already marked as finished, don't do anything to it
                if block_id in to_pop:
                    continue

                assert block.open_inside >= block.closed_inside

                block.closed_inside += 1

                if block.open_inside == block.closed_inside:
                    # All open brackets are closed
                    block.end = span + 1
                    to_pop.append(block_id)

        if not to_pop:
            return None

        # Pop these elements and return
        return pop(data=self.blocks, ids=to_pop)


def is_equal(a, b) -> bool:
    """ if there is a str <-> int mismatch or a list/tuple mismatch, we compensate for it."""
    if a == b:
        return True
    if type(a) in [int, str] and type(b) in [int, str]:
        if str(a) == str(b):
            return True
    if type(a) in [list, tuple] and type(b) in [list, tuple]:
        if tuple(a) == tuple(b):
            return True
    if type(a) is dict and type(b) is dict:
        is_same = True
        for k, v in a.items():
            if type(k) is str:
                try:
                    _v = b[int(k)]
                    if v != _v:
                        is_same = False
                        break
                except ValueError:
                    # What to do?
                    ...
                except KeyError:
                    is_same = False
                    break
            elif type(k) is int:
                try:
                    _v = b[str(k)]
                    if v != _v:
                        is_same = False
                        break
                except ValueError:
                    # What to do?
                    ...
                except KeyError:
                    is_same = False
                    break
            else:
                if not (k in b and v == b[k]):
                    is_same = False
                    break
        if is_same:
            return True
    return False


def check_dumped_config(config: SerializedBertConfig, old: Union[dict, Path, SerializedBertConfig],
                        verbose: bool = True, find_alternatives: bool = True) -> bool:
    """
        If the config stored in the dir mismatches the config passed as param, find out places where it does mismatch
        And second, find alternatives in the parent dir if there are any similar ones.

        Some keys we're okay to be different. e.g. trim.

    :param config: a BertConfig object with custom fields that we're using included in there as well.
    :param old: the directory where we expect this config to be stored OR the actual dict pulled already.
    :param verbose: if true, we print out the differences in the given and stored config
    :param find_alternatives: if True, we go through the dict and try to find if there are
        other configs that match up the given one.
    """

    keys_to_ignore: List[str] = [
        'trim',
        'loss_scales',
        'epochs',
        'lr',
        'ner_class_weights',
        'freeze_encoder',
        'filter_candidates_pos_threshold',
        'skip_instance_after_nspan',
        'learning_rate',
        'bias_in_last_layers',
        'encoder_learning_rate',
        'encoder_weight_decay',
        'device',
        'wandb',
        'wandb_comment',
        'wandb_trial',
        'wandbid',
        'savedir',
        'debug',
        'coref_loss_mean',
        'coref_higher_order',
        'curdir',
        'n_classes_ner',
        'trainer',
        'params',
        'shared_compressor',
        'task',
        'task_2',
        'dense_layers',
        'use_speakers',
        'unary_hdim',
        'encoder_dropout',
        'pruner_dropout',
        'ner_dropout',
        'pos_dropout',
        'pruner_top_span_ratio',
        'pruner_max_num_spans',
        'pruner_use_width',
        'coref_dropout',
        'coref_depth',
        'coref_use_metadata',
        'coref_loss_type',
        'train_on_dev',
        '_config',
        '_tokenizer',
        '_encoder',
        '_sampling_ratio',
        'wandb_name'
    ]

    # If old is a dict, we don't need to pull
    if isinstance(old, dict):
        old_config = old
    elif isinstance(old, BertConfig):
        old_config = old.to_dict()
    else:
        # Pull the old config
        try:
            with (old / 'config.json').open('r', encoding='utf8') as f:
                old_config = json.load(f)
        except FileNotFoundError:
            return False

    config_d = config.to_dict() if isinstance(config, BertConfig) else deepcopy(config)
    mismatches = {}
    for k, v in config_d.items():

        if k in keys_to_ignore:
            continue

        if k not in old_config:
            mismatches[k] = None
            continue
        if not is_equal(v, old_config[k]):
            mismatches[k] = old_config[k]

    if not mismatches:
        # They're all same!
        return True
    else:
        # There are some discrepancies
        if verbose:
            print("Following are the differences found in the configs.")
            key_maxlen = max(len(k) for k in mismatches) + 4
            for k, v in mismatches.items():
                print(f"Old: {k: >{key_maxlen}}: {v}")
                print(f"New: {k: >{key_maxlen}}: {config_d[k]}")

        if find_alternatives:
            alternative_dirs = old.parent.glob('*')
            suitable_alternatives: List[Path] = []
            for alternative_dir in alternative_dirs:
                if check_dumped_config(config=config, old=alternative_dir, verbose=False,
                                       find_alternatives=False):
                    suitable_alternatives.append(alternative_dir)

            if not suitable_alternatives:
                # No other folder has a similar config also
                print("No other saved runs have a similar config. You should save a new run altogether."
                      "For that, re-run the same command but without the -resume-dir arg.")
                return False
            else:
                print(f"Similar config found in directories {', '.join(dirnm.name for dirnm in suitable_alternatives)}."
                      f"You could replace the -resume-dir with either of these if you want to resume them instead.")
                return False
        else:
            return False


def merge_configs(old, new):
    """
        we copy over elements from old and add them to new IF the element does not exist in new.
            If the element is a dict, we do this recursively.

        arg new may be a dict or a FancyDict or a BertConfig
    """

    if type(new) is dict:
        new = FancyDict(new)

    if isinstance(old, BertConfig):
        old = old.to_dict()

    for k, v in old.items():

        try:
            _ = new.__getattribute__(k) if type(new) in [BertConfig, SerializedBertConfig] else new.__getattr__(k)

            # Check if the Value is nested
            if type(v) in [BertConfig, FancyDict, dict]:
                # If so, call the fn recursively
                v = merge_configs(v, new.__getattribute__(k) if type(new) in [BertConfig, SerializedBertConfig] \
                    else new.__getattr__(k))
                new.__setattr__(k, v)
        except (AttributeError, KeyError) as _:
            new.__setattr__(k, v)

    return new


def compute_class_weight_sparse(class_names, class_frequencies: np.ndarray, class_zero_freq: int = 0) -> np.ndarray:
    """ if class zero freq is provided, we replace the first value of bincount with it """
    if class_zero_freq > 0:
        class_frequencies[0] = class_zero_freq

    total = np.sum(class_frequencies)
    return np.array([total / (len(class_names) * freq) for freq in class_frequencies])


def argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)


def deterministic_hash(obj: Any) -> str:
    """ Any serializable obj is serialized and hashed using md5 to get a key which should be deterministic."""

    # If obj is dict, sort by keys
    if isinstance(obj, dict):
        obj = type(obj)(**{k: obj[k] for k in sorted(obj.keys())})

    return hashlib.md5(json.dumps(obj).encode()).hexdigest()


def get_duplicates_format_listoflist(spans: List[List[int]]) -> (dict, list):
    """ Return a dict of {(spstart, spend): n,..} and duplicate indices (every occurance of a span after the first)"""
    return _get_duplicates_([tuple(span) for span in spans])


def get_duplicates_format_startend(start: Union[list, torch.tensor], end: Union[list, torch.tensor]) -> (dict, list):
    """ Return a dict of {(spstart, spend): n,..} and duplicate indices (every occurance of a span after the first)"""

    if isinstance(start, torch.Tensor):
        start = start.tolist()
    if isinstance(end, torch.Tensor):
        end = end.tolist()

    if not len(start) == len(end):
        raise AssertionError(f"Span lengths are not equal. Start: {len(start)}. End: {len(end)}.")

    return _get_duplicates_([tup for tup in zip(start, end)])


def _get_duplicates_(spantuples: List[tuple]) -> (dict, list):
    counter = {}
    dupls = []
    for i, tup in enumerate(spantuples):
        counter[tup] = counter.get(tup, 0) + 1

    duplicates = {k: v for k, v in counter.items() if v > 1}
    indices = []
    for i, tup in enumerate(duplicates.keys()):
        ind_occ = [i for i in range(len(spantuples)) if spantuples[i] == tup]
        indices.append(ind_occ)
    return duplicates, indices


def convert_to_fancydict(obj: Union[Type[BertConfig], Type[dict]]) -> FancyDict:
    if isinstance(obj, BertConfig):
        obj = obj.to_dict()

    op = FancyDict()
    for k, v in obj.items():
        if isinstance(v, dict) or isinstance(v, BertConfig):
            # noinspection PyTypeChecker
            op[k] = convert_to_fancydict(v)
        else:
            op[k] = v

    return op
