import json
import torch
import numpy as np
import transformers
from pathlib import Path
from transformers import BertConfig
from dataclasses import dataclass, field
from typing import List, Union, Optional


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
            if 'device' == 'cpu':
                instance[k] = v.detach().to('cpu')
            else:
                instance[k] = v.to(device)
        elif type(v) is dict:
            instance[k] = change_device(v, device)

    return instance


def weighted_addition_losses(losses, tasks, scales):
    # Sort the tasks
    stacked = torch.hstack([losses[task_nm] for task_nm in tasks])
    weighted = stacked * scales
    return torch.sum(weighted)


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


def check_dumped_config(config: transformers.BertConfig, old: Union[dict, Path, BertConfig],
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
        'tasks',
        'debug',
        'coref_loss_mean',
        'curdir',
    ]

    # If old is a dict, we don't need to pull
    if type(old) is dict:
        old_config = old
    elif type(old) is BertConfig:
        old_config = old.to_dict()
    else:
        # Pull the old config
        try:
            with (old / 'config.json').open('r', encoding='utf8') as f:
                old_config = json.load(f)
        except FileNotFoundError:
            return False

    config_d = config.to_dict()
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
        # Go through all elements of old config and put it on the new one

        # TODO: i removed this pulling old items from disk thing. this is okay, right?
        # for k, v in old_config.items():
        #     if k not in config_d:
        #         setattr(config, k, v)
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
