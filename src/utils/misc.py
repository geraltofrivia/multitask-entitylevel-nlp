import json
import torch
import numpy as np
import transformers
from pathlib import Path
from copy import deepcopy
from collections import Counter
from transformers import BertConfig
from dataclasses import dataclass, field
from typing import List, Union, Tuple, Optional, Iterable


def pop(data: list, ids: Union[np.ndarray, List[int]]) -> Optional[list]:
    """Pop multiple elements from a list"""
    if len(ids) == 0:
        return []

    popped: list = []

    ids = sorted(ids, reverse=True)
    # Fix datatype of ids
    ids = [int(x) for x in ids]

    assert ids[0] < len(
        data
    ), f"Data has only {len(data)} elements, but we have to pop {ids[0]}."

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
    tasks = sorted(deepcopy(tasks))
    stacked = torch.hstack([losses[task_nm] for task_nm in tasks])
    weighted = stacked * scales
    return torch.sum(weighted)


@dataclass
class AnnotationBlock:
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


class ClusterAnnotationBlockStack:
    """
    NOT USED ANYMORE

    A collection of aforementioned Blocks of annotation
    Designed to parse nested coref cluster information in the processed CONLL2012 format.
    """

    def __init__(self):
        self.blocks: List[AnnotationBlock] = []

    @property
    def expected(self) -> List[Union[int, str]]:
        """If no open annotation were to close, what would be the cluster info of this token."""
        return [block.tag for block in self.blocks]

    def find_blocks_of_clusters(
            self, ids: Iterable[int], oldest: bool = True, return_indices: bool = False
    ) -> List[Union[int, AnnotationBlock]]:
        """For each cluster ID, find the oldest/newest block"""

        # If descending, multiply start indices by -1; else by +1
        asc_desc_sign = -1 if oldest else 1

        block_indices: List[int] = []
        for cluster in ids:
            # Find all blocks of this cluster (which we haven't already selected)
            block_indices_this_cluster = [
                i
                for i, block in enumerate(self.blocks)
                if block.tag == cluster and i not in block_indices
            ]

            # Find the oldest amongst them all
            # i.e. sort this list based on their corresponding block's start index (mul w -1 if oldest; 1 if newest)
            selected_block_index = sorted(
                block_indices_this_cluster,
                key=lambda x: asc_desc_sign * self.blocks[x].start,
            )[0]

            # Append it
            block_indices.append(selected_block_index)

        # All these indices are now to be returned
        if return_indices:
            return sorted(block_indices)
        else:
            return [self.blocks[i] for i in sorted(block_indices)]

    @staticmethod
    def intersection(a: List[int], b: List[int]) -> List[int]:
        """
        Intersection of two lists including duplicates -
        https://stackoverflow.com/questions/37645053/intersection-of-two-lists-including-duplicates
        """
        return list((Counter(a) & Counter(b)).elements())

    @staticmethod
    def subtraction(a: List[int], b: List[int]) -> List[int]:
        """Subtraction of one list from another (with duplicates)"""
        if not b:
            return deepcopy(a)

        b_ = deepcopy(b)
        res = []
        for ele in a:
            if ele in b_:
                b_.pop(b_.index(ele))
            else:
                res.append(ele)

        return res

    def process(
            self, token, cluster: Union[int, Tuple[int]], span_id: int
    ) -> Optional[List[AnnotationBlock]]:
        """
        Given: this token's actual annotation; and expected annotation.

        1. If the annotation is empty, ensure that the stack is also empty. Something is wrong otherwise.
        2. Do an intersection of Current (`c`) and Expected annotations (`e`)
        3. If annotation belongs in `c - c ∩ e` -> it is a "new" annotation; make a block
        4. If annotation belongs in `e - c ∩ e` -> it is an annotation that closed. Pop the block and send it back.

        """

        if type(cluster) is int:
            cluster = [cluster]
        elif type(cluster) in [tuple, list]:
            cluster = list(cluster)
        else:
            raise TypeError(f"Cluster {cluster} is of an unknown type")

        while -1 in cluster:
            cluster.pop(cluster.index(-1))

        if cluster == [] and len(self.blocks) == 0:
            return []

        # Fetch the "expected" value
        expected = self.expected.copy()
        intersection = self.intersection(expected, cluster)

        # These (intersection) annotations were expected to continue, and they do. Just add the current word to them
        # Find the actual AnnotationBlock objects corresponding to this
        continuing_blocks = self.find_blocks_of_clusters(intersection)
        for block in continuing_blocks:
            block.words.append(token)

        to_add: List[AnnotationBlock] = []
        # to_del: List[int] = []

        # Elements in `cluster - (expected ∩ cluster)` are NEW annotations
        for new_annotation in self.subtraction(cluster, intersection):
            # Create a block and add it to the stack, later
            to_add.append(
                AnnotationBlock(start=span_id, words=[token], tag=new_annotation)
            )

        # Elements in `expected - (expected ∩ cluster)` are now complete annotations, to be popped
        to_del: List[int] = self.find_blocks_of_clusters(
            self.subtraction(expected, intersection), oldest=False, return_indices=True
        )

        # Add the span end ID to these blocks
        for block in to_del:
            self.blocks[block].end = span_id
            self.blocks[block].finalise()

        # Pop them from the internal list
        popped: List[AnnotationBlock] = pop(data=self.blocks, ids=to_del)

        if not cluster:
            assert (
                    self.blocks == []
            ), f"There should be no more open annotations. There are {self.blocks}"

        # Add the new ones
        self.blocks += to_add

        return popped


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
        'ner_class_weights'
        'device',
        'wandb',
        'wandb_comment',
        'wandb_trial',
        'wandbid'
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
        if k not in old_config:
            mismatches[k] = None
            continue
        if not is_equal(v, old_config[k]):
            mismatches[k] = old_config[k]

    # Ignore some mismatches
    for key in keys_to_ignore:
        if key in mismatches:
            mismatches.pop(key)

    if not mismatches:
        # They're all same!
        # Go through all elements of old config and put it on the new one
        for k, v in old_config.items():
            if k not in config_d:
                setattr(config, k, v)
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
