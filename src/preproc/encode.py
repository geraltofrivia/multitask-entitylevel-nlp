import pickle
import statistics
import warnings
from pathlib import Path
from typing import Callable, Union

import torch
from mytorch.utils.goodies import FancyDict
from tqdm.auto import tqdm
from transformers import BertModel

# Local Imports
try:
    import _pathfix
except ImportError:
    from . import _pathfix
from config import LOCATIONS as LOC
from utils.misc import SerializedBertConfig, change_device
from dataiter import MultiTaskDataIter, MultiDomainDataCombiner


class Cacher:
    """
        Give it a dataiter and a config.
        We're going to whip up BERT and encode whatever a dataiter gives me.

        If we need to move away from BERT, we shall make a custom module and use it there.
    """

    def __init__(
            self,
            dataset_partial: Callable,
            config: Union[dict, FancyDict, SerializedBertConfig]
    ):
        base_config = SerializedBertConfig(vocab_size=config.vocab_size)
        self.config = config
        self.bert = BertModel(base_config, add_pooling_layer=False)
        self.dataset: Union[MultiTaskDataIter, MultiDomainDataCombiner] = dataset_partial()

        self._device = config.device

        # If there are sampling ratios involved, we should warn
        if (
                isinstance(self.dataset, MultiDomainDataCombiner) and
                self.dataset.sampling_ratio is not None and
                not statistics.mean(self.dataset.sampling_ratio) == 1
        ) or (self.config.trim):
            warnings.warn(f"We are not seeing all the samples")

    def run(self):
        """
            Write instances to disk
            Write the config as well
        :return:
        """
        self._write_instances_()
        self._write_config_()

    def _get_write_location_(self, instance: dict) -> Path:

        if isinstance(self.dataset, MultiTaskDataIter):
            loc: Path = LOC.encoded / self.dataset._src_ / self.dataset._split_
        elif isinstance(self.dataset, MultiDomainDataCombiner):
            # Get the MTDataIter from the Combiner
            dataset_ind = [task.dataset for task in self.dataset.tasks].index(instance['domain'])
            dataset = self.dataset.dataiters[dataset_ind]
            loc: Path = LOC.encoded / dataset._src_ / dataset._split_
        else:
            raise TypeError(f"Type of dataset is not understood: {type(self.dataset)}.")

        loc.mkdir(parents=True, exist_ok=True)
        return loc / f"{instance['hash']}.torch"

    def _write_to_disk_(self, instance: dict, output: torch.tensor):
        """
            Get the location based on instance, write the tensor
        """
        loc = self._get_write_location_(instance)
        with loc.open('wb+') as f:
            torch.save(output, f)

    def _write_config_(self):

        if isinstance(self.dataset, MultiTaskDataIter):
            # Just write here
            loc: Path = LOC.encoded / self.dataset._src_ / self.dataset._split_ / 'config.pkl'
            with loc.open('w+') as f:
                pickle.dump(self.config, f)
        elif isinstance(self.dataset, MultiDomainDataCombiner):

            for dataset in self.dataset.dataiters:
                loc: Path = LOC.encoded / dataset._src_ / dataset._split_ / 'config.pkl'
                with loc.open('wb+') as f:
                    # TODO: does this need to be more sophisticated? do we need to remove the other task etc?
                    pickle.dump(self.config, f)

    def _write_instances_(self):

        with torch.no_grad():
            for i, instance in enumerate(tqdm(self.dataset)):
                instance = change_device(instance, self._device)
                output = self.bert(input_ids=instance['input_ids'], attention_mask=instance['attention_mask'])[0]
                self._write_to_disk_(instance, output)

        print(f"Wrote {i} instances, encoded to disk.")
