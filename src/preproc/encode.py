import statistics
import warnings
from pathlib import Path
from typing import Callable, Union

import torch
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
from utils.exceptions import DatasetNotEncoded, InstanceNotEncoded, MismatchedConfig


def get_write_location(dataset, instance: dict, create_mode: bool = True) -> Path:
    if isinstance(dataset, MultiTaskDataIter):
        loc: Path = LOC.encoded / dataset._src_ / dataset._split_
    elif isinstance(dataset, MultiDomainDataCombiner):
        # Get the MTDataIter from the Combiner
        dataset_ind = [task.dataset for task in dataset.tasks].index(instance['domain'])
        dataset = dataset.dataiters[dataset_ind]
        loc: Path = LOC.encoded / dataset._src_ / dataset._split_
    else:
        raise TypeError(f"Type of dataset is not understood: {type(dataset)}.")

    if create_mode:
        loc.mkdir(parents=True, exist_ok=True)

    return loc


class Encoder:
    """
        Give it a dataiter and a config.
        We're going to whip up BERT and encode whatever a dataiter gives me.

        If we need to move away from BERT, we shall make a custom module and use it there.
        TODO: device it up to cuda if available.
    """

    def __init__(
            self,
            dataset_partial: Callable,
            vocab_size: str,  # the name like 'bert-base-uncased' etc
            device: Union[str, torch.device] = 'cpu'
    ):
        self._vocab_size = vocab_size
        self._device = device

        base_config = SerializedBertConfig(vocab_size=vocab_size)
        self.bert = BertModel(base_config, add_pooling_layer=False).to(self._device)
        self.dataset: Union[MultiTaskDataIter, MultiDomainDataCombiner] = dataset_partial()

        # If there are sampling ratios involved, we should warn
        if isinstance(self.dataset, MultiDomainDataCombiner) and self.dataset.sampling_ratio is not None and \
                not statistics.mean(self.dataset.sampling_ratio) == 1:
            warnings.warn(f"We are not seeing all the samples")

    def run(self):
        """
            Write instances to disk
            Write the config as well
        :return:
        """
        self._write_instances_()

    def _write_to_disk_(self, instance: dict, output: torch.tensor):
        """
            Get the location based on instance, write the tensor
        """
        loc = get_write_location(self.dataset, instance)
        loc = loc / f"{instance['hash']}.torch"
        with loc.open('wb+') as f:
            torch.save(output, f)

    # def _write_config_(self):
    #
    #     if isinstance(self.dataset, MultiTaskDataIter):
    #         # Just write here
    #         loc: Path = LOC.encoded / self.dataset._src_ / self.dataset._split_ / 'config.pkl'
    #         with loc.open('w+') as f:
    #             pickle.dump(self.config, f)
    #     elif isinstance(self.dataset, MultiDomainDataCombiner):
    #
    #         for dataset in self.dataset.dataiters:
    #             loc: Path = LOC.encoded / dataset._src_ / dataset._split_ / 'config.pkl'
    #             with loc.open('wb+') as f:
    #                 # TODO: does this need to be more sophisticated? do we need to remove the other task etc?
    #                 pickle.dump(self.config, f)

    def _write_instances_(self):

        with torch.no_grad():
            for i, instance in enumerate(tqdm(self.dataset)):
                instance = change_device(instance, self._device)
                output = self.bert(input_ids=instance['input_ids'], attention_mask=instance['attention_mask'])[0]
                output = {'encoded': output, 'vocab_size': self._vocab_size}
                self._write_to_disk_(instance, output)

        print(f"Wrote {i} instances, encoded to disk.")


class Retriever:
    """
        Give me hash and I'll give you the tensor. Lightweight stuff.
        Every time we check the vocab size.
    """

    def __init__(
            self,
            vocab_size: str,
            device: Union[str, torch.device],
            dataiter: Union[MultiTaskDataIter, MultiDomainDataCombiner]):
        self.vocab_size = vocab_size
        self.device = device
        self.dataset = dataiter

    def load(self, instance: dict) -> torch.tensor:
        # need src, split, hash
        loc = get_write_location(self.dataset, instance, create_mode=False)

        if not loc.exists():
            raise DatasetNotEncoded(loc)

        if not loc.exists():
            raise InstanceNotEncoded(loc=loc, hash=instance['hash'])

        loc = loc / f"{instance['hash']}.torch"

        encoded, vocab_size = torch.load(loc.open('rb'))

        if vocab_size != self.vocab_size:
            raise MismatchedConfig(f"The current vocab size: {self.vocab_size}. The one on disk: {vocab_size}.")

        return encoded

# @click.command()
# @click.option("--encoder", "-enc", type=str, default=None, help="Which BERT model (for now) to load.")
# @click.option("--tokenizer", "-tok", type=str, default=None, help="Put in value here in case value differs from enc")
# @click.option("--device", "-dv", type=str, default=None, help="The device to use: cpu, cuda, cuda:0, ...")
# def run(
#         encoder,
#         tokenizer,
#         device,
# )
#
# if __name__ == '__main__':
#
#     run()
