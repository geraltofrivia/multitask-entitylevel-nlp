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
from utils.misc import change_device
from dataiter import MultiTaskDataIter, MultiDomainDataCombiner
from utils.exceptions import DatasetNotEncoded, InstanceNotEncoded, MismatchedConfig


def get_write_location(domain: str, create_mode: bool = True) -> Path:
    loc: Path = LOC.encoded / domain
    if create_mode:
        loc.mkdir(parents=True, exist_ok=True)
    return loc


class PreEncoder:
    """
        Give it a dataiter and a config.
        We're going to whip up BERT and encode whatever a dataiter gives me.

        If we need to move away from BERT, we shall make a custom module and use it there.
        TODO: device it up to cuda if available.
    """

    def __init__(
            self,
            dataset_partial: Callable,
            enc_nm: str,  # the name like 'bert-base-uncased' etc
            device: Union[str, torch.device] = 'cpu'
    ):
        self._vocab_size = enc_nm
        self._device = device

        # base_config = SerializedBertConfig(enc_nm=enc_nm)
        self.bert = BertModel.from_pretrained(enc_nm, add_pooling_layer=False).to(self._device)
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

    @staticmethod
    def _write_to_disk_(instance: dict, output: torch.tensor):
        """
            Get the location based on instance, write the tensor
        """
        loc = get_write_location(instance['domain'])
        loc = loc / f"{instance['hash']}.torch"
        with loc.open('wb+') as f:
            torch.save(output, f)

    def _write_instances_(self) -> None:

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
            device: Union[str, torch.device]
    ):
        self.vocab_size = vocab_size
        self.device = device

    def load(self, domain: str, hash: int) -> torch.tensor:
        # need src, split, hash
        loc: Path = get_write_location(domain, create_mode=False)

        if not loc.exists():
            raise DatasetNotEncoded(loc)

        if not loc.exists():
            raise InstanceNotEncoded(loc=loc, hash=hash)

        loc: Path = loc / f"{hash}.torch"

        encoded, vocab_size = torch.load(loc.open('rb'))

        if vocab_size != self.vocab_size:
            raise MismatchedConfig(f"The current vocab size: {self.vocab_size}. The one on disk: {vocab_size}.")

        encoded = encoded.to(self.device)

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
