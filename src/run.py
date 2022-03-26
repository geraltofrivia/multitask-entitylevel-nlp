import click
import torch
import numpy as np
import transformers
from pathlib import Path
from copy import deepcopy
from functools import partial
from mytorch.utils.goodies import Timer
from typing import List, Callable, Union

from tqdm.auto import tqdm
from config import LOCATIONS as LOC
from models.multitask import BasicMTL
from dataiter import MultiTaskDataset
from utils.exceptions import BadParameters


def get_pretrained_dirs(nm: str):
    """Check if the given nm is stored locally. If so, load that. Else, pass it on as is."""
    plausible_parent_dir: Path = LOC.root / "models" / "huggingface" / nm

    if (
            (plausible_parent_dir / "config").exists()
            and (plausible_parent_dir / "tokenizer").exists()
            and (plausible_parent_dir / "encoder").exists()
    ):
        return (
            str(plausible_parent_dir / "config"),
            str(plausible_parent_dir / "tokenizer"),
            str(plausible_parent_dir / "encoder"),
        )
    else:
        return nm, nm, nm


def simplest_loop(
        epochs: int,
        data: dict,
        opt: torch.optim,
        loss_fn: torch.nn,
        train_fn: Callable,
        predict_fn: Callable,
        device: Union[str, torch.device],
        data_fn: Callable,
        eval_fn: Callable = None,
) -> (list, list, list):
    """
    A fn which can be used to train a language model.

    The model doesn't need to be an nn.Module,
        but have an eval (optional), a train and a predict function.

    Data should be a dict like so:
        {"train":{"x":np.arr, "y":np.arr}, "val":{"x":np.arr, "y":np.arr} }

    Train_fn must return both loss and y_pred

    :param epochs: number of epochs to train for
    :param data: a dict having keys train_x, test_x, train_y, test_y
    :param device: torch device to create new tensor from data
    :param opt: optimizer
    :param loss_fn: loss function
    :param train_fn: function to call with x and y
    :param predict_fn: function to call with x (test)
    :param data_fn: a class to which we can pass X and Y, and get an iterator.
    :param eval_fn: (optional) function which when given pred and true, returns acc
    :return: traces
    """

    train_loss = []
    train_acc = []
    valid_acc = []
    lrs = []

    # Epoch level
    for e in range(epochs):

        per_epoch_loss = []
        per_epoch_tr_acc = []

        # Train
        with Timer() as timer:

            # Make data
            trn_dl, val_dl = data_fn(split="train"), data_fn(split="valid")

            # Train Loop
            for instance in tqdm(trn_dl):
                opt.zero_grad()

                outputs = train_fn(**instance)
                # TODO: change the loss parser when tasks change
                loss = outputs["loss"]["ner"]

                # per_epoch_tr_acc.append(eval_fn(y_pred=y_pred, y_true=_y).item())
                per_epoch_loss.append(loss.item())

                loss.backward()
                opt.step()

            # Val
            with torch.no_grad():

                per_epoch_vl_acc = []
                for instance in tqdm(val_dl):
                    outputs = predict_fn(**instance)

                    per_epoch_vl_acc.append(eval_fn(y_pred, _y).item())

        # Bookkeep
        train_acc.append(np.mean(per_epoch_tr_acc))
        train_loss.append(np.mean(per_epoch_loss))
        valid_acc.append(np.mean(per_epoch_vl_acc))

        print(
            "Epoch: %(epo)03d | Loss: %(loss).5f | Tr_c: %(tracc)0.5f | Vl_c: %(vlacc)0.5f | Time: %(time).3f min"
            % {
                "epo": e,
                "loss": float(np.mean(per_epoch_loss)),
                "tracc": float(np.mean(per_epoch_tr_acc)),
                "vlacc": float(np.mean(per_epoch_vl_acc)),
                "time": timer.interval / 60.0,
            }
        )

    return train_acc, valid_acc, train_loss


#
# def loop(model, config, train_dataset, valid_dataset):
#
#     inter_epoch_loss = []
#
#     for e in range(config.epochs):
#
#         intra_epoch_loss = []
#         for data in tqdm(train_dataset()):
#
#             outputs = model.pred_with_labels(**data)
#
#             loss = outputs['loss']
#             intra_epoch_loss.append(loss)


@click.command()
@click.option(
    "--dataset", "-d", type=str, help="The name of dataset e.g. ontonotes etc"
)
@click.option(
    "--tasks",
    "-t",
    type=str,
    default=None,
    multiple=True,
    help="Multiple values are okay e.g. -t coref -t ner or just one of these",
)
@click.option(
    "--encoder",
    "-enc",
    type=str,
    default=None,
    help="Which BERT model (for now) to load.",
)
@click.option(
    "--epochs",
    "-e",
    type=int,
    default=None,
    help="Specify the number of epochs for which to train.",
)
@click.option(
    "--device",
    "-dv",
    type=str,
    default=None,
    help="The device to use: cpu, cuda, cuda:0, ...",
)
def run(
        dataset: str,
        epochs: int = 10,
        encoder: str = "bert-base-uncased",
        tasks: List[str] = None,
        device: str = "cpu",
):
    dir_config, dir_tokenizer, dir_encoder = get_pretrained_dirs(encoder)

    # Parsing the task
    if tasks != ["coref"]:
        raise BadParameters("Only coref is supported for now.")

    tokenizer = transformers.BertTokenizer.from_pretrained(dir_tokenizer)
    config = transformers.BertConfig(dir_config)
    config.max_span_width = 5
    config.coref_dropout = 0.3
    config.metadata_feature_size = 20
    config.unary_hdim = 1000
    config.binary_hdim = 2000
    config.top_span_ratio = 0.4
    config.max_top_antecedents = 50
    config.device = device
    config.epochs = epochs

    # TODO: need to figure this out: how many classes per dataset.
    # Need to figure out the number of classes. Load a DL. Get the number. Delete the DL.
    temp_ds = MultiTaskDataset(
        src=dataset,
        config=config,
        tasks=("ner_gold",),
        split="train",
        tokenizer=tokenizer,
    )
    config.n_classes_ner = deepcopy(temp_ds.ner_tag_dict.__len__())
    del temp_ds

    model = BasicMTL(dir_encoder, config=config)

    # Load the data
    train_ds = partial(
        MultiTaskDataset, src=dataset, config=config, tasks=("ner_gold",), split="train"
    )
    valid_ds = partial(
        MultiTaskDataset,
        src=dataset,
        config=config,
        tasks=("coref",),
        split="development",
    )

    # outputs = loop(model=model, train_dataset=train_ds, valid_dataset=valid_ds, config=config)


if __name__ == "__main__":
    pass
