import click
import torch
import numpy as np
import transformers
from pathlib import Path
from copy import deepcopy
from tqdm.auto import tqdm
from functools import partial
from mytorch.utils.goodies import Timer
from typing import List, Callable, Union

# Local imports
try:
    import _pathfix
except ImportError:
    from . import _pathfix
from eval import ner_all as ner
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
        opt: torch.optim,
        train_fn: Callable,
        predict_fn: Callable,
        trn_dl: Callable,
        dev_dl: Callable,
        eval_fn: Callable = None,
) -> (list, list, list):

    train_loss = []
    train_acc = []
    valid_acc = []

    # Epoch level
    for e in range(epochs):

        per_epoch_loss = []
        per_epoch_tr_acc = []
        per_epoch_vl_acc = []

        # Train
        with Timer() as timer:

            # Make data
            trn_ds, dev_ds = trn_dl(), dev_dl()

            # Train Loop
            for instance in tqdm(trn_ds):
                opt.zero_grad()

                outputs = train_fn(**instance)
                # TODO: change the loss parser when tasks change
                loss = outputs["loss"]["ner"]
                # TODO: add other metrics here
                acc = eval_fn(outputs["ner"]["logits"], outputs["ner"]["labels"])

                per_epoch_loss.append(loss.item())
                per_epoch_tr_acc.append(acc)

                loss.backward()
                opt.step()

            # Val
            with torch.no_grad():

                per_epoch_vl_acc = []
                for instance in tqdm(dev_ds):
                    outputs = predict_fn(**instance)
                    ner_logits = outputs["ner"]["logits"]
                    # Can we make the following generalised? So far its too specific to the loop.
                    ner_labels = instance["ner"]["gold_labels"]
                    acc = eval_fn(ner_logits, ner_labels)

                    # TODO: write an eval function (somewhere)
                    per_epoch_vl_acc.append(acc)

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
        tasks=("ner",),
        split="train",
        tokenizer=tokenizer,
    )
    config.n_classes_ner = deepcopy(temp_ds.ner_tag_dict.__len__())
    del temp_ds

    model = BasicMTL(dir_encoder, config=config)

    # Load the data
    train_ds = partial(
        MultiTaskDataset,
        src=dataset,
        config=config,
        tasks=("ner",),
        split="train",
        tokenizer=tokenizer,
    )
    valid_ds = partial(
        MultiTaskDataset,
        src=dataset,
        config=config,
        tasks=("ner",),
        split="development",
        tokenizer=tokenizer,
    )
    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    outputs = simplest_loop(
        epochs=epochs,
        trn_dl=train_ds,
        dev_dl=valid_ds,
        train_fn=model.pred_with_labels,
        predict_fn=model.forward,
        eval_fn=ner,
        opt=opt,
    )
    print("potato")


if __name__ == "__main__":
    run()
