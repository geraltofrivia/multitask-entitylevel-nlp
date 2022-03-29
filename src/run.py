import click
import torch
import numpy as np
import transformers
from pathlib import Path
from copy import deepcopy
from tqdm.auto import tqdm
from functools import partial
from mytorch.utils.goodies import Timer
from typing import List, Callable, Iterable

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


def make_optimizer(model, optimizer_class: Callable, lr: float, freeze_encoder: bool):
    if freeze_encoder:
        return optimizer_class(
            [param for name, param in model.named_parameters() if not name.startswith("encoder")],
            lr=lr
        )
    else:
        return optimizer_class(model.parameters(), lr=lr)


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
        tasks: Iterable[str],
        opt: torch.optim,
        train_fn: Callable,
        predict_fn: Callable,
        trn_dl: Callable,
        dev_dl: Callable,
        eval_fn: Callable = None,
) -> (list, list, list):
    train_loss = {task_nm: [] for task_nm in tasks}
    train_acc = {task_nm: [] for task_nm in tasks}
    valid_acc = {task_nm: [] for task_nm in tasks}

    # Make data
    trn_ds, dev_ds = trn_dl(), dev_dl()

    # Epoch level
    for e in range(epochs):

        per_epoch_loss = {task_nm: [] for task_nm in tasks}
        per_epoch_tr_acc = {task_nm: [] for task_nm in tasks}
        per_epoch_vl_acc = {task_nm: [] for task_nm in tasks}

        # Train
        with Timer() as timer:

            # Train Loop
            for instance in tqdm(trn_ds):
                opt.zero_grad()

                outputs = train_fn(**instance)

                """
                    Depending on instance.tasks list, do the following:
                        - task specific loss (added to losses)
                        - task specific metrics (added to metrics)
                """
                for task_nm in instance['tasks']:
                    loss = outputs["loss"][task_nm]
                    per_epoch_loss[task_nm].append(loss.item())

                    # TODO: add other metrics here
                    acc = eval_fn(outputs[task_nm]["logits"], outputs[task_nm]["labels"])
                    per_epoch_tr_acc[task_nm].append(acc)

                loss.backward()
                opt.step()

            # Val
            with torch.no_grad():

                for instance in tqdm(dev_ds):
                    outputs = predict_fn(**instance)

                    for task_nm in instance["tasks"]:
                        logits = outputs[task_nm]["logits"]
                        # TODO: make the label puller task specific somehow
                        labels = instance["ner"]["gold_labels"]
                        acc = eval_fn(logits, labels)

                        per_epoch_vl_acc[task_nm].append(acc)

        # Bookkeep
        for task_nm in tasks:
            train_acc[task_nm].append(np.mean(per_epoch_tr_acc[task_nm]))
            train_loss[task_nm].append(np.mean(per_epoch_loss[task_nm]))
            valid_acc[task_nm].append(np.mean(per_epoch_vl_acc[task_nm]))

        print(f"Epoch: {e:3d}" +
              ''.join([f" | {task_nm} Loss: {float(np.mean(per_epoch_loss[task_nm])):.5f}" +
                       f" | {task_nm} Tr_c: {float(np.mean(per_epoch_tr_acc[task_nm])):.5f}" +
                       f" | {task_nm} Vl_c: {float(np.mean(per_epoch_vl_acc[task_nm])):.5f}" for task_nm in tasks]))

    return train_acc, valid_acc, train_loss


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
@click.option('--trim', is_flag=True,
              help="If True, We only consider 50 documents in one dataset. For quick iterations. ")
@click.option('--train-encoder', is_flag=True, default=False,
              help="If True, We only consider 50 documents in one dataset. For quick iterations. ")
def run(
        dataset: str,
        epochs: int = 10,
        encoder: str = "bert-base-uncased",
        tasks: List[str] = None,
        device: str = "cpu",
        trim: bool = False,
        train_encoder: bool = False
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
    config.trim = trim
    config.freeze_encoder = not train_encoder

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

    # Make the model
    model = BasicMTL(dir_encoder, config=config)

    # Load the data
    train_ds = partial(
        MultiTaskDataset,
        src=dataset,
        config=config,
        tasks=tasks,
        split="train",
        tokenizer=tokenizer,
    )
    valid_ds = partial(
        MultiTaskDataset,
        src=dataset,
        config=config,
        tasks=tasks,
        split="development",
        tokenizer=tokenizer,
    )

    opt = make_optimizer(model=model, optimizer_class=torch.optim.SGD, lr=0.001, freeze_encoder=config.freeze_encoder)
    # opt = torch.optim.SGD(model.parameters(), lr=0.001)

    print(config)
    print("Training commences!")

    outputs = simplest_loop(
        epochs=epochs,
        trn_dl=train_ds,
        dev_dl=valid_ds,
        train_fn=model.pred_with_labels,
        predict_fn=model.forward,
        eval_fn=ner,
        opt=opt,
        tasks=tasks
    )
    print("potato")


if __name__ == "__main__":
    run()
