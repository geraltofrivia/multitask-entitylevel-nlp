import click
import torch
import numpy as np
import transformers
from pathlib import Path
from copy import deepcopy
from tqdm.auto import tqdm
from functools import partial
from mytorch.utils.goodies import Timer
from typing import List, Callable, Iterable, Dict

# Local imports
try:
    import _pathfix
except ImportError:
    from . import _pathfix
from config import LOCATIONS as LOC
from models.multitask import BasicMTL
from dataiter import MultiTaskDataset
from utils.exceptions import BadParameters
from eval import ner_all, ner_only_annotated, ner_span_recog_recall, ner_span_recog_precision


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


def compute_metrics(metrics: Dict[str, Callable], logits, labels) -> Dict[str, float]:
    return {metric_nm: metric_fn(logits=logits, labels=labels).item() for metric_nm, metric_fn in metrics.items()}


def aggregate_metrics(inter_epoch: dict, intra_epoch: dict):
    for task_nm in inter_epoch.keys():
        for metric_nm, metric_list in intra_epoch[task_nm].items():
            inter_epoch[task_nm][metric_nm].append(np.mean(metric_list))
    return inter_epoch


def simplest_loop(
        epochs: int,
        tasks: Iterable[str],
        opt: torch.optim,
        forward_fn: Callable,
        predict_fn: Callable,
        trn_dl: Callable,
        dev_dl: Callable,
        eval_fns: dict,
) -> (list, list, list):
    train_loss = {task_nm: [] for task_nm in tasks}
    train_metrics = {task_nm: {metric_nm: [] for metric_nm in eval_fns[task_nm].keys()} for task_nm in tasks}
    valid_metrics = {task_nm: {metric_nm: [] for metric_nm in eval_fns[task_nm].keys()} for task_nm in tasks}

    # Make data
    trn_ds = trn_dl()
    dev_ds = dev_dl()

    # Epoch level
    for e in range(epochs):

        per_epoch_loss = {task_nm: [] for task_nm in tasks}
        per_epoch_tr_metrics = {task_nm: {metric_nm: [] for metric_nm in eval_fns[task_nm].keys()} for task_nm in tasks}
        per_epoch_vl_metrics = {task_nm: {metric_nm: [] for metric_nm in eval_fns[task_nm].keys()} for task_nm in tasks}

        # Train
        with Timer() as timer:

            # Train Loop
            for instance in tqdm(trn_ds):

                # Reset the gradients.
                opt.zero_grad()

                # Forward Pass
                outputs = forward_fn(**instance)

                """
                    Depending on instance.tasks list, do the following:
                        - task specific loss (added to losses)
                        - task specific metrics (added to metrics)
                """
                for task_nm in instance['tasks']:
                    loss = outputs["loss"][task_nm]
                    per_epoch_loss[task_nm].append(loss.item())

                    # TODO: add other metrics here
                    instance_metrics = compute_metrics(eval_fns[task_nm],
                                                       logits=outputs[task_nm]["logits"],
                                                       labels=outputs[task_nm]["labels"])
                    for metric_nm, metric_vl in instance_metrics.items():
                        per_epoch_tr_metrics[task_nm][metric_nm].append(metric_vl)

                loss.backward()
                opt.step()

            # Val
            with torch.no_grad():

                for instance in tqdm(dev_ds):
                    outputs = forward_fn(**instance)

                    for task_nm in instance["tasks"]:
                        logits = outputs[task_nm]["logits"]
                        labels = outputs[task_nm]["labels"]

                        instance_metrics = compute_metrics(eval_fns[task_nm], logits=logits, labels=labels)
                        for metric_nm, metric_vl in instance_metrics.items():
                            per_epoch_vl_metrics[task_nm][metric_nm].append(metric_vl)

        # Bookkeep
        for task_nm in tasks:
            train_loss[task_nm].append(np.mean(per_epoch_loss[task_nm]))
            train_metrics = aggregate_metrics(train_metrics, per_epoch_tr_metrics)
            valid_metrics = aggregate_metrics(valid_metrics, per_epoch_vl_metrics)

        print(f"\nEpoch: {e:3d}" +
              ''.join([f" | {task_nm} Loss: {float(np.mean(per_epoch_loss[task_nm])):.5f}" +
                       ''.join([f" | {task_nm} Tr_{metric_nm}: {float(metric_vls[-1]):.3f}"
                                for metric_nm, metric_vls in train_metrics[task_nm].items()]) +
                       ''.join([f" | {task_nm} Vl_{metric_nm}: {float(metric_vls[-1]):.3f}"
                                for metric_nm, metric_vls in valid_metrics[task_nm].items()])
                       # f" | {task_nm} Tr_c: {float(np.mean(per_epoch_tr_acc[task_nm])):.5f}" +
                       # f" | {task_nm} Vl_c: {float(np.mean(per_epoch_vl_acc[task_nm])):.5f}"
                       for task_nm in tasks]))

    return train_metrics, valid_metrics, train_loss


@click.command()
@click.option("--dataset", "-d", type=str, help="The name of dataset e.g. ontonotes etc")
@click.option("--tasks", "-t", type=str, default=None, multiple=True,
              help="Multiple values are okay e.g. -t coref -t ner or just one of these", )
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
              help="If enabled, the BERTish encoder is not frozen but trains also.")
@click.option('--ner-unweighted', is_flag=True, default=False,
              help="If True, we do not input priors of classes (estimated from the dev set) into Model -> NER CE loss.")
def run(
        dataset: str,
        epochs: int = 10,
        encoder: str = "bert-base-uncased",
        tasks: List[str] = None,
        device: str = "cpu",
        trim: bool = False,
        train_encoder: bool = False,
        ner_unweighted: bool = False
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
    config.ner_ignore_weights = ner_unweighted

    # Need to figure out the number of classes. Load a DL. Get the number. Delete the DL.
    temp_ds = MultiTaskDataset(
        src=dataset,
        config=config,
        tasks=("ner",),
        split="development",
        tokenizer=tokenizer,
    )
    config.ner_n_classes = deepcopy(temp_ds.ner_tag_dict.__len__())
    config.ner_class_weights = temp_ds.estimate_class_weights()
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

    # Make the optimizer
    opt = make_optimizer(model=model, optimizer_class=torch.optim.SGD, lr=0.005, freeze_encoder=config.freeze_encoder)

    # Make the evaluation suite (may compute multiple metrics corresponding to the tasks)
    eval_fns = {
        'ner': {'acc': ner_all,
                'acc_l': ner_only_annotated,
                'span_p': ner_span_recog_precision,
                'span_r': ner_span_recog_recall},
        'coref': {

        }
    }

    print(config)
    print("Training commences!")

    outputs = simplest_loop(
        epochs=epochs,
        trn_dl=train_ds,
        dev_dl=valid_ds,
        forward_fn=model.pred_with_labels,
        predict_fn=model.forward,
        eval_fns=eval_fns,
        opt=opt,
        tasks=tasks
    )
    print("potato")


if __name__ == "__main__":
    run()
