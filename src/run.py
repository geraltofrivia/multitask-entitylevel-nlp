import wandb
import click
import torch
import transformers
from pathlib import Path
from copy import deepcopy
from functools import partial
from typing import List, Callable, Dict, Iterable

# Local imports
try:
    import _pathfix
except ImportError:
    from . import _pathfix
from loops import training_loop
from config import LOCATIONS as LOC, CONFIG
from models.multitask import BasicMTL
from dataiter import MultiTaskDataset
from eval import ner_all, ner_only_annotated, ner_span_recog_recall, ner_span_recog_precision, \
    pruner_p, pruner_r


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


def pick_loss_scale(options: dict, tasks: Iterable[str]):
    key = 'loss_scales_' + '_'.join(sorted(tasks))
    return options[key]


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
@click.option("--epochs", "-e", type=int, default=None,
              help="Specify the number of epochs for which to train.")
@click.option("--learning-rate", "-lr", type=float, default=0.005,
              help="Learning rate. Defaults to 0.005 if not specified")
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
@click.option('--use-wandb', '-wb', is_flag=True, default=False,
              help="If True, we report this run to WandB")
@click.option('--wandb-comment', '-wbm', type=str, default=None,
              help="If use-wandb is enabled, whatever comment you write will be included in WandB runs.")
@click.option('--wandb-trial', '-wbt', is_flag=True, default=False,
              help="If true, the wandb run is placed in a group of 'trial' runs.")
@click.option('--filter-candidates-pos', '-filtercp', is_flag=True, default=False,
              help="If true, dataiter ignores those candidates which have verbs in them "
                   "IF the doc has more than 10k candidates.")
def run(
        dataset: str,
        epochs: int = 10,
        learning_rate: float = 0.005,
        use_wandb: bool = False,
        encoder: str = "bert-base-uncased",
        tasks: List[str] = None,
        device: str = "cpu",
        trim: bool = False,
        train_encoder: bool = False,
        ner_unweighted: bool = False,
        wandb_comment: str = '',
        wandb_trial: bool = False,
        filter_candidates_pos: bool = False
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
    config.lr = learning_rate
    config.tasks = tasks
    config.filter_candidates_pos_threshold = CONFIG['filter_candidates_pos_threshold'] if filter_candidates_pos else -1

    # Assign loss scales based on task
    loss_scales = pick_loss_scale(CONFIG, tasks)
    config.loss_scales = loss_scales.tolist() if not type(loss_scales) is list else loss_scales

    if 'ner' in tasks or 'pruner' in tasks:
        # Need to figure out the number of classes. Load a DL. Get the number. Delete the DL.
        temp_ds = MultiTaskDataset(
            src=dataset,
            config=config,
            tasks=tasks,
            split="development",
            tokenizer=tokenizer,
        )
        if 'ner' in tasks:
            config.ner_n_classes = deepcopy(temp_ds.ner_tag_dict.__len__())
            config.ner_class_weights = temp_ds.estimate_class_weights('ner')
        else:
            config.ner_n_classes = 1
            config.ner_class_weights = [1.0, ]
        if 'pruner' in tasks:
            config.pruner_class_weights = temp_ds.estimate_class_weights('pruner')
        del temp_ds
    else:
        config.ner_n_classes = 1
        config.ner_class_weights = [1.0, ]

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
    opt = make_optimizer(model=model, optimizer_class=torch.optim.SGD, lr=config.lr,
                         freeze_encoder=config.freeze_encoder)

    # Make the evaluation suite (may compute multiple metrics corresponding to the tasks)
    eval_fns: Dict[str, Dict[str, Callable]] = {
        'ner': {'acc': ner_all,
                'acc_l': ner_only_annotated,
                'span_p': ner_span_recog_precision,
                'span_r': ner_span_recog_recall},
        'coref': {

        },
        'pruner': {'p': pruner_p,
                   'r': pruner_r}
    }

    print(config)
    print("Training commences!")

    # WandB stuff
    if use_wandb:
        wandb.init(project="entitymention-mtl", entity="magnet", notes=wandb_comment,
                   group="trial" if wandb_trial or trim else "main", config=config.to_dict())

    outputs = training_loop(
        epochs=epochs,
        trn_dl=train_ds,
        dev_dl=valid_ds,
        forward_fn=model.pred_with_labels,
        device=device,
        eval_fns=eval_fns,
        opt=opt,
        tasks=tasks,
        loss_scales=torch.tensor(loss_scales, dtype=torch.float, device=device),
        use_wandb=use_wandb
    )
    print("potato")


if __name__ == "__main__":
    run()
