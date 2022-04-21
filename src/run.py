import json
import wandb
import click
import torch
import transformers
from pathlib import Path
from copy import deepcopy
from functools import partial
from typing import List, Callable, Iterable, Union
from mytorch.utils.goodies import mt_save_dir

# Local imports
try:
    import _pathfix
except ImportError:
    from . import _pathfix
from loops import training_loop
from models.multitask import BasicMTL
from dataiter import MultiTaskDataIter
from utils.misc import check_dumped_config
from utils.exceptions import ImproperDumpDir
from config import LOCATIONS as LOC, CONFIG, KNOWN_SPLITS
from eval import Evaluator, NERAcc, NERSpanRecognitionPR, PrunerPR, CorefBCubed, CorefMUC, CorefCeafe


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


def get_saved_wandb_id(loc: Path):
    with (loc / 'config.json').open('r', encoding='utf8') as f:
        config = json.load(f)

    return config['wandbid']


def get_save_parent_dir(parentdir: Path, tasks: List[str], config: Union[transformers.BertConfig, dict]) -> Path:
    """
        Normally returns parentdir/'_'.join(sorted(tasks)).
        E.g. if tasks are ['coref', 'ner']:
                parentdir/coref_ner
            but if they are arranged like ['ner', coref'], the output would still be
                parentdir/coref_ner
            if tasks are ['ner', 'pruner', 'coref']:
                parentdir/coref_ner_pruner

        However, if we find that trim flag is active in config, or that the run is going to wandb-trials
            then the output is
                parentdir/trial/<tasks concatenated with _ in alphabetical order>
    """

    if config.trim or config.wandb_trial:
        return parentdir / 'trial' / '_'.join(sorted(tasks))
    else:
        return parentdir / '_'.join(sorted(tasks))


@click.command()
@click.option("--dataset", "-d", type=str, help="The name of dataset e.g. ontonotes etc")
@click.option("--epochs", "-e", type=int, default=None, help="Specify the number of epochs for which to train.")
@click.option("--learning-rate", "-lr", type=float, default=0.005, help="Learning rate. Defaults to 0.005.")
@click.option("--encoder", "-enc", type=str, default=None, help="Which BERT model (for now) to load.")
@click.option("--tasks", "-t", type=str, default=None, multiple=True,
              help="Multiple values are okay e.g. -t coref -t ner or just one of these", )
@click.option("--device", "-dv", type=str, default=None, help="The device to use: cpu, cuda, cuda:0, ...")
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
@click.option('--save', '-s', is_flag=True, default=False, help="If true, the model is dumped to disk at every epoch.")
@click.option('--resume-dir', default=-1, type=int,
              help="In case you want to continue from where we left off, give the folder number. "
                   "The lookup will go like /models/<task combination>/<resume_dir>/model.torch.")
def run(
        dataset: str,
        epochs: int = 10,
        learning_rate: float = 0.005,
        encoder: str = "bert-base-uncased",
        tasks: List[str] = None,
        device: str = "cpu",
        trim: bool = False,
        train_encoder: bool = False,
        ner_unweighted: bool = False,
        use_wandb: bool = False,
        wandb_comment: str = '',
        wandb_trial: bool = False,
        filter_candidates_pos: bool = False,
        save: bool = False,
        resume_dir: int = -1
):
    # If trim is enabled, we WILL turn the wandb_trial flag on
    if trim:
        wandb_trial = True

    # If we are to "resume" training things from somewhere, we should also have the save flag enabled
    if resume_dir >= 0:
        save = True

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
    config.wandb = use_wandb
    config.wandb_comment = wandb_comment
    config.wandb_trial = wandb_trial

    # Assign loss scales based on task
    loss_scales = pick_loss_scale(CONFIG, tasks)
    config.loss_scales = loss_scales.tolist() if not type(loss_scales) is list else loss_scales

    if 'ner' in tasks or 'pruner' in tasks:
        # Need to figure out the number of classes. Load a DL. Get the number. Delete the DL.
        temp_ds = MultiTaskDataIter(
            src=dataset,
            config=config,
            tasks=tasks,
            split=KNOWN_SPLITS[dataset].dev,
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
    print("Model params: ", sum([param.nelement() for param in model.parameters()]))

    # Load the data
    train_ds = partial(
        MultiTaskDataIter,
        src=dataset,
        config=config,
        tasks=tasks,
        split=KNOWN_SPLITS[dataset].train,
        tokenizer=tokenizer,
    )
    dev_ds = partial(
        MultiTaskDataIter,
        src=dataset,
        config=config,
        tasks=tasks,
        split=KNOWN_SPLITS[dataset].dev,
        tokenizer=tokenizer,
    )

    # Make the optimizer
    opt = make_optimizer(model=model, optimizer_class=torch.optim.SGD, lr=config.lr,
                         freeze_encoder=config.freeze_encoder)

    # Make the evaluation suite (may compute multiple metrics corresponding to the tasks)
    metrics = []
    if 'ner' in tasks:
        metrics += [NERAcc(), NERSpanRecognitionPR()]
    if 'pruner' in tasks:
        metrics += [PrunerPR()]
    if 'coref' in tasks:
        metrics += [CorefBCubed(), CorefMUC(), CorefCeafe()]
    train_eval = Evaluator(
        predict_fn=model.pred_with_labels,
        dataset_partial=train_ds,
        metrics=metrics,
        device=device
    )
    dev_eval = Evaluator(
        predict_fn=model.pred_with_labels,
        dataset_partial=dev_ds,
        metrics=metrics,
        device=device
    )

    print(config)
    print("Training commences!")

    # Saving stuff
    if save:
        savedir = get_save_parent_dir(LOC.models, tasks=tasks, config=config)
        savedir.mkdir(parents=True, exist_ok=True)

        if resume_dir >= 0:
            # We already know which dir to save the model to.
            savedir = savedir / str(resume_dir)
            assert savedir.exists(), f"No subfolder {resume_dir} in {savedir.parent}. Can not resume!"
        else:
            # This is a new run and we should just save the model in a new place
            savedir = mt_save_dir(parentdir=savedir, _newdir=True)

        save_config = config.to_dict()
        # save_objs = [tosave('tokenizer.pkl', tokenizer), tosave('config.pkl', )]
    else:
        savedir, save_config, save_objs = None, None, None

    # Resuming stuff
    if resume_dir >= 0:
        # We are resuming the model
        savedir = mt_save_dir(parentdir=get_save_parent_dir(LOC.models, tasks=tasks, config=config), _newdir=False)

        """
            First check if the config matches. If not, then
                - report the mismatches
                - try to find other saved models which have the same config.
            
            Get the WandB ID (if its there, and if WandB is enabled.)
            Second, pull the model weights and put them on the model.            
         """

        # Check config
        if not check_dumped_config(config, old=savedir):
            raise ImproperDumpDir(f"No config.json file found in {savedir}. Exiting.")

        # See WandB stuff
        if use_wandb:
            # Try to find WandB ID in saved stuff
            config.wandbid = get_saved_wandb_id(savedir)

        # Pull checkpoint and update opt, model
        checkpoint = torch.load(savedir / 'torch.save')
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Successfully resuming training from Epoch {config.epoch}")

    # WandB stuff
    if use_wandb:
        if 'wandbid' not in config.to_dict():
            config.wandbid = wandb.util.generate_id()

        wandb.init(project="entitymention-mtl", entity="magnet", notes=wandb_comment,
                   id=config.wandbid, resume="allow", group="trial" if wandb_trial or trim else "main")
        wandb.config.update(config.to_dict(), allow_val_change=True)

    outputs = training_loop(
        model=model,
        epochs=epochs,
        trn_dl=train_ds,
        forward_fn=model.pred_with_labels,
        device=device,
        train_eval=train_eval,
        dev_eval=dev_eval,
        opt=opt,
        tasks=tasks,
        loss_scales=torch.tensor(loss_scales, dtype=torch.float, device=device),
        flag_wandb=use_wandb,
        flag_save=save,
        save_dir=savedir,
        save_config=save_config,
        epochs_last_run=config.epoch if hasattr(config, 'epoch') else 0
    )
    print("potato")


if __name__ == "__main__":
    run()
