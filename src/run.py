import json
import wandb
import click
import torch
import transformers
from pathlib import Path
from copy import deepcopy
from functools import partial
from typing import List, Callable, Union, Optional
from mytorch.utils.goodies import mt_save_dir

# Local imports
try:
    import _pathfix
except ImportError:
    from . import _pathfix
from utils.data import Tasks
from loops import training_loop
from models.multitask import BasicMTL
from dataiter import MultiTaskDataIter, DataIterCombiner
from utils.misc import check_dumped_config
from config import LOCATIONS as LOC, CONFIG, KNOWN_SPLITS
from utils.exceptions import ImproperDumpDir, LabelDictNotFound, BadParameters
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


def pick_loss_scale(options: dict, tasks: Tasks):
    key = 'loss_scales_' + '_'.join(sorted(tasks))
    return options[key]


def get_saved_wandb_id(loc: Path):
    with (loc / 'config.json').open('r', encoding='utf8') as f:
        config = json.load(f)

    return config['wandbid']


def get_n_classes(task: str, dataset: str) -> int:
    try:
        with (LOC.manual / f"{task}_{dataset}_tag_dict.json").open("r") as f:
            ner_tag_dict = json.load(f)
            return len(ner_tag_dict)
    except FileNotFoundError:
        # The tag dictionary does not exist. Report and quit.
        raise LabelDictNotFound(f"No label dict found for ner task for {dataset}: {task}")


def get_save_parent_dir(parentdir: Path, dataset: str, tasks: Tasks,
                        dataset_2: Optional[str], tasks_2: Optional[Tasks],
                        config: Union[transformers.BertConfig, dict]) -> Path:
    """
        Normally returns parentdir/dataset+dataset2/'_'.join(sorted(tasks))+'-'+'_'.join(sorted(tasks_2)).
        E.g. if dataset, tasks are ontonotes and ['coref', 'pruner'] and
            dataset_2, tasks_2 are scierc and ['ner'], the output will be
            parentdir/ontonotes_scierc/coref_pruner-ner

        However, if we find that trim flag is active in config, or that the run is going to wandb-trials
            then the output is
                parentdir/trial/dataset+dataset2/'_'.join(sorted(tasks+tasks_2)).
    """
    # if dataset_2 is alphabetically before dataset, start with it
    if dataset_2 and dataset_2[0] < dataset[0]:
        dataset_2, dataset = dataset, dataset_2
        tasks_2, tasks = tasks, tasks_2

    dataset_prefix = dataset + '_' + dataset_2 if dataset_2 else dataset
    tasks_prefix = '_'.join(tasks)
    if tasks_2:
        tasks_prefix += '-'
        tasks_prefix += '_'.join(tasks_2)

    if config.trim or config.wandb_trial:
        return parentdir / 'trial' / dataset_prefix / tasks_prefix
    else:
        return parentdir / dataset_prefix / tasks_prefix


def get_dataiter_partials(
        config: Union[dict, transformers.BertConfig],
        tasks: Tasks,
        dataset: str,
        tokenizer: transformers.BertTokenizer
):
    # Assign loss scales based on task
    loss_scales = pick_loss_scale(CONFIG, tasks)
    # loss_scales = loss_scales.tolist() if not type(loss_scales) is list else loss_scales

    # Load the data
    train_ds = partial(
        MultiTaskDataIter,
        src=dataset,
        config=config,
        tasks=tasks,
        split=KNOWN_SPLITS[dataset].train,
        tokenizer=tokenizer,
        loss_scales=loss_scales
    )
    dev_ds = partial(
        MultiTaskDataIter,
        src=dataset,
        config=config,
        tasks=tasks,
        split=KNOWN_SPLITS[dataset].dev,
        tokenizer=tokenizer,
        loss_scales=loss_scales
    )

    return train_ds, dev_ds


@click.command()
@click.option("--dataset", "-d", type=str, help="The name of dataset e.g. ontonotes etc")
@click.option("--tasks", "-t", type=str, multiple=True,
              help="Multiple values are okay e.g. -t coref -t ner or just one of these", )
@click.option("--dataset-2", "-d2", type=str, help="The name of dataset e.g. ontonotes etc for a secondary thing.")
@click.option("--tasks_2", "-t2", type=str, default=None, multiple=True,
              help="Multiple values are okay e.g. -t2 coref -t2 ner or just one of these", )
@click.option("--epochs", "-e", type=int, default=None, help="Specify the number of epochs for which to train.")
@click.option("--learning-rate", "-lr", type=float, default=0.005, help="Learning rate. Defaults to 0.005.")
@click.option("--encoder", "-enc", type=str, default=None, help="Which BERT model (for now) to load.")
@click.option("--device", "-dv", type=str, default=None, help="The device to use: cpu, cuda, cuda:0, ...")
@click.option('--trim', is_flag=True,
              help="If True, We only consider 50 documents in one dataset. For quick iterations. ")
@click.option('--train-encoder', is_flag=True, default=False,
              help="If enabled, the BERTish encoder is not frozen but trains also.")
@click.option('--ner-unweighted', is_flag=True, default=False,
              help="If True, we do not input priors of classes into Model -> NER CE loss.")
@click.option('--pruner-unweighted', is_flag=True, default=False,
              help="If True, we do not input priors of classes into Model -> Pruner BCEwL loss.")
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
        tasks: List[str],
        dataset_2: str,
        tasks_2: List[str] = None,
        epochs: int = 10,
        learning_rate: float = 0.005,
        encoder: str = "bert-base-uncased",
        device: str = "cpu",
        trim: bool = False,
        train_encoder: bool = False,
        ner_unweighted: bool = False,
        pruner_unweighted: bool = False,
        use_wandb: bool = False,
        wandb_comment: str = '',
        wandb_trial: bool = False,
        filter_candidates_pos: bool = False,
        save: bool = False,
        resume_dir: int = -1
):
    # TODO: enable specifying data sampling ratio when we have 2 datasets
    # TODO: enable specifying loss ratios for different tasks.

    # If trim is enabled, we WILL turn the wandb_trial flag on
    if trim:
        wandb_trial = True

    # If we are to "resume" training things from somewhere, we should also have the save flag enabled
    if resume_dir >= 0:
        save = True

    # Sanity Checks
    if dataset_2:
        if not tasks_2:
            raise BadParameters(f"No tasks specified for dataset 2: {dataset_2}.")
        if not dataset_2 in KNOWN_SPLITS.keys():
            raise BadParameters(f"Unknown dataset: {dataset_2}.")
    if dataset not in KNOWN_SPLITS:
        raise BadParameters(f"Unknown dataset: {dataset}.")
    # If there is overlap in tasks and tasks_2
    if set(tasks).intersection(tasks_2):
        raise BadParameters("Tasks are overlapping between the two sources. That should not happen.")

    # Convert task args to a proper tasks obj
    tasks = Tasks(tasks)
    tasks_2 = Tasks(tasks_2)

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
    config.pruner_ignore_weights = pruner_unweighted
    config.lr = learning_rate
    config.filter_candidates_pos_threshold = CONFIG['filter_candidates_pos_threshold'] if filter_candidates_pos else -1
    config.wandb = use_wandb
    config.wandb_comment = wandb_comment
    config.wandb_trial = wandb_trial

    # if NER is a task, we need to find number of NER classes. We can't have NER in both dataset_a and dataset_b
    if 'ner' in tasks:
        n_classes_ner = get_n_classes(task='ner', dataset=dataset)
    if 'ner' in tasks_2:
        n_classes_ner = get_n_classes(task='ner', dataset=dataset_2)

    # Make the model
    model = BasicMTL(dir_encoder, config=config, n_classes_ner=n_classes_ner)
    print("Model params: ", sum([param.nelement() for param in model.parameters()]))

    # Make the optimizer
    opt = make_optimizer(model=model, optimizer_class=torch.optim.SGD, lr=config.lr,
                         freeze_encoder=config.freeze_encoder)

    """
        Prep datasets.
        For both d1 and d2,
            - prep loss scales
            - prep class weights (by instantiating a temp dataiter)
            - make suitable partials.
            
        Then, IF d2 is specified, make a data combiner thing, otherwise just use this partial in the loop.
        Do the same for evaluators.
        
    """

    train_ds, dev_ds = get_dataiter_partials(config, tasks, dataset=dataset, tokenizer=tokenizer)

    # Collect all metrics
    metrics = []
    if 'ner' in tasks + tasks_2:
        metrics += [NERAcc(), NERSpanRecognitionPR()]
    if 'pruner' in tasks + tasks_2:
        metrics += [PrunerPR()]
    if 'coref' in tasks + tasks_2:
        metrics += [CorefBCubed(), CorefMUC(), CorefCeafe()]

    if dataset_2 and tasks_2:
        train_ds_2, dev_ds_2 = get_dataiter_partials(config, tasks_2, dataset=dataset_2, tokenizer=tokenizer)
        # Make combined iterators since we have two sets of datasets and tasks
        train_ds = partial(DataIterCombiner, srcs=[train_ds, train_ds_2])
        dev_ds = partial(DataIterCombiner, srcs=[dev_ds, dev_ds_2])

    # Make evaluators
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
        savedir = get_save_parent_dir(LOC.models, tasks=tasks, config=config, dataset=dataset,
                                      tasks_2=tasks_2, dataset_2=dataset_2)
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
    config.savedir = savedir

    # Resuming stuff
    if resume_dir >= 0:
        # We are resuming the model
        savedir = mt_save_dir(parentdir=get_save_parent_dir(LOC.models, tasks=tasks, config=config, dataset=dataset,
                                                            tasks_2=tasks_2, dataset_2=dataset_2), _newdir=False)

        """
            First check if the config matches. If not, then
                - report the mismatches
                - try to find other saved models which have the same config.
            
            Get the WandB ID (if its there, and if WandB is enabled.)
            Second, pull the model weights and put them on the model.            
         """

        # Check config
        if not check_dumped_config(config, old=savedir, verbose=True):
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
            save_config = config.to_dict()

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
        tasks=tasks + tasks_2,  # This is used only for bookkeeping. We're assuming empty entries in logs are fine.
        flag_wandb=use_wandb,
        flag_save=save,
        save_dir=savedir,
        save_config=save_config,
        epochs_last_run=config.epoch if hasattr(config, 'epoch') else 0
    )
    print("potato")


if __name__ == "__main__":
    run()
