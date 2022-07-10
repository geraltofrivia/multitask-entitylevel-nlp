import json
import random
from functools import partial
from pathlib import Path
from typing import List, Callable, Union, Optional, Tuple, Type

import click
import numpy as np
import torch
import transformers
import wandb
from mytorch.utils.goodies import mt_save_dir, FancyDict

# Local imports
try:
    import _pathfix
except ImportError:
    from . import _pathfix
from utils.data import Tasks
from loops import training_loop
from models.multitask import BasicMTL, MangoesMTL
from dataiter import MultiTaskDataIter, MultiDomainDataCombiner
from utils.misc import check_dumped_config, merge_configs, SerializedBertConfig
from config import LOCATIONS as LOC, DEFAULTS, KNOWN_SPLITS, _SEED_ as SEED, SCHEDULER_CONFIG
from utils.exceptions import ImproperDumpDir, LabelDictNotFound, BadParameters
from eval import Evaluator, NERAcc, NERSpanRecognitionMicro, NERSpanRecognitionMacro, \
    PrunerPRMicro, PrunerPRMacro, CorefBCubed, CorefMUC, CorefCeafe, TraceCandidates

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# def make_optimizer(model, optimizer_class: Callable, lr: float, freeze_encoder: bool):
#     if freeze_encoder:
#         return optimizer_class(
#             [param for name, param in model.named_parameters() if not name.startswith("encoder")],
#             lr=lr
#         )
#     else:
#         return optimizer_class(model.parameters(), lr=lr)
def make_optimizer(
        model: BasicMTL,
        base_keyword: str,
        task_weight_decay: Optional[float],
        task_learning_rate: Optional[float],
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_epsilon: float = 1e-6,
        encoder_learning_rate: float = 2e-05,
        encoder_weight_decay: float = 0.0,
        freeze_encoder: bool = False,
        optimizer_class: Callable = torch.optim.AdamW,
):
    """
    Setup the optimizer and the learning rate scheduler.

    This will use AdamW. If you want to use something else (ie, a different optimizer and multiple learn rates), you
    can subclass and override this method in a subclass.
    """

    if task_learning_rate is None:
        task_learning_rate = encoder_learning_rate
    if task_weight_decay is None:
        task_weight_decay = encoder_weight_decay

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and
                       base_keyword in n],
            "weight_decay": encoder_weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and
                       base_keyword in n],
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and
                       base_keyword not in n],
            "weight_decay": task_weight_decay,
            "lr": task_learning_rate,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and
                       base_keyword not in n],
            "weight_decay": 0.0,
            "lr": task_learning_rate,
        },
    ]

    if freeze_encoder:
        # Drop the first two groups from the params
        optimizer_grouped_parameters = optimizer_grouped_parameters[2:]

    optimizer_kwargs = {"betas": (adam_beta1, adam_beta2), "eps": adam_epsilon, "lr": encoder_learning_rate}
    return optimizer_class(optimizer_grouped_parameters, **optimizer_kwargs)


def make_scheduler(opt, lr_schedule: Optional[str], lr_schedule_val: Optional[float]) \
        -> Optional[Type[torch.optim.lr_scheduler._LRScheduler]]:
    if not lr_schedule:
        return None

    if lr_schedule == 'gamma':
        hyperparam = lr_schedule_val if lr_schedule_val >= 0 else SCHEDULER_CONFIG['gamma']['decay_rate']
        lambda_1 = lambda epoch: hyperparam ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda_1)
        return scheduler

    else:
        raise BadParameters(f"Unknown LR Schedule Recipe Name - {lr_schedule}")


def get_pretrained_dirs(enc_nm: str, tok_nm: str):
    """Check if the given nm is stored locally. If so, load that. Else, pass it on as is."""
    plausible_parent_dir: Path = LOC.root / "models" / "huggingface"

    if ((plausible_parent_dir / enc_nm / "encoder").exists() and (plausible_parent_dir / enc_nm / "config").exists()
            and (plausible_parent_dir / tok_nm / "tokenizer").exists()):
        return (
            str(plausible_parent_dir / enc_nm / "config"),
            str(plausible_parent_dir / tok_nm / "tokenizer"),
            str(plausible_parent_dir / enc_nm / "encoder"),
        )
    else:
        return enc_nm, tok_nm, enc_nm


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
                        config: Union[SerializedBertConfig, dict]) -> Path:
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
    tasks_prefix = '_'.join(tasks.names)
    if not tasks_2.isempty():
        tasks_prefix += '-'
        tasks_prefix += '_'.join(tasks_2.names)

    if config.trim or config.wandb_trial:
        return parentdir / 'trial' / dataset_prefix / tasks_prefix
    else:
        return parentdir / dataset_prefix / tasks_prefix


def get_dataiter_partials(
        config: Union[dict, SerializedBertConfig],
        tasks: Tasks,
        tokenizer: transformers.BertTokenizer,
):

    # Load the data
    train_ds = partial(
        MultiTaskDataIter,
        src=tasks.dataset,
        config=config,
        tasks=tasks,
        split=KNOWN_SPLITS[tasks.dataset].train,
        tokenizer=tokenizer,
    )
    dev_ds = partial(
        MultiTaskDataIter,
        src=tasks.dataset,
        config=config,
        tasks=tasks,
        split=KNOWN_SPLITS[tasks.dataset].dev,
        tokenizer=tokenizer,
    )

    return train_ds, dev_ds


# noinspection PyDefaultArgument
@click.command()
@click.option("--dataset", "-d", type=str, required=True,
              help="The name of the first (or only) dataset e.g. ontonotes etc")
@click.option("--tasks", "-t", type=(str, float, bool), multiple=True, required=True,
              help="We are expected to have a tuple of three elements where each signifies: "
                   "1. a string denoting task name (in coref, ner, pruner) "
                   "2. a float denoting loss weight. if its negative, we ignore the value "
                   "3. a bool signifying if the class should be weighted or not."
                   "Some example of correct: -t coref -1 True -t pruner 3.5 False")
@click.option("--dataset-2", "-d2", type=str,
              help="The name of dataset e.g. ontonotes etc for a secondary task. Optional. ")
@click.option("--tasks_2", "-t2", type=(str, float, bool), default=None, multiple=True,
              help="We are expected to have a tuple of three elements where each signifies: "
                   "1. a string denoting task name (in coref, ner, pruner) "
                   "2. a float denoting loss weight. if its negative, we ignore the value "
                   "3. a bool signifying if the class should be weighted or not."
                   "Some example of correct: -t coref -1 True -t pruner 3.5 False")
# TODO: understand the semantics of sampling ratios
@click.option("--sampling-ratio", "-sr", type=(float, float), default=(1.0, 1.0), multiple=False,
              help="A set of floats signifying sampling ratios. (1.0, 1.0) is normal (fully sample)."
                   "(0.5, 1.0) would only get half instances from the first. ")
@click.option("--epochs", "-e", type=int, default=None, help="Specify the number of epochs for which to train.")
@click.option("--learning-rate", "-lr", type=float, default=DEFAULTS.trainer.learning_rate,
              help="Learning rate. Defaults to 0.005.")
@click.option("--lr-schedule", "-lrs", default=(None, None), type=(str, float),
              help="Write 'gamma' to decay the lr. Add another param to init the hyperparam for this lr schedule." \
                   "TODO: add more recipes here")
@click.option("--encoder", "-enc", type=str, default=None, help="Which BERT model (for now) to load.")
@click.option("--tokenizer", "-tok", type=str, default=None, help="Put in value here in case value differs from enc")
@click.option("--device", "-dv", type=str, default=None, help="The device to use: cpu, cuda, cuda:0, ...")
@click.option('--trim', is_flag=True,
              help="If True, We only consider 50 documents in one dataset. For quick iterations. NOTE:"
                   "if d1, d2 are both provided, documents are trimmed for both.")
@click.option('--debug', is_flag=True,
              help="If True, we may break code where previously we would have paved through regardless. More verbose.")
@click.option('--train-encoder', is_flag=True, default=False,
              help="If enabled, the BERTish encoder is not frozen but trains also.")
@click.option('--filter-candidates-pos', '-filtercp', is_flag=True, default=False,
              help="If true, dataiter ignores those candidates which have verbs in them "
                   "IF the doc has more than 10k candidates.")
@click.option('--save', '-s', is_flag=True, default=False, help="If true, the model is dumped to disk at every epoch.")
@click.option('--resume-dir', default=-1, type=int,
              help="In case you want to continue from where we left off, give the folder number. The lookup will go: "
                   "/models/trained/<dataset combination>/<task combination>/<resume_dir>/model.torch.")
@click.option('--max-span-width', '-msw', type=int, default=DEFAULTS['max_span_width'],
              help="Max subwords to consider when making span. Use carefully. 5 already is too high.")
@click.option('--coref-loss-mean', type=bool, default=DEFAULTS['coref_loss_mean'], is_flag=True,
              help='If True, coref loss will range from -1 to 1, where it normally can go in tens of thousands.')
@click.option('--coref-higher-order', '-cho', type=int, default=DEFAULTS['coref_higher_order'],
              help='Number of times we run the higher order loop. ')
@click.option('--use-pretrained-model', default=None, type=str,
              help="If you want the model parameters (as much as can be loaded) from a particular place on disk,"
                   "maybe from another run for e.g., you want to specify the directory here.")
@click.option('--use-wandb', '-wb', is_flag=True, default=False,
              help="If True, we report this run to WandB")
@click.option('--wandb-comment', '-wbm', type=str, default=None,
              help="If use-wandb is enabled, whatever comment you write will be included in WandB runs.")
@click.option('--wandb-name', '-wbname', type=str, default=None,
              help="You can specify a short name for the run here as well. ")
def run(
        tokenizer: str,
        encoder: str,
        epochs: int,
        dataset: str,
        tasks: List[Tuple[str, float, bool]],
        dataset_2: str,
        tasks_2: List[Tuple[str, float, bool]] = [],
        device: str = "cpu",
        trim: bool = False,
        debug: bool = False,
        train_encoder: bool = False,
        use_wandb: bool = False,
        wandb_comment: str = '',
        wandb_name: str = None,
        filter_candidates_pos: bool = False,
        save: bool = False,
        resume_dir: int = -1,
        use_pretrained_model: str = None,
        lr_schedule: (str, float) = (None, None),
        sampling_ratio: (float, float) = (1.0, 1.0),
        learning_rate: float = DEFAULTS.trainer.learning_rate,
        max_span_width: int = DEFAULTS.max_span_width,
        coref_loss_mean: bool = DEFAULTS.coref_loss_mean,
        coref_higher_order: int = DEFAULTS.coref_higher_order,
):
    # TODO: enable specifying data sampling ratio when we have 2 datasets
    # TODO: implement soft loading the model parameters somehow.

    if not tokenizer:
        tokenizer = encoder

    # If trim OR debug is enabled, we WILL turn the wandb_trial flag on
    wandb_trial = trim or debug

    # If we are to "resume" training things from somewhere, we should also have the save flag enabled
    if resume_dir >= 0:
        save = True

    """
        Sanity Checks
        -> At least one dataset and task 1 are provided
        -> If dataset2 is specified, at least one task2 is specified
        -> If at least one task2 is specified, dataset2 is specified
        -> Datasets, Tasks are all known
    """
    if not dataset or not tasks:
        raise BadParameters(f"Dataset 1, or Task 1 or both are not provided! ")
    if (not tasks_2 and dataset_2) or (tasks_2 and not dataset_2):
        raise BadParameters(f"Either one of dataset or task is not provided for the second domain")
    if dataset not in KNOWN_SPLITS:
        raise BadParameters(f"Unknown dataset: {dataset}.")
    if dataset_2 not in list(KNOWN_SPLITS.keys()) + [None]:
        raise BadParameters(f"Unknown dataset: {dataset_2}")

    _is_multidomain: bool = dataset_2 is not None

    tasks = Tasks.parse(dataset, position='primary', tuples=tasks)
    tasks_2 = Tasks.parse(dataset_2, position='secondary', tuples=tasks_2)

    dir_config, dir_tokenizer, dir_encoder = get_pretrained_dirs(encoder, tokenizer)

    tokenizer = transformers.BertTokenizer.from_pretrained(dir_tokenizer)
    config = SerializedBertConfig(dir_config)
    config.max_span_width = max_span_width
    config.coref_dropout = 0.3
    config.metadata_feature_size = 20
    config.unary_hdim = 3000
    config.binary_hdim = 2000
    config.top_span_ratio = 0.4
    config.max_top_antecedents = 50
    config.device = device
    config.trim = trim
    config.debug = debug
    config.filter_candidates_pos_threshold = DEFAULTS[
        'filter_candidates_pos_threshold'] if filter_candidates_pos else -1
    config.wandb = use_wandb
    config.wandb_comment = wandb_comment
    config.wandb_trial = wandb_trial
    config.coref_loss_mean = coref_loss_mean
    config.uncased = encoder.endswith('uncased')
    config.curdir = str(Path('.').absolute())
    config.coref_higher_order = coref_higher_order
    config.vocab_size = tokenizer.get_vocab().__len__()

    # Make a trainer dict and also
    config.trainer = FancyDict()
    config.trainer.learning_rate = learning_rate
    config.trainer.epochs = epochs
    config.trainer.freeze_encoder = not train_encoder
    config.trainer.lr_schedule = lr_schedule[0]
    config.trainer.lr_schedule_param = lr_schedule[1]
    # config.trainer.adam_beta1

    config = merge_configs(old=DEFAULTS, new=config)

    # if NER is a task, we need to find number of NER classes. We can't have NER in both dataset_a and dataset_b
    if 'ner' in tasks:
        n_classes_ner = get_n_classes(task='ner', dataset=dataset)
        tasks.n_classes_ner = n_classes_ner
    if 'ner' in tasks_2:
        n_classes_ner = get_n_classes(task='ner', dataset=dataset_2)
        tasks_2.n_classes_ner = n_classes_ner

    n_classes_pruner = 2
    tasks.n_classes_pruner = n_classes_pruner
    tasks_2.n_classes_pruner = n_classes_pruner

    # Log the tasks var into config as well
    config.task_1 = tasks
    config.task_2 = tasks_2

    # Make the model
    model = MangoesMTL.from_pretrained(dir_encoder, config=config, **config.to_dict()).to(device)
    # model = BasicMTL.from_pretrained(dir_encoder, config=config, **config.to_dict())
    print("Model params: ", sum([param.nelement() for param in model.parameters()]))

    # Make the optimizer
    # opt_base = torch.optim.Adam
    opt = make_optimizer(
        model=model,
        task_learning_rate=config.trainer.learning_rate,
        freeze_encoder=config.trainer.freeze_encoder,
        base_keyword='bert',
        task_weight_decay=None,
        encoder_learning_rate=config.trainer.encoder_learning_rate,
        encoder_weight_decay=config.trainer.encoder_weight_decay,
        adam_beta1=config.trainer.adam_beta1,
        adam_beta2=config.trainer.adam_beta2,
        adam_epsilon=config.trainer.adam_epsilon,
    )
    scheduler = make_scheduler(opt, lr_schedule[0], lr_schedule[1])

    """
        Prep datasets.
        For both d1 and d2,
            - prep loss scales
            - prep class weights (by instantiating a temp dataiter)
            - make suitable partials.
            
        Then, IF d2 is specified, make a data combiner thing, otherwise just use this partial in the loop.
        Do the same for evaluators.
        
    """

    train_ds, dev_ds = get_dataiter_partials(config, tasks, tokenizer=tokenizer)
    if _is_multidomain:
        # Make a specific single domain multitask dataiter
        train_ds_2, dev_ds_2 = get_dataiter_partials(config, tasks_2, tokenizer=tokenizer)

        # Combine the two single domain dataset to make a multidomain dataiter
        train_ds = partial(MultiDomainDataCombiner, srcs=[train_ds, train_ds_2])
        dev_ds = partial(MultiDomainDataCombiner, srcs=[dev_ds, dev_ds_2])

    # Collect all metrics
    metrics, metrics_2 = [TraceCandidates], []
    if 'ner' in tasks:
        metrics += [NERAcc,
                    partial(NERSpanRecognitionMicro, device=config.device),
                    partial(NERSpanRecognitionMacro, n_classes=tasks.n_classes_ner, device=config.device)]
    if 'pruner' in tasks:
        metrics += [partial(PrunerPRMicro, device=config.device),
                    partial(PrunerPRMacro, n_classes=tasks.n_classes_pruner, device=config.device)]
    if 'coref' in tasks:
        metrics += [CorefBCubed, CorefMUC, CorefCeafe]
    if _is_multidomain:
        metrics_2 += [TraceCandidates]
        if 'ner' in tasks_2:
            metrics_2 += [NERAcc,
                          partial(NERSpanRecognitionMicro, device=config.device),
                          partial(NERSpanRecognitionMacro, n_classes=tasks_2.n_classes_ner, device=config.device)]
        if 'pruner' in tasks_2:
            metrics += [partial(PrunerPRMicro, device=config.device),
                        partial(PrunerPRMacro, n_classes=tasks_2.n_classes_pruner, device=config.device)]
        if 'coref' in tasks_2:
            metrics += [CorefBCubed, CorefMUC, CorefCeafe]

    # Make evaluators
    train_eval = Evaluator(
        predict_fn=model.pred_with_labels,
        dataset_partial=train_ds,
        metrics_primary=metrics,
        metrics_secondary=metrics_2,
        device=device
    )
    dev_eval = Evaluator(
        predict_fn=model.pred_with_labels,
        dataset_partial=dev_ds,
        metrics_primary=metrics,
        metrics_secondary=metrics_2,
        device=device
    )

    # Saving stuff
    if save:
        raise NotImplementedError
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
    config.savedir = str(savedir)

    # Resuming stuff
    if resume_dir >= 0:
        raise NotImplementedError
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
        print(f"Successfully resuming training from Epoch {config.epochs_last_run}")

    # WandB stuff
    if use_wandb:
        if 'wandbid' not in config.to_dict():
            config.wandbid = wandb.util.generate_id()
            save_config = config.to_dict()
            save_config['dataset'] = dataset
            save_config['dataset_2'] = dataset_2
            save_config['tasks'] = list(tasks)
            save_config['tasks_2'] = list(tasks_2)

        wandb.init(project="entitymention-mtl", entity="magnet", notes=wandb_comment, name=wandb_name,
                   id=config.wandbid, resume="allow", group="trial" if wandb_trial or trim else "main")
        wandb.config.update(save_config, allow_val_change=True)

    print(config)
    print("Training commences!")

    training_loop(
        model=model,
        epochs=config.trainer.epochs,
        trn_dl=train_ds,
        forward_fn=model.pred_with_labels,
        device=device,
        train_eval=train_eval,
        dev_eval=dev_eval,
        opt=opt,
        tasks=[tasks, tasks_2] if _is_multidomain else [tasks],
        # This is used only for bookkeeping. We're assuming empty entries in logs are fine.
        flag_wandb=use_wandb,
        flag_save=save,
        save_dir=savedir,
        save_config=save_config,
        epochs_last_run=config.epochs_last_run if hasattr(config, 'epochs_last_run') else 0,
        filter_candidates_len_threshold=int(config.filter_candidates_pos_threshold / config.max_span_width),
        debug=config.debug,
        clip_grad_norm=config.trainer.clip_gradients_norm,
        scheduler=scheduler
    )
    print("potato")


if __name__ == "__main__":
    run()

    # si tu veux executer le code manuellment,
    # max_span_width = 5
    # dataset = 'ontonotes'
    # tasks: List[str] = ['coref', 'pruner']
    # dataset_2: str = None
    # tasks_2: List[str] = []
    # epochs: int = 10
    # learning_rate: float = 0.005
    # encoder: str = "bert-base-uncased"
    # device: str = "cuda"
    # trim: bool = True
    # train_encoder: bool = False
    # ner_unweighted: bool = False
    # pruner_unweighted: bool = False
    # t1_ignore_task: str = None
    # t2_ignore_task: str = None
    # use_wandb: bool = False
    # wandb_comment: str = ''
    # wandb_trial: bool = False
    # filter_candidates_pos: bool = False
    # save: bool = False
    # resume_dir: int = -1
    # use_pretrained_model: str = None
