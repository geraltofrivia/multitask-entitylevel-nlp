import json
import os
import random
from copy import deepcopy
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
from preproc.encode import PreEncoder
from models.multitask import MTLModel
from dataiter import MultiTaskDataIter, MultiDomainDataCombiner
from utils.misc import merge_configs, SerializedBertConfig, safely_pull_config
from config import LOCATIONS as LOC, DEFAULTS, KNOWN_SPLITS, _SEED_ as SEED, SCHEDULER_CONFIG, DOMAIN_HAS_NER_MULTILABEL
from utils.exceptions import BadParameters, UnknownDomainException
from eval import Evaluator, NERAcc, NERSpanRecognitionMicro, PrunerPRMicro, CorefBCubed, CorefMUC, CorefCeafe, \
    TraceCandidates, NERSpanRecognitionMicroMultiLabel, NERMultiLabelAcc, POSPRMacro, POSAcc

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True


def make_optimizer(
        model: MTLModel,
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
        Setup the optimizer.
        TODO: fix this (update with frozen encoder thing; check if the submodules get LR propagated to them properly)
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


# noinspection PyProtectedMember
def make_scheduler(opt, lr_schedule: Optional[str], lr_schedule_val: Optional[float], n_updates: int) \
        -> Optional[Type[torch.optim.lr_scheduler._LRScheduler]]:
    if not lr_schedule:
        return None, None

    if lr_schedule == 'gamma':
        hyperparam = lr_schedule_val if lr_schedule_val >= 0 else SCHEDULER_CONFIG['gamma']['decay_rate']
        lambda_1 = lambda epoch: hyperparam ** epoch
        scheduler_per_epoch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda_1)
        scheduler_per_iter = None
    elif lr_schedule == 'warmup':
        # TODO: model both optimizers here
        warmup_ratio = lr_schedule_val if lr_schedule_val >= 0 else SCHEDULER_CONFIG['warmup']['warmup']
        warmup_steps = int(n_updates * warmup_ratio)

        def lr_lambda_bert(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(
                0.0, float(n_updates - current_step) / float(max(1, n_updates - warmup_steps))
            )

        def lr_lambda_task(current_step):
            return max(0.0, float(n_updates - current_step) / float(max(1, n_updates)))

        scheduler_per_iter = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_task)
        scheduler_per_epoch = None
    else:
        raise BadParameters(f"Unknown LR Schedule Recipe Name - {lr_schedule}")

    if scheduler_per_iter is not None and scheduler_per_epoch is not None:
        raise ValueError(f"Both Scheduler per iter and Scheduler per epoch are non-none. This won't fly.")

    return scheduler_per_epoch, scheduler_per_iter


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


def get_save_parent_dir(parentdir: Path, tasks: Tasks, tasks_2: Optional[Tasks],
                        trial: bool = False) -> Path:
    """
        Normally returns parentdir/dataset+dataset2/'_'.join(sorted(tasks))+'-'+'_'.join(sorted(tasks_2)).
        E.g. if dataset, tasks are ontonotes and ['coref', 'pruner'] and
            dataset_2, tasks_2 are scierc and ['ner'], the output will be
            parentdir/ontonotes_scierc/coref_pruner-ner

            However, if we find that trim flag is active in config, or that the run is going to wandb-trials
            then the output is
                parentdir/trial/dataset+dataset2/'_'.join(sorted(tasks+tasks_2)).
    """

    dataset = tasks.dataset
    dataset_2 = None if tasks_2.isempty() else tasks_2.dataset

    # if dataset_2 is alphabetically before dataset, start with it
    if dataset_2 and dataset_2[0] < dataset[0]:
        dataset_2, dataset = dataset, dataset_2
        tasks_2, tasks = tasks, tasks_2

    dataset_prefix = dataset + '_' + dataset_2 if dataset_2 else dataset
    tasks_prefix = '_'.join(tasks.names)
    if not tasks_2.isempty():
        tasks_prefix += '-'
        tasks_prefix += '_'.join(tasks_2.names)

    if trial:
        return parentdir / 'trial' / dataset_prefix / tasks_prefix
    else:
        return parentdir / dataset_prefix / tasks_prefix


def get_dataiter_partials(
        config: Union[dict, SerializedBertConfig],
        tasks: Tasks,
        tokenizer: transformers.BertTokenizer,
):
    train_split = KNOWN_SPLITS[tasks.dataset].train
    dev_split = KNOWN_SPLITS[tasks.dataset].dev
    if config.train_on_dev:
        try:
            test_split = KNOWN_SPLITS[tasks.dataset].test
        except KeyError:
            raise UnknownDomainException(f"The dataset: {tasks.dataset} does not have a test split. "
                                         f"This can either be because the dataset itself does not have a test split, "
                                         f"or because we haven't configured KNOWN_SPLITS (src/config.py) properly.")
    else:
        test_split = None

    # Load the data
    train_ds = partial(
        MultiTaskDataIter,
        src=tasks.dataset,
        config=config,
        tasks=tasks,
        split=[train_split, dev_split] if config.train_on_dev else train_split,
        tokenizer=tokenizer,
        use_speakers=config.use_speakers
    )
    dev_ds = partial(
        MultiTaskDataIter,
        src=tasks.dataset,
        config=config,
        tasks=tasks,
        split=test_split if config.train_on_dev else dev_split,
        tokenizer=tokenizer,
        use_speakers=config.use_speakers
    )

    return train_ds, dev_ds


# noinspection PyDefaultArgument,PyProtectedMember
@click.group()
@click.pass_context
@click.option("--dataset", "-d", type=str, required=True,
              help="The name of the first (or only) dataset e.g. ontonotes etc")
@click.option("--tasks", "-t", type=(str, float, bool), multiple=True, required=True,
              help="We are expected to have a tuple of three elements where each signifies: "
                   "1. a string denoting task name (in coref, ner, pos, pruner) "
                   "2. a float denoting loss weight. if its negative, we ignore the value "
                   "3. a bool signifying if the class should be weighted or not."
                   "Some example of correct: -t coref -1 True -t pruner 3.5 False")
@click.option("--dataset-2", "-d2", type=str,
              help="The name of dataset e.g. ontonotes etc for a secondary task. Optional. ")
@click.option("--tasks-2", "-t2", type=(str, float, bool), default=None, multiple=True,
              help="We are expected to have a tuple of three elements where each signifies: "
                   "1. a string denoting task name (in coref, ner, pos,  pruner) "
                   "2. a float denoting loss weight. if its negative, we ignore the value "
                   "3. a bool signifying if the class should be weighted or not."
                   "Some example of correct: -t coref -1 True -t pruner 3.5 False")
# TODO: understand the semantics of sampling ratios
@click.option("--sampling-ratio", "-sr", type=(float, float), default=(1.0, 1.0), multiple=False,
              help="A set of floats signifying sampling ratios. (1.0, 1.0) is normal (fully sample)."
                   "(0.5, 1.0) would only get half instances from the first. ")
@click.option("--epochs", "-e", type=int, default=None, help="Specify the number of epochs for which to train.")
@click.option("--learning-rate", "-lr", type=float, default=DEFAULTS.trainer.learning_rate,
              help="lr for task stuff. defaults to 2e-4")
@click.option("--encoder-learning-rate", "-elr", type=float, default=DEFAULTS.trainer.encoder_learning_rate,
              help="lr for encoder (bert stuff). defaults to 1e-5")
@click.option("--lr-schedule", "-lrs", default=(None, None), type=(str, float),
              help="Write 'gamma' to decay the lr. Add another param to init the hyperparam for this lr schedule."
                   "E.g.: `gamma 0.98`. \nTODO: add more recipes here")
@click.option("--encoder", "-enc", type=str, default=None, help="Which BERT model (for now) to load.")
@click.option("--tokenizer", "-tok", type=str, default=None, help="Put in value here in case value differs from enc")
@click.option("--device", "-dv", type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
              help="Force a device- ('cpu'/'cuda'). If not specified, defaults to cuda if available else cpu.")
@click.option('--trim', type=int, default=-1,
              help="If True, We only consider <n> documents in one dataset. NOTE:"
                   "if d1, d2 are both provided, documents are trimmed for both.")
@click.option('--trim-deterministic', is_flag=True,
              help="If given, you will also see first <trim> instances, not randomly shuffled everytime.")
@click.option('--debug', is_flag=True,
              help="If True, we may break code where previously we would have paved through regardless. More verbose.")
@click.option('--train-encoder', type=bool, default=False,
              help="If enabled, the BERTish encoder is not frozen but trains also.")
@click.option('--filter-candidates-pos', '-filtercp', type=bool, default=False,
              help="If true, dataiter ignores those candidates which have verbs in them "
                   "IF the doc has more than 10k candidates.")
@click.option('--max-training-segments', '-mts', type=int, default=DEFAULTS['max_training_segments'],
              help="Specify the num of segments (n*512 word pieces) to keep. Only holds for train split.")
@click.option('--dense-layers', '-dl', type=int, default=DEFAULTS['dense_layers'],
              help="Specify the number of n cross n FFNN layers right after encoder. They're task agnostic.")
@click.option('--unary-hdim', type=int, default=DEFAULTS['unary_hdim'],
              help="Specify the dimensions for unary hdim. Its used in multiple places and may trim some fat")
@click.option('--save', '-s', is_flag=True, default=False, help="If true, the model is dumped to disk at every epoch.")
@click.option('--resume-dir', default=-1, type=int,
              help="In case you want to continue from where we left off, give the folder number. The lookup will go: "
                   "/models/trained/<dataset combination>/<task combination>/<resume_dir>/model.torch.")
@click.option('--max-span-width', '-msw', type=int, default=DEFAULTS['max_span_width'],
              help="Max subwords to consider when making span. Use carefully. 5 already is too high.")
@click.option('--pruner_top_span_ratio', '-ptsr', type=float, default=DEFAULTS['pruner_top_span_ratio'],
              help="A float b/w 0 and 1 which may help decide how many spans to keep post pruning depending also"
                   "on pruner_max_num_spans")
@click.option('--coref-loss-mean', type=bool, default=DEFAULTS['coref_loss_mean'],
              help='If True, coref loss will range from -1 to 1, where it normally can go in tens of thousands.')
@click.option('--coref-higher-order', '-cho', type=str, default=DEFAULTS['coref_higher_order'],
              help='Whether we do cluster merging or something else for higher order aggregation')
@click.option('--coref-depth', type=int, default=DEFAULTS['coref_depth'],
              help="Number of times we run the higher order loop. Defaults to one.")
@click.option('--use-speakers', type=bool, default=True,
              help="If False, we ignore speaker ID info even if we have access to it")
@click.option('--use-pretrained-model', default=None, type=str,
              help="If you want the model parameters (as much as can be loaded) from a particular place on disk,"
                   "maybe from another run for e.g., you want to specify the directory here.")
@click.option('--shared-compressor', '-sc', type=bool, default=False,
              help="If true, the hidden layers turn BERT's hidden embeddings down to 1/3 its size and "
                   "also decreases the unary hdim by a third. "
                   "If this flag is enabled but dense layers are zero, we raise an error.")
@click.option('--use-wandb', '-wb', is_flag=True, default=False,
              help="If True, we report this run to WandB")
@click.option('--wandb-comment', '-wbm', type=str, default=None,
              help="If use-wandb is enabled, whatever comment you write will be included in WandB runs.")
@click.option('--wandb-name', '-wbname', type=str, default=None,
              help="You can specify a short name for the run here as well. ")
@click.option('--wandb-tags', '-wt', type=str, default=None, multiple=True,
              help="Space seperated tags, as many as you want!")
@click.option('--train-on-dev', is_flag=True, help="If enabled, test<-dev & train<-train+dev set.")
def run(
        ctx,  # The ctx obj click uses to pass things around in commands
        tokenizer: str,
        encoder: str,
        epochs: int,
        device: str,
        trim: int,
        trim_deterministic: bool,
        dense_layers: int,
        unary_hdim: int,
        dataset: str,
        tasks: List[Tuple[str, float, bool]],
        dataset_2: str,
        tasks_2: List[Tuple[str, float, bool]],
        debug: bool,
        train_encoder: bool,
        use_wandb: bool,
        wandb_comment: str,
        wandb_name: str,
        wandb_tags: List[str],
        filter_candidates_pos: bool,
        use_speakers: bool,
        shared_compressor: bool,
        save: bool,
        resume_dir: int,
        use_pretrained_model: str,
        lr_schedule: (str, float),
        sampling_ratio: (float, float),
        learning_rate: float,
        encoder_learning_rate: float,
        max_span_width: int,
        max_training_segments: int,
        coref_loss_mean: bool,
        coref_higher_order: str,
        coref_depth: int,
        pruner_top_span_ratio: float,
        train_on_dev: bool
):
    # TODO: enable specifying data sampling ratio when we have 2 datasets
    # TODO: implement soft loading the model parameters somehow.
    # TODO: when train encoder is given we have to upend a lot of things and ignore the cached stuff

    ctx.ensure_object(dict)

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
        raise BadParameters(f"Unknown dataset: {dataset}")
    if dataset_2 not in list(KNOWN_SPLITS.keys()) + [None]:
        raise BadParameters(f"Unknown dataset: {dataset_2}")

    tasks = Tasks.parse(dataset, tuples=tasks)
    tasks_2 = Tasks.parse(dataset_2, tuples=tasks_2)

    _is_multidomain: bool = not tasks_2.isempty()

    """
        At this point we ask whether we want to resume something or we want to have a regular run.
        If resume dir is non negative, it indicates that we ignore every other parameter and infer things from the task
        and saved config alone
    """

    if resume_dir < 0:

        if epochs is None:
            raise BadParameters("You never specified how many epochs you want to train this model."
                                "This is fine when we're resuming something. Not when its a new run.")

        """
            Here we either go for train part of things, or we go to resume part (which just requires this stuff).
        """

        if not tokenizer:
            tokenizer = encoder

        if shared_compressor and dense_layers < 1:
            raise BadParameters(f"You want the shared layers to compress the BERT embeddings but"
                                f"also want to have zero (specifically: {dense_layers}) layers. That's not going to work.")

        # If trim OR debug is enabled, we WILL turn the wandb_trial flag on
        wandb_trial = trim > 0 or debug

        if not use_speakers:
            tasks.n_speakers = -1
            tasks_2.n_speakers = -1

        dir_config, dir_tokenizer, dir_encoder = get_pretrained_dirs(encoder, tokenizer)

        tokenizer = transformers.BertTokenizer.from_pretrained(dir_tokenizer)
        config = SerializedBertConfig(dir_config)

        """
            TODO: HACK: Fix later
        """
        if dir_config == 'SpanBERT/spanbert-large-cased':
            config.hidden_size = 1024
        elif dir_config == 'bert-large-cased' or dir_config == 'bert-large-uncased':
            config.hidden_size = 1024

        # These things are stored to help restoring down the line
        config._config = dir_config
        config._tokenizer = dir_tokenizer
        config._encoder = dir_encoder
        config._sampling_ratio = sampling_ratio

        config.curdir = os.uname().nodename + ':' + os.getcwd()
        config.max_span_width = max_span_width
        config.max_training_segments = max_training_segments
        config.use_speakers = use_speakers
        config.device = device
        config.trim = trim
        config.trim_deterministic = trim_deterministic
        config.debug = debug
        config.skip_instance_after_nspan = DEFAULTS['skip_instance_after_nspan'] if filter_candidates_pos else -1
        config.wandb = use_wandb
        config.wandb_comment = wandb_comment
        config.wandb_trial = wandb_trial
        config.wandb_name = wandb_name
        config.coref_loss_mean = coref_loss_mean
        config.dense_layers = dense_layers
        config.unary_hdim = unary_hdim
        config.shared_compressor = shared_compressor
        config.uncased = encoder.endswith('uncased')
        config.pruner_top_span_ratio = pruner_top_span_ratio
        config.coref_higher_order = coref_higher_order
        config.coref_num_speakers = tasks.n_speakers + tasks_2.n_speakers if config.use_speakers else 0
        config.coref_num_genres = sum(task.n_genres for task in [tasks, tasks_2])
        config.coref_depth = coref_depth
        config.vocab_size = tokenizer.get_vocab().__len__()
        config.freeze_encoder = not train_encoder
        config.train_on_dev = train_on_dev
        if config.shared_compressor:
            config.unary_hdim = unary_hdim // 3

        # Make a trainer dict and also
        config.trainer = FancyDict()
        config.trainer.learning_rate = learning_rate
        config.trainer.encoder_learning_rate = encoder_learning_rate
        config.trainer.epochs = epochs
        config.trainer.lr_schedule = lr_schedule[0]
        config.trainer.lr_schedule_param = lr_schedule[1]
        # config.trainer.adam_beta1

        config = merge_configs(old=DEFAULTS, new=config)

        # Log the tasks var into config as well
        config.task_1 = tasks
        config.task_2 = tasks_2

        # Saving stuff
        if save:
            savedir = get_save_parent_dir(LOC.models, tasks=tasks, tasks_2=tasks_2,
                                          trial=config.trim > 0 or config.wandb_trial)
            savedir.mkdir(parents=True, exist_ok=True)
            savedir = mt_save_dir(parentdir=savedir, _newdir=True)
            config.savedir = str(savedir)
            save_config = config.to_dict()
        else:
            savedir, save_config = None, None

    else:

        # Figure out where we pull the model and everything from
        savedir = get_save_parent_dir(LOC.models, tasks=tasks, tasks_2=tasks_2, trial=trim > 0 or debug)
        savedir = savedir / str(resume_dir)
        assert savedir.exists(), f"No subfolder {resume_dir} in {savedir.parent}. Can not resume!"

        with (savedir / 'config.json').open('r', encoding='utf8') as f:
            config = safely_pull_config(json.load(f))

        # We need to make task objects from the config as well. This is because loss scales, class weights may differ.
        tasks = Tasks(**config.task_1)
        tasks_2 = Tasks(**config.task_2)

        # Pull config, tokenizer and encoder stuff from
        dir_config = config._config
        dir_tokenizer = config._tokenizer
        dir_encoder = config._encoder
        sampling_ratio = config._sampling_ratio

        tokenizer = transformers.BertTokenizer.from_pretrained(dir_tokenizer)
        config = merge_configs(old=SerializedBertConfig(dir_config), new=config)
        save_config = deepcopy(config)

        # There. now we can continue as normal, and will have to interject just once
        # #### when model, optimizer and scheduler are inited

        # Saving stuff
        if save:
            config.savedir = str(savedir)
        else:
            savedir = None

    """
        Speaker ID logic | Genre ID logic
        If ANY of the domains has speaker IDs, and we're using them, we should enable speakers for all datasets.
        This is because there is going to be a shape problem with the slow antecedent scorer.
        So for example, you gave ` -d codicrac-light -d2 ontonotes` (72 speakers and no speaker respectively)
        we will have 73 speakers where first 72 are for Light, and the last one is for Ontonotes (always constant).
    
        The genres also work in the same manner. We concat the dict, and offset values for subsequent domains
    """
    # This is to be given to a MultiDomainDataCombiner IF we are working in a multidomain setting.
    # This is for task 1 and 2 respectively. If you add more datasets, change!
    speaker_offsets = [0, tasks.n_speakers]
    genre_offsets = [0, tasks.n_genres]

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
        train_ds = partial(MultiDomainDataCombiner, srcs=[train_ds, train_ds_2], sampling_ratio=sampling_ratio,
                           speaker_offsets=speaker_offsets, genre_offsets=genre_offsets)
        dev_ds = partial(MultiDomainDataCombiner, srcs=[dev_ds, dev_ds_2], sampling_ratio=sampling_ratio,
                         speaker_offsets=speaker_offsets, genre_offsets=genre_offsets)

    # Init them once to note the length
    len_train = train_ds().__len__()

    """
        Prepare Context Object
        So far, we've done some common stuff. At this point, based on what was invoked, we can change a lot of things.
        You may have ended the command with 'train' in which case the rest of the stuff will happen in the train fn.
        There will be an 'unpacking of context' there. Likewise in all functions decorated with `click.command`.
    """
    # Pack things in context to invoke the right command
    ctx.obj['dir_encoder'] = dir_encoder
    ctx.obj['config'] = config
    ctx.obj['device'] = device
    ctx.obj['tasks'] = tasks
    ctx.obj['tasks_2'] = tasks_2
    ctx.obj['tokenizer'] = tokenizer
    ctx.obj['_is_multidomain'] = _is_multidomain
    ctx.obj['speaker_offsets'] = speaker_offsets
    ctx.obj['genre_offsets'] = genre_offsets
    ctx.obj['save'] = save
    ctx.obj['resume_dir'] = resume_dir
    ctx.obj['dataset'] = dataset
    ctx.obj['dataset_2'] = dataset_2
    ctx.obj['trim'] = trim
    ctx.obj['train_ds'] = train_ds
    ctx.obj['dev_ds'] = dev_ds
    ctx.obj['len_train'] = len_train
    ctx.obj['savedir'] = savedir
    ctx.obj['save_config'] = save_config
    ctx.obj['wandb_tags'] = wandb_tags


@run.command()
@click.pass_context
def train(ctx):
    """
        This is the default function. We make model, make the scheduler and train the model and everything.
    """
    # Unpacking the context (courtesy of click)
    dir_encoder = ctx.obj['dir_encoder']
    config = ctx.obj['config']
    device = ctx.obj['device']
    tasks = ctx.obj['tasks']
    tasks_2 = ctx.obj['tasks_2']
    _is_multidomain = ctx.obj['_is_multidomain']
    save = ctx.obj['save']
    resume_dir = ctx.obj['resume_dir']
    dataset = ctx.obj['dataset']
    dataset_2 = ctx.obj['dataset_2']
    trim = ctx.obj['trim']
    train_ds = ctx.obj['train_ds']
    dev_ds = ctx.obj['dev_ds']
    len_train = ctx.obj['len_train']
    savedir = ctx.obj['savedir']
    save_config = ctx.obj['save_config']
    wandb_tags = ctx.obj['wandb_tags']

    # Make the model
    model = MTLModel(dir_encoder, config=config, coref_false_new_delta=config.trainer.coref_false_new_delta,
                     **config.to_dict() if isinstance(config, SerializedBertConfig) else config).to(device)
    # model = BasicMTL.from_pretrained(dir_encoder, config=config, **config.to_dict())
    n_params = sum([param.nelement() for param in model.parameters()])
    print("Model params: ", n_params)

    # Make the optimizer
    # opt_base = torch.optim.Adam
    opt = make_optimizer(
        model=model,
        task_learning_rate=config.trainer.learning_rate,
        freeze_encoder=config.freeze_encoder,
        base_keyword='bert',
        task_weight_decay=None,
        encoder_learning_rate=config.trainer.encoder_learning_rate,
        encoder_weight_decay=config.trainer.encoder_weight_decay,
        adam_beta1=config.trainer.adam_beta1,
        adam_beta2=config.trainer.adam_beta2,
        adam_epsilon=config.trainer.adam_epsilon,
    )
    scheduler_per_epoch, scheduler_per_iter = make_scheduler(
        opt=opt,
        lr_schedule=config.trainer.lr_schedule,
        lr_schedule_val=config.trainer.lr_schedule_param,
        n_updates=len_train * config.trainer.epochs)

    # Collect all metrics
    metrics = {task.dataset: [] for task in [tasks, tasks_2]}
    for task in [tasks, tasks_2]:

        if task.isempty():
            continue

        metrics[task.dataset] += [TraceCandidates]

        if 'ner' in task:
            metrics[task.dataset] += [
                NERAcc if not task.dataset in DOMAIN_HAS_NER_MULTILABEL else \
                    partial(NERMultiLabelAcc, nc=task.n_classes_ner, threshold=config.ner_threshold),
                NERSpanRecognitionMicro if task.dataset not in DOMAIN_HAS_NER_MULTILABEL else NERSpanRecognitionMicroMultiLabel,
                # partial(NERSpanRecognitionMacro, n_classes=task.n_classes_ner, device=config.device)
            ]
        if 'pruner' in task:
            metrics[task.dataset] += [PrunerPRMicro]
        if 'coref' in task:
            metrics[task.dataset] += [CorefBCubed, CorefMUC, CorefCeafe]
        if 'pos' in task:
            metrics[task.dataset] += [partial(POSPRMacro, n_classes=task.n_classes_pos), POSAcc]

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
        device=device,
        model=model
    )

    # WandB stuff
    if config.wandb:

        if resume_dir < 0:
            # Its a new run
            config.wandbid = wandb.util.generate_id()
            if save_config:
                save_config['wandbid'] = config.wandbid
            wandb_config = config.to_dict()
            wandb_config['dataset'] = dataset
            wandb_config['dataset_2'] = dataset_2
            wandb_config['tasks'] = list(tasks)
            wandb_config['tasks_2'] = list(tasks_2)
            run = wandb.init(project="entitymention-mtl", entity="magnet",
                             notes=config.wandb_comment, name=config.wandb_name,
                             id=config.wandbid, resume="allow",
                             group="trial" if config.wandb_trial or trim > 0 else "main")
            wandb.config.update(wandb_config, allow_val_change=True)

            if wandb_tags:
                run.tags = run.tags + wandb_tags
        else:

            wandb.init(project="entitymention-mtl", entity="magnet",
                       notes=config.wandb_comment, name=config.wandb_name,
                       id=config.wandbid, resume="allow", group="trial" if config.wandb_trial or trim > 0 else "main")

    if resume_dir >= 0:
        """ We're actually resuming a run. So now we need to load params, state dicts"""
        checkpoint = torch.load(savedir / 'torch.save')
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler_per_epoch:
            scheduler_per_epoch.load_state_dict(checkpoint['scheduler_per_epoch_state_dict'])
        if scheduler_per_iter:
            scheduler_per_iter.load_state_dict(checkpoint['scheduler_per_iter_state_dict'])
    else:
        config.params = n_params

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
        flag_wandb=config.wandb,
        flag_save=save,
        save_dir=savedir,
        save_config=save_config,
        epochs_last_run=config.epochs_last_run if hasattr(config, 'epochs_last_run') else 0,
        debug=config.debug,
        clip_grad_norm=config.trainer.clip_gradients_norm,
        scheduler_per_epoch=scheduler_per_epoch,
        scheduler_per_iter=scheduler_per_iter
    )
    print("potato")


@run.command()
@click.pass_context
def encode(ctx):
    """
        Here we aim to encode the datasets based on the BERT model that we're working with.
    """
    # Unpack the context
    dir_encoder = ctx.obj['dir_encoder']
    train_ds = ctx.obj['train_ds']
    dev_ds = ctx.obj['dev_ds']
    device = ctx.obj['device']
    config = ctx.obj['config']

    encoder = PreEncoder(dataset_partial=train_ds, enc_nm=dir_encoder, device=device)
    encoder.run()
    del encoder
    encoder = PreEncoder(dataset_partial=dev_ds, enc_nm=dir_encoder, device=device)
    encoder.run()
    del encoder

    print("Done encoding.")


@run.command()
@click.pass_context
def infer(ctx):
    """ Generate Universal Anaphora compatible predictions and score"""
    raise NotImplementedError


if __name__ == "__main__":
    run()
