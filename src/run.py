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
from preproc.encode import PreEncoder
from models.multitask import MangoesMTL
from dataiter import MultiTaskDataIter, MultiDomainDataCombiner
from utils.misc import check_dumped_config, merge_configs, SerializedBertConfig
from config import LOCATIONS as LOC, DEFAULTS, KNOWN_SPLITS, _SEED_ as SEED, SCHEDULER_CONFIG, NER_IS_MULTILABEL
from utils.exceptions import ImproperDumpDir, BadParameters
from eval import Evaluator, NERAcc, NERSpanRecognitionMicro, NERSpanRecognitionMacro, \
    PrunerPRMicro, PrunerPRMacro, CorefBCubed, CorefMUC, CorefCeafe, TraceCandidates, NERMultiLabelAcc

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def make_optimizer(
        model: MangoesMTL,
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


def get_save_parent_dir(parentdir: Path, tasks: Tasks, tasks_2: Optional[Tasks],
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
        allow_speaker_ids=not config.ignore_speakers
    )
    dev_ds = partial(
        MultiTaskDataIter,
        src=tasks.dataset,
        config=config,
        tasks=tasks,
        split=KNOWN_SPLITS[tasks.dataset].dev,
        tokenizer=tokenizer,
        allow_speaker_ids=not config.ignore_speakers
    )

    return train_ds, dev_ds


# noinspection PyDefaultArgument
@click.group()
@click.pass_context
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
@click.option("--tasks-2", "-t2", type=(str, float, bool), default=None, multiple=True,
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
              help="Write 'gamma' to decay the lr. Add another param to init the hyperparam for this lr schedule."
                   "E.g.: `gamma 0.98`. \nTODO: add more recipes here")
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
@click.option('--max-training-segments', '-mts', type=int, default=DEFAULTS['max_training_segments'],
              help="Specify the num of segments (n*512 word pieces) to keep. Only holds for train split.")
@click.option('--dense-layers', '-dl', type=int, default=DEFAULTS['dense_layers'],
              help="Specify the number of n cross n FFNN layers right after encoder. They're task agnostic.")
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
@click.option('--ignore-speakers', is_flag=True, help="If True, we ignore speaker ID info even if we have access to it")
@click.option('--use-pretrained-model', default=None, type=str,
              help="If you want the model parameters (as much as can be loaded) from a particular place on disk,"
                   "maybe from another run for e.g., you want to specify the directory here.")
@click.option('--shared-compressor', '-sc', is_flag=True,
              help="If true, the hidden layers turn BERT's hidden embeddings down to 1/3 its size and "
                   "also decreases the unary hdim by a third. "
                   "If this flag is enabled but dense layers are zero, we raise an error.")
@click.option('--use-wandb', '-wb', is_flag=True, default=False,
              help="If True, we report this run to WandB")
@click.option('--wandb-comment', '-wbm', type=str, default=None,
              help="If use-wandb is enabled, whatever comment you write will be included in WandB runs.")
@click.option('--wandb-name', '-wbname', type=str, default=None,
              help="You can specify a short name for the run here as well. ")
def run(
        ctx,  # The ctx obj click uses to pass things around in commands
        tokenizer: str,
        encoder: str,
        epochs: int,
        device: str,
        trim: bool,
        dense_layers: int,
        dataset: str,
        tasks: List[Tuple[str, float, bool]],
        dataset_2: str,
        tasks_2: List[Tuple[str, float, bool]],
        debug: bool,
        train_encoder: bool,
        use_wandb: bool,
        wandb_comment: str,
        wandb_name: str,
        filter_candidates_pos: bool,
        ignore_speakers: bool,
        shared_compressor: bool,
        save: bool,
        resume_dir: int,
        use_pretrained_model: str,
        lr_schedule: (str, float),
        sampling_ratio: (float, float),
        learning_rate: float,
        max_span_width: int,
        max_training_segments: int,
        coref_loss_mean: bool,
        coref_higher_order: int,
):
    # TODO: enable specifying data sampling ratio when we have 2 datasets
    # TODO: implement soft loading the model parameters somehow.
    # TODO: when train encoder is given we have to upend a lot of things and ignore the cached stuff

    ctx.ensure_object(dict)

    if not tokenizer:
        tokenizer = encoder

    if shared_compressor and dense_layers < 1:
        raise BadParameters(f"You want the shared layers to compress the BERT embeddings but"
                            f"also want to have zero (specifically: {dense_layers}) layers. That's not going to work.")

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

    tasks = Tasks.parse(dataset, tuples=tasks)
    tasks_2 = Tasks.parse(dataset_2, tuples=tasks_2)

    if ignore_speakers:
        tasks.n_speakers = -1
        tasks_2.n_speakers = -1

    dir_config, dir_tokenizer, dir_encoder = get_pretrained_dirs(encoder, tokenizer)

    tokenizer = transformers.BertTokenizer.from_pretrained(dir_tokenizer)
    config = SerializedBertConfig(dir_config)
    config.max_span_width = max_span_width
    config.max_training_segments = max_training_segments
    config.ignore_speakers = ignore_speakers
    config.device = device
    config.trim = trim
    config.debug = debug
    config.skip_instance_after_nspan = DEFAULTS[
        'skip_instance_after_nspan'] if filter_candidates_pos else -1
    config.wandb = use_wandb
    config.wandb_comment = wandb_comment
    config.wandb_trial = wandb_trial
    config.coref_loss_mean = coref_loss_mean
    config.dense_layers = dense_layers
    config.shared_compressor = shared_compressor
    config.uncased = encoder.endswith('uncased')
    config.curdir = str(Path('.').absolute())
    config.coref_higher_order = coref_higher_order
    config.coref_num_speakers = 0 if config.ignore_speakers else tasks.n_speakers + tasks_2.n_speakers
    config.vocab_size = tokenizer.get_vocab().__len__()
    config.freeze_encoder = not train_encoder
    if shared_compressor:
        config.unary_hdim = DEFAULTS.unary_hdim // 3

    # Make a trainer dict and also
    config.trainer = FancyDict()
    config.trainer.learning_rate = learning_rate
    config.trainer.epochs = epochs
    config.trainer.lr_schedule = lr_schedule[0]
    config.trainer.lr_schedule_param = lr_schedule[1]
    # config.trainer.adam_beta1

    config = merge_configs(old=DEFAULTS, new=config)

    # Log the tasks var into config as well
    config.task_1 = tasks
    config.task_2 = tasks_2

    """
        Speaker ID logic
        If ANY of the domains has speaker IDs, and we're using them, we should enable speakers for all datasets.
        This is because there is going to be a shape problem with the slow antecedent scorer.
        So for example, you gave ` -d codicrac-light -d2 ontonotes` (72 speakers and no speaker respectively)
        we will have 73 speakers where first 72 are for Light, and the last one is for Ontonotes (always constant).
    """
    # This is to be given to a MultiDomainDataCombiner IF we are working in a multidomain setting.
    # This is for task 1 and 2 respectively. If you add more datasets, change!
    speaker_offsets = [0, tasks.n_speakers]

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
                           speaker_offsets=speaker_offsets)
        dev_ds = partial(MultiDomainDataCombiner, srcs=[dev_ds, dev_ds_2], sampling_ratio=sampling_ratio,
                         speaker_offsets=speaker_offsets)

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
    ctx.obj['lr_schedule'] = lr_schedule
    ctx.obj['tasks'] = tasks
    ctx.obj['tasks_2'] = tasks_2
    ctx.obj['tokenizer'] = tokenizer
    ctx.obj['_is_multidomain'] = _is_multidomain
    ctx.obj['speaker_offsets'] = speaker_offsets
    ctx.obj['save'] = save
    ctx.obj['resume_dir'] = resume_dir
    ctx.obj['use_wandb'] = use_wandb
    ctx.obj['dataset'] = dataset
    ctx.obj['dataset_2'] = dataset_2
    ctx.obj['wandb_comment'] = wandb_comment
    ctx.obj['wandb_name'] = wandb_name
    ctx.obj['wandb_trial'] = wandb_trial
    ctx.obj['trim'] = trim
    ctx.obj['train_ds'] = train_ds
    ctx.obj['dev_ds'] = dev_ds


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
    lr_schedule = ctx.obj['lr_schedule']
    tasks = ctx.obj['tasks']
    tasks_2 = ctx.obj['tasks_2']
    _is_multidomain = ctx.obj['_is_multidomain']
    save = ctx.obj['save']
    resume_dir = ctx.obj['resume_dir']
    use_wandb = ctx.obj['use_wandb']
    dataset = ctx.obj['dataset']
    dataset_2 = ctx.obj['dataset_2']
    wandb_comment = ctx.obj['wandb_comment']
    wandb_name = ctx.obj['wandb_name']
    wandb_trial = ctx.obj['wandb_trial']
    trim = ctx.obj['trim']
    train_ds = ctx.obj['train_ds']
    dev_ds = ctx.obj['dev_ds']

    # Make the model
    model = MangoesMTL(dir_encoder, config=config, **config.to_dict()).to(device)
    # model = BasicMTL.from_pretrained(dir_encoder, config=config, **config.to_dict())
    n_params = sum([param.nelement() for param in model.parameters()])
    print("Model params: ", n_params)
    config.params = n_params

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
    scheduler = make_scheduler(opt, lr_schedule[0], lr_schedule[1])

    # Collect all metrics
    metrics = {task.dataset: [] for task in [tasks, tasks_2]}
    for task in [tasks, tasks_2]:

        if task.isempty():
            continue

        metrics[task.dataset] += [TraceCandidates]

        if 'ner' in task:
            metrics[task.dataset] += [NERAcc if not task.dataset in NER_IS_MULTILABEL else \
                                          partial(NERMultiLabelAcc, nc=task.n_classes_ner,
                                                  threshold=config.ner_threshold),
                                      partial(NERSpanRecognitionMicro, device=config.device),
                                      partial(NERSpanRecognitionMacro, n_classes=task.n_classes_ner,
                                              device=config.device)]
        if 'pruner' in task:
            metrics[task.dataset] += [partial(PrunerPRMicro, device=config.device),
                                      partial(PrunerPRMacro, n_classes=task.n_classes_pruner, device=config.device)]
        if 'coref' in task:
            metrics[task.dataset] += [CorefBCubed, CorefMUC, CorefCeafe]

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

    # Saving stuff
    if save:

        # TODO: check if we want to resume. If so, put the resume dir here instead after checking for consistency etc
        # raise NotImplementedError
        savedir = get_save_parent_dir(LOC.models, tasks=tasks, config=config, tasks_2=tasks_2)
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
        # raise NotImplementedError
        # We are resuming the model
        savedir = mt_save_dir(parentdir=get_save_parent_dir(LOC.models, tasks=tasks, config=config,
                                                            tasks_2=tasks_2), _newdir=False)

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
        if scheduler:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
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
        debug=config.debug,
        clip_grad_norm=config.trainer.clip_gradients_norm,
        scheduler=scheduler
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
