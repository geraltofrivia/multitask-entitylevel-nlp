"""
    Same as run.py but here we work with mangoes code instead.

"""
import json
import random
from functools import partial
from pathlib import Path
from typing import List, Callable, Union, Optional

import click
import numpy as np
import torch
import transformers
import wandb
from mytorch.utils.goodies import mt_save_dir

# Local imports
try:
    import _pathfix
except ImportError:
    from . import _pathfix
from utils.data import Tasks
from loops import training_loop
from models.multitask import BasicMTL
from utils.misc import check_dumped_config
from dataiter import MultiTaskDataIter, MultiDomainDataCombiner
from config import LOCATIONS as LOC, DEFAULTS, KNOWN_SPLITS, LOSS_SCALES, _SEED_ as SEED
from utils.exceptions import ImproperDumpDir, LabelDictNotFound, BadParameters
from mangoes.modeling import BERTForCoreferenceResolution
from eval import Evaluator, NERAcc, NERSpanRecognitionPR, PrunerPR, CorefBCubed, CorefMUC, CorefCeafe, TraceCandidates

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class MangoesCorefWrapper():

    def __init__(self, *args, **kwargs):
        self.model = BERTForCoreferenceResolution.load("bert-base-cased", "SpanBERT/spanbert-base-cased",
                                                       max_top_antecendents=50, max_training_segments=3,
                                                       top_span_ratio=0.4,
                                                       ffnn_hidden_size=3000, coref_dropout=0.3, coref_depth=2,
                                                       use_metadata=False)

    def forward(self, input_ids, attention_mask, sentence_map, coref, **kwargs):

        gold_starts = coref['gold_starts']
        gold_ends = coref['gold_ends']
        cluster_ids = coref['gold_label_values']

        outputs = self.model.model.forward(
            input_ids, attention_mask, sentence_map, gold_starts=gold_starts,
            gold_ends=gold_ends, cluster_ids=cluster_ids, return_dict=True)

        # This here is joe's code to do this. we don't know if its right or not.
        # we're going to assume it works. if this model doesn't work too well either then we start worrying.

        gold_clusters = {}
        for i in range(len(cluster_ids)):
            assert len(cluster_ids) == len(gold_starts) == len(gold_ends)
            cid = cluster_ids[i].item()
            if cid in gold_clusters:
                gold_clusters[cid].append((gold_starts[i].item(),
                                           gold_ends[i].item()))
            else:
                gold_clusters[cid] = [(gold_starts[i].item(),
                                       gold_ends[i].item())]

        gold_clusters = [tuple(v) for v in gold_clusters.values()]
        mention_to_gold = {}
        for c in gold_clusters:
            for mention in c:
                mention_to_gold[mention] = c

        top_indices = torch.argmax(outputs["top_antecedents_score"], dim=-1, keepdim=False)
        ids = outputs["flattened_ids"]
        top_span_starts = outputs["top_span_starts"]
        top_span_ends = outputs["top_span_ends"]
        top_antecedents = outputs["top_antecedents"]
        mention_indices = []
        antecedent_indices = []
        predicted_antecedents = []
        for i in range(len(outputs["top_span_ends"])):
            if top_indices[i] > 0:
                mention_indices.append(i)
                antecedent_indices.append(top_antecedents[i][top_indices[i] - 1].item())
                predicted_antecedents.append(top_indices[i] - 1)

        cluster_sets = []
        for i in range(len(mention_indices)):
            new_cluster = True
            for j in range(len(cluster_sets)):
                if mention_indices[i] in cluster_sets[j] or antecedent_indices[i] in cluster_sets[j]:
                    cluster_sets[j].add(mention_indices[i])
                    cluster_sets[j].add(antecedent_indices[i])
                    new_cluster = False
                    break
            if new_cluster:
                cluster_sets.append({mention_indices[i], antecedent_indices[i]})

        cluster_dicts = []
        clusters = []
        for i in range(len(cluster_sets)):
            cluster_mentions = sorted(list(cluster_sets[i]))
            current_ids = []
            current_start_end = []
            for mention_index in cluster_mentions:
                current_ids.append(ids[top_span_starts[mention_index]:top_span_ends[mention_index] + 1])
                current_start_end.append((top_span_starts[mention_index].item(), top_span_ends[mention_index].item()))
            cluster_dicts.append({"cluster_ids": current_ids})
            clusters.append(tuple(current_start_end))

        mention_to_predicted = {}
        for c in clusters:
            for mention in c:
                mention_to_predicted[mention] = c

        #         coref_evaluator.update(clusters, gold_clusters, mention_to_predicted, mention_to_gold)

        # Now to convert things back to the way our code base needs them

        #         outputs['_clusters'] = clusters
        #         outputs['_gold_clusters'] = gold_clusters
        #         outputs['_mention_to_predicted'] = mention_to_predicted
        #         outputs['_mention_to_gold'] = mention_to_gold
        coref_eval = {
            "clusters": clusters,
            "gold_clusters": gold_clusters,
            "mention_to_predicted": mention_to_predicted,
            "mention_to_gold": mention_to_gold
        }

        outputs['num_candidates'] = outputs['candidate_ends'].shape[0]
        # outputs['coref'] = {}
        outputs['coref'] = coref_eval
        outputs['loss'] = {'coref': outputs['loss']}

        return outputs


def make_optimizer(
        model: BasicMTL,
        base_keyword: str,
        task_weight_decay: Optional[float],
        task_learning_rate: Optional[float],
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_epsilon: float = 1e-8,
        encoder_learning_rate: float = 5e-05,
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


def pick_loss_scale(tasks: Tasks, ignore_task: str):
    key = 'loss_scales_' + '_'.join(sorted(tasks))
    scales = LOSS_SCALES[key]
    if ignore_task:
        ignore_index = tasks.index(ignore_task)
        scales[ignore_index] = 0
    return scales


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
        Normally returns parentdir/dataset+dataset2/'_'.join(sorted(names))+'-'+'_'.join(sorted(tasks_2)).
        E.g. if dataset, names are ontonotes and ['coref', 'pruner'] and
            dataset_2, tasks_2 are scierc and ['ner'], the output will be
            parentdir/ontonotes_scierc/coref_pruner-ner

        However, if we find that trim flag is active in config, or that the run is going to wandb-trials
            then the output is
                parentdir/trial/dataset+dataset2/'_'.join(sorted(names+tasks_2)).
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
        tokenizer: transformers.BertTokenizer,
        ignore_task: str
):
    # Assign loss scales based on task
    loss_scales = pick_loss_scale(tasks, ignore_task=ignore_task)
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

# noinspection PyDefaultArgument
@click.command()
@click.option("--dataset", "-d", type=str, help="The name of dataset e.g. ontonotes etc")
@click.option("--tasks", "-t", type=str, multiple=True,
              help="Multiple values are okay e.g. -t coref -t ner or just one of these", )
@click.option("--dataset-2", "-d2", type=str, help="The name of dataset e.g. ontonotes etc for a secondary thing.")
@click.option("--tasks_2", "-t2", type=str, default=None, multiple=True,
              help="Multiple values are okay e.g. -t2 coref -t2 ner or just one of these", )
@click.option("--epochs", "-e", type=int, default=None, help="Specify the number of epochs for which to train.")
@click.option("--learning-rate", "-lr", type=float, default=DEFAULTS['learning_rate'],
              help="Learning rate. Defaults to 0.005.")
@click.option("--encoder", "-enc", type=str, default=None, help="Which BERT model (for now) to load.")
@click.option("--device", "-dv", type=str, default=None, help="The device to use: cpu, cuda, cuda:0, ...")
@click.option('--trim', is_flag=True,
              help="If True, We only consider 50 documents in one dataset. For quick iterations. ")
@click.option('--debug', is_flag=True,
              help="If True, we may break code where previously we would have paved through regardless. More verbose.")
@click.option('--train-encoder', is_flag=True, default=False,
              help="If enabled, the BERTish encoder is not frozen but trains also.")
@click.option('--ner-unweighted', is_flag=True, default=DEFAULTS['ner_unweighted'],
              help="If True, we do not input priors of classes into Model -> NER CE loss.")
@click.option('--pruner-unweighted', is_flag=True, default=DEFAULTS['pruner_unweighted'],
              help="If True, we do not input priors of classes into Model -> Pruner BCEwL loss.")
@click.option('--t1-ignore-task', default=None, type=str,
              help="Whatever task is mentioned here, we'll set its loss scale to zero. So it does not train.")
@click.option('--t2-ignore-task', default=None, type=str,
              help="Whatever task is mentioned here, we'll set its loss scale to zero. So it does not train.")
@click.option('--use-wandb', '-wb', is_flag=True, default=False,
              help="If True, we report this run to WandB")
@click.option('--wandb-comment', '-wbm', type=str, default=None,
              help="If use-wandb is enabled, whatever comment you write will be included in WandB runs.")
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
def run(
        epochs: int,
        dataset: str,
        tasks: List[str],
        dataset_2: str,
        tasks_2: List[str] = [],
        encoder: str = "bert-base-uncased",
        device: str = "cpu",
        trim: bool = False,
        debug: bool = False,
        train_encoder: bool = False,
        ner_unweighted: bool = False,
        pruner_unweighted: bool = False,
        t1_ignore_task: str = None,
        t2_ignore_task: str = None,
        use_wandb: bool = False,
        wandb_comment: str = '',
        filter_candidates_pos: bool = False,
        save: bool = False,
        resume_dir: int = -1,
        use_pretrained_model: str = None,  # @TODO: integrate this someday
        learning_rate: float = DEFAULTS['learning_rate'],
        max_span_width: int = DEFAULTS['max_span_width'],
        coref_loss_mean: bool = DEFAULTS['coref_loss_mean'],
        coref_higher_order: int = DEFAULTS['coref_higher_order'],
):
    # TODO: enable specifying data sampling ratio when we have 2 datasets
    # TODO: enable specifying loss ratios for different tasks.
    # TODO: implement soft loading the model parameters somehow.

    # If trim OR debug is enabled, we WILL turn the wandb_trial flag on
    wandb_trial = trim or debug

    # If we are to "resume" training things from somewhere, we should also have the save flag enabled
    if resume_dir >= 0:
        save = True

    # Sanity Checks
    if dataset_2:
        if not tasks_2:
            raise BadParameters(f"No tasks specified for dataset 2: {dataset_2}.")
        if dataset_2 not in KNOWN_SPLITS.keys():
            raise BadParameters(f"Unknown dataset: {dataset_2}.")
    if dataset not in KNOWN_SPLITS:
        raise BadParameters(f"Unknown dataset: {dataset}.")
    # If there is overlap in tasks and tasks_2
    if set(tasks).intersection(tasks_2):
        raise BadParameters("Tasks are overlapping between the two sources. That should not happen.")

    raise NotImplementedError("We haven't changed this code to work with multiple domains. "
                              "We'll have to start with changing the click args, and keep going down after that.")
    # Convert task args to a proper tasks obj
    tasks = Tasks(tasks)
    tasks_2 = Tasks(tasks_2)

    dir_config, dir_tokenizer, dir_encoder = get_pretrained_dirs(encoder)

    tokenizer = transformers.BertTokenizer.from_pretrained(dir_tokenizer)
    config = transformers.BertConfig(dir_config)
    config.max_span_width = max_span_width
    config.coref_dropout = 0.3
    config.metadata_feature_size = 20
    config.unary_hdim = 1000
    config.binary_hdim = 2000
    config.top_span_ratio = 0.4
    config.max_top_antecedents = 50
    config.device = device
    config.epochs = epochs
    config.trim = trim
    config.debug = debug
    config.freeze_encoder = not train_encoder
    config.ner_unweighted = ner_unweighted
    config.pruner_unweighted = pruner_unweighted
    config.learning_rate = learning_rate
    config.filter_candidates_pos_threshold = DEFAULTS[
        'filter_candidates_pos_threshold'] if filter_candidates_pos else -1
    config.wandb = use_wandb
    config.wandb_comment = wandb_comment
    config.wandb_trial = wandb_trial
    config.coref_loss_mean = coref_loss_mean
    config.uncased = encoder.endswith('uncased')
    config.curdir = str(Path('.').absolute())
    config.coref_higher_order = coref_higher_order

    # merge all pre-typed config values into this bertconfig obj
    for k, v in DEFAULTS.items():
        try:
            _ = config.__getattribute__(k)
        except AttributeError:
            config.__setattr__(k, v)

    # if NER is a task, we need to find number of NER classes. We can't have NER in both dataset_a and dataset_b
    if 'ner' in tasks:
        n_classes_ner = get_n_classes(task='ner', dataset=dataset)
    elif 'ner' in tasks_2:
        n_classes_ner = get_n_classes(task='ner', dataset=dataset_2)
    else:
        n_classes_ner = 1

    # Make the model
    # model = BasicMTL(dir_encoder, config=config, n_classes_ner=n_classes_ner)
    # print("Model params: ", sum([param.nelement() for param in model.parameters()]))
    model = MangoesCorefWrapper()
    model.model.model.to(config.device)

    opt = make_optimizer(
        model=model.model.model,
        task_learning_rate=config.learning_rate,
        freeze_encoder=config.freeze_encoder,
        base_keyword='bert',
        task_weight_decay=None,
        encoder_learning_rate=config.encoder_learning_rate,
        encoder_weight_decay=config.encoder_weight_decay
    )

    # # Make the optimiser
    # base_keyword = 'bert'
    # task_learn_rate = config.learning_rate
    # weight_decay = 0.0
    # no_decay = ["bias", "LayerNorm.weight"]
    # task_weight_decay = weight_decay
    # adam_beta1 = 0.9
    # adam_beta2 = 0.999
    # adam_epsilon = 1e-8
    # optimizer_grouped_parameters = [
    #     {
    #         "params": [p for n, p in model.model.model.named_parameters() if not any(nd in n for nd in no_decay) and
    #                    base_keyword in n],
    #         "weight_decay": weight_decay,
    #     },
    #     {
    #         "params": [p for n, p in model.model.model.named_parameters() if any(nd in n for nd in no_decay) and
    #                    base_keyword in n],
    #         "weight_decay": 0.0,
    #     },
    #     {
    #         "params": [p for n, p in model.model.model.named_parameters() if not any(nd in n for nd in no_decay) and
    #                    base_keyword not in n],
    #         "weight_decay": task_weight_decay,
    #         "lr": task_learn_rate,
    #     },
    #     {
    #         "params": [p for n, p in model.model.model.named_parameters() if any(nd in n for nd in no_decay) and
    #                    base_keyword not in n],
    #         "weight_decay": 0.0,
    #         "lr": task_learn_rate,
    #     },
    # ]
    #
    # optimizer_cls = transformers.AdamW
    # optimizer_kwargs = {"betas": (adam_beta1, adam_beta2), "eps": adam_epsilon, "lr": config.learning_rate}
    # opt = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    """
        Prep datasets.
        For both d1 and d2,
            - prep loss scales
            - prep class weights (by instantiating a temp dataiter)
            - make suitable partials.

        Then, IF d2 is specified, make a data combiner thing, otherwise just use this partial in the loop.
        Do the same for evaluators.

    """

    train_ds, dev_ds = get_dataiter_partials(config, tasks, dataset=dataset, tokenizer=tokenizer,
                                             ignore_task=t1_ignore_task)

    # Collect all metrics
    metrics = []
    metrics += [TraceCandidates(debug=config.debug)]
    if 'ner' in tasks + tasks_2:
        metrics += [NERAcc(debug=config.debug), NERSpanRecognitionPR(debug=config.debug)]
    if 'pruner' in tasks + tasks_2:
        metrics += [PrunerPR(debug=config.debug)]
    if 'coref' in tasks + tasks_2:
        metrics += [CorefBCubed(debug=config.debug), CorefMUC(debug=config.debug), CorefCeafe(debug=config.debug)]

    if dataset_2 and tasks_2:
        train_ds_2, dev_ds_2 = get_dataiter_partials(config, tasks_2, dataset=dataset_2, tokenizer=tokenizer,
                                                     ignore_task=t2_ignore_task)
        # Make combined iterators since we have two sets of datasets and tasks
        train_ds = partial(MultiDomainDataCombiner, srcs=[train_ds, train_ds_2])
        dev_ds = partial(MultiDomainDataCombiner, srcs=[dev_ds, dev_ds_2])

    # Make evaluators
    train_eval = Evaluator(
        predict_fn=model.forward,
        dataset_partial=train_ds,
        metrics=metrics,
        device=device
    )
    dev_eval = Evaluator(
        predict_fn=model.forward,
        dataset_partial=dev_ds,
        metrics=metrics,
        device=device
    )

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
    config.savedir = str(savedir)

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

    print(config)
    print("Training commences!")

    training_loop(
        model=model,
        epochs=epochs,
        trn_dl=train_ds,
        forward_fn=model.forward,
        device=device,
        train_eval=train_eval,
        dev_eval=dev_eval,
        opt=opt,
        tasks=tasks + tasks_2,  # This is used only for bookkeeping. We're assuming empty entries in logs are fine.
        flag_wandb=use_wandb,
        flag_save=save,
        save_dir=savedir,
        save_config=save_config,
        epochs_last_run=config.epoch if hasattr(config, 'epoch') else 0,
        filter_candidates_len_threshold=int(config.filter_candidates_pos_threshold / config.max_span_width),
        debug=config.debug
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
