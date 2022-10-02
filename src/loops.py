import atexit
import json
import pickle
from pathlib import Path
from typing import List, Callable, Union, Optional, Type

import numpy as np
import torch
import wandb
from termcolor import colored
from tqdm.auto import tqdm

# Local imports
try:
    import _pathfix
except ImportError:
    from . import _pathfix
from eval import Evaluator
from utils.misc import change_device, weighted_addition_losses
from utils.exceptions import AnticipateOutOfMemException
from utils.data import Tasks


def aggregate_metrics(inter_epoch: dict, intra_epoch: dict):
    for task_nm in inter_epoch.keys():
        for metric_nm, metric_list in intra_epoch[task_nm].items():
            inter_epoch[task_nm][metric_nm].append(np.mean(metric_list))
    return inter_epoch


def goodbye(loc: Path):
    prefix = colored("loops.py", "green", attrs=['bold'])
    value = colored(f"{loc}", "green", attrs=['bold'])
    print(prefix, f"Find the summary of this run at ", value)


# noinspection PyProtectedMember
def training_loop(
        model: torch.nn.Module,
        epochs: int,
        tasks: List[Tasks],
        opt: torch.optim,
        forward_fn: Callable,
        device: Union[str, torch.device],
        trn_dl: Callable,
        train_eval: Evaluator,
        dev_eval: Evaluator,
        flag_wandb: bool = False,
        flag_save: bool = False,
        save_dir: Optional[Path] = None,
        epochs_last_run: int = 0,
        save_config: dict = None,
        debug: bool = False,
        scheduler_per_epoch: Optional[Type[torch.optim.lr_scheduler._LRScheduler]] = None,
        scheduler_per_iter: Optional[Type[torch.optim.lr_scheduler._LRScheduler]] = None,
        clip_grad_norm: float = 0.0,
) -> (list, list, list):
    """
    TODO: write about every param

    :param debug:
    :param clip_grad_norm:
    :param model:
    :param epochs:
    :param tasks:
    :param opt:
    :param forward_fn:
    :param device:
    :param trn_dl:
    :param train_eval:
    :param dev_eval:
    :param flag_wandb:
    :param flag_save:
    :param save_dir:
    :param epochs_last_run:
    :param save_config:
    :param scheduler: a lr_scheduler object (or none), preconfigured with our optimizer.
    """
    if flag_save and save_config is None:
        save_config = {}

    train_loss = {task_obj.dataset: {task_nm: [] for task_nm in task_obj} for task_obj in tasks}
    train_metrics = {}
    dev_metrics = {}
    skipped = {task_obj.dataset: [] for task_obj in tasks}

    trn_dataset = trn_dl()

    if flag_save:
        atexit.register(goodbye, save_dir)

    bert_params, task_params = model.get_params()

    # Epoch level
    for e in range(epochs_last_run + 1, epochs + epochs_last_run + 1):

        if device == 'cuda':
            torch.cuda.empty_cache()

        # Make data
        # trn_dataset = trn_dl()
        per_epoch_loss = {task_obj.dataset: {task_nm: [] for task_nm in task_obj.names} for task_obj in tasks}
        per_epoch_skipped = {task_obj.dataset: 0 for task_obj in tasks}

        model.train()

        # Training (on the train set)
        for i, instance in enumerate(tqdm(trn_dataset)):

            # Reset the gradients.
            model.zero_grad()

            instance["prep_coref_eval"] = True

            # Move all input tensors to the right devices
            instance = change_device(instance, device)

            # Forward Pass
            try:
                outputs = forward_fn(**instance)
            except AnticipateOutOfMemException:
                per_epoch_skipped[instance["domain"]] += 1
                continue

            # Calc loss
            loss = weighted_addition_losses(outputs["loss"], instance["tasks"], instance['loss_scales'])

            # Calc gradients
            loss.backward()

            # Clip Gradients
            if clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(bert_params, clip_grad_norm)
                torch.nn.utils.clip_grad_norm_(task_params, clip_grad_norm)

            # Backward Pass
            # opt.step()
            for optimizer in opt:
                optimizer.step()

            # Throw the outputs to the eval benchmark also
            train_eval.update(instance=instance, outputs=outputs)

            for task_nm in instance['tasks']:
                per_epoch_loss[instance['domain']][task_nm].append(outputs["loss"][task_nm].item())

            # If LR scheduler is provided, run it
            if scheduler_per_iter:
                scheduler_per_iter[0].step()
                scheduler_per_iter[1].step()

            trn_dataset[i] = change_device(instance, 'cpu')

        model.eval()
        # Evaluation (on the validation set)
        dev_eval.run()

        # If LR scheduler (per epoch) is provided, run it
        if scheduler_per_epoch:
            scheduler_per_epoch[0].step()
            scheduler_per_epoch[1].step()

        # Bookkeeping (summarise the train and valid evaluations, and the loss)
        train_metrics = train_eval.aggregate_reports(train_metrics, train_eval.report())
        dev_metrics = dev_eval.aggregate_reports(dev_metrics, dev_eval.report())
        for task in tasks:
            for task_nm in task:
                train_loss[task.dataset][task_nm].append(np.mean(per_epoch_loss[task.dataset][task_nm]))

            if flag_wandb:
                _loss_logs = {task_nm: {"loss": train_loss[task.dataset][task_nm][-1]} for task_nm in task}
                wandb.log({task.dataset: _loss_logs}, step=e)

        for k in skipped.keys():
            skipped[k].append(per_epoch_skipped[k])
        lrs = [pg['lr'] for pg in opt[0].param_groups] + [pg['lr'] for pg in opt[1].param_groups]
        if flag_wandb:
            wandb.log({"train": train_eval.report(), "valid": dev_eval.report()}, step=e)
            wandb.log({f'lr_{i}': lrs[i] for i in range(len(lrs))}, step=e)
            wandb.log({"skipped": {k: v[-1] for k, v in skipped.items()}})

        # print(train_metrics)

        # Printing
        print(f"\nEpoch: {e:5d}" +
              '\n\t'.join([f" | {task.dataset + task_nm} Loss: "
                           f"{float(np.mean(per_epoch_loss[task.dataset][task_nm])):.5f}\n" +
                           ''.join([f" | {task.dataset + task_nm} Tr_{metric_nm}: {float(metric_vls[-1]):.3f}"
                                    for metric_nm, metric_vls in
                                    train_metrics[task.dataset][task_nm].items()]) + '\n' +
                           ''.join([f" | {task.dataset + task_nm} Vl_{metric_nm}: {float(metric_vls[-1]):.3f}"
                                    for metric_nm, metric_vls in dev_metrics[task.dataset][task_nm].items()])
                           # f" | {task_nm} Tr_c: {float(np.mean(per_epoch_tr_acc[task_nm])):.5f}" +
                           # f" | {task_nm} Vl_c: {float(np.mean(per_epoch_vl_acc[task_nm])):.5f}"
                           for task in tasks for task_nm in task]))

        # Saving code
        if flag_save:
            # TODO: add condition to save above a certain metric

            # Save config
            with (save_dir / 'config.json').open('w+', encoding='utf8') as f:
                json.dump({**save_config, **{'epochs_last_run': e}}, f)

            # Save Traces
            with (save_dir / 'traces.pkl').open('wb+') as f:
                pickle.dump([train_metrics, dev_metrics, train_loss], f)

            # Save Model
            torch.save({
                'epochs_last_run': e,
                'model_state_dict': model.state_dict(),
                'optimizer_bert_state_dict': opt[0].state_dict(),
                'optimizer_task_state_dict': opt[1].state_dict(),
                'scheduler_per_epoch_bert_state_dict': scheduler_per_epoch[
                    0].state_dict() if scheduler_per_epoch else None,
                'scheduler_per_epoch_task_state_dict': scheduler_per_epoch[
                    0].state_dict() if scheduler_per_epoch else None,
                'scheduler_per_iter_bert_state_dict': scheduler_per_iter[
                    0].state_dict() if scheduler_per_iter else None,
                'scheduler_per_iter_task_state_dict': scheduler_per_iter[
                    1].state_dict() if scheduler_per_iter else None,
            }, Path(save_dir) / 'torch.save')
            print(f"Model saved on Epoch {e} at {save_dir}.")

        # Reset eval benches
        train_eval.reset()
        trn_dataset.reset()
        dev_eval.reset()

    return train_metrics, dev_metrics, train_loss
