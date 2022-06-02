import json
import pickle
import warnings
from pathlib import Path
from typing import Iterable, Callable, Union, Optional

import numpy as np
import torch
import wandb
from tqdm.auto import tqdm

# Local imports
try:
    import _pathfix
except ImportError:
    from . import _pathfix
from utils.misc import change_device, weighted_addition_losses
from eval import Evaluator


def aggregate_metrics(inter_epoch: dict, intra_epoch: dict):
    for task_nm in inter_epoch.keys():
        for metric_nm, metric_list in intra_epoch[task_nm].items():
            inter_epoch[task_nm][metric_nm].append(np.mean(metric_list))
    return inter_epoch


def training_loop(
        model: torch.nn.Module,
        epochs: int,
        tasks: Iterable[str],
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
        filter_candidates_len_threshold: int = -1,
        debug: bool = False,
        clip_grad_norm: float = 0.0,
) -> (list, list, list):
    """
    TODO: write about every param

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
    :param filter_candidates_len_threshold: this is LEN threshold, not POS threshold.
        You divide the POS threshold by max span width and send that here.
    :return:
    """
    if flag_save and save_config is None:
        save_config = {}

    train_loss = {task_nm: [] for task_nm in tasks}
    train_metrics = {}
    dev_metrics = {}

    trn_dataset = trn_dl()

    # Epoch level
    for e in range(epochs_last_run + 1, epochs + epochs_last_run + 1):

        # Make data
        # trn_dataset = trn_dl()
        per_epoch_loss = {task_nm: [] for task_nm in tasks}

        # Training (on the train set)
        for i, instance in enumerate(tqdm(trn_dataset)):

            # Reset the gradients.
            opt.zero_grad()

            instance["prep_coref_eval"] = True

            # if there are more than a certain amount of (roughly) candidates, we skip the instance. save mem
            if instance['attention_mask'].view(-1).sum().item() > filter_candidates_len_threshold > 0:
                warnings.warn(f"Skipping {i:5d}. Too many candidates. "
                              f"Input: {instance['attention_mask'].view(-1).sum().item()}.")
                continue

            # Move all input tensors to the right devices
            instance = change_device(instance, device)

            # # DEBUG: if there are more than 9k candidates, skip the instance. save mem.
            # if instance['candidate_starts'].shape[0] > 9000:
            #     warnings.warn(f"Skipping {i:5d}. Too many candidates. "
            #                   f"Input: {instance['input_ids'].shape}."
            #                   f"Spans: {instance['candidate_starts'].shape}")
            #     continue

            # Forward Pass
            outputs = forward_fn(**instance)

            # Calc loss
            loss = weighted_addition_losses(outputs["loss"], instance["tasks"], instance['loss_scales'])

            # Calc gradients
            loss.backward()

            # Clip Gradients
            if clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_([param for group in opt.param_groups for param in group['params']],
                                               clip_grad_norm)

            # Backward Pass
            opt.step()

            # Throw the outputs to the eval benchmark also
            train_eval.update(instance=instance, outputs=outputs)

            for task_nm in instance['tasks']:
                per_epoch_loss[task_nm].append(outputs["loss"][task_nm].item())

        # Evaluation (on the validation set)
        dev_eval.run()


        # Bookkeeping (summarise the train and valid evaluations, and the loss)
        train_metrics = train_eval.aggregate_reports(train_metrics, train_eval.report())
        dev_metrics = dev_eval.aggregate_reports(dev_metrics, dev_eval.report())
        if flag_wandb:
            wandb.log({"train": train_eval.report(), "valid": dev_eval.report()}, step=e)
        for task_nm in tasks:
            train_loss[task_nm].append(np.mean(per_epoch_loss[task_nm]))

            if flag_wandb:
                task_specific_wandb_logs = {"loss": train_loss[task_nm][-1]}
                wandb.log({task_nm: task_specific_wandb_logs}, step=e)

        # Printing
        print(f"\nEpoch: {e:3d}" +
              '\n\t'.join([f" | {task_nm} Loss: {float(np.mean(per_epoch_loss[task_nm])):.5f}\n" +
                           ''.join([f" | {task_nm} Tr_{metric_nm}: {float(metric_vls[-1]):.3f}"
                                    for metric_nm, metric_vls in train_metrics[task_nm].items()]) + '\n' +
                           ''.join([f" | {task_nm} Vl_{metric_nm}: {float(metric_vls[-1]):.3f}"
                                    for metric_nm, metric_vls in dev_metrics[task_nm].items()])
                           # f" | {task_nm} Tr_c: {float(np.mean(per_epoch_tr_acc[task_nm])):.5f}" +
                           # f" | {task_nm} Vl_c: {float(np.mean(per_epoch_vl_acc[task_nm])):.5f}"
                           for task_nm in tasks]))

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
                'optimizer_state_dict': opt.state_dict(),
            }, Path(save_dir) / 'torch.save')
            print(f"Model saved on Epoch {e} at {save_dir}.")

        # Reset eval benches
        train_eval.reset()
        dev_eval.reset()

    return train_metrics, dev_metrics, train_loss
