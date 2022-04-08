import wandb
import torch
import warnings
import numpy as np
from copy import deepcopy
from tqdm.auto import tqdm
from eval import compute_metrics
from mytorch.utils.goodies import Timer
from typing import Iterable, Callable, Union, Dict

# Local imports
try:
    import _pathfix
except ImportError:
    from . import _pathfix



def aggregate_metrics(inter_epoch: dict, intra_epoch: dict):
    for task_nm in inter_epoch.keys():
        for metric_nm, metric_list in intra_epoch[task_nm].items():
            inter_epoch[task_nm][metric_nm].append(np.mean(metric_list))
    return inter_epoch


def change_device(instance: dict, device: Union[str, torch.device]) -> dict:
    """ Go through every k, v in a dict and change its device (make it recursive) """
    for k, v in instance.items():
        if type(v) is torch.Tensor:
            if 'device' == 'cpu':
                instance[k] = v.detach().to('cpu')
            else:
                instance[k] = v.to(device)
        elif type(v) is dict:
            instance[k] = change_device(v, device)

    return instance


def weighted_addition_losses(losses, tasks, scales):
    # Sort the tasks
    tasks = sorted(deepcopy(tasks))
    stacked = torch.hstack([losses[task_nm] for task_nm in tasks])
    weighted = stacked * scales
    return torch.sum(weighted)


def training_loop(
        epochs: int,
        tasks: Iterable[str],
        opt: torch.optim,
        forward_fn: Callable,
        device: Union[str, torch.device],
        trn_dl: Callable,
        dev_dl: Callable,
        eval_fns: dict,
        loss_scales: torch.tensor,
        use_wandb: bool = False
) -> (list, list, list):
    train_loss = {task_nm: [] for task_nm in tasks}
    train_metrics = {task_nm: {metric_nm: [] for metric_nm in eval_fns[task_nm].keys()} for task_nm in tasks}
    valid_metrics = {task_nm: {metric_nm: [] for metric_nm in eval_fns[task_nm].keys()} for task_nm in tasks}

    # Epoch level
    for e in range(epochs):

        # Make data
        trn_ds = trn_dl()
        dev_ds = dev_dl()

        per_epoch_loss = {task_nm: [] for task_nm in tasks}
        per_epoch_tr_metrics = {task_nm: {metric_nm: [] for metric_nm in eval_fns[task_nm].keys()} for task_nm in tasks}
        per_epoch_vl_metrics = {task_nm: {metric_nm: [] for metric_nm in eval_fns[task_nm].keys()} for task_nm in tasks}

        # Train
        with Timer() as timer:

            # Train Loop
            for i, instance in enumerate(tqdm(trn_ds)):

                # Reset the gradients.
                opt.zero_grad()

                # Avoid excessive computations of prepping data for evaluating coref
                instance["prep_coref_eval"] = False

                # Move all input tensors to the right devices
                instance = change_device(instance, device)

                # DEBUG
                if instance['candidate_starts'].shape[0] > 10000:
                    warnings.warn(f"Skipping {i:5d}. Too many candidates. "
                                  f"Input: {instance['input_ids'].shape}."
                                  f"Spans: {instance['candidate_starts'].shape}")
                    continue

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

                    if task_nm == 'pruner':
                        # For the pruner task, we don't want "logits" we want "logits_after_pruning"
                        instance_metrics = compute_metrics(eval_fns[task_nm],
                                                           logits=outputs[task_nm]["logits_after_pruning"],
                                                           labels=outputs[task_nm]["labels"])
                    elif task_nm == 'ner':
                        instance_metrics = compute_metrics(eval_fns[task_nm],
                                                           logits=outputs[task_nm]["logits"],
                                                           labels=outputs[task_nm]["labels"])
                    else:
                        continue

                    # elif task_nm == 'coref':
                    #     # we use completely different things for coref eval

                    for metric_nm, metric_vl in instance_metrics.items():
                        per_epoch_tr_metrics[task_nm][metric_nm].append(metric_vl)

                # TODO: losses need to be mixed!
                # loss = torch.sum(torch.hstack([outputs["loss"][task_nm] for task_nm in instance['tasks']]))
                loss = weighted_addition_losses(outputs["loss"], tasks, loss_scales)
                loss.backward()
                opt.step()

                # Try to plug mem leaks
                del loss
                del outputs
                trn_ds[i] = change_device(instance, 'cpu')
                # del instance

            # Val
            with torch.no_grad():

                for i, instance in enumerate(tqdm(dev_ds)):

                    # # DEBUG
                    # if not i == 36:
                    #     continue

                    # Move the instance to the right device
                    instance = change_device(instance, device)

                    # Ensure that data is prepped for coref eval
                    instance["prep_coref_eval"] = True

                    # Forward Pass
                    outputs = forward_fn(**instance)

                    for task_nm in instance["tasks"]:

                        if task_nm == 'pruner':
                            # For the pruner task, we don't want "logits" we want "logits_after_pruning"
                            instance_metrics = compute_metrics(eval_fns[task_nm],
                                                               logits=outputs[task_nm]["logits_after_pruning"],
                                                               labels=outputs[task_nm]["labels"])
                        elif task_nm == 'ner':
                            instance_metrics = compute_metrics(eval_fns[task_nm],
                                                               logits=outputs[task_nm]["logits"],
                                                               labels=outputs[task_nm]["labels"])
                        elif task_nm == 'coref':
                            instance_metrics = compute_metrics(eval_fns[task_nm],
                                                               **outputs['coref']['eval'])

                        for metric_nm, metric_vl in instance_metrics.items():
                            per_epoch_vl_metrics[task_nm][metric_nm].append(metric_vl)

                    # Try to plug mem leaks
                    del outputs
                    # del instance

            del trn_ds, dev_ds

        # Bookkeep
        for task_nm in tasks:
            train_loss[task_nm].append(np.mean(per_epoch_loss[task_nm]))
            train_metrics = aggregate_metrics(train_metrics, per_epoch_tr_metrics)
            valid_metrics = aggregate_metrics(valid_metrics, per_epoch_vl_metrics)

            if use_wandb:
                task_specific_wandb_logs = {"loss": train_loss[task_nm][-1]}
                train_metrics_this_epoch_wandb, valid_metrics_this_epoch_wandb = {}, {}
                for metric_nm, metric_vl in train_metrics[task_nm].items():
                    train_metrics_this_epoch_wandb[metric_nm] = train_metrics[task_nm][metric_nm][-1]
                for metric_nm, metric_vl in valid_metrics[task_nm].items():
                    valid_metrics_this_epoch_wandb[metric_nm] = valid_metrics[task_nm][metric_nm][-1]
                task_specific_wandb_logs["train"] = train_metrics_this_epoch_wandb
                task_specific_wandb_logs["valid"] = valid_metrics_this_epoch_wandb
                wandb.log({task_nm: task_specific_wandb_logs}, step=e)

        print(f"\nEpoch: {e:3d}" +
              '\n\t'.join([f" | {task_nm} Loss: {float(np.mean(per_epoch_loss[task_nm])):.5f}" +
                           ''.join([f" | {task_nm} Tr_{metric_nm}: {float(metric_vls[-1]):.3f}"
                                    for metric_nm, metric_vls in train_metrics[task_nm].items()]) + '\n\t\t' +
                           ''.join([f" | {task_nm} Vl_{metric_nm}: {float(metric_vls[-1]):.3f}"
                                    for metric_nm, metric_vls in valid_metrics[task_nm].items()])
                           # f" | {task_nm} Tr_c: {float(np.mean(per_epoch_tr_acc[task_nm])):.5f}" +
                           # f" | {task_nm} Vl_c: {float(np.mean(per_epoch_vl_acc[task_nm])):.5f}"
                           for task_nm in tasks]))

    return train_metrics, valid_metrics, train_loss
