import wandb
import torch
import warnings
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from eval import Evaluator
from mytorch.utils.goodies import Timer
from typing import Iterable, Callable, Union, Optional
from mytorch.utils.goodies import tosave, mt_save

# Local imports
try:
    import _pathfix
except ImportError:
    from . import _pathfix
from utils.misc import change_device, weighted_addition_losses


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
        loss_scales: torch.tensor,
        flag_wandb: bool = False,
        flag_save: bool = False,
        save_dir: Optional[Path] = None,
        save_params: dict = {}
) -> (list, list, list):
    train_loss = {task_nm: [] for task_nm in tasks}
    train_metrics = {}
    dev_metrics = {}

    # Epoch level
    for e in range(epochs):

        # Make data
        trn_ds = trn_dl()
        per_epoch_loss = {task_nm: [] for task_nm in tasks}

        # Train
        with Timer() as timer:

            # Training (on the train set)
            for i, instance in enumerate(tqdm(trn_ds)):

                # Reset the gradients.
                opt.zero_grad()

                # TODO should we avoid excessive computations of prepping data for evaluating coref (set flag to false)
                instance["prep_coref_eval"] = True

                # Move all input tensors to the right devices
                instance = change_device(instance, device)

                # DEBUG
                if instance['candidate_starts'].shape[0] > 9000:
                    warnings.warn(f"Skipping {i:5d}. Too many candidates. "
                                  f"Input: {instance['input_ids'].shape}."
                                  f"Spans: {instance['candidate_starts'].shape}")
                    continue

                # Forward Pass
                outputs = forward_fn(**instance)
                loss = weighted_addition_losses(outputs["loss"], tasks, loss_scales)
                loss.backward()
                opt.step()

                # Throw the outputs to the eval benchmark also
                train_eval.update(instance=instance, outputs=outputs)

                for task_nm in instance['tasks']:
                    per_epoch_loss[task_nm].append(outputs["loss"][task_nm].item())

                # Try to plug mem leaks
                del loss
                change_device(outputs, 'cpu')
                del outputs
                trn_ds[i] = change_device(instance, 'cpu')

            # Evaluation (on the validation set)
            dev_eval.run()

            del trn_ds

        # Bookkeeping (summarise the train and valid evaluations, and the loss)
        train_metrics = train_eval.aggregate_reports(train_metrics, train_eval.report())
        dev_metrics = train_eval.aggregate_reports(dev_metrics, dev_eval.report())
        if flag_wandb:
            wandb.log({"train": train_eval.report(), "valid": dev_eval.report()}, step=e)
        for task_nm in tasks:
            train_loss[task_nm].append(np.mean(per_epoch_loss[task_nm]))

            if flag_wandb:
                task_specific_wandb_logs = {"loss": train_loss[task_nm][-1]}
                wandb.log({task_nm: task_specific_wandb_logs}, step=e)

        # Printing
        print(f"\nEpoch: {e:3d}" +
              '\n\t'.join([f" | {task_nm} Loss: {float(np.mean(per_epoch_loss[task_nm])):.5f}" +
                           ''.join([f" | {task_nm} Tr_{metric_nm}: {float(metric_vls[-1]):.3f}"
                                    for metric_nm, metric_vls in train_metrics[task_nm].items()]) + '\n\t\t' +
                           ''.join([f" | {task_nm} Vl_{metric_nm}: {float(metric_vls[-1]):.3f}"
                                    for metric_nm, metric_vls in dev_metrics[task_nm].items()])
                           # f" | {task_nm} Tr_c: {float(np.mean(per_epoch_tr_acc[task_nm])):.5f}" +
                           # f" | {task_nm} Vl_c: {float(np.mean(per_epoch_vl_acc[task_nm])):.5f}"
                           for task_nm in tasks]))

        # Saving code
        if flag_save:
            # TODO: add condition to save above a certain metric

            save_content = {
                'torch_stuff': [tosave('model.torch', model.state_dict())],
                'json_stuff': [tosave('config.json', {**save_params, **{'epoch': e}})],
                'pickle_stuff': [tosave('traces.pkl', [train_metrics, dev_metrics, train_loss])]
            }

            mt_save(
                save_dir,
                **save_content
            )
            print(f"Model saved on Epoch {e} at {save_dir}.")

        # Reset eval benches
        train_eval.reset()
        dev_eval.reset()

    return train_metrics, dev_metrics, train_loss
