"""
    Collection of evaluation functions for different tasks.
"""
import torch
import numpy as np
from tqdm.auto import tqdm
from collections import Counter
from abc import ABC, abstractmethod
from typing import Dict, Callable, List, Union, Optional
from scipy.optimize import linear_sum_assignment as linear_assignment

# Local imports
try:
    import _pathfix
except ImportError:
    from . import _pathfix
from utils.misc import change_device

"""
    Make a overall (macro) eval system. 
    Configure it like you configure the training loop and throw it in the loop. Do not give it the model.
    The model is fed every time the loop is run. 
    
    All the metrics below, all helper functions are situated strategically.
    
    At every point it can return a dict,
        or append to a dict
        or summarise a dict ( across epochs ) 
    
    and has a nice _repr_ function. This is your goal for 09/04/2022!
    
    In a way that can return everything that needs to be returned. At any point of time.
    
"""


class Metric(ABC):

    def __init__(self):
        # If not None, will be concatenated before self.name values.
        self.prefix: Optional[str] = None
        self.values: List[str] = []
        self.task: str = ''

        # Here, we store the interim values which can be returned by a simple 'mean' at the end
        self.logs: Dict[str, List] = {}

    def summarise(self):
        summary = {
            nm: torch.mean(torch.tensor(vl, dtype=torch.float, device='cpu')).item()
            for nm, vl in self.logs.items()
        }
        if self.prefix is not None:
            return {
                self.prefix + '_' + nm: vl for nm, vl in summary.items()
            }
        else:
            return summary

    def __repr__(self):
        return self.task if self.prefix is None else self.task + '_' + self.prefix

    @abstractmethod
    def compute(self, *args, **kwargs):
        """ the fn called with inputs and outputs for each instance. you're expected to store results in self.logs """
        ...

    def reset(self):
        self.logs = Dict[str, List] = {}


class MacroMetric(Metric, ABC):

    def __init__(self, beta=1):
        super().__init__()
        self.prefix = None
        self.values = ['p', 'r', 'f1']

        self.p_num = 0
        self.p_den = 0
        self.r_num = 0
        self.r_den = 0
        self.beta = beta

    @abstractmethod
    def compute(self, *args, **kwargs):
        ...

    @staticmethod
    def f1(p_num, p_den, r_num, r_den, beta=1):
        p = 0 if p_den == 0 else p_num / float(p_den)
        r = 0 if r_den == 0 else r_num / float(r_den)
        return 0 if p + r == 0 else (1 + beta * beta) * p * r / (beta * beta * p + r)

    def get_f1(self):
        return self.f1(self.p_num, self.p_den, self.r_num, self.r_den, beta=self.beta)

    def get_recall(self):
        return 0 if self.r_num == 0 else self.r_num / float(self.r_den)

    def get_precision(self):
        return 0 if self.p_num == 0 else self.p_num / float(self.p_den)

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()

    def get_counts(self):
        return self.p_num, self.p_den, self.r_num, self.r_den

    def reset(self):
        self.p_num = 0
        self.p_den = 0
        self.r_num = 0
        self.r_den = 0

    def summarise(self):
        summary = {
            'p': self.get_precision(),
            'r': self.get_recall(),
            'f1': self.get_f1()
        }
        if self.prefix is not None:
            return {
                self.prefix + '_' + nm: vl for nm, vl in summary.items()
            }
        else:
            return summary


class Evaluator:
    """
        An evaluator takes the model, and upon executing the fn "run",
            - it inits a new dataset instance
            - it iterates through the dataset, and for each instance
                - it takes the predictions and runs through the metrics.

        The metrics micro and metrics macro are passed independently.
        Micro metrics need only work on a instance to instance basis to return a score that can be simply aggregated
        Macro metrics can not make this aggregation independently. So we expect them to return s'thing once at the end.

        The metric each in turn is a class object. The micro ones simply inherit object and
            can be thought of as static fns. The macros can't. The macro objects register the scores internally
            and return the aggregate when signaled.
    """

    def __init__(
            self,
            predict_fn: Callable,
            dataset_partial: Callable,
            metrics: List[Metric],
            device: Union[str, torch.device]
    ):
        """
        :param predict_fn: The function that gives the model outputs (forward class or whatever)
        :param dataset_partial: a partial encapsulating the dataset so that it can be reset whenever needed.
        :param metrics: a list of class objects inheriting class Metric.
            Micro metrics are expected to return their output at every point and can be aggregated by a simple mean
        """
        self.predict_fn = predict_fn
        self.ds_partial = dataset_partial
        self.metrics = {}
        for metric in metrics:
            self.metrics[metric.task] = self.metrics.get(metric.task, []) + [metric]
        self.device = device

        self.results = {}

    def update(self, instance: dict, outputs: dict):
        """
            Depending on the tasks contained in instance['tasks'], invoke different metrics and
                ask them to consider this instance

        :param instance:
        :param outputs:
        :return: None
        """

        for task_nm in instance['tasks']:

            if task_nm == 'coref':
                for metric in self.metrics['coref']:
                    metric.compute(**outputs['coref']['eval'])

            else:
                for metric in self.metrics[task_nm]:
                    metric.compute(**outputs[task_nm])

    def run(self):

        # Init the dataset
        ds = self.ds_partial()

        with torch.no_grad():
            for i, instance in enumerate(tqdm(ds)):
                # Move the instance to the right device
                instance = change_device(instance, self.device)

                # Ensure that data is prepped for coref eval
                instance["prep_coref_eval"] = True

                # Forward Pass
                outputs = self.predict_fn(**instance)

                # Now we can pass these outputs out and collect the metrics
                self.update(instance, outputs)

                # Try to plug mem leaks
                change_device(outputs, 'cpu')
                del outputs
                ds[i] = change_device(instance, 'cpu')

        del ds
        return self.report()

    def report(self):

        if self.results:
            return self.results

        for task_nm, task_metrics in self.metrics.items():
            if task_nm not in self.results:
                self.results[task_nm] = {}

            for metric in task_metrics:
                summary = metric.summarise()
                for nm, vl in summary.items():
                    self.results[task_nm][nm] = vl

        return self.results

    # def __repr__(self):
    #     raise NotImplementedError

    def reset(self):
        for metrics in self.metrics.values():
            for metric in metrics:
                metric.reset()

        self.results = {}

    @staticmethod
    def aggregate_reports(aggregate, current):
        """ expect every value in 'current' to also be there in the aggregate """
        for task_nm, task_metrics in current.items():
            if task_nm not in aggregate:
                aggregate[task_nm] = {}

            for metric_nm, metric_vl in task_metrics.items():
                if metric_nm not in aggregate[task_nm]:
                    aggregate[task_nm][metric_nm] = []
                aggregate[task_nm][metric_nm].append(metric_vl)

        return aggregate


class NERAcc(Metric):

    def __init__(self):
        super().__init__()
        self.values = ['acc', 'acc_nonzero']
        self.task = 'ner'
        self.prefix = None

    def compute(self, logits, labels, *args, **kwargs):
        """
            Does not distinguish b/w invalid spans, and actually annotated spans.
        :param logits: n_spans, n_classes
        :param labels: n_spans
        :return: scalar
        """
        op = {
            'acc': torch.mean((torch.argmax(logits, dim=1) == labels).float()),
            'acc_nonzero': torch.mean((torch.argmax(logits[labels != 0], dim=1) == labels[labels != 0]).float())
        }

        for k, v in op.items():
            self.logs[k] = self.logs.get(k, []) + [v.item()]


class NERSpanRecognitionPR(Metric):

    def __init__(self):
        super().__init__()
        self.values = ['p', 'r']
        self.prefix = 'spanrec'
        self.task = 'ner'

    def compute(self, logits: torch.Tensor, labels: torch.Tensor, *args, **kwargs):
        """
            Treat as binary clf. And find proportion of spans which were correctly recognized as being spans
            (regardless of the label).
        """
        _logits = torch.argmax(logits, dim=1)  # n_spans, 1
        p = torch.sum((_logits > 0).to(float) * (labels > 0).to(float)) / torch.sum((labels > 0).to(float))
        r = torch.sum((_logits > 0).to(float) * (labels > 0).to(float)) / torch.sum((_logits > 0).to(float))
        op = {'p': p, 'r': r}

        for k, v in op.items():
            self.logs[k] = self.logs.get(k, []) + [v.item()]


class PrunerPR(Metric):

    def __init__(self):
        super().__init__()
        self.values = ['p', 'r']
        self.task = 'pruner'

    def compute(self, logits_after_pruning, labels, *args, **kwargs):
        """
        :param logits_after_pruning: n_spans
        :param labels: n_spans
        :return: scalar
        """
        p = torch.sum((logits_after_pruning > 0).to(float) * (labels > 0).to(float)) \
            / torch.sum((labels > 0).to(float))
        r = torch.sum((logits_after_pruning > 0).to(float) * (labels > 0).to(float)) \
            / torch.sum((logits_after_pruning > 0).to(float))
        # TODO: add f1
        op = {'p': p, 'r': r}

        for k, v in op.items():
            self.logs[k] = self.logs.get(k, []) + [v.item()]


# noinspection PyUnusedLocal
class CorefCeafe(MacroMetric):

    def __init__(self, beta=1):
        super().__init__(beta=beta)
        self.prefix = 'ceafe'
        self.task = 'coref'

    def _compute_(self, clusters, gold_clusters):
        clusters = [c for c in clusters if len(c) != 1]
        scores = np.zeros((len(gold_clusters), len(clusters)))
        for i in range(len(gold_clusters)):
            for j in range(len(clusters)):
                scores[i, j] = self._coref_phi4_(gold_clusters=gold_clusters[i], clusters=clusters[j])
        matching = linear_assignment(-scores)
        similarity = sum(scores[matching[0], matching[1]])

        # similarity = sum(scores[matching[:, 0], matching[:, 1]])
        return similarity, len(clusters), similarity, len(gold_clusters)

    def compute(self, clusters, gold_clusters, mention_to_gold, *args, **kwargs):
        pn, pd, rn, rd = self._compute_(clusters, gold_clusters)
        self.p_num += pn
        self.p_den += pd
        self.r_num += rn
        self.r_den += rd

    @staticmethod
    def _coref_phi4_(clusters, gold_clusters, *args, **kwargs):
        return 2 * len([m for m in clusters if m in gold_clusters]) / float(len(clusters) + len(gold_clusters))


class CorefBCubed(MacroMetric):

    def __init__(self):
        super().__init__()
        self.prefix = 'b_cubed'
        self.task = 'coref'

    def compute(self, clusters, gold_clusters, mention_to_predicted, mention_to_gold, *args, **kwargs):
        pn, pd = self._compute_(clusters, mention_to_gold)
        rn, rd = self._compute_(gold_clusters, mention_to_predicted)
        self.p_num += pn
        self.p_den += pd
        self.r_num += rn
        self.r_den += rd

    @staticmethod
    def _compute_(clusters, mention_to_gold):
        num, dem = 0, 0

        for c in clusters:
            if len(c) == 1:
                continue

            gold_counts = Counter()
            correct = 0
            for m in c:
                if m in mention_to_gold:
                    gold_counts[tuple(mention_to_gold[m])] += 1
            for c2, count in gold_counts.items():
                if len(c2) != 1:
                    correct += count * count

            num += correct / float(len(c))
            dem += len(c)

        return num, dem


class CorefMUC(MacroMetric):

    def __init__(self):
        super().__init__()
        self.prefix = 'muc'
        self.task = 'coref'

    @staticmethod
    def _compute_(clusters, mention_to_gold):
        tp, p = 0, 0
        for c in clusters:
            p += len(c) - 1
            tp += len(c)
            linked = set()
            for m in c:
                if m in mention_to_gold:
                    linked.add(mention_to_gold[m])
                else:
                    tp -= 1
            tp -= len(linked)
        return tp, p

    def compute(self, clusters, gold_clusters, mention_to_predicted, mention_to_gold, *args, **kwargs):
        pn, pd = self._compute_(clusters, mention_to_gold)
        rn, rd = self._compute_(gold_clusters, mention_to_predicted)
        self.p_num += pn
        self.p_den += pd
        self.r_num += rn
        self.r_den += rd
