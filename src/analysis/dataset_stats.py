"""
    This file tries to give a notion of what all datasets look like.
    Stuff we report:
        - number of documents
        - number of cluster in a document
        - number of tokens in a document
        - number of length of a mention
        - ??
"""
from pprint import pprint
from typing import Union, Iterable

import numpy as np
from mytorch.utils.goodies import FancyDict

# Local imports
from dataiter import DocumentReader
from utils.data import Document, Tasks


class Analyser:

    def __init__(self, label: str, documents: Union[Iterable[Document], DocumentReader]):
        self.data = documents
        self.label = label

        self.run()

    @staticmethod
    def get_document_length(doc: Document):
        return sum([len(sent) for sent in doc.document])

    def report(self, stats: FancyDict):
        print(f" ############################################################ ")
        print(f" # Reporting Statistics on: {self.label:31s} # ")
        print(f" ############################################################ ")
        pprint(stats)

    def run(self):
        stats = FancyDict()

        stats.n_documents = len(self.data)
        stats.avg_len_documents = np.mean([self.get_document_length(doc) for doc in self.data])
        stats.avg_len_documents_ = np.std([self.get_document_length(doc) for doc in self.data])
        stats.avg_len_span_coref, stats.avg_len_span_coref_ = self.get_span_length('coref', self.data)
        stats.avg_num_span_coref, stats.avg_num_span_coref_ = self.get_span_num('coref', self.data)

        self.report(stats)

    @staticmethod
    def get_span_num(task: str, data: Union[Iterable[Document], DocumentReader]):
        nums = []
        for datum in data:
            if task == 'coref':
                nums.append(len(datum.coref.get_all_spans()))
        return np.mean(nums), np.std(nums)

    @staticmethod
    def get_span_length(task: str, data: Union[Iterable[Document], DocumentReader]):
        length = []
        for datum in data:
            if task == 'coref':
                length += [span[1] - span[0] for span in datum.coref.get_all_spans()]
        return np.mean(length), np.std(length)


if __name__ == '__main__':
    # Get a dataset (temp)
    # TODO: wire up click here

    task = Tasks.parse(datasrc='ontonotes', tuples=[('coref', 1.0, True)])
    dr = DocumentReader('ontonotes', 'train', tasks=task)
    Analyser("ontonotes", dr)
