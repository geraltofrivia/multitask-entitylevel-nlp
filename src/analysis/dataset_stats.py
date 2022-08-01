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
        stats.avg_len_span_coref = self.get_span_length('coref', self.data)

        self.report(stats)

    def get_span_length(self, task: str, data: Union[Iterable[Document], DocumentReader]):
        length = []
        for datum in data:
            length += [span[1] - span[0] for span in datum.coref.get_all_spans()]
        return np.mean(length)


if __name__ == '__main__':
    # Get a dataset (temp)
    # TODO: wire up click here

    task = Tasks.parse(datasrc='codicrac-persuasion', tuples=[('coref', 1.0, True)])
    dr = DocumentReader('codicrac-persuasion', 'train')
    Analyser("CODICRAC 22 - Persuasion", dr)
