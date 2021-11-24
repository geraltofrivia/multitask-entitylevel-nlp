"""
    Here be iterators

"""
import pickle
import numpy as np
from abc import ABC
from typing import List, Optional
from pathlib import Path

from config import LOCATIONS as LOC, NPRSEED
from utils.data import Document

np.random.seed(NPRSEED)


class BaseLoader(ABC):

    def __init__(self):

        self.dir: Path = Path()
        self.fnames: List[str] = []
        self._n_: int = -1

    @staticmethod
    def _count_instances_(fdir: Path, fnames: List[str]) -> List[int]:
        """ Return a list of bridging instances in each file by loading each and storing its length """
        n: List[int] = []
        for fname in fnames:
            with (fdir / fname).open('rb') as f:
                bridging_instances_batch: List[Document] = pickle.load(f)

            n.append(len(bridging_instances_batch))

        return n

    def __len__(self) -> int:
        """
            Go through all the files and count the number of instances.
            If you did do this before. don't reopen all files
        """

        if self._n_ < 0:
            self._n_ = sum(self._count_instances_(self.dir, self.fnames))

        return self._n_

    @staticmethod
    def get_fnames(dataset: str, split: str):
        return [fnm for fnm in (LOC.parsed / dataset / split).glob('*.pkl')]


class DataLoader(BaseLoader):
    """
        Provide a dataset name, and split, and we load it for you.
        We do lazy loading so you never have to worry about overwhelming the mem
            unless of course the data is stored in one giant pickle file.
            in which case, we can't help you and please contact the preprocessing department

    """

    def __init__(self, dataset: str, split: str, shuffle: bool = False, ignore_empty_coref: bool=False):
        super().__init__()

        self.dataset = dataset
        self.split = split
        self._shuffle_: bool = shuffle
        self._ignore_empty_coref_: bool = ignore_empty_coref

        self.fnames = self.get_fnames(dataset, split)

        self.data: Optional[List[Document]] = None

    def _count_instances_(self, fdir: Path, fnames: List[str]) -> List[int]:
        """ Return a list of bridging instances in each file by loading each and storing its length """
        n: List[int] = []
        for fname in fnames:
            n_doc = 0
            with (fdir / fname).open('rb') as f:
                for instance in pickle.load(f):

                    if self._ignore_empty_coref_ and instance.coref.isempty:
                        continue

                    n_doc += 1

            n.append(n_doc)

        return n

    def all(self):

        for fname in self.fnames:
            with fname.open('rb') as f:
                data_batch: List[Document] = pickle.load(f)

            if self._shuffle_:
                np.random.shuffle(data_batch)

            for instance in data_batch:

                if self._ignore_empty_coref_ and not instance.coref.isempty:
                    continue

                # instance.finalise()
                yield instance

    def __iter__(self):
        """ This function enables simply calling dl() instead of dl.all(). """
        return self.all()
