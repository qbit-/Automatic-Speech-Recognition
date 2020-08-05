import abc
from typing import List, Tuple
import numpy as np
import pandas as pd
from tensorflow import keras


class Dataset(keras.utils.Sequence):
    """
    The `Dataset` represents the sequence of samples used for
    Keras models.
    It has a view by the `reference` to sample sources,
    so we do not keep an entire dataset in the memory.

    The class contains two essential methods `len` and `getitem`,
    which are required to use the `keras.utils.Sequence` interface.
    This structure guarantee that the network only trains once on each
    sample per epoch.
    """

    def __init__(self,
                 references: pd.DataFrame,
                 batch_size: int,
                 group_size: int = 1,
                 rank: int = 0):
        """
        :param references: dataframe with paths to data files
        :param batch_size: batch size
        :param group_size: number of independent consumers of data for
         parallel use. Each consumer will get different part
         of the dataset
        :param rank: rank of the consumer in range(0, group_size)
        """
        self._batch_size = batch_size
        self._group_size = group_size
        self._rank = rank

        # slice references if parallel execution is required
        local_len = int(np.floor(len(references.index) / group_size))
        assert local_len > 0, "less than 1 element per worker"
        assert rank in range(group_size), "rank should be in range(0, group_size)"
        start, end = rank * local_len, (rank + 1) * local_len

        self._references = references[start:end].reset_index(
            drop=True, inplace=False)

        self._indices = np.arange(len(self))

    @property
    def indices(self):
        return self._indices

    def __len__(self) -> int:
        """ Indicate the number of batches per epoch. """
        return int(np.floor(len(self._references.index) / self._batch_size))

    def __getitem__(self, index: int) -> Tuple[List[np.ndarray], List[str]]:
        """ Get the batch data. We have an auxiliary index to have more
        control of the order, because basically model uses it sequentially. """
        aux_index = self._indices[index]
        return self.get_batch(aux_index)

    @abc.abstractmethod
    def get_batch(self, index: int) -> Tuple[List[np.ndarray], List[str]]:
        pass

    def shuffle_indices(self):
        """ Set up the order of return batches. """
        np.random.shuffle(self._indices)

        
def map_dataset(dataset, func):
    def decorator(get_batch_func):
        def new_get_batch(i):
            return func(get_batch_func(i))
        return new_get_batch
    
    dataset.get_batch = decorator(dataset.get_batch)
    return dataset


def cache_dataset(dataset):
    dataset.get_batch = lru_cache(max_size=None)(dataset.get_batch)
    return dataset
